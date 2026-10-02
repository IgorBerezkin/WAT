import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from wat.common import CausalConv1d, GLUMerge, RMSNorm, match_pointers


class TreeSearch(nn.Module):
    def __init__(self, d, mode="beam8", heads=4):
        super().__init__()
        self.mode, self.heads = mode, heads
        self.norm = RMSNorm(d)
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_u = nn.Linear(d, d)
        self.W_o = nn.Linear(d, d)
        self.extend(d)

    def extend(self, d):
        self.beta = nn.Parameter(torch.ones(self.heads))

    def split(self, z):
        B, T, D = z.shape
        return z.view(B, T, self.heads, D // self.heads).transpose(1, 2).float()

    def forward(self, x):
        B, T, D = x.shape
        h = self.norm(x)
        q, k, u = self.split(self.W_q(h)), self.split(self.W_k(h)), self.split(self.W_u(h))
        u_next = torch.cat([u[:, :, 1:], torch.zeros_like(u[:, :, :1])], dim=2)
        with torch.autocast("cuda", enabled=False):
            out = self.read(h, q, k, u_next)
        return self.W_o(out.transpose(1, 2).reshape(B, T, D).to(x.dtype))

    def read(self, h, q, k, u_next):
        if self.mode == "search":
            return torch.utils.checkpoint.checkpoint(self._search, q, k, u_next, use_reentrant=False)
        half = torch.float16 if q.is_cuda else q.dtype
        return self._beam(q.to(half), k.to(half), u_next.to(half), int(self.mode[4:] or 8)).float()

    def _search(self, q, k, u_next):
        B, H, T, _ = q.shape
        L = max(1, (T - 1).bit_length())
        K = 1 << L
        if K > T:
            k = torch.cat([k, k.new_zeros(B, H, K - T, k.size(-1))], dim=2)
            u_next = torch.cat([u_next, u_next.new_zeros(B, H, K - T, u_next.size(-1))], dim=2)
        sums = [k]
        while sums[-1].size(2) > 1:
            sums.append(torch.maximum(sums[-1][:, :, 0::2], sums[-1][:, :, 1::2]))
        beta = self.beta.view(1, H, 1).float()
        t = torch.arange(T, device=q.device)
        roots = []
        for m in range(L + 1):
            node = sums[m][:, :, ((t >> m) - 1).clamp(min=0)]
            logit = beta * (q * node).sum(-1)
            roots.append(logit.masked_fill(((t >> m) & 1) == 0, float("-inf")))
        roots = torch.stack(roots, -1)
        roots = torch.where((t > 0).view(1, 1, -1, 1), roots, torch.zeros_like(roots))
        roots = F.log_softmax(roots, -1)
        tc = t.view(-1, 1)
        P = None
        for m in range(L, -1, -1):
            n = torch.arange(K >> m, device=q.device).view(1, -1)
            inside = ((n + 1) << m) <= tc
            parent_inside = (((n >> 1) + 1) << (m + 1)) <= tc if m < L else torch.zeros_like(inside)
            start = roots[..., m:m + 1].expand(B, H, T, K >> m)
            if m < L:
                delta = sums[m][:, :, 1::2] - sums[m][:, :, 0::2]
                z = (beta.unsqueeze(-1) * (q @ delta.transpose(-1, -2))).repeat_interleave(2, dim=-1)
                sign = ((n & 1) * 2 - 1).float()
                down = P.repeat_interleave(2, dim=-1) + F.logsigmoid(z * sign)
            else:
                down = torch.full_like(start, float("-inf"))
            P = torch.where(inside & ~parent_inside, start,
                            torch.where(inside & parent_inside, down, torch.full_like(start, float("-inf"))))
        return torch.exp(P) @ u_next

    def _tree(self, q, k, u_next):
        B, H, T, _ = q.shape
        L = max(1, (T - 1).bit_length())
        K = 1 << L
        if K > T:
            k = torch.cat([k, k.new_zeros(B, H, K - T, k.size(-1))], dim=2)
            u_next = torch.cat([u_next, u_next.new_zeros(B, H, K - T, u_next.size(-1))], dim=2)
        sums = [k]
        while sums[-1].size(2) > 1:
            sums.append(torch.maximum(sums[-1][:, :, 0::2], sums[-1][:, :, 1::2]))
        acc = torch.float32 if q.dtype in (torch.float16, torch.bfloat16) else q.dtype
        beta = self.beta.view(1, H, 1).to(acc)
        t = torch.arange(T, device=q.device)
        roots = []
        for m in range(L + 1):
            node = sums[m][:, :, ((t >> m) - 1).clamp(min=0)]
            roots.append((beta * (q * node).sum(-1).to(acc)).masked_fill(((t >> m) & 1) == 0, float("-inf")))
        roots = torch.stack(roots, -1)
        valid = torch.isfinite(roots)
        roots = F.log_softmax(torch.where((t > 0).view(1, 1, -1, 1), roots, torch.zeros_like(roots)), -1)
        return L, sums, u_next, beta, t, roots.masked_fill(~valid, float("-inf"))

    @staticmethod
    def _rows(table, idx):
        B, H, T, K = idx.shape
        flat = idx.reshape(B, H, T * K, 1).expand(B, H, T * K, table.size(-1))
        return torch.gather(table, 2, flat).view(B, H, T, K, table.size(-1))

    def _beam(self, q, k, u_next, width):
        B, H, T, _ = q.shape
        L, sums, u_next, beta, t, roots = self._tree(q, k, u_next)
        idx, lp = None, None
        for m in range(L, -1, -1):
            cand_idx = [((t >> m) - 1).clamp(min=0).view(1, 1, T, 1).expand(B, H, T, 1)]
            cand_lp = [roots[..., m:m + 1]]
            if idx is not None:
                delta = sums[m][:, :, 1::2] - sums[m][:, :, 0::2]
                z = beta.unsqueeze(-1) * torch.einsum("bhtd,bhtkd->bhtk", q, self._rows(delta, idx)).to(beta.dtype)
                cand_idx += [2 * idx, 2 * idx + 1]
                cand_lp += [lp + F.logsigmoid(-z), lp + F.logsigmoid(z)]
            cand_idx, cand_lp = torch.cat(cand_idx, -1), torch.cat(cand_lp, -1)
            lp, pos = cand_lp.topk(min(width, cand_lp.size(-1)), dim=-1)
            idx = cand_idx.gather(-1, pos)
        found = torch.isfinite(lp)
        weights = torch.softmax(torch.where(found, lp, torch.full_like(lp, -1e9)), -1) * found
        return torch.einsum("bhtk,bhtkd->bhtd", weights.to(u_next.dtype), self._rows(u_next, idx))


class MainBlock(nn.Module):
    def __init__(self, d, max_len=2048, mem="beam8"):
        super().__init__()
        levels = (max_len - 1).bit_length()
        self.conv = CausalConv1d(d, 3)
        self.W_gate = nn.Linear(d, d)
        self.tree_merge = GLUMerge(d)
        self.W_global = nn.Linear(d, d)
        self.ffn = nn.Sequential(nn.Linear(d, d * 4), nn.GELU(), nn.Linear(d * 4, d))
        self.norm_conv = RMSNorm(d)
        self.norm_ffn = RMSNorm(d)
        self.level_gain = nn.Parameter(torch.ones(levels, d))
        self.W_read = nn.Linear(d, d)
        self.level_emb = nn.Parameter(torch.zeros(levels, d))
        self.ctx_norm_m = RMSNorm(d)
        self.memory = TreeSearch(d, mem) if mem else None

    def tree(self, x):
        levels = [x]
        while levels[-1].size(2) > 1:
            B, C, N, D = levels[-1].shape
            pairs = levels[-1].view(B, C, N // 2, 2, D)
            levels.append(self.tree_merge(pairs[:, :, :, 0], pairs[:, :, :, 1]))
        return levels

    def read(self, levels, x):
        q = self.W_read(x)
        out = torch.zeros_like(levels[0])
        for level in range(len(levels) - 1):
            prev = levels[level][:, :, :-1] * self.level_gain[level]
            add = torch.cat([torch.zeros_like(prev[:, :, :1]), prev], dim=2).repeat_interleave(2 ** level, dim=2)
            out = out + 2 * torch.sigmoid(q + self.level_emb[level]) * add
        return out

    def forward(self, x):
        B, T, D = x.shape
        h = self.conv(self.norm_conv(x))
        x = x + h * torch.sigmoid(self.W_gate(h))
        K = 1 << (T - 1).bit_length()
        x_padded = x if K == T else torch.cat([x, x[:, -1:, :].expand(-1, K - T, -1)], dim=1)
        grid = x_padded.view(B, 1, K, D)
        ctx = self.ctx_norm_m(self.read(self.tree(grid), grid).reshape(B, K, D))
        h_ctx = (x_padded + self.W_global(ctx))[:, :T, :]
        x = x + (h_ctx - x.detach()) * 0.5
        if self.memory is not None:
            x = x + self.memory(x)
        return x + self.ffn(self.norm_ffn(x))


class MainBackbone(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_layers=3, max_len=2048, dropout=0.0, mem="beam8",
                 ptr=(4, 8, 16, 32), **kw):
        super().__init__()
        self.embed_dim, self.vocab = embed_dim, vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.ptr = sorted(ptr) if ptr else []
        self.ptr_emb = nn.ModuleList([nn.Embedding(vocab_size + 1, embed_dim) for _ in self.ptr])
        self.input_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([MainBlock(embed_dim, max_len, mem) for _ in range(n_layers)])
        self.layer_dropout = nn.Dropout(dropout)
        self.output_norm = RMSNorm(embed_dim)
        scale = 0.02 / (2 * max(1, n_layers)) ** 0.5
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, scale)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, x):
        h = self.embedding(x)
        for emb, cand in zip(self.ptr_emb, match_pointers(x, self.ptr, self.vocab)):
            h = h + emb(cand)
        h = self.input_dropout(h)
        for layer in self.layers:
            h = self.layer_dropout(layer(h))
        return self.output_norm(h)
