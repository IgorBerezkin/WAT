import contextlib
import math
import os
import random
import time
import urllib.request
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class CausalConv1d(nn.Module):
    def __init__(self, d, kernel_size=3):
        super().__init__()
        self.padding = kernel_size - 1
        self.conv = nn.Conv1d(d, d, kernel_size=kernel_size)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.pad(x, (self.padding, 0))
        return self.conv(x).transpose(1, 2)


class GLUMerge(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.W_val = nn.Linear(2 * d, d)
        self.W_gate = nn.Linear(2 * d, d)
        self.W_res = nn.Linear(2 * d, d)
        self.norm = nn.RMSNorm(d)

    def forward(self, left, right):
        combined = torch.cat([left, right], dim=-1)
        val = self.W_val(combined)
        gate = torch.sigmoid(self.W_gate(combined))
        merged = self.norm(val * gate)
        res_gate = torch.sigmoid(self.W_res(combined))
        residual = (left + right) * 0.5
        return res_gate * merged + (1.0 - res_gate) * residual


class WATBlockX(nn.Module):
    def __init__(self, d, chunk_size=32, ctx_mode="mean", intra=False):
        super().__init__()
        self.d, self.K = d, chunk_size
        self.ctx_mode, self.intra = ctx_mode, intra

        self.conv = CausalConv1d(d, 3)
        self.W_gate = nn.Linear(d, d)
        self.tree_merge = GLUMerge(d)
        self.W_global = nn.Linear(d, d)
        self.ffn = nn.Sequential(nn.Linear(d, d * 4), nn.GELU(),
                                 nn.Linear(d * 4, d))
        self.norm_conv = nn.RMSNorm(d)
        self.norm_ffn = nn.RMSNorm(d)

        if ctx_mode == "prefix_tree":
            self.scan_merge = GLUMerge(d)
        elif ctx_mode == "gated":
            self.W_gr = nn.Linear(2 * d, d)
            self.W_sr = nn.Linear(2 * d, d)
        elif ctx_mode == "attn":
            self.W_q = nn.Linear(d, d)
            self.W_k = nn.Linear(d, d)
            self.W_v = nn.Linear(d, d)
            self.null_summary = nn.Parameter(torch.zeros(1, 1, d))
        if intra:
            self.intra_merge = GLUMerge(d)
            self.W_intra = nn.Linear(d, d)

    def _tree_reduction_all(self, chunks):
        B, C, K, D = chunks.shape
        curr, size = chunks, K
        while size > 1:
            next_size = (size + 1) // 2
            if size % 2 != 0:
                curr = torch.cat([curr, curr[:, :, -1:, :]], dim=2)
            curr = curr.view(B, C, next_size, 2, D)
            curr = self.tree_merge(curr[:, :, :, 0, :], curr[:, :, :, 1, :])
            size = next_size
        return curr.squeeze(2)

    def _ctx_mean(self, s):
        B, C, D = s.shape
        if C == 1:
            return torch.zeros_like(s)
        cumsum = torch.cumsum(s, dim=1)
        counts = torch.arange(1, C + 1, device=s.device,
                              dtype=s.dtype).view(1, -1, 1)
        means = cumsum / counts
        return torch.cat([torch.zeros(B, 1, D, device=s.device, dtype=s.dtype),
                          means[:, :-1, :]], dim=1)

    def _ctx_prefix_tree(self, s):
        B, C, D = s.shape
        curr, step = s, 1
        while step < C:
            merged = self.scan_merge(curr[:, :-step], curr[:, step:])
            curr = torch.cat([curr[:, :step], merged], dim=1)
            step *= 2
        return torch.cat([torch.zeros(B, 1, D, device=s.device, dtype=s.dtype),
                          curr[:, :-1, :]], dim=1)

    def _ctx_gated(self, s):
        B, C, D = s.shape
        S = torch.zeros(B, D, device=s.device, dtype=s.dtype)
        ctxs = [S]
        for i in range(C - 1):
            inp = torch.cat([S, s[:, i]], dim=-1)
            g = torch.sigmoid(self.W_gr(inp))
            cand = torch.tanh(self.W_sr(inp))
            S = g * S + (1.0 - g) * cand
            ctxs.append(S)
        return torch.stack(ctxs, dim=1)

    def _ctx_attn(self, x_padded, s):
        B, Tp, D = x_padded.shape
        C = s.size(1)
        kv = torch.cat([self.null_summary.expand(B, 1, D).to(s.dtype), s], dim=1)
        q = self.W_q(x_padded)
        k, v = self.W_k(kv), self.W_v(kv)
        att = torch.einsum("btd,bcd->btc", q, k) / math.sqrt(D)
        chunk_idx = (torch.arange(Tp, device=x_padded.device) // self.K)
        jj = torch.arange(C + 1, device=x_padded.device)
        valid = (jj.view(1, -1) == 0) | \
                ((jj.view(1, -1) - 1) < chunk_idx.view(-1, 1))
        att = att.masked_fill(~valid.unsqueeze(0), float("-inf"))
        att = F.softmax(att, dim=-1)
        return torch.einsum("btc,bcd->btd", att, v)

    def _intra_prefix(self, x_padded):
        B, Tp, D = x_padded.shape
        C = Tp // self.K
        chunks = x_padded.view(B, C, self.K, D)
        curr, step = chunks, 1
        while step < self.K:
            merged = self.intra_merge(curr[:, :, :-step], curr[:, :, step:])
            curr = torch.cat([curr[:, :, :step], merged], dim=2)
            step *= 2
        return curr.reshape(B, Tp, D)

    def forward(self, x):
        B, T, D = x.shape
        K = self.K
        h = self.norm_conv(x)
        h = self.conv(h)
        h = h * torch.sigmoid(self.W_gate(h))
        x = x + h
        pad_len = (K - T % K) % K
        x_padded = x if pad_len == 0 else torch.cat(
            [x, x[:, -1:, :].expand(-1, pad_len, -1)], dim=1)
        Tp = x_padded.size(1)
        C = Tp // K
        chunks = x_padded.unfold(1, K, K).transpose(2, 3)
        summaries = self._tree_reduction_all(chunks)
        if self.ctx_mode == "attn":
            ctx_pos = self._ctx_attn(x_padded, summaries)
        else:
            ctx_chunk = {"mean": self._ctx_mean,
                         "prefix_tree": self._ctx_prefix_tree,
                         "gated": self._ctx_gated}[self.ctx_mode](summaries)
            ctx_pos = ctx_chunk.unsqueeze(2).expand(-1, -1, K, -1) \
                               .reshape(B, Tp, D)
        h_ctx = x_padded + self.W_global(ctx_pos)
        h_ctx = h_ctx[:, :T, :]
        x = x + (h_ctx - x.detach()) * 0.5
        if self.intra:
            xp = x if pad_len == 0 else torch.cat(
                [x, x[:, -1:, :].expand(-1, pad_len, -1)], dim=1)
            pref = self._intra_prefix(xp)[:, :T, :]
            x = x + self.W_intra(pref)
        h = self.norm_ffn(x)
        x = x + self.ffn(h)
        return x


VARIANTS = OrderedDict(
    v0=dict(ctx_mode="mean", intra=False),
    v1=dict(ctx_mode="prefix_tree", intra=False),
    v2=dict(ctx_mode="gated", intra=False),
    v3=dict(ctx_mode="mean", intra=True),
    v4=dict(ctx_mode="prefix_tree", intra=True),
    v5=dict(ctx_mode="attn", intra=False),
)


class WATBackboneX(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_layers=3, chunk_size=32,
                 max_len=2048, dropout=0.1, ctx_mode="mean", intra=False, **kw):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Embedding(max_len, embed_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [WATBlockX(embed_dim, chunk_size, ctx_mode, intra)
             for _ in range(n_layers)])
        self.layer_dropout = nn.Dropout(dropout)
        self.output_norm = nn.RMSNorm(embed_dim)
        scale = 0.02 / (2 * max(1, n_layers)) ** 0.5
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, scale)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device)
        h = self.embedding(x) + self.pos_encoding(positions)
        h = self.input_dropout(h)
        for layer in self.layers:
            h = layer(h)
            h = self.layer_dropout(h)
        return self.output_norm(h)


class CausalSelfAttention(nn.Module):
    def __init__(self, d, n_heads, dropout=0.1, max_len=2048):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, d // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(d, d * 3)
        self.proj = nn.Linear(d, d)
        self.attn_dropout = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(max_len, max_len, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, max_len, max_len),
                             persistent=False)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = [t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
                   for t in self.qkv(x).chunk(3, dim=-1)]
        att = (q @ k.transpose(-2, -1)) * self.scale
        att = att.masked_fill(~self.mask[:, :, :T, :T], float("-inf"))
        att = self.attn_dropout(F.softmax(att, dim=-1))
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(out)


class TransformerBackbone(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_layers=3, max_len=2048,
                 dropout=0.1, **kw):
        super().__init__()
        self.embed_dim = embed_dim
        n_heads = 1
        for h in (1, 2, 4):
            if embed_dim % h == 0 and embed_dim // h >= 8:
                n_heads = h
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Embedding(max_len, embed_dim)
        self.input_dropout = nn.Dropout(dropout)
        blocks = []
        for _ in range(n_layers):
            b = nn.Module()
            b.norm1 = nn.RMSNorm(embed_dim)
            b.attn = CausalSelfAttention(embed_dim, n_heads, dropout, max_len)
            b.norm2 = nn.RMSNorm(embed_dim)
            b.ffn = nn.Sequential(nn.Linear(embed_dim, embed_dim * 4),
                                  nn.GELU(),
                                  nn.Linear(embed_dim * 4, embed_dim))
            b.drop = nn.Dropout(dropout)
            blocks.append(b)
        self.layers = nn.ModuleList(blocks)
        self.output_norm = nn.RMSNorm(embed_dim)
        scale = 0.02 / (2 * max(1, n_layers)) ** 0.5
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, scale)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device)
        h = self.embedding(x) + self.pos_encoding(positions)
        h = self.input_dropout(h)
        for b in self.layers:
            h = h + b.drop(b.attn(b.norm1(h)))
            h = h + b.drop(b.ffn(b.norm2(h)))
        return self.output_norm(h)


class LSTMBackbone(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_layers=3, dropout=0.1, **kw):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_dim, embed_dim, num_layers=n_layers,
                            batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.output_norm = nn.RMSNorm(embed_dim)

    def forward(self, x):
        h = self.input_dropout(self.embedding(x))
        h, _ = self.lstm(h)
        return self.output_norm(h)


class LMModel(nn.Module):
    def __init__(self, backbone, vocab_size):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.embed_dim, vocab_size)

    def forward(self, x):
        return self.head(self.backbone(x))


class CLSModel(nn.Module):
    def __init__(self, backbone, n_classes, pad_id):
        super().__init__()
        self.backbone, self.pad_id = backbone, pad_id
        self.head = nn.Linear(backbone.embed_dim, n_classes)

    def forward(self, x):
        h = self.backbone(x)
        mask = (x != self.pad_id).unsqueeze(-1).to(h.dtype)
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return self.head(pooled)


class CLSRootModel(nn.Module):
    def __init__(self, backbone, n_classes, pad_id):
        super().__init__()
        self.backbone, self.pad_id = backbone, pad_id
        self.root_merge = GLUMerge(backbone.embed_dim)
        self.head = nn.Linear(backbone.embed_dim * 2, n_classes)

    def forward(self, x):
        h = self.backbone(x)
        mask = (x != self.pad_id).unsqueeze(-1).to(h.dtype)
        mean = (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        curr = h * mask
        while curr.size(1) > 1:
            if curr.size(1) % 2 != 0:
                curr = torch.cat([curr, curr[:, -1:, :]], dim=1)
            curr = self.root_merge(curr[:, 0::2], curr[:, 1::2])
        return self.head(torch.cat([mean, curr[:, 0]], dim=-1))


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def match_embed_dim(make_fn, target):
    best_ed, best_n = 16, float("inf")
    for ed in range(16, 512, 8):
        try:
            n = n_params(make_fn(ed))
        except Exception:
            continue
        if abs(n - target) < abs(best_n - target):
            best_n, best_ed = n, ed
        if n > target * 1.6:
            break
    return best_ed, best_n


@torch.no_grad()
def causality_probe(backbone, vocab):
    model = LMModel(backbone, vocab).eval()
    x = torch.randint(0, vocab, (1, 96))
    base = model(x)
    for p in (10, 40, 70, 95):
        x2 = x.clone()
        x2[0, p] = (x2[0, p] + 7) % vocab
        d = (model(x2) - base).abs().max(dim=-1).values[0]
        if p > 0 and d[:p].max().item() > 1e-4:
            return False, p, d[:p].max().item()
    return True, None, 0.0


def load_shakespeare(data_dir):
    path = os.path.join(data_dir, "shakespeare.txt")
    if not os.path.exists(path):
        os.makedirs(data_dir, exist_ok=True)
        print("  скачиваю TinyShakespeare...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/karpathy/char-rnn/"
            "master/data/tinyshakespeare/input.txt", path)
    text = open(path, encoding="utf-8").read()
    chars = sorted(set(text))
    vocab = {c: i for i, c in enumerate(chars)}
    return np.array([vocab[c] for c in text], dtype=np.int64), len(chars)


class LMDataset(Dataset):
    def __init__(self, data, seq_len, stride):
        self.d = torch.from_numpy(np.ascontiguousarray(data))
        self.seq = seq_len
        self.starts = list(range(0, len(data) - seq_len - 1, stride))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = self.starts[i]
        return self.d[s:s + self.seq], self.d[s + 1:s + self.seq + 1]


def make_copy(n, seq_len, n_mem, seed):
    rng = np.random.RandomState(seed)
    V_CONTENT, NOISE, MARK = 16, 16, 17
    xs = np.full((n, seq_len), NOISE, dtype=np.int64)
    ys = np.full((n, seq_len), -100, dtype=np.int64)
    body = seq_len - n_mem
    for i in range(n):
        pos = np.sort(rng.choice(body, size=n_mem, replace=False))
        toks = rng.randint(0, V_CONTENT, size=n_mem)
        xs[i, pos] = toks
        xs[i, body:] = MARK
        ys[i, body:] = toks
    return torch.from_numpy(xs), torch.from_numpy(ys), 18


PAIRS = {"(": ")", "[": "]", "{": "}"}
FLIP = {"(": ")", ")": "(", "[": "]", "]": "[", "{": "}", "}": "{"}


def gen_balanced(rng, length):
    opens = list(PAIRS.keys())
    seq, stack = [], []
    for i in range(length):
        remaining = length - i
        if not stack:
            c = rng.choice(opens); seq.append(c); stack.append(c)
        elif remaining == len(stack):
            seq.append(PAIRS[stack.pop()])
        else:
            if rng.random() < 0.5:
                c = rng.choice(opens); seq.append(c); stack.append(c)
            else:
                seq.append(PAIRS[stack.pop()])
    return seq


def is_balanced(seq):
    inv = {v: k for k, v in PAIRS.items()}
    stack = []
    for c in seq:
        if c in PAIRS:
            stack.append(c)
        else:
            if not stack or stack.pop() != inv[c]:
                return False
    return not stack


def corrupt_n(rng, seq, n_mut):
    for _ in range(30):
        s2 = list(seq)
        for _ in range(n_mut):
            op = rng.randrange(3)
            i = rng.randrange(len(s2))
            if op == 0:
                s2[i] = rng.choice(list("()[]{}"))
            elif op == 1:
                j = rng.randrange(len(s2))
                s2[i], s2[j] = s2[j], s2[i]
            else:
                s2[i] = FLIP[s2[i]]
        if not is_balanced(s2):
            return s2
    return None


def make_brackets2(n, lo, hi, seed):
    rng = random.Random(seed)
    cmap = {c: i for i, c in enumerate("()[]{}")}
    PAD = 6
    xs, ys, muts = [], [], []
    diffs = [1, 2, 4, 8]
    di = 0
    while len(xs) < n:
        L = rng.randrange(lo // 2, hi // 2 + 1) * 2
        bal = gen_balanced(rng, L)
        if len(xs) % 2 == 0:
            xs.append([cmap[c] for c in bal]); ys.append(1); muts.append(0)
        else:
            m = diffs[di % 4]; di += 1
            bad = corrupt_n(rng, bal, m)
            if bad is None:
                continue
            xs.append([cmap[c] for c in bad]); ys.append(0); muts.append(m)
    return xs, ys, muts, 7, PAD


def make_depth(n, lo, hi, seed):
    rng = random.Random(seed)
    cmap = {c: i for i, c in enumerate("()[]{}")}
    PAD = 6
    pool = []
    while len(pool) < n * 3:
        L = rng.randrange(lo // 2, hi // 2 + 1) * 2
        s = gen_balanced(rng, L)
        depth, d = 0, 0
        for c in s:
            d += 1 if c in PAIRS else -1
            depth = max(depth, d)
        pool.append((s, depth))
    depths = sorted(d for _, d in pool)
    q = [depths[len(depths) // 4], depths[len(depths) // 2],
         depths[3 * len(depths) // 4]]

    def cls(d):
        return 0 if d <= q[0] else 1 if d <= q[1] else 2 if d <= q[2] else 3

    per = n // 4
    got = {0: 0, 1: 0, 2: 0, 3: 0}
    xs, ys = [], []
    for s, d in pool:
        c = cls(d)
        if got[c] < per:
            xs.append([cmap[ch] for ch in s]); ys.append(c); got[c] += 1
        if len(xs) >= per * 4:
            break
    return xs, ys, 7, PAD, q


LO_OPS = ["MAX", "MIN", "MED", "SM"]


def _lo_apply(op, vals):
    if op == "MAX":
        return max(vals)
    if op == "MIN":
        return min(vals)
    if op == "MED":
        return sorted(vals)[(len(vals) - 1) // 2]
    return sum(vals) % 10


def gen_listops(rng, max_depth=6, max_args=4):
    def rec(depth):
        if depth == 0 or rng.random() < 0.3:
            v = rng.randrange(10)
            return [str(v)], v
        op = rng.choice(LO_OPS)
        n = rng.randint(2, max_args)
        toks, vals = ["[", op], []
        for _ in range(n):
            t, v = rec(depth - 1)
            toks += t
            vals.append(v)
        toks.append("]")
        return toks, _lo_apply(op, vals)

    return rec(max_depth)


def make_listops(n, max_len, seed):
    rng = random.Random(seed)
    vocab = {str(i): i for i in range(10)}
    for i, op in enumerate(LO_OPS):
        vocab[op] = 10 + i
    vocab["["], vocab["]"] = 14, 15
    PAD = 16
    xs, ys = [], []
    while len(xs) < n:
        toks, val = gen_listops(rng)
        if not (48 <= len(toks) <= max_len):
            continue
        xs.append([vocab[t] for t in toks])
        ys.append(val)
    return xs, ys, 17, PAD


class PaddedCLSDataset(Dataset):
    def __init__(self, xs, ys, pad_id, max_len):
        self.xs, self.ys, self.pad, self.max_len = xs, ys, pad_id, max_len

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, i):
        x = self.xs[i][:self.max_len]
        x = x + [self.pad] * (self.max_len - len(x))
        return torch.tensor(x), torch.tensor(self.ys[i])


def make_recall(n, seq_len, n_pairs, seed):
    rng = np.random.RandomState(seed)
    N_KEYS, N_VALS = 16, 16
    KEY0, VAL0 = 0, N_KEYS
    NOISE, MARK = 32, 33
    xs = np.full((n, seq_len), NOISE, dtype=np.int64)
    ys = np.full((n, seq_len), -100, dtype=np.int64)
    body = seq_len - 2
    slots = np.arange(0, body - 1, 2)
    for i in range(n):
        keys = rng.permutation(N_KEYS)[:n_pairs]
        vals = rng.randint(0, N_VALS, size=n_pairs)
        pos = rng.choice(len(slots), size=n_pairs, replace=False)
        for k, v, p in zip(keys, vals, slots[pos]):
            xs[i, p] = KEY0 + k
            xs[i, p + 1] = VAL0 + v
        qi = rng.randint(0, n_pairs)
        xs[i, body] = MARK
        xs[i, body + 1] = KEY0 + keys[qi]
        ys[i, body + 1] = VAL0 + vals[qi]
    return torch.from_numpy(xs), torch.from_numpy(ys), 34


def make_sched(opt, total_steps, warmup):
    def fn(s):
        if s < warmup:
            return (s + 1) / warmup
        p = (s - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


def autocast_ctx(device):
    if device.type == "cuda":
        dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast("cuda", dtype=dt)
    return contextlib.nullcontext()


@torch.no_grad()
def evaluate(model, loader, device, loss_kind):
    model.eval()
    nll, corr, tot = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with autocast_ctx(device):
            out = model(x)
        out = out.float()
        if loss_kind == "lm":
            m = (y != -100)
            nll += F.cross_entropy(out.reshape(-1, out.size(-1)),
                                   y.reshape(-1), ignore_index=-100,
                                   reduction="sum").item()
            corr += (out.argmax(-1)[m] == y[m]).sum().item()
            tot += m.sum().item()
        else:
            nll += F.cross_entropy(out, y, reduction="sum").item()
            corr += (out.argmax(-1) == y).sum().item()
            tot += y.numel()
    return corr / max(1, tot), nll / max(1, tot) / math.log(2)


def train_model(model, loaders, device, epochs, lr, loss_kind, name,
                patience=4, log_every=100):
    train_loader, val_loader = loaders
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = epochs * len(train_loader)
    sched = make_sched(opt, total, warmup=max(20, total // 20))
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(device.type == "cuda" and
                         not torch.cuda.is_bf16_supported()))
    best = {"val_acc": 0.0, "val_bpc": float("inf"), "epoch": 0}
    best_state = {k: v.detach().cpu().clone()
                  for k, v in model.state_dict().items()}
    bad = 0
    t00 = time.time()
    for ep in range(epochs):
        model.train()
        t0, run_loss, nstep = time.time(), 0.0, 0
        for bi, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with autocast_ctx(device):
                out = model(x)
                if loss_kind == "lm":
                    loss = F.cross_entropy(out.reshape(-1, out.size(-1)),
                                           y.reshape(-1), ignore_index=-100)
                else:
                    loss = F.cross_entropy(out, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item(); nstep += 1
            if (bi + 1) % log_every == 0:
                el = time.time() - t0
                eta = el / (bi + 1) * (len(train_loader) - bi - 1)
                print(f"    [{name}] ep{ep+1} {bi+1}/{len(train_loader)} "
                      f"loss={run_loss/nstep:.4f} eta={eta:.0f}s", flush=True)
        va, vb = evaluate(model, val_loader, device, loss_kind)
        if va > best["val_acc"] + 1e-4 or vb < best["val_bpc"] - 1e-4:
            best = {"val_acc": va, "val_bpc": vb, "epoch": ep + 1}
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        print(f"  [{name}] ep {ep+1}/{epochs}  loss={run_loss/nstep:.4f}  "
              f"val_acc={va*100:.2f}%  val_bpc={vb:.3f}  ({time.time()-t0:.0f}s)",
              flush=True)
        if bad >= patience:
            print(f"  [{name}] early stop (patience={patience})", flush=True)
            break
    model.load_state_dict(best_state)
    best["train_time_s"] = round(time.time() - t00, 1)
    return best
