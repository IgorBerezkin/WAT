import argparse, json, math, os, random, sys, time, urllib.request
from collections import OrderedDict

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "results", "benchmark", "massive")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


class CausalConv1d(nn.Module):
    def __init__(self, embed_dim: int, kernel_size: int = 3):
        super().__init__()
        self.padding = kernel_size - 1
        self.conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=kernel_size)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.pad(x, (self.padding, 0))
        x = self.conv(x)
        return x.transpose(1, 2)


class WATBlock(nn.Module):
    def __init__(self, embed_dim: int, chunk_size: int = 32):
        super().__init__()
        self.embed_dim = embed_dim
        self.chunk_size = chunk_size
        self.conv = CausalConv1d(embed_dim, kernel_size=3)
        self.W_gate = nn.Linear(embed_dim, embed_dim)
        self.W_merge_val = nn.Linear(embed_dim * 2, embed_dim)
        self.W_merge_gate = nn.Linear(embed_dim * 2, embed_dim)
        self.W_res_gate = nn.Linear(embed_dim * 2, embed_dim)
        self.tree_norm = nn.RMSNorm(embed_dim)
        self.W_global = nn.Linear(embed_dim, embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4), nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim))
        self.norm_conv = nn.RMSNorm(embed_dim)
        self.norm_ffn = nn.RMSNorm(embed_dim)

    def _glu_merge(self, left, right):
        combined = torch.cat([left, right], dim=-1)
        val = self.W_merge_val(combined)
        gate = torch.sigmoid(self.W_merge_gate(combined))
        merged = self.tree_norm(val * gate)
        res_gate = torch.sigmoid(self.W_res_gate(combined))
        residual = (left + right) * 0.5
        return res_gate * merged + (1.0 - res_gate) * residual

    def _tree_reduction_all(self, chunks):
        B, C, K, D = chunks.shape
        curr = chunks
        size = K
        while size > 1:
            next_size = (size + 1) // 2
            if size % 2 != 0:
                curr = torch.cat([curr, curr[:, :, -1:, :]], dim=2)
            curr = curr.view(B, C, next_size, 2, D)
            left = curr[:, :, :, 0, :]
            right = curr[:, :, :, 1, :]
            curr = self._glu_merge(left, right)
            size = next_size
        return curr.squeeze(2)

    def _build_global_ctx(self, summaries, batch_size, embed_dim):
        n_chunks = summaries.size(1)
        device = summaries.device
        if n_chunks == 1:
            return torch.zeros(batch_size, 1, embed_dim, device=device)
        cumsum = torch.cumsum(summaries, dim=1)
        counts = torch.arange(1, n_chunks + 1, device=device,
                              dtype=summaries.dtype).view(1, -1, 1)
        means = cumsum / counts
        ctx = torch.cat([torch.zeros(batch_size, 1, embed_dim, device=device,
                                     dtype=summaries.dtype),
                         means[:, :-1, :]], dim=1)
        return ctx

    def forward(self, x):
        B, T, D = x.shape
        K = self.chunk_size
        h = self.norm_conv(x)
        h = self.conv(h)
        h = h * torch.sigmoid(self.W_gate(h))
        x = x + h
        pad_len = (K - T % K) % K
        x_padded = x if pad_len == 0 else torch.cat(
            [x, x[:, -1:, :].expand(-1, pad_len, -1)], dim=1)
        T_padded = x_padded.size(1)
        n_chunks = T_padded // K
        chunks = x_padded.unfold(1, K, K).transpose(2, 3)
        summaries = self._tree_reduction_all(chunks)
        global_ctx = self._build_global_ctx(summaries, B, D)
        ctx_expanded = global_ctx.unsqueeze(2).expand(-1, -1, K, -1) \
                                 .reshape(B, T_padded, D)
        h_ctx = x_padded + self.W_global(ctx_expanded)
        h_ctx = h_ctx[:, :T, :]
        x = x + (h_ctx - x.detach()) * 0.5
        h = self.norm_ffn(x)
        x = x + self.ffn(h)
        return x


class WATBackbone(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_layers=2, chunk_size=32,
                 max_len=2048, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Embedding(max_len, embed_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [WATBlock(embed_dim, chunk_size) for _ in range(n_layers)])
        self.layer_dropout = nn.Dropout(dropout)
        self.output_norm = nn.RMSNorm(embed_dim)
        init_scaled(self, n_layers)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device)
        h = self.embedding(x) + self.pos_encoding(positions)
        h = self.input_dropout(h)
        for layer in self.layers:
            h = layer(h)
            h = self.layer_dropout(h)
        return self.output_norm(h)


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout=0.1, max_len=2048):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(max_len, max_len, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, max_len, max_len),
                             persistent=False)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
                   for t in qkv]
        att = (q @ k.transpose(-2, -1)) * self.scale
        att = att.masked_fill(~self.mask[:, :, :T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout=0.1, max_len=2048):
        super().__init__()
        self.norm1 = nn.RMSNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, n_heads, dropout, max_len)
        self.norm2 = nn.RMSNorm(embed_dim)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, embed_dim * 4), nn.GELU(),
                                 nn.Linear(embed_dim * 4, embed_dim))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class TransformerBackbone(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_layers=2, max_len=2048,
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
        self.layers = nn.ModuleList(
            [TransformerBlock(embed_dim, n_heads, dropout, max_len)
             for _ in range(n_layers)])
        self.output_norm = nn.RMSNorm(embed_dim)
        init_scaled(self, n_layers)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device)
        h = self.embedding(x) + self.pos_encoding(positions)
        h = self.input_dropout(h)
        for layer in self.layers:
            h = layer(h)
        return self.output_norm(h)


class LSTMBackbone(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_layers=2, dropout=0.1, **kw):
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


_OFFICIAL_MAMBA = False
try:
    from mamba_ssm import Mamba as _OfficialMamba
    _OFFICIAL_MAMBA = True
except Exception:
    _OFFICIAL_MAMBA = False


def ssd_scan(x, dt, A_log, Bm, Cm, chunk=32):
    b, T, d = x.shape
    N = Bm.shape[-1]
    a = -torch.exp(A_log.float())
    dt = F.softplus(dt.float())
    logdec = dt * a
    xin = x.float() * dt
    Bm = Bm.float(); Cm = Cm.float()
    y = torch.empty(b, T, d, device=x.device, dtype=torch.float32)
    S = torch.zeros(b, d, N, device=x.device, dtype=torch.float32)
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        Q = e - s
        lq = logdec[:, s:e]
        P = lq.cumsum(1)
        Bq, Cq, xq = Bm[:, s:e], Cm[:, s:e], xin[:, s:e]
        y_inter = torch.einsum("bqn,bdn->bqd", Cq, S) * P.exp()
        M = torch.einsum("bqn,bpn->bqp", Cq, Bq)
        dec = P.unsqueeze(2) - P.unsqueeze(1)
        tri = torch.ones(Q, Q, device=x.device, dtype=torch.bool).tril()
        dec = dec.masked_fill(~tri.view(1, Q, Q, 1), float("-inf")).exp()
        y_intra = torch.einsum("bqp,bqpd,bpd->bqd", M, dec, xq)
        y[:, s:e] = y_inter + y_intra
        wdec = (P[:, -1].unsqueeze(1) - P).exp()
        S = P[:, -1].exp().unsqueeze(-1) * S + \
            torch.einsum("bqd,bqn->bdn", wdec * xq, Bq)
    return y.to(x.dtype)


class MambaBlockMinimal(nn.Module):
    def __init__(self, embed_dim, d_state=16, expand=2, d_conv=4):
        super().__init__()
        d_inner = expand * embed_dim
        self.d_inner = d_inner
        self.norm = nn.RMSNorm(embed_dim)
        self.in_proj = nn.Linear(embed_dim, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, groups=d_inner,
                                padding=d_conv - 1)
        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, d_inner, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(
            d_inner, 1).mean(dim=1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, embed_dim, bias=False)
        with torch.no_grad():
            u = torch.rand(d_inner) * (0.1 - 1e-3) + 1e-3
            self.dt_proj.bias.copy_(u + torch.log(-torch.expm1(-u)))

    def forward(self, x):
        res = x
        x = self.norm(x)
        xz = self.in_proj(x)
        xs, z = xz.chunk(2, dim=-1)
        T = xs.size(1)
        xs = self.conv1d(xs.transpose(1, 2))[:, :, :T].transpose(1, 2)
        xs = F.silu(xs)
        bcd = self.x_proj(xs)
        N = (bcd.shape[-1] - 1) // 2
        Bm, Cm, dt0 = bcd[..., :N], bcd[..., N:2 * N], bcd[..., 2 * N:]
        dt = self.dt_proj(dt0)
        y = ssd_scan(xs, dt, self.A_log, Bm, Cm)
        y = y + self.D * xs
        y = y * F.silu(z)
        return res + self.out_proj(y)


class MambaBackbone(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_layers=2, dropout=0.1, **kw):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.input_dropout = nn.Dropout(dropout)
        if _OFFICIAL_MAMBA:
            class _Blk(nn.Module):
                def __init__(self, d):
                    super().__init__()
                    self.norm = nn.RMSNorm(d)
                    self.mixer = _OfficialMamba(d_model=d, d_state=16,
                                                d_conv=4, expand=2)
                def forward(self, x):
                    return x + self.mixer(self.norm(x))
            self.layers = nn.ModuleList([_Blk(embed_dim)
                                         for _ in range(n_layers)])
        else:
            self.layers = nn.ModuleList([MambaBlockMinimal(embed_dim)
                                         for _ in range(n_layers)])
        self.layer_dropout = nn.Dropout(dropout)
        self.output_norm = nn.RMSNorm(embed_dim)

    def forward(self, x):
        h = self.input_dropout(self.embedding(x))
        for layer in self.layers:
            h = layer(h)
            h = self.layer_dropout(h)
        return self.output_norm(h)


def init_scaled(model, n_layers):
    scale = 0.02 / (2 * max(1, n_layers)) ** 0.5
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, scale)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)


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
        self.backbone = backbone
        self.pad_id = pad_id
        self.head = nn.Linear(backbone.embed_dim, n_classes)

    def forward(self, x):
        h = self.backbone(x)
        mask = (x != self.pad_id).unsqueeze(-1).to(h.dtype)
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return self.head(pooled)


BACKBONES = OrderedDict(
    wat=WATBackbone, transformer=TransformerBackbone,
    lstm=LSTMBackbone, mamba=MambaBackbone)


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def match_embed_dim(cls, vocab, target, n_layers, max_len):
    best_ed, best_n = 16, float("inf")
    for ed in range(16, 512, 8):
        try:
            n = n_params(cls(vocab, ed, n_layers=n_layers, max_len=max_len))
        except Exception:
            continue
        if abs(n - target) < abs(best_n - target):
            best_n, best_ed = n, ed
        if n > target * 1.6:
            break
    return best_ed, best_n


def load_shakespeare():
    path = os.path.join(DATA_DIR, "shakespeare.txt")
    if not os.path.exists(path):
        print("  скачиваю TinyShakespeare...")
        url = ("https://raw.githubusercontent.com/karpathy/char-rnn/"
               "master/data/tinyshakespeare/input.txt")
        urllib.request.urlretrieve(url, path)
    text = open(path, encoding="utf-8").read()
    chars = sorted(set(text))
    vocab = {c: i for i, c in enumerate(chars)}
    data = np.array([vocab[c] for c in text], dtype=np.int64)
    return data, len(chars)


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


def gen_balanced(rng, length):
    pairs = {"(": ")", "[": "]", "{": "}"}
    opens = list(pairs.keys())
    seq, stack = [], []
    for i in range(length):
        remaining = length - i
        if not stack:
            c = rng.choice(opens); seq.append(c); stack.append(c)
        elif remaining == len(stack):
            seq.append(pairs[stack.pop()])
        else:
            if rng.random() < 0.5:
                c = rng.choice(opens); seq.append(c); stack.append(c)
            else:
                seq.append(pairs[stack.pop()])
    return seq


def is_balanced(seq):
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    for c in seq:
        if c in "([{":
            stack.append(c)
        else:
            if not stack or stack.pop() != pairs[c]:
                return False
    return not stack


def corrupt(rng, seq):
    seq = list(seq)
    for _ in range(20):
        s2 = list(seq)
        op = rng.randrange(3)
        if op == 0:
            i = rng.randrange(len(s2))
            s2[i] = rng.choice(list("()[]{}"))
        elif op == 1:
            i, j = rng.randrange(len(s2)), rng.randrange(len(s2))
            s2[i], s2[j] = s2[j], s2[i]
        else:
            i = rng.randrange(len(s2))
            flip = {"(": ")", ")": "(", "[": "]", "]": "[",
                    "{": "}", "}": "{"}
            s2[i] = flip[s2[i]]
        if not is_balanced(s2):
            return s2
    return None


def make_brackets(n, lo, hi, seed):
    rng = random.Random(seed)
    cmap = {c: i for i, c in enumerate("()[]{}")}
    PAD = 6
    xs, ys = [], []
    while len(xs) < n:
        L = rng.randrange(lo // 2, hi // 2 + 1) * 2
        bal = gen_balanced(rng, L)
        assert is_balanced(bal)
        if len(xs) % 2 == 0:
            xs.append([cmap[c] for c in bal]); ys.append(1)
        else:
            bad = corrupt(rng, bal)
            if bad is None:
                continue
            xs.append([cmap[c] for c in bad]); ys.append(0)
    return xs, ys, 7, PAD


class PaddedCLSDataset(Dataset):
    def __init__(self, xs, ys, pad_id, max_len):
        self.xs, self.ys, self.pad, self.max_len = xs, ys, pad_id, max_len

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, i):
        x = self.xs[i][: self.max_len]
        x = x + [self.pad] * (self.max_len - len(x))
        return torch.tensor(x), torch.tensor(self.ys[i])


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
    import contextlib
    return contextlib.nullcontext()


def train_model(model, loaders, device, epochs, lr, loss_kind, name,
                patience=5, log_every=50):
    train_loader, val_loader = loaders
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = epochs * len(train_loader)
    sched = make_sched(opt, total, warmup=max(20, total // 20))
    scaler = torch.amp.GradScaler("cuda",
                                  enabled=(device.type == "cuda" and
                                           not torch.cuda.is_bf16_supported()))
    best = {"val_acc": 0.0, "val_bpc": float("inf"), "epoch": 0}
    bad_epochs = 0
    t_start = time.time()
    for ep in range(epochs):
        model.train()
        t0 = time.time()
        run_loss, nstep = 0.0, 0
        for bi, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
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
                print(f"    [{name}] ep{ep+1} step {bi+1}/{len(train_loader)} "
                      f"loss={run_loss/nstep:.4f} eta_ep={eta:.0f}s", flush=True)
        va, vb = evaluate(model, val_loader, device, loss_kind)
        improved = va > best["val_acc"] + 1e-4 or vb < best["val_bpc"] - 1e-4
        if improved:
            best = {"val_acc": va, "val_bpc": vb, "epoch": ep + 1}
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        print(f"  [{name}] ep {ep+1}/{epochs}  train_loss={run_loss/nstep:.4f}  "
              f"val_acc={va*100:.2f}%  val_bpc={vb:.3f}  "
              f"({time.time()-t0:.0f}s)", flush=True)
        if bad_epochs >= patience:
            print(f"  [{name}] early stop (patience={patience})", flush=True)
            break
    model.load_state_dict(best_state)
    best["train_time_s"] = round(time.time() - t_start, 1)
    return best


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


def ngram_reference(train, eval_pairs, V, orders=(3, 4, 5)):
    results = {}
    tables = {}
    for o in sorted(set(list(orders) + [1, 2])):
        codes = np.zeros(len(train) - o, dtype=np.int64)
        for k in range(o):
            codes = codes * V + train[k:len(train) - o + k]
        nxt = train[o:]
        key = codes * V + nxt
        uk, cnt = np.unique(key, return_counts=True)
        ctx = uk // V
        order_sort = np.lexsort((cnt, ctx))
        ctx_s, uk_s = ctx[order_sort], uk[order_sort]
        last = np.r_[ctx_s[1:] != ctx_s[:-1], True]
        tables[o] = (ctx_s[last], (uk_s[last] % V))
    uni = np.bincount(train, minlength=V).argmax()

    for o in orders:
        corr = tot = 0
        for ctx_arr, tgt in eval_pairs:
            pred = uni
            for oo in range(min(o, len(ctx_arr)), 0, -1):
                code = 0
                for k in range(oo):
                    code = code * V + int(ctx_arr[len(ctx_arr) - oo + k])
                keys, vals = tables[oo]
                i = np.searchsorted(keys, code)
                if i < len(keys) and keys[i] == code:
                    pred = int(vals[i]); break
            corr += int(pred == tgt); tot += 1
        results[o] = corr / max(1, tot)
    return results


def task_speed(cfg, results):
    print("\n" + "=" * 78 + "\nTASK: SPEED (fwd+bwd, tok/s)\n" + "=" * 78)
    device = cfg.device
    V = 65
    seqs = [256, 512] if cfg.quick else [256, 512, 1024, 2048]
    B = 2 if cfg.quick else 4
    rows = {}
    for name in cfg.models:
        ed, npar = match_embed_dim(BACKBONES[name], V, cfg.target_params,
                                   cfg.n_layers, max_len=max(seqs))
        model = LMModel(BACKBONES[name](V, ed, n_layers=cfg.n_layers,
                                        max_len=max(seqs), dropout=0.0), V
                        ).to(device)
        rows[name] = {"params": npar, "toks_per_s": {}}
        for T in seqs:
            x = torch.randint(0, V, (B, T), device=device)
            y = torch.randint(0, V, (B, T), device=device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
            try:
                for _ in range(2):
                    loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
                    loss.backward(); opt.step(); opt.zero_grad()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.time(); iters = 3 if cfg.quick else 5
                for _ in range(iters):
                    loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))
                    loss.backward(); opt.step(); opt.zero_grad()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                tps = B * T * iters / (time.time() - t0)
                rows[name]["toks_per_s"][T] = round(tps)
                print(f"  {name:<12} T={T:<5} {tps:>10,.0f} tok/s "
                      f"({npar:,} params)", flush=True)
            except torch.cuda.OutOfMemoryError:
                rows[name]["toks_per_s"][T] = None
                print(f"  {name:<12} T={T:<5} OOM", flush=True)
                torch.cuda.empty_cache()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    results["speed"] = rows


def task_lm(cfg, results):
    print("\n" + "=" * 78 + "\nTASK: CHAR-LM, FULL TinyShakespeare\n" + "=" * 78)
    device = cfg.device
    data, V = load_shakespeare()
    n = len(data)
    tr_end, va_end = int(n * 0.90), int(n * 0.95)
    train, val, test = data[:tr_end], data[tr_end:va_end], data[va_end:]
    seq = 256 if cfg.quick else 512
    if cfg.quick:
        train = train[:60_000]; val = val[:8_000]; test = test[:8_000]
    stride = seq if cfg.quick else 128
    tl = DataLoader(LMDataset(train, seq, stride), batch_size=cfg.bs_lm,
                    shuffle=True, drop_last=True)
    vl = DataLoader(LMDataset(val, seq, seq), batch_size=cfg.bs_lm)
    sl = DataLoader(LMDataset(test, seq, seq), batch_size=cfg.bs_lm)
    print(f"  train {len(train):,} chars | val {len(val):,} | test {len(test):,}"
          f" | seq={seq} stride={stride} | {len(tl)} steps/ep")
    rows = {}
    for name in cfg.models:
        ed, npar = match_embed_dim(BACKBONES[name], V, cfg.target_params,
                                   cfg.n_layers, max_len=seq)
        model = LMModel(BACKBONES[name](V, ed, n_layers=cfg.n_layers,
                                        max_len=seq, dropout=cfg.dropout), V)
        print(f"\n  --- {name} (ed={ed}, {n_params(model):,} params) ---")
        best = train_model(model, (tl, vl), device, cfg.epochs_lm, cfg.lr,
                           "lm", name, patience=cfg.patience)
        ta, tb = evaluate(model, sl, device, "lm")
        best.update({"test_acc": ta, "test_bpc": tb, "params": n_params(model)})
        print(f"  [{name}] TEST acc={ta*100:.2f}%  bpc={tb:.3f}")
        rows[name] = best
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    print("\n  --- n-gram reference ---")
    pairs = []
    for s in range(0, len(test) - seq - 1, seq):
        w = test[s:s + seq + 1]
        for p in range(1, seq + 1):
            pairs.append((w[max(0, p - 6):p], int(w[p])))
    ng = ngram_reference(train, pairs, V, orders=(3, 4, 5))
    for o, a in ng.items():
        print(f"  ngram-{o}: test acc={a*100:.2f}%")
    rows["ngram"] = {f"order{o}_test_acc": a for o, a in ng.items()}
    results["lm"] = rows


def task_brackets(cfg, results):
    print("\n" + "=" * 78 + "\nTASK: BRACKET BALANCE (512-1024, одинаковые головы)"
          "\n" + "=" * 78)
    device = cfg.device
    lo, hi = (64, 128) if cfg.quick else (512, 1024)
    ntr, nva = (300, 100) if cfg.quick else (4000, 800)
    xs, ys, V, PAD = make_brackets(ntr + nva, lo, hi, seed=cfg.seed)
    tl = DataLoader(PaddedCLSDataset(xs[:ntr], ys[:ntr], PAD, hi),
                    batch_size=cfg.bs_cls, shuffle=True, drop_last=True)
    vl = DataLoader(PaddedCLSDataset(xs[ntr:], ys[ntr:], PAD, hi),
                    batch_size=cfg.bs_cls)
    print(f"  {ntr} train / {nva} val, длины {lo}-{hi}, баланс классов 50/50")
    rows = {}
    for name in cfg.models:
        ed, npar = match_embed_dim(BACKBONES[name], V, cfg.target_params,
                                   cfg.n_layers, max_len=hi)
        model = CLSModel(BACKBONES[name](V, ed, n_layers=cfg.n_layers,
                                         max_len=hi, dropout=cfg.dropout),
                         2, PAD)
        print(f"\n  --- {name} (ed={ed}, {n_params(model):,} params) ---")
        best = train_model(model, (tl, vl), device, cfg.epochs_cls, cfg.lr,
                           "cls", name, patience=cfg.patience)
        best["params"] = n_params(model)
        rows[name] = best
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    results["brackets"] = rows


def task_copy(cfg, results):
    print("\n" + "=" * 78 + "\nTASK: SELECTIVE COPYING (диагностика Mamba)\n"
          + "=" * 78)
    device = cfg.device
    T = 128 if cfg.quick else 512
    ntr, nva = (300, 100) if cfg.quick else (6000, 1000)
    n_mem = 8 if cfg.quick else 16
    xtr, ytr, V = make_copy(ntr, T, n_mem, seed=cfg.seed)
    xva, yva, _ = make_copy(nva, T, n_mem, seed=cfg.seed + 1)
    tl = DataLoader(torch.utils.data.TensorDataset(xtr, ytr),
                    batch_size=cfg.bs_lm, shuffle=True, drop_last=True)
    vl = DataLoader(torch.utils.data.TensorDataset(xva, yva),
                    batch_size=cfg.bs_lm)
    print(f"  {ntr} train / {nva} val, seq={T}, {n_mem} токенов для копирования")
    rows = {}
    for name in cfg.models:
        ed, npar = match_embed_dim(BACKBONES[name], V, cfg.target_params,
                                   cfg.n_layers, max_len=T)
        model = LMModel(BACKBONES[name](V, ed, n_layers=cfg.n_layers,
                                        max_len=T, dropout=cfg.dropout), V)
        print(f"\n  --- {name} (ed={ed}, {n_params(model):,} params) ---")
        best = train_model(model, (tl, vl), device, cfg.epochs_cls, cfg.lr,
                           "lm", name, patience=cfg.patience)
        best["params"] = n_params(model)
        rows[name] = best
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    results["copy"] = rows


def summarize(results):
    lines = ["# MASSIVE BENCHMARK — итог", ""]
    if "speed" in results:
        lines += ["## Speed (tok/s, fwd+bwd)", ""]
        seqs = sorted({t for r in results["speed"].values()
                       for t in r["toks_per_s"]})
        lines.append("| model | params | " +
                     " | ".join(f"T={t}" for t in seqs) + " |")
        lines.append("|" + "---|" * (len(seqs) + 2))
        for m, r in results["speed"].items():
            cells = [f"{r['toks_per_s'].get(t, '-') or 'OOM'}" for t in seqs]
            lines.append(f"| {m} | {r['params']:,} | " + " | ".join(cells) + " |")
        lines.append("")
    for task, title, cols in [
        ("lm", "Char-LM (полный TinyShakespeare)",
         [("val_bpc", "val bpc"), ("test_bpc", "test bpc"),
          ("test_acc", "test acc"), ("train_time_s", "time,s")]),
        ("brackets", "Bracket balance 512-1024",
         [("val_acc", "val acc"), ("epoch", "best ep"),
          ("train_time_s", "time,s")]),
        ("copy", "Selective copying",
         [("val_acc", "val acc"), ("epoch", "best ep"),
          ("train_time_s", "time,s")]),
    ]:
        if task not in results:
            continue
        lines += [f"## {title}", ""]
        lines.append("| model | params | " +
                     " | ".join(c[1] for c in cols) + " |")
        lines.append("|" + "---|" * (len(cols) + 2))
        for m, r in results[task].items():
            if m == "ngram":
                continue
            cells = []
            for key, _ in cols:
                v = r.get(key)
                if v is None:
                    cells.append("-")
                elif "acc" in key:
                    cells.append(f"{v*100:.2f}%")
                elif "time" in key:
                    cells.append(f"{v:.0f}")
                elif isinstance(v, float):
                    cells.append(f"{v:.3f}")
                else:
                    cells.append(str(v))
            lines.append(f"| {m} | {r.get('params', 0):,} | " +
                         " | ".join(cells) + " |")
        if task == "lm" and "ngram" in results["lm"]:
            ng = results["lm"]["ngram"]
            lines.append("")
            lines.append("n-gram reference (test acc): " + ", ".join(
                f"{k}={v*100:.2f}%" for k, v in ng.items()))
        lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="all",
                   help="all | speed,lm,brackets,copy (через запятую)")
    p.add_argument("--models", default="wat,transformer,lstm,mamba")
    p.add_argument("--scale", default="base", choices=["small", "base", "big"])
    p.add_argument("--epochs-lm", type=int, default=None)
    p.add_argument("--epochs-cls", type=int, default=None)
    p.add_argument("--quick", action="store_true", help="смок-тест 5-10 минут")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    cfg = p.parse_args()

    random.seed(cfg.seed); np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.benchmark = True

    cfg.device = torch.device(cfg.device or
                              ("cuda" if torch.cuda.is_available() else "cpu"))
    scale_map = {"small": (200_000, 2), "base": (1_000_000, 3),
                 "big": (5_000_000, 4)}
    cfg.target_params, cfg.n_layers = scale_map[cfg.scale]
    if cfg.quick:
        cfg.target_params, cfg.n_layers = 60_000, 2
    cfg.dropout = 0.1
    cfg.lr = 3e-4
    cfg.patience = 4
    cfg.bs_lm = {"small": 32, "base": 32, "big": 16}[cfg.scale]
    cfg.bs_cls = {"small": 16, "base": 16, "big": 8}[cfg.scale]
    if cfg.device.type == "cpu":
        cfg.bs_lm, cfg.bs_cls = min(cfg.bs_lm, 16), min(cfg.bs_cls, 8)
    cfg.epochs_lm = cfg.epochs_lm or (1 if cfg.quick else 8)
    cfg.epochs_cls = cfg.epochs_cls or (1 if cfg.quick else 15)
    cfg.models = [m.strip() for m in cfg.models.split(",") if m.strip()]
    tasks = (["speed", "lm", "brackets", "copy"] if cfg.task == "all"
             else [t.strip() for t in cfg.task.split(",")])

    print("=" * 78)
    print("MASSIVE BENCHMARK — WAT (Igor Berezkin) vs baselines")
    print(f"device={cfg.device}  scale={cfg.scale}  "
          f"target_params={cfg.target_params:,}  layers={cfg.n_layers}")
    print(f"models={cfg.models}  tasks={tasks}  quick={cfg.quick}")
    print(f"mamba: {'официальный mamba_ssm' if _OFFICIAL_MAMBA else 'встроенный SSD-скан (pure PyTorch)'}")
    print("=" * 78)

    results = {"config": {k: (str(v) if isinstance(v, torch.device) else v)
                          for k, v in vars(cfg).items()}}
    t0 = time.time()
    for t in tasks:
        {"speed": task_speed, "lm": task_lm,
         "brackets": task_brackets, "copy": task_copy}[t](cfg, results)

    results["total_time_s"] = round(time.time() - t0, 1)
    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    summary = summarize(results)
    with open(os.path.join(OUT_DIR, "summary.md"), "w", encoding="utf-8") as f:
        f.write(summary)
    print("\n" + summary)
    print(f"\nГотово за {results['total_time_s']/60:.1f} мин. "
          f"Результаты: results/benchmark/massive/results.json, summary.md")


if __name__ == "__main__":
    main()
