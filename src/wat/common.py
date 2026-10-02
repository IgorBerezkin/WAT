import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.RMSNorm):
    def forward(self, x):
        weight = self.weight.to(x.dtype) if self.weight is not None else None
        return F.rms_norm(x, self.normalized_shape, weight, self.eps)


class CausalConv1d(nn.Module):
    def __init__(self, d, kernel_size=3, groups=1):
        super().__init__()
        self.padding = kernel_size - 1
        self.conv = nn.Conv1d(d, d, kernel_size=kernel_size, groups=groups)

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
        self.norm = RMSNorm(d)

    def forward(self, left, right):
        combined = torch.cat([left, right], dim=-1)
        val = self.W_val(combined)
        gate = torch.sigmoid(self.W_gate(combined))
        merged = self.norm(val * gate)
        res_gate = torch.sigmoid(self.W_res(combined))
        residual = (left + right) * 0.5
        return res_gate * merged + (1.0 - res_gate) * residual


def match_pointers(x, lengths, vocab):
    B, T = x.shape
    t = torch.arange(T, device=x.device)
    shift = lambda z: torch.cat([torch.zeros_like(z[:, :1]), z[:, :-1]], dim=1)
    h1, h2, out = torch.zeros_like(x), torch.zeros_like(x), []
    for n in range(1, (max(lengths) if lengths else 0) + 1):
        h1 = (shift(h1) * 257 + x + 1) % 2147483647
        h2 = (shift(h2) * 263 + x + 1) % 2147483629
        if n not in lengths:
            continue
        key = torch.where(t >= n - 1, h1 * 2147483629 + h2, -(t + 1))
        sk, idx = torch.sort(key, dim=1, stable=True)
        prev_sorted = torch.where(sk[:, 1:] == sk[:, :-1], idx[:, :-1], -1)
        prev_sorted = torch.cat([torch.full_like(idx[:, :1], -1), prev_sorted], dim=1)
        prev = torch.full_like(idx, -1).scatter(1, idx, prev_sorted)
        nxt = x.gather(1, (prev + 1).clamp(max=T - 1))
        out.append(torch.where(prev >= 0, nxt, torch.full_like(nxt, vocab)))
    return out


class LMModel(nn.Module):
    def __init__(self, backbone, vocab_size):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.embed_dim, vocab_size)

    def forward(self, x):
        return self.head(self.backbone(x))


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


def make_sched(opt, total_steps, warmup):
    def fn(s):
        if s < warmup:
            return (s + 1) / warmup
        p = (s - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)
