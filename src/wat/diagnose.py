import math

import numpy as np
import torch
import torch.nn.functional as F

LENS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)


def ngram_hashes(w, n):
    h = np.zeros((w.shape[0], w.shape[1] - n + 1), dtype=np.uint64)
    with np.errstate(over="ignore"):
        for k in range(n):
            h = h * np.uint64(1000003) + (w[:, k:w.shape[1] - n + 1 + k].astype(np.uint64) + np.uint64(1))
    return h


def previous_occurrence(h):
    rows, cols = h.shape
    key = h.ravel()
    row = np.repeat(np.arange(rows), cols)
    col = np.tile(np.arange(cols), rows)
    order = np.lexsort((col, row, key))
    same = np.zeros(len(order), dtype=bool)
    same[1:] = (key[order][1:] == key[order][:-1]) & (row[order][1:] == row[order][:-1])
    prev = np.full(len(order), -1)
    prev[1:] = np.where(same[1:], col[order][:-1], -1)
    out = np.full(rows * cols, -1)
    out[order] = prev
    return out.reshape(rows, cols)


def repeat_lengths(w):
    T = w.shape[1] - 1
    copy = np.zeros((w.shape[0], T), dtype=np.int64)
    match = np.zeros((w.shape[0], T), dtype=np.int64)
    for L in LENS:
        if L + 1 >= w.shape[1]:
            break
        hit = previous_occurrence(ngram_hashes(w, L + 1)) >= 0
        t = np.arange(L - 1, T)
        copy[:, t] = np.where(hit & (L > copy[:, t]), L, copy[:, t])
        hit = previous_occurrence(ngram_hashes(w[:, :-1], L)) >= 0
        t = np.arange(L - 1, T)
        match[:, t] = np.where(hit & (L > match[:, t]), L, match[:, t])
    return copy, match


@torch.no_grad()
def token_bits(model, task, split, device, autocast, batch_size, max_windows):
    model.eval()
    bits, windows = [], []
    for i, (x, y) in enumerate(task.eval_batches(split, batch_size)):
        if i * batch_size >= max_windows:
            break
        with autocast():
            logits = model(x.to(device))
        nll = F.cross_entropy(logits.float().transpose(1, 2), y.to(device), reduction="none") / math.log(2)
        bits.append(nll.cpu().numpy())
        windows.append(torch.cat([x[:, :1], y], dim=1).numpy())
    model.train()
    return np.concatenate(bits)[:max_windows], np.concatenate(windows)[:max_windows]


def copy_stats(bits, windows):
    copy, match = repeat_lengths(windows)
    pos = np.broadcast_to(np.arange(bits.shape[1]), bits.shape)
    groups = {
        "all": np.ones_like(copy, dtype=bool),
        "copy0": copy == 0, "copy1-3": (copy >= 1) & (copy < 4), "copy4-7": (copy >= 4) & (copy < 8),
        "copy8-15": (copy >= 8) & (copy < 16), "copy16+": copy >= 16,
        "pos0-63": pos < 64, "pos64-255": (pos >= 64) & (pos < 256), "pos256+": pos >= 256,
        "match8_same": (match >= 8) & (copy >= 8), "match8_diff": (match >= 8) & (copy < 8),
    }
    return {name: {"frac": round(float(g.mean()), 5), "bpc": round(float(bits[g].mean()), 5) if g.any() else None}
            for name, g in groups.items()}


def diagnose(model, task, device, autocast, batch_size, max_windows=4000):
    out = {}
    for split in ("val", "test"):
        bits, windows = token_bits(model, task, split, device, autocast, batch_size, max_windows)
        out[split] = copy_stats(bits, windows)
    return out
