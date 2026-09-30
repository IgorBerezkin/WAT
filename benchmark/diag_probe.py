# -*- coding: utf-8 -*-
"""
diag_probe.py — есть ли информация в точке чтения? (линейный пробинг)
Положи рядом с wat_lab.py.  Запуск:  python diag_probe.py
GPU ~2-3 мин (CPU ~6-8). Обучается только крошечная линейная проба,
бэкбоны ЗАМОРОЖЕНЫ на инициализации.

Задача-канарейка: copy T=128, n_mem=1, один слой (L=1, транспорт обязан
работать в одном блоке). Для каждой конфигурации меряем two-point probe:
  probe(ctx)    — линейный классификатор токена по вектору глобального
                  контекста читающего чанка (ДО инжекции)
  probe(reader) — то же по hidden читателя, позиция 127 (ПОСЛЕ всего блока)

Конфигурации:
  C0  as-is                — контроль
  C4  sum-res + inj8 + lane (прошлый фикс)
  C5  ГРОМКАЯ ПОЛОСА       — lane std 0.3, дерево в саммари задавлено x0.1
  C6  ТОЛЬКО ПОЛОСА        — саммари = lane, дерево выключено
  TR  transformer          — эталон (probe(reader) only)

ЧТЕНИЕ РЕЗУЛЬТАТОВ:
  probe(ctx) >> 6.25%, probe(reader) ~ 6.25%  -> инжекция хоронит
  probe(ctx) ~ 6.25%                          -> саммари хоронит
  оба высокие, а e2e (из diag_fix) на шансе    -> проблема оптимизации,
                                                 лечим масштабы/шаги/lr
"""
import sys, math, types
sys.path.insert(0, ".")
import torch
import torch.nn as nn
import torch.nn.functional as F

from wat_lab import (WATBlockX, WATBackboneX, TransformerBackbone, GLUMerge,
                     make_copy)

torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ED, K, T, NTR, NTE = 96, 32, 128, 4000, 1000


class GLUMergeSumRes(GLUMerge):
    def forward(self, left, right):
        combined = torch.cat([left, right], dim=-1)
        val = self.W_val(combined)
        gate = torch.sigmoid(self.W_gate(combined))
        merged = self.norm(val * gate)
        res_gate = torch.sigmoid(self.W_res(combined))
        return res_gate * merged + (1.0 - res_gate) * (left + right)


def _tree_lane(self, chunks):
    s = WATBlockX._tree_reduction_all(self, chunks) * self.tree_damp
    g = torch.sigmoid(self.lane_g(chunks))
    v = self.lane_v(chunks)
    return s + (g * v).sum(dim=2) / math.sqrt(chunks.size(2))


def build_wat(config):
    bb = WATBackboneX(18, ED, n_layers=1, chunk_size=K, max_len=T,
                      dropout=0.0, ctx_mode="mean", intra=False)
    scale = 0.02 / (2 * 1) ** 0.5
    blk = bb.layers[0]
    if config == "C4":
        m = GLUMergeSumRes(ED)
        for mm in m.modules():
            if isinstance(mm, nn.Linear):
                nn.init.normal_(mm.weight, 0.0, scale)
                nn.init.zeros_(mm.bias)
        blk.tree_merge = m
        with torch.no_grad():
            blk.W_global.weight.mul_(8.0)
    if config in ("C4", "C5", "C6"):
        blk.lane_g = nn.Linear(ED, 1)
        blk.lane_v = nn.Linear(ED, ED)
        std = {"C4": scale * 4, "C5": 0.3, "C6": 0.3}[config]
        nn.init.normal_(blk.lane_v.weight, 0.0, std)
        nn.init.zeros_(blk.lane_v.bias)
        nn.init.zeros_(blk.lane_g.weight)
        nn.init.zeros_(blk.lane_g.bias)
        blk.tree_damp = {"C4": 1.0, "C5": 0.1, "C6": 0.0}[config]
        blk._tree_reduction_all = types.MethodType(_tree_lane, blk)
    else:
        blk.tree_damp = 1.0
    if config == "C5":
        with torch.no_grad():
            blk.W_global.weight.mul_(8.0)
    if config == "C6":
        with torch.no_grad():
            blk.W_global.weight.mul_(8.0)
    return bb.to(DEVICE).eval()


@torch.no_grad()
def extract_wat(bb, x):
    """Возвращает (ctx-вектор чанка читателя, hidden читателя)."""
    blk = bb.layers[0]
    positions = torch.arange(x.size(1), device=x.device)
    h = bb.embedding(x) + bb.pos_encoding(positions)
    hh = blk.norm_conv(h)
    hh = blk.conv(hh)
    hh = hh * torch.sigmoid(blk.W_gate(hh))
    x1 = h + hh
    chunks = x1.unfold(1, K, K).transpose(2, 3)
    s = blk._tree_reduction_all(chunks)
    ctx = blk._ctx_mean(s)                        # (B, C, D)
    reader_ctx = ctx[:, -1, :]                    # контекст чанка читателя
    full = bb(x)                                  # (B, T, D)
    reader_h = full[:, -1, :]
    return reader_ctx, reader_h


@torch.no_grad()
def extract_tr(bb, x):
    return None, bb(x)[:, -1, :]


def linear_probe(feats_tr, y_tr, feats_te, y_te, epochs=400):
    """Логистическая регрессия на замороженных фичах."""
    f_tr = (feats_tr - feats_tr.mean(0)) / (feats_tr.std(0) + 1e-6)
    f_te = (feats_te - feats_tr.mean(0)) / (feats_tr.std(0) + 1e-6)
    probe = nn.Linear(f_tr.size(1), 16).to(DEVICE)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(probe(f_tr), y_tr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = (probe(f_te).argmax(-1) == y_te).float().mean().item()
    return acc


def dataset():
    x, y, V = make_copy(NTR + NTE, T, 1, seed=42)
    labels = y[:, -1] - 0                        # токен на позиции 127
    # в make_copy контентные токены 0..15, метка = сам токен
    return (x[:NTR].to(DEVICE), labels[:NTR].to(DEVICE),
            x[NTR:].to(DEVICE), labels[NTR:].to(DEVICE))


def run():
    xtr, ytr, xte, yte = dataset()
    assert ytr.min() >= 0 and ytr.max() <= 15, "метки вне 0..15"
    print(f"device={DEVICE}  train={NTR} test={NTE}  шанс=6.25%")
    print(f"{'config':<6} {'probe(ctx)':>11} {'probe(reader)':>14}")
    for config in ("C0", "C4", "C5", "C6"):
        bb = build_wat(config)
        ctx_tr, rd_tr = [], []
        ctx_te, rd_te = [], []
        for xs, dst_c, dst_r in ((xtr, ctx_tr, rd_tr), (xte, ctx_te, rd_te)):
            for i in range(0, xs.size(0), 256):
                c, r = extract_wat(bb, xs[i:i + 256])
                dst_c.append(c); dst_r.append(r)
        a_ctx = linear_probe(torch.cat(ctx_tr), ytr, torch.cat(ctx_te), yte)
        a_rd = linear_probe(torch.cat(rd_tr), ytr, torch.cat(rd_te), yte)
        print(f"{config:<6} {a_ctx*100:>10.1f}% {a_rd*100:>13.1f}%", flush=True)
        del bb
    tr = TransformerBackbone(18, ED, n_layers=1, max_len=T,
                             dropout=0.0).to(DEVICE).eval()
    rd_tr, rd_te = [], []
    for xs, dst in ((xtr, rd_tr), (xte, rd_te)):
        for i in range(0, xs.size(0), 256):
            _, r = extract_tr(tr, xs[i:i + 256])
            dst.append(r)
    a_rd = linear_probe(torch.cat(rd_tr), ytr, torch.cat(rd_te), yte)
    print(f"{'TR':<6} {'—':>10} {a_rd*100:>13.1f}%")
    print("\nИнтерпретация — в шапке файла.")


if __name__ == "__main__":
    run()