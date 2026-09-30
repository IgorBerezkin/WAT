# -*- coding: utf-8 -*-
"""
diag_marathon.py — финальный тест теории "тихий сигнал + время = обучение".
Положи рядом с wat_lab.py.  Запуск:  python diag_marathon.py
GPU ~5-10 мин. Всё в fp32, без autocast (убираем bf16-конфаундер).

ЗАБЕГ 1 — марафон головы: бэкбон заморожен (init как в анатомии, seed 42),
  hidden@reader кэшируется один раз, голова учится 500 full-batch эпох,
  lr 1e-2. Условия головы = e2e (сырые фичи, 18 классов, без стандартизации);
  единственное отличие от прежних провалов — БЮДЖЕТ.
ЗАБЕГ 2 — марафон e2e: бэкбон разморожен, 250 эпох, bs 256,
  head lr 1e-2 / бэкбон 3e-4. Телеметрия каждые 25 эпох: dSig — амплитуда
  следа токена на входе головы (растит бэкбон громкость или глушит).

КРИТЕРИИ (зафиксированы до прогона):
  забег 1 >= 40%  -> теория amplitude-limited optimization ПОДТВЕРЖДЕНА
  15-40%          -> частично, ищем что доедает
  < 15%           -> теория неверна, назад к колбе
  ориентир-потолок: проба на этих же фичах давала 53.1% (слепая зона)
"""
import sys, math, time
sys.path.insert(0, ".")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from wat_lab import WATBackboneX, make_copy

torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ED, K, T, NTR, NTE = 96, 32, 128, 4000, 1000
CHANCE = 6.25


def fresh_backbone(V):
    torch.manual_seed(42)                     # тот же init, что видела колба
    return WATBackboneX(V, ED, n_layers=1, chunk_size=K, max_len=T,
                        dropout=0.0, ctx_mode="mean", intra=False)


def data():
    xtr, ytr, V = make_copy(NTR, T, 1, seed=42)
    xva, yva, _ = make_copy(NTE, T, 1, seed=43)
    lab_tr = ytr[:, -1]                       # значение токена = метка 0..15
    lab_va = yva[:, -1]
    return (xtr.to(DEVICE), lab_tr.to(DEVICE),
            xva.to(DEVICE), lab_va.to(DEVICE), V)


@torch.no_grad()
def reader_features(bb, x, bs=256):
    bb.eval()
    outs = []
    for i in range(0, x.size(0), bs):
        outs.append(bb(x[i:i + bs])[:, -1, :].float())
    return torch.cat(outs)


@torch.no_grad()
def dsig_at_head(bb, x_probe, pos_probe):
    """Амплитуда следа токена на входе головы, % от потока."""
    bb.eval()
    ar = torch.arange(x_probe.size(0), device=DEVICE)
    x2 = x_probe.clone()
    x2[ar, pos_probe] = (x2[ar, pos_probe] + 7) % 16
    h1 = bb(x_probe)[:, -1, :].float()
    h2 = bb(x2)[:, -1, :].float()
    rel = (h2 - h1).norm(dim=1) / (h1.pow(2).mean(1).sqrt()
                                   * math.sqrt(h1.size(1)) + 1e-9)
    return rel.mean().item() * 100


def milestones(acc, hit, tag, ep):
    for m in (15.0, 40.0):
        if acc * 100 >= m and m not in hit:
            hit.add(m)
            print(f"  >>> [{tag}] РУБЕЖ {m:.0f}% взят на эпохе {ep} <<<",
                  flush=True)


def race1():
    print("=" * 70)
    print("ЗАБЕГ 1 — марафон головы (бэкбон заморожен, 500 full-batch эпох)")
    print("=" * 70)
    xtr, ytr, xva, yva, V = data()
    bb = fresh_backbone(V).to(DEVICE)
    f_tr = reader_features(bb, xtr)
    f_va = reader_features(bb, xva)
    head = nn.Linear(ED, V).to(DEVICE)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2, weight_decay=1e-4)
    best, hit = 0.0, set()
    t0 = time.time()
    for ep in range(1, 501):
        head.train()
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(head(f_tr), ytr)
        loss.backward()
        opt.step()
        if ep % 20 == 0 or ep == 1:
            head.eval()
            with torch.no_grad():
                acc = (head(f_va).argmax(-1) == yva).float().mean().item()
            best = max(best, acc)
            milestones(acc, hit, "забег1", ep)
            print(f"  [забег1] ep{ep:3d}  loss={loss.item():.4f}  "
                  f"val_acc={acc*100:5.1f}%  ({time.time()-t0:.0f}s)",
                  flush=True)
    return best


def race2():
    print("=" * 70)
    print("ЗАБЕГ 2 — марафон e2e (бэкбон разморожен, 250 эпох, bs 256)")
    print("=" * 70)
    xtr, ytr, xva, yva, V = data()
    # для dSig-телеметрии нужен полный x с позициями токена
    x_raw, _, _ = make_copy(256, T, 1, seed=77)
    x_probe = x_raw.to(DEVICE)
    pos_probe = (x_probe[:, :T - 1] != 16).float().argmax(dim=1)

    bb = fresh_backbone(V).to(DEVICE)
    head = nn.Linear(ED, V).to(DEVICE)
    opt = torch.optim.Adam([
        {"params": bb.parameters(), "lr": 3e-4},
        {"params": head.parameters(), "lr": 1e-2},
    ], weight_decay=1e-4)
    tl = DataLoader(TensorDataset(xtr, ytr), batch_size=256, shuffle=True)
    best, hit = 0.0, set()
    t0 = time.time()
    print(f"  dSig@старт = {dsig_at_head(bb, x_probe, pos_probe):.2f}%")
    for ep in range(1, 251):
        bb.train(); head.train()
        for xb, yb in tl:
            opt.zero_grad(set_to_none=True)
            h = bb(xb)[:, -1, :]
            loss = F.cross_entropy(head(h), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(bb.parameters()) + list(head.parameters()), 1.0)
            opt.step()
        if ep % 10 == 0 or ep == 1:
            bb.eval(); head.eval()
            with torch.no_grad():
                acc = (head(reader_features(bb, xva)).argmax(-1)
                       == yva).float().mean().item()
            best = max(best, acc)
            milestones(acc, hit, "забег2", ep)
            extra = ""
            if ep % 25 == 0 or ep == 1:
                extra = f"  dSig={dsig_at_head(bb, x_probe, pos_probe):.2f}%"
            print(f"  [забег2] ep{ep:3d}  loss={loss.item():.4f}  "
                  f"val_acc={acc*100:5.1f}%{extra}  ({time.time()-t0:.0f}s)",
                  flush=True)
    return best


if __name__ == "__main__":
    b1 = race1()
    b2 = race2()
    print("=" * 70)
    print(f"ИТОГ: забег1 best={b1*100:.1f}%  забег2 best={b2*100:.1f}%  "
          f"(шанс {CHANCE}%, ориентир пробы 53.1%)")
    print("Критерии из шапки: >=40 подтверждена / 15-40 частично / <15 неверна")