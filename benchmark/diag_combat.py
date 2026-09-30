# -*- coding: utf-8 -*-
"""
diag_combat.py — боевое сравнение до сходимости: copy T=512/nm=16 и recall.
Положи рядом с wat_lab.py.  Запуск:
  python diag_combat.py              # оба блока (~25-30 мин)
  python diag_combat.py --task copy  # только боевой copy
  python diag_combat.py --task recall
GPU, fp32, без autocast. Все модели учатся ДО ПЛАТО (ранний выход:
30 эпох без улучшения best после перехода) или до капа.

Блок A — COPY T=512, n_mem=16 (протокол massive_benchmark, tiny-масштаб):
  wat_v0 (кап 200), wat_v1 лестница (кап 200), transformer (кап 60),
  lstm (кап 60). Слепая зона ~3.2% -> потолок ~97%.
Блок B — RECALL T=256, 12 пар (протокол wat_lab):
  wat_v0 (кап 150), transformer (кап 30), lstm (кап 30).

В таблице: переход, best, эпох, СЕКУНД — сравнивай и по эпохам, и по
wall-clock (эпоха WAT на T=512 ~2x дешевле трансформерной).
"""
import argparse, sys, time
sys.path.insert(0, ".")
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from wat_lab import (WATBackboneX, TransformerBackbone, LSTMBackbone,
                     LMModel, make_copy, make_recall)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ED, K = 96, 32
CHANCE = 6.25


def build(kind, V, T):
    torch.manual_seed(42)
    if kind == "wat_v0":
        bb = WATBackboneX(V, ED, n_layers=1, chunk_size=K, max_len=T,
                          dropout=0.0, ctx_mode="mean", intra=False)
    elif kind == "wat_v1":
        bb = WATBackboneX(V, ED, n_layers=1, chunk_size=K, max_len=T,
                          dropout=0.0, ctx_mode="prefix_tree", intra=False)
    elif kind == "transformer":
        bb = TransformerBackbone(V, ED, n_layers=1, max_len=T, dropout=0.0)
    else:
        bb = LSTMBackbone(V, ED, n_layers=1, dropout=0.0)
    return LMModel(bb, V).to(DEVICE)


@torch.no_grad()
def val_acc(model, xva, yva, bs):
    model.eval()
    corr = tot = 0
    for i in range(0, xva.size(0), bs):
        out = model(xva[i:i + bs])
        y = yva[i:i + bs]
        m = (y != -100)
        corr += (out.argmax(-1)[m] == y[m]).sum().item()
        tot += m.sum().item()
    return corr / max(1, tot)


def train_one(tag, model, xtr, ytr, xva, yva, bs, cap, patience=30):
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    tl = DataLoader(TensorDataset(xtr, ytr), batch_size=bs, shuffle=True,
                    generator=torch.Generator().manual_seed(42))
    best, best_ep, cross = 0.0, 0, None
    t0 = time.time()
    ep = 0
    for ep in range(1, cap + 1):
        model.train()
        for xb, yb in tl:
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = F.cross_entropy(out.reshape(-1, out.size(-1)),
                                   yb.reshape(-1), ignore_index=-100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        acc = val_acc(model, xva, yva, bs)
        if acc > best + 1e-4:
            best, best_ep = acc, ep
        if cross is None and acc * 100 >= 15.0:
            cross = ep
        if ep % 10 == 0 or ep == 1 or cross == ep:
            note = "  <- ПЕРЕХОД" if cross == ep else ""
            print(f"  [{tag}] ep{ep:3d}  loss={loss.item():.3f}  "
                  f"val_acc={acc*100:5.1f}%{note}  ({time.time()-t0:.0f}s)",
                  flush=True)
        if cross is not None and ep - best_ep >= patience:
            print(f"  [{tag}] полка стабильна, выход (ep{ep})", flush=True)
            break
    return best, cross, ep, round(time.time() - t0)


def block_copy(results):
    T, NM = 512, 16
    print("=" * 70)
    print(f"БЛОК A: БОЕВОЙ COPY  T={T}, n_mem={NM}  (потолок ~97%)")
    print("=" * 70)
    xtr, ytr, V = make_copy(4000, T, NM, seed=42)
    xva, yva, _ = make_copy(1000, T, NM, seed=43)
    xtr, ytr = xtr.to(DEVICE), ytr.to(DEVICE)
    xva, yva = xva.to(DEVICE), yva.to(DEVICE)
    for kind, cap in (("wat_v0", 200), ("wat_v1", 200),
                      ("transformer", 60), ("lstm", 60)):
        model = build(kind, V, T)
        npar = sum(p.numel() for p in model.parameters())
        print(f"--- {kind} ({npar:,} params, кап {cap}) ---")
        b, c, e, sec = train_one(f"copy {kind}", model, xtr, ytr, xva, yva,
                                 bs=128, cap=cap)
        results.append(("copy", kind, b, c, e, sec))
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()


def block_recall(results):
    T, NP = 256, 12
    print("=" * 70)
    print(f"БЛОК B: RECALL  T={T}, {NP} пар ключ-значение")
    print("=" * 70)
    xtr, ytr, V = make_recall(6000, T, NP, seed=42)
    xva, yva, _ = make_recall(1000, T, NP, seed=43)
    xtr, ytr = xtr.to(DEVICE), ytr.to(DEVICE)
    xva, yva = xva.to(DEVICE), yva.to(DEVICE)
    for kind, cap in (("wat_v0", 150), ("transformer", 30), ("lstm", 30)):
        model = build(kind, V, T)
        npar = sum(p.numel() for p in model.parameters())
        print(f"--- {kind} ({npar:,} params, кап {cap}) ---")
        b, c, e, sec = train_one(f"recall {kind}", model, xtr, ytr, xva, yva,
                                 bs=256, cap=cap)
        results.append(("recall", kind, b, c, e, sec))
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="all", choices=["all", "copy", "recall"])
    args = p.parse_args()
    results = []
    if args.task in ("all", "copy"):
        block_copy(results)
    if args.task in ("all", "recall"):
        block_recall(results)
    print("\n" + "=" * 70)
    print(f"{'задача':<8} {'модель':<12} {'переход':>8} {'best':>8} "
          f"{'эпох':>6} {'сек':>7}")
    print("-" * 70)
    for task, kind, b, c, e, sec in results:
        print(f"{task:<8} {kind:<12} {str(c or '—'):>8} {b*100:>7.1f}% "
              f"{e:>6} {sec:>7}")
    print(f"(шанс {CHANCE}%; старые нули WAT на этих задачах — бюджет 15 эпох)")


if __name__ == "__main__":
    main()