import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from wat.data import make_copy
from wat.history.lab import WATBackboneX

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ED, K, T, EPOCHS = 96, 32, 128, 60
CHANCE = 6.25


def data():
    xtr, ytr, _ = make_copy(4000, T, 1, seed=42)
    xva, yva, V = make_copy(1000, T, 1, seed=43)
    return (xtr.to(DEVICE), ytr[:, -1].to(DEVICE),
            xva.to(DEVICE), yva[:, -1].to(DEVICE), V)


@torch.no_grad()
def val_acc(bb, head, xva, yva, bs=256):
    bb.eval(); head.eval()
    outs = []
    for i in range(0, xva.size(0), bs):
        outs.append(head(bb(xva[i:i + bs])[:, -1, :]).argmax(-1))
    return (torch.cat(outs) == yva).float().mean().item()


def run(tag, bs, head_lr, seed, xtr, ytr, xva, yva, V):
    torch.manual_seed(seed)
    bb = WATBackboneX(V, ED, n_layers=1, chunk_size=K, max_len=T,
                      dropout=0.0, ctx_mode="mean", intra=False).to(DEVICE)
    head = nn.Linear(ED, V).to(DEVICE)
    opt = torch.optim.Adam([
        {"params": bb.parameters(), "lr": 3e-4},
        {"params": head.parameters(), "lr": head_lr},
    ], weight_decay=1e-4)
    tl = DataLoader(TensorDataset(xtr, ytr), batch_size=bs, shuffle=True,
                    generator=torch.Generator().manual_seed(seed))
    best, cross = 0.0, None
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        bb.train(); head.train()
        for xb, yb in tl:
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(bb(xb)[:, -1, :]), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(bb.parameters()) + list(head.parameters()), 1.0)
            opt.step()
        acc = val_acc(bb, head, xva, yva)
        best = max(best, acc)
        if cross is None and acc * 100 >= 15.0:
            cross = ep
        if ep % 10 == 0 or ep == 1 or (cross == ep):
            note = "  <- ПЕРЕХОД" if cross == ep else ""
            print(f"  [{tag}] ep{ep:2d}  loss={loss.item():.3f}  "
                  f"val_acc={acc*100:5.1f}%{note}  ({time.time()-t0:.0f}s)",
                  flush=True)
    return best, cross


def main():
    xtr, ytr, xva, yva, V = data()
    grid = [
        ("G1 bs32  hlr3e-4 s42", 32, 3e-4, 42),
        ("G2 bs32  hlr1e-2 s42", 32, 1e-2, 42),
        ("G3 bs256 hlr3e-4 s42", 256, 3e-4, 42),
        ("G4 bs256 hlr1e-2 s42", 256, 1e-2, 42),
        ("G4 bs256 hlr1e-2 s1 ", 256, 1e-2, 1),
        ("G4 bs256 hlr1e-2 s2 ", 256, 1e-2, 2),
    ]
    results = []
    for tag, bs, hlr, seed in grid:
        print("=" * 66)
        print(f"{tag}  ({EPOCHS} эпох)")
        print("=" * 66)
        best, cross = run(tag, bs, hlr, seed, xtr, ytr, xva, yva, V)
        results.append((tag, best, cross))
    print("\n" + "=" * 66)
    print(f"{'конфиг':<24} {'грокнул?':>9} {'эпоха>15%':>10} {'best':>8}")
    print("-" * 66)
    for tag, best, cross in results:
        ok = "ДА" if best * 100 >= 40 else ("частично" if best * 100 >= 15
                                            else "нет")
        print(f"{tag:<24} {ok:>9} {str(cross or '—'):>10} "
              f"{best*100:>7.1f}%")
    print(f"(шанс {CHANCE}%, полка марафона 75.4%)")


if __name__ == "__main__":
    main()
