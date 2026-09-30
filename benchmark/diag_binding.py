import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from wat_lab import WATBackboneX, LMModel, make_copy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T, NM, K = 128, 4, 32
CHANCE = 6.25


def data():
    xtr, ytr, V = make_copy(4000, T, NM, seed=42)
    xva, yva, _ = make_copy(1000, T, NM, seed=43)
    return (xtr.to(DEVICE), ytr.to(DEVICE),
            xva.to(DEVICE), yva.to(DEVICE), V)


@torch.no_grad()
def val_acc(model, xva, yva, bs=256):
    model.eval()
    corr = tot = 0
    for i in range(0, xva.size(0), bs):
        out = model(xva[i:i + bs])
        y = yva[i:i + bs]
        m = (y != -100)
        corr += (out.argmax(-1)[m] == y[m]).sum().item()
        tot += m.sum().item()
    return corr / max(1, tot)


def run(tag, ctx_mode, ed, layers, epochs, xtr, ytr, xva, yva, V,
        patience_after=25):
    torch.manual_seed(42)
    bb = WATBackboneX(V, ed, n_layers=layers, chunk_size=K, max_len=T,
                      dropout=0.0, ctx_mode=ctx_mode, intra=False)
    model = LMModel(bb, V).to(DEVICE)
    npar = sum(p.numel() for p in model.parameters())
    print(f"--- {tag} ({npar:,} params, до {epochs} эпох) ---")
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    tl = DataLoader(TensorDataset(xtr, ytr), batch_size=256, shuffle=True,
                    generator=torch.Generator().manual_seed(42))
    best, best_ep, cross = 0.0, 0, None
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in tl:
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = F.cross_entropy(out.reshape(-1, out.size(-1)),
                                   yb.reshape(-1), ignore_index=-100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        acc = val_acc(model, xva, yva)
        if acc > best + 1e-4:
            best, best_ep = acc, ep
        if cross is None and acc * 100 >= 15.0:
            cross = ep
        if ep % 15 == 0 or ep == 1 or cross == ep:
            note = "  <- ПЕРЕХОД" if cross == ep else ""
            print(f"  [{tag}] ep{ep:3d}  loss={loss.item():.3f}  "
                  f"val_acc={acc*100:5.1f}%{note}  ({time.time()-t0:.0f}s)",
                  flush=True)
        if cross is not None and ep - best_ep >= patience_after:
            print(f"  [{tag}] полка стабильна, выход (ep{ep}, "
                  f"best={best*100:.1f}%)", flush=True)
            break
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return best, cross, ep


def main():
    xtr, ytr, xva, yva, V = data()
    runs = [
        ("R1 v0 tiny 300ep", "mean", 96, 1, 300),
        ("R2 v0 wide ED192/L2", "mean", 192, 2, 100),
        ("R3 v5 attn tiny", "attn", 96, 1, 100),
        ("R4 v1 ladder tiny", "prefix_tree", 96, 1, 100),
    ]
    results = []
    for tag, mode, ed, L, cap in runs:
        b, c, r = run(tag, mode, ed, L, cap, xtr, ytr, xva, yva, V)
        results.append((tag, b, c, r))
    print("\n" + "=" * 62)
    print(f"{'прогон':<22} {'переход':>8} {'best':>8} {'эпох':>6}")
    print("-" * 62)
    for tag, b, c, r in results:
        print(f"{tag:<22} {str(c or '—'):>8} {b*100:>7.1f}% {r:>6}")
    print(f"(шанс {CHANCE}%; v0@80эп=31.1%; потолок ~77%; TR tiny@15эп=72.2%)")


if __name__ == "__main__":
    main()
