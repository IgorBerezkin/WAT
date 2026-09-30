import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from wat_lab import WATBackboneX, TransformerBackbone, LSTMBackbone, LMModel, make_copy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ED, K = 96, 32
CHANCE = 6.25

STAGES = [
    ("S1", 128, 1, 60),
    ("S2", 128, 4, 80),
    ("S3", 256, 8, 100),
    ("S4", 512, 16, 140),
]


def stage_data(T, nm):
    xtr, ytr, V = make_copy(4000, T, nm, seed=42)
    xva, yva, _ = make_copy(1000, T, nm, seed=43)
    return (xtr.to(DEVICE), ytr.to(DEVICE),
            xva.to(DEVICE), yva.to(DEVICE), V)


def build(model_name, V, T):
    torch.manual_seed(42)
    if model_name == "wat":
        bb = WATBackboneX(V, ED, n_layers=1, chunk_size=K, max_len=T,
                          dropout=0.0, ctx_mode="mean", intra=False)
    elif model_name == "transformer":
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


def train_one(tag, model, xtr, ytr, xva, yva, bs, max_epochs,
              patience_after=15):
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    tl = DataLoader(TensorDataset(xtr, ytr), batch_size=bs, shuffle=True,
                    generator=torch.Generator().manual_seed(42))
    best, best_ep, cross = 0.0, 0, None
    t0 = time.time()
    for ep in range(1, max_epochs + 1):
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
        show = (ep % 10 == 0 or ep == 1 or cross == ep)
        if show:
            note = "  <- ПЕРЕХОД" if cross == ep else ""
            print(f"  [{tag}] ep{ep:3d}  loss={loss.item():.3f}  "
                  f"val_acc={acc*100:5.1f}%{note}  ({time.time()-t0:.0f}s)",
                  flush=True)
        if cross is not None and ep - best_ep >= patience_after:
            print(f"  [{tag}] полка стабильна, ранний выход (ep{ep})",
                  flush=True)
            break
    return best, cross, ep


def main():
    results = []
    for sname, T, nm, cap in STAGES:
        bs = 128 if T >= 512 else 256
        blind = (K - nm) / (T - nm) * 100
        print("=" * 70)
        print(f"{sname}: T={T}, n_mem={nm}  |  слепая зона ~{blind:.1f}% "
              f"токенов -> потолок-ориентир ~{100-blind:.0f}%+")
        print("=" * 70)
        xtr, ytr, xva, yva, V = stage_data(T, nm)
        for mname, cap_m in (("wat", cap), ("transformer", 15), ("lstm", 15)):
            model = build(mname, V, T)
            npar = sum(p.numel() for p in model.parameters())
            print(f"--- {mname} ({npar:,} params, до {cap_m} эпох) ---")
            best, cross, ran = train_one(f"{sname} {mname}", model,
                                         xtr, ytr, xva, yva, bs, cap_m)
            results.append((sname, T, nm, mname, best, cross, ran))
            del model
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
    print("\n" + "=" * 70)
    print(f"{'ступень':<10} {'модель':<12} {'переход':>8} {'best':>8} "
          f"{'эпох':>6}")
    print("-" * 70)
    for sname, T, nm, mname, best, cross, ran in results:
        print(f"{sname+f' T{T}/nm{nm}':<10} {mname:<12} "
              f"{str(cross or '—'):>8} {best*100:>7.1f}% {ran:>6}")
    print(f"(шанс {CHANCE}%; референс 1M @S4: TR 51.7 / LSTM 6.3 / "
          f"WAT-старый-бюджет 6.5)")


if __name__ == "__main__":
    main()
