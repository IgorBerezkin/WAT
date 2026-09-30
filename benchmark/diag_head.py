import sys, time
sys.path.insert(0, ".")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from wat_lab import WATBackboneX, TransformerBackbone, LMModel, make_copy, evaluate
from wat_anatomy import gradient_snr, grad_groups_wat, grad_groups_tr

torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ED, K, T = 96, 32, 128


def data(bs_train):
    xtr, ytr, V = make_copy(4000, T, 1, seed=42)
    xva, yva, _ = make_copy(1000, T, 1, seed=43)
    tl = DataLoader(TensorDataset(xtr, ytr), batch_size=bs_train, shuffle=True)
    vl = DataLoader(TensorDataset(xva, yva), batch_size=256)
    return tl, vl, V


def fresh_wat(V):
    torch.manual_seed(42)
    return WATBackboneX(V, ED, n_layers=1, chunk_size=K, max_len=T,
                        dropout=0.0, ctx_mode="mean", intra=False)


def train_loop(model, tl, vl, opt, epochs, tag, log_every=3):
    best = 0.0
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        if getattr(model, "_freeze_bb", False):
            model.backbone.eval()
        for x, y in tl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = F.cross_entropy(out.reshape(-1, out.size(-1)),
                                   y.reshape(-1), ignore_index=-100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        va, _ = evaluate(model, vl, DEVICE, "lm")
        best = max(best, va)
        if (ep + 1) % log_every == 0 or ep == epochs - 1:
            print(f"  [{tag}] ep{ep+1:2d}  loss={loss.item():.3f}  "
                  f"val_acc={va*100:5.1f}%  ({time.time()-t0:.0f}s)",
                  flush=True)
    return best


def e1():
    print("=" * 70)
    print("E1 — ЗАМОРОЖЕННЫЙ БЭКБОН, учится только голова (шанс 6.25%)")
    print("=" * 70)
    results = {}
    for tag, lr, bs, epochs in (("A: e2e-рецепт", 3e-4, 32, 15),
                                ("B: probe-рецепт", 1e-2, 256, 30)):
        tl, vl, V = data(bs)
        model = LMModel(fresh_wat(V), V).to(DEVICE)
        for p in model.backbone.parameters():
            p.requires_grad_(False)
        model._freeze_bb = True
        opt = torch.optim.Adam(model.head.parameters(), lr=lr,
                               weight_decay=1e-4)
        print(f"--- {tag} (lr={lr}, bs={bs}) ---")
        results[tag] = train_loop(model, tl, vl, opt, epochs, tag)
        del model
    return results


def e2():
    print("=" * 70)
    print("E2 — ПОЛНЫЙ E2E, голове lr x33 (head 1e-2, бэкбон 3e-4)")
    print("=" * 70)
    tl, vl, V = data(32)
    model = LMModel(fresh_wat(V), V).to(DEVICE)
    opt = torch.optim.Adam([
        {"params": model.backbone.parameters(), "lr": 3e-4},
        {"params": model.head.parameters(), "lr": 1e-2},
    ], weight_decay=1e-4)
    return train_loop(model, tl, vl, opt, 15, "E2 head-boost")


def e3():
    print("=" * 70)
    print("E3 — SNR НА ПЛАТО (после 2 эпох e2e; сравни с init из анатомии)")
    print("=" * 70)
    tl, vl, V = data(32)
    wat = LMModel(fresh_wat(V), V).to(DEVICE)
    opt = torch.optim.Adam(wat.parameters(), lr=3e-4)
    train_loop(wat, tl, vl, opt, 2, "warm WAT", log_every=1)
    xs, ys, _ = make_copy(16 * 12, T, 1, seed=100)
    snr_w = gradient_snr(wat, grad_groups_wat(wat),
                         xs.to(DEVICE), ys.to(DEVICE), 16)
    torch.manual_seed(42)
    tr = LMModel(TransformerBackbone(V, ED, n_layers=1, max_len=T,
                                     dropout=0.0), V).to(DEVICE)
    opt = torch.optim.Adam(tr.parameters(), lr=3e-4)
    train_loop(tr, tl, vl, opt, 2, "warm TR", log_every=1)
    snr_t = gradient_snr(tr, grad_groups_tr(tr),
                         xs.to(DEVICE), ys.to(DEVICE), 16)
    print(f"{'группа':<12} {'WAT@плато':>12} {'TR@плато':>12}")
    keys = list(dict.fromkeys(list(snr_w) + list(snr_t)))
    for g in keys:
        a = f"{snr_w[g]:.4f}" if g in snr_w else "—"
        b = f"{snr_t[g]:.4f}" if g in snr_t else "—"
        print(f"{g:<12} {a:>12} {b:>12}")


if __name__ == "__main__":
    r1 = e1()
    r2 = e2()
    e3()
    print("\nИТОГ: " + "  ".join(f"[{k}] {v*100:.1f}%" for k, v in r1.items())
          + f"  [E2] {r2*100:.1f}%")
    print("Дерево выводов — в шапке файла. Шли всё целиком.")
