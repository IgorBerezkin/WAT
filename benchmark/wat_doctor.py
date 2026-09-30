import argparse, sys, time, types
sys.path.insert(0, ".")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from wat_lab import (WATBackboneX, WATBlockX, LMModel, make_copy, make_recall)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K = 32
CHANCE = 6.25


class GainedLinear(nn.Module):
    def __init__(self, lin, alpha0):
        super().__init__()
        self.lin = lin
        self.alpha = nn.Parameter(torch.tensor(float(alpha0)))

    def forward(self, x):
        return self.alpha * self.lin(x)


def _ctx_attn_pos(self, x_padded, s):
    import math as _m
    B, Tp, D = x_padded.shape
    C = s.size(1)
    pos = self.sum_pos(torch.arange(C + 1, device=s.device)).unsqueeze(0)
    kv = torch.cat([self.null_summary.expand(B, 1, D).to(s.dtype), s], dim=1)
    kv = kv + pos
    q = self.W_q(x_padded)
    k, v = self.W_k(kv), self.W_v(kv)
    att = torch.einsum("btd,bcd->btc", q, k) / _m.sqrt(D)
    chunk_idx = (torch.arange(Tp, device=x_padded.device) // self.K)
    jj = torch.arange(C + 1, device=x_padded.device)
    valid = (jj.view(1, -1) == 0) | \
            ((jj.view(1, -1) - 1) < chunk_idx.view(-1, 1))
    att = att.masked_fill(~valid.unsqueeze(0), float("-inf"))
    att = F.softmax(att, dim=-1)
    return torch.einsum("btc,bcd->btd", att, v)


def _ctx_ladder_safe(self, s):
    return 0.5 * WATBlockX._ctx_prefix_tree(self, s) + \
           0.5 * WATBlockX._ctx_mean(self, s)


def build(kind, V, T):
    torch.manual_seed(42)
    if kind == "wide":
        bb = WATBackboneX(V, 192, n_layers=2, chunk_size=K, max_len=T,
                          dropout=0.0, ctx_mode="mean", intra=False)
    elif kind == "gain":
        bb = WATBackboneX(V, 96, n_layers=1, chunk_size=K, max_len=T,
                          dropout=0.0, ctx_mode="mean", intra=False)
        for blk in bb.layers:
            blk.W_global = GainedLinear(blk.W_global, 8.0)
    elif kind == "attn_pos":
        bb = WATBackboneX(V, 96, n_layers=1, chunk_size=K, max_len=T,
                          dropout=0.0, ctx_mode="attn", intra=False)
        for blk in bb.layers:
            blk.sum_pos = nn.Embedding(64, bb.embed_dim)
            nn.init.normal_(blk.sum_pos.weight, 0.0, 0.02)
            blk._ctx_attn = types.MethodType(_ctx_attn_pos, blk)
    elif kind == "ladder_safe":
        bb = WATBackboneX(V, 96, n_layers=1, chunk_size=K, max_len=T,
                          dropout=0.0, ctx_mode="prefix_tree", intra=False)
        for blk in bb.layers:
            blk._ctx_prefix_tree = types.MethodType(_ctx_ladder_safe, blk)
    else:
        bb = WATBackboneX(V, 96, n_layers=1, chunk_size=K, max_len=T,
                          dropout=0.0, ctx_mode="mean", intra=False)
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


def train_one(tag, model, xtr, ytr, xva, yva, bs, cap, wd, patience=60):
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=wd)
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
            g = ""
            if hasattr(model.backbone.layers[0], "W_global") and \
               isinstance(model.backbone.layers[0].W_global, GainedLinear):
                g = f"  alpha={model.backbone.layers[0].W_global.alpha.item():.2f}"
            print(f"  [{tag}] ep{ep:3d}  loss={loss.item():.3f}  "
                  f"val_acc={acc*100:5.1f}%{note}{g}  ({time.time()-t0:.0f}s)",
                  flush=True)
        if cross is not None and ep - best_ep >= patience:
            print(f"  [{tag}] полка стабильна, выход (ep{ep})", flush=True)
            break
    return best, cross, ep, round(time.time() - t0)


RUNS = {
    "wide":        ("copy", 160),
    "gain":        ("copy", 200),
    "attn_pos":    ("copy", 160),
    "ladder_safe": ("copy", 160),
    "recall_wd":   ("recall", 120),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="all",
                   help="all | wide,gain,attn_pos,ladder_safe,recall_wd")
    args = p.parse_args()
    todo = list(RUNS) if args.run == "all" else \
        [r.strip() for r in args.run.split(",")]

    data_cache = {}
    results = []
    for kind in todo:
        task, cap = RUNS[kind]
        if task not in data_cache:
            if task == "copy":
                xtr, ytr, V = make_copy(4000, 512, 16, seed=42)
                xva, yva, _ = make_copy(1000, 512, 16, seed=43)
                bs = 128
            else:
                xtr, ytr, V = make_recall(6000, 256, 12, seed=42)
                xva, yva, _ = make_recall(1000, 256, 12, seed=43)
                bs = 256
            data_cache[task] = (xtr.to(DEVICE), ytr.to(DEVICE),
                                xva.to(DEVICE), yva.to(DEVICE), V, bs)
        xtr, ytr, xva, yva, V, bs = data_cache[task]
        T = xtr.size(1)
        wd = 1e-2 if kind == "recall_wd" else 1e-4
        model = build(kind, V, T)
        npar = sum(p.numel() for p in model.parameters())
        print("=" * 70)
        print(f"{kind}  [{task} T={T}]  ({npar:,} params, кап {cap}, "
              f"wd={wd})")
        print("=" * 70)
        b, c, e, sec = train_one(kind, model, xtr, ytr, xva, yva, bs, cap, wd)
        results.append((kind, task, b, c, e, sec))
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print(f"{'кандидат':<13} {'задача':<7} {'переход':>8} {'best':>8} "
          f"{'эпох':>6} {'сек':>7}")
    print("-" * 70)
    for kind, task, b, c, e, sec in results:
        print(f"{kind:<13} {task:<7} {str(c or '—'):>8} {b*100:>7.1f}% "
              f"{e:>6} {sec:>7}")
    print(f"(шанс {CHANCE}%; ориентиры: copy v0-tiny=18.9%(срез), "
          f"TR-tiny полка=40.7%; recall v0 wd1e-4=14.6% меморизация)")


if __name__ == "__main__":
    main()
