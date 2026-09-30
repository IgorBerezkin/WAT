import json, os, time, traceback, types
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from wat_lab import (WATBackboneX, WATBlockX, LMModel, make_copy, make_recall,
                     load_shakespeare, LMDataset)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K = 32
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "benchmark", "night")
os.makedirs(OUT, exist_ok=True)
RES_PATH = os.path.join(OUT, "results.json")
RESULTS = json.load(open(RES_PATH, encoding="utf-8")) \
    if os.path.exists(RES_PATH) else {}


def save(rid, payload):
    RESULTS[rid] = payload
    with open(RES_PATH, "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, indent=1, ensure_ascii=False)
    write_summary()


class GainedLinear(nn.Module):
    def __init__(self, lin, alpha0=8.0):
        super().__init__()
        self.lin = lin
        self.alpha = nn.Parameter(torch.tensor(float(alpha0)))

    def forward(self, x):
        return self.alpha * self.lin(x)


def _ctx_ladder_safe(self, s):
    return 0.5 * WATBlockX._ctx_prefix_tree(self, s) + \
           0.5 * WATBlockX._ctx_mean(self, s)


def _tree_with_lane(self, chunks):
    import math as _m
    s = WATBlockX._tree_reduction_all(self, chunks)
    g = torch.sigmoid(self.lane_g(chunks))
    v = self.lane_v(chunks)
    return s + (g * v).sum(dim=2) / _m.sqrt(chunks.size(2))


def build(V, T, ed=96, layers=1, gain=None, ladder=False, lane=False,
          seed=42):
    torch.manual_seed(seed)
    mode = "prefix_tree" if ladder else "mean"
    bb = WATBackboneX(V, ed, n_layers=layers, chunk_size=K, max_len=T,
                      dropout=0.0, ctx_mode=mode, intra=False)
    scale = 0.02 / (2 * max(1, layers)) ** 0.5
    for blk in bb.layers:
        if gain is not None:
            blk.W_global = GainedLinear(blk.W_global, gain)
        if ladder:
            blk._ctx_prefix_tree = types.MethodType(_ctx_ladder_safe, blk)
        if lane:
            blk.lane_g = nn.Linear(ed, 1)
            blk.lane_v = nn.Linear(ed, ed)
            nn.init.normal_(blk.lane_v.weight, 0.0, scale * 4)
            nn.init.zeros_(blk.lane_v.bias)
            nn.init.zeros_(blk.lane_g.weight)
            nn.init.zeros_(blk.lane_g.bias)
            blk._tree_reduction_all = types.MethodType(_tree_with_lane, blk)
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


def train(rid, model, xtr, ytr, xva, yva, bs, cap, patience=80, lr=3e-4,
          wd=1e-4, seed=42, log_every=10):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    tl = DataLoader(TensorDataset(xtr, ytr), batch_size=bs, shuffle=True,
                    generator=torch.Generator().manual_seed(seed))
    best, best_ep, cross = 0.0, 0, None
    curve = []
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
        if ep % log_every == 0 or ep == 1 or cross == ep:
            curve.append([ep, round(acc * 100, 2)])
            a = ""
            blk0 = model.backbone.layers[0]
            if isinstance(getattr(blk0, "W_global", None), GainedLinear):
                a = f"  alpha={blk0.W_global.alpha.item():.2f}"
            note = "  <- ПЕРЕХОД" if cross == ep else ""
            print(f"  [{rid}] ep{ep:3d} loss={loss.item():.3f} "
                  f"val={acc*100:5.1f}%{note}{a} ({time.time()-t0:.0f}s)",
                  flush=True)
        if cross is not None and ep - best_ep >= patience:
            print(f"  [{rid}] полка стабильна, выход ep{ep}", flush=True)
            break
    return {"best": round(best * 100, 2), "cross": cross, "epochs": ep,
            "sec": round(time.time() - t0), "curve": curve}


def run_guarded(rid, fn):
    if rid in RESULTS and "error" not in RESULTS[rid]:
        print(f"[skip] {rid} уже готов: {RESULTS[rid].get('best')}%",
              flush=True)
        return
    print("=" * 70)
    print(f"RUN {rid}   ({time.strftime('%H:%M:%S')})")
    print("=" * 70)
    try:
        payload = fn()
        save(rid, payload)
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        save(rid, {"error": tb[-1500:]})
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


_CACHE = {}


def copy_data(T, nm):
    key = ("copy", T, nm)
    if key not in _CACHE:
        xtr, ytr, V = make_copy(4000, T, nm, seed=42)
        xva, yva, _ = make_copy(1000, T, nm, seed=43)
        _CACHE[key] = (xtr.to(DEVICE), ytr.to(DEVICE),
                       xva.to(DEVICE), yva.to(DEVICE), V)
    return _CACHE[key]


def recall_data(n):
    key = ("recall", n)
    if key not in _CACHE:
        xtr, ytr, V = make_recall(n, 256, 12, seed=42)
        xva, yva, _ = make_recall(1000, 256, 12, seed=43)
        _CACHE[key] = (xtr.to(DEVICE), ytr.to(DEVICE),
                       xva.to(DEVICE), yva.to(DEVICE), V)
    return _CACHE[key]


def block_combat():
    xtr, ytr, xva, yva, V = copy_data(512, 16)

    def mk(rid, cap, bs=128, seed=42, **kw):
        def fn():
            m = build(V, 512, seed=seed, **kw)
            return train(rid, m, xtr, ytr, xva, yva, bs, cap, seed=seed)
        run_guarded(rid, fn)

    mk("c_v0_tiny", 400)
    mk("c_gain_tiny", 400, gain=8.0)
    mk("c_lane_tiny", 300, lane=True)
    mk("c_gain_ladder", 300, gain=8.0, ladder=True)
    mk("c_wide", 350, ed=192, layers=2)
    mk("c_wide_gain", 350, ed=192, layers=2, gain=8.0)
    mk("c_v2_full", 350, ed=192, layers=2, gain=8.0, ladder=True)
    mk("c_big_gain", 200, ed=288, layers=3, gain=8.0, bs=96)
    mk("c_gain_seed1", 250, gain=8.0, seed=1)
    mk("c_gain_seed2", 250, gain=8.0, seed=2)


def block_curriculum():
    def fn():
        V = 18
        model = build(V, 512, gain=8.0)
        total = {"phases": [], "sec": 0}
        for T, nm, cap in ((128, 4, 60), (256, 8, 60), (512, 16, 200)):
            xtr, ytr, xva, yva, _ = copy_data(T, nm)
            r = train(f"cur_{T}", model, xtr, ytr, xva, yva,
                      bs=128 if T >= 512 else 256, cap=cap, patience=60)
            total["phases"].append({"T": T, **{k: r[k] for k in
                                               ("best", "cross", "epochs")}})
            total["sec"] += r["sec"]
        total["best"] = total["phases"][-1]["best"]
        total["cross"] = total["phases"][-1]["cross"]
        return total
    run_guarded("cur_gain_128_256_512", fn)


def block_recall():
    for rid, n, cap in (("r_ctrl_6k", 6000, 250), ("r_data_20k", 20000, 250)):
        def fn(n=n, cap=cap, rid=rid):
            xtr, ytr, xva, yva, V = recall_data(n)
            m = build(V, 256)
            return train(rid, m, xtr, ytr, xva, yva, bs=256, cap=cap,
                         patience=60)
        run_guarded(rid, fn)


def block_lm():
    data, V = load_shakespeare()
    n = len(data)
    tr, va = data[:int(n * .9)], data[int(n * .9):int(n * .95)]
    tl = DataLoader(LMDataset(tr, 512, 256), batch_size=64, shuffle=True,
                    generator=torch.Generator().manual_seed(42))
    vl = DataLoader(LMDataset(va, 512, 512), batch_size=64)

    def lm_run(rid, **kw):
        def fn():
            m = build(V, 512, **kw)
            opt = torch.optim.Adam(m.parameters(), lr=3e-4, weight_decay=1e-4)
            best = 0.0
            t0 = time.time()
            for ep in range(1, 13):
                m.train()
                for xb, yb in tl:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    opt.zero_grad(set_to_none=True)
                    out = m(xb)
                    loss = F.cross_entropy(out.reshape(-1, V),
                                           yb.reshape(-1))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                    opt.step()
                corr = tot = 0
                m.eval()
                with torch.no_grad():
                    for xb, yb in vl:
                        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                        p = m(xb).argmax(-1)
                        corr += (p == yb).sum().item()
                        tot += yb.numel()
                acc = corr / tot
                best = max(best, acc)
                print(f"  [{rid}] ep{ep} val_acc={acc*100:.2f}% "
                      f"({time.time()-t0:.0f}s)", flush=True)
            return {"best": round(best * 100, 2), "cross": None,
                    "epochs": 12, "sec": round(time.time() - t0)}
        run_guarded(rid, fn)

    lm_run("lm_v0_tiny")
    lm_run("lm_gain_tiny", gain=8.0)
    lm_run("lm_wide", ed=192, layers=2)


def block_s2():
    xtr, ytr, xva, yva, V = copy_data(128, 4)
    for rid, kw in (("s2_v0", {}), ("s2_gain", {"gain": 8.0}),
                    ("s2_ladder_safe", {"ladder": True})):
        def fn(kw=kw, rid=rid):
            m = build(V, 128, **kw)
            return train(rid, m, xtr, ytr, xva, yva, bs=256, cap=400,
                         patience=60)
        run_guarded(rid, fn)


def write_summary():
    lines = ["# НОЧНОЙ ПРОГОН — сводка", "",
             "| run | best | переход | эпох | сек |", "|---|---|---|---|---|"]
    for rid, r in RESULTS.items():
        if "error" in r:
            lines.append(f"| {rid} | ERROR | — | — | — |")
        else:
            lines.append(f"| {rid} | {r.get('best','—')}% | "
                         f"{r.get('cross','—')} | {r.get('epochs','—')} | "
                         f"{r.get('sec','—')} |")
    lines += ["", "Ориентиры: боевой copy TR-tiny=40.7%, v0-старый=18.9%; "
              "recall v0=14.6% (меморизация); шанс 6.25%."]
    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    t_start = time.time()
    print(f"НОЧНОЙ СТАРТ {time.strftime('%H:%M:%S')}  device={DEVICE}")
    block_combat()
    block_curriculum()
    block_recall()
    block_lm()
    block_s2()
    try:
        from wat_night_x import run_block_x
        run_block_x()
    except Exception:
        import traceback as _tb
        print(_tb.format_exc(), flush=True)
    write_summary()
    print(f"\nГОТОВО за {(time.time()-t_start)/3600:.1f} ч. "
          f"results/benchmark/night/summary.md + results.json")
