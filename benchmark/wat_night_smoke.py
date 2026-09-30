import sys, time, traceback
sys.path.insert(0, ".")
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import wat_night as N
import wat_night_x as X
from wat_lab import make_copy, make_recall, load_shakespeare, LMDataset, LMModel

DEVICE = N.DEVICE
FAILS = []


def check(name, fn):
    t0 = time.time()
    try:
        fn()
        print(f"PASS  {name:<26} ({time.time()-t0:.1f}s)", flush=True)
    except Exception:
        tb = traceback.format_exc()
        print(f"FAIL  {name:<26} ({time.time()-t0:.1f}s)\n{tb}", flush=True)
        FAILS.append((name, tb.strip().splitlines()[-1]))
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def tiny_copy(T, nm, n=256, nv=128):
    xtr, ytr, V = make_copy(n, T, nm, seed=42)
    xva, yva, _ = make_copy(nv, T, nm, seed=43)
    return (xtr.to(DEVICE), ytr.to(DEVICE),
            xva.to(DEVICE), yva.to(DEVICE), V)


def smoke_core():
    xtr, ytr, xva, yva, V = tiny_copy(512, 16)

    core = [
        ("c_v0_tiny",    dict(), 128),
        ("c_gain_tiny",  dict(gain=8.0), 128),
        ("c_lane_tiny",  dict(lane=True), 128),
        ("c_gain_ladder", dict(gain=8.0, ladder=True), 128),
        ("c_wide",       dict(ed=192, layers=2), 128),
        ("c_wide_gain",  dict(ed=192, layers=2, gain=8.0), 128),
        ("c_v2_full",    dict(ed=192, layers=2, gain=8.0, ladder=True), 128),
        ("c_big_gain",   dict(ed=288, layers=3, gain=8.0), 96),
        ("c_gain_seed1", dict(gain=8.0), 128),
    ]
    for rid, kw, bs in core:
        def fn(rid=rid, kw=kw, bs=bs):
            m = N.build(V, 512, seed=1 if rid.endswith("seed1") else 42, **kw)
            r = N.train(f"smoke:{rid}", m, xtr, ytr, xva, yva, bs, cap=1,
                        log_every=1)
            assert 0 <= r["best"] <= 100
        check(rid, fn)

    def fn_cur():
        m = N.build(18, 512, gain=8.0)
        for T, nm in ((128, 4), (256, 8), (512, 16)):
            a, b, c, d, _ = tiny_copy(T, nm)
            N.train(f"smoke:cur_{T}", m, a, b, c, d,
                    128 if T >= 512 else 256, cap=1, log_every=1)
    check("cur_gain_128_256_512", fn_cur)

    def fn_recall():
        xtr, ytr, V = make_recall(512, 256, 12, seed=42)
        xva, yva, _ = make_recall(128, 256, 12, seed=43)
        m = N.build(V, 256)
        N.train("smoke:recall", m, xtr.to(DEVICE), ytr.to(DEVICE),
                xva.to(DEVICE), yva.to(DEVICE), 256, cap=1, log_every=1)
    check("r_ctrl/r_data (recall)", fn_recall)

    def fn_lm():
        data, V = load_shakespeare()
        tr, va = data[:20000], data[20000:24000]
        tl = DataLoader(LMDataset(tr, 512, 512), batch_size=32, shuffle=True)
        vl = DataLoader(LMDataset(va, 512, 512), batch_size=32)
        for kw in (dict(), dict(gain=8.0), dict(ed=192, layers=2)):
            m = N.build(V, 512, **kw)
            opt = torch.optim.Adam(m.parameters(), lr=3e-4)
            for xb, yb in tl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                loss = F.cross_entropy(m(xb).reshape(-1, V), yb.reshape(-1))
                opt.zero_grad(); loss.backward(); opt.step()
                break
            with torch.no_grad():
                for xb, yb in vl:
                    m(xb.to(DEVICE)); break
    check("lm_v0/gain/wide", fn_lm)

    def fn_s2():
        a, b, c, d, V = tiny_copy(128, 4)
        for kw in (dict(), dict(gain=8.0), dict(ladder=True)):
            m = N.build(V, 128, **kw)
            N.train("smoke:s2", m, a, b, c, d, 256, cap=1, log_every=1)
    check("s2_v0/gain/ladder_safe", fn_s2)


def smoke_x():
    xtr, ytr, xva, yva, V = tiny_copy(512, 16)

    simple = ["x_matrix", "x_kv_rope", "x_rotor", "x_registers",
              "x_gate_bias", "x_ema", "x_cheatsheet", "x_hybrid_attn"]
    for kind in simple:
        def fn(kind=kind):
            ok, pos = X.verify(kind, V)
            assert ok, f"причинность: утечка на p={pos}"
            m = LMModel(X.make_backbone(kind, V, 512), V).to(DEVICE)
            X.train(f"smoke:{kind}", m, xtr, ytr, xva, yva, 128, cap=1)
        check(kind, fn)

    def fn_anneal():
        ok, pos = X.verify("x_ladder_anneal", V)
        assert ok, pos
        m = LMModel(X.make_backbone("x_ladder_anneal", V, 512), V).to(DEVICE)

        def hook(model, ep):
            model.backbone.layers[0].ladder_w = 0.5
        X.train("smoke:x_ladder_anneal", m, xtr, ytr, xva, yva, 128, cap=1,
                epoch_hook=hook)
        assert m.backbone.layers[0].ladder_w == 0.5
    check("x_ladder_anneal", fn_anneal)

    def fn_chunk():
        ok, pos = X.verify("x_chunk_curr", V)
        assert ok, pos
        m = LMModel(X.make_backbone("x_chunk_curr", V, 512), V).to(DEVICE)
        for Kc in (128, 64, 32):
            m.backbone.layers[0].K = Kc
            X.train(f"smoke:x_chunk K{Kc}", m, xtr, ytr, xva, yva, 128, cap=1)
    check("x_chunk_curr", fn_chunk)

    def fn_echo():
        m = LMModel(X.make_backbone("x_echo", V, 512), V).to(DEVICE)
        y_echo = torch.full_like(xtr, -100)
        y_echo[:, 64:] = xtr[:, :-64]
        X.train("smoke:x_echo1", m, xtr, y_echo, xva, yva, 128, cap=1)
        X.train("smoke:x_echo2", m, xtr, ytr, xva, yva, 128, cap=1)
    check("x_echo", fn_echo)

    def fn_aux():
        import types
        from wat_lab import WATBlockX
        bb = X.make_backbone("x_aux_ctx", V, 512)
        blk = bb.layers[0]
        orig = WATBlockX._ctx_mean

        def stash(self, s):
            out = orig(self, s)
            self._ctx_last = out
            return out
        blk._ctx_mean = types.MethodType(stash, blk)
        model = LMModel(bb, V).to(DEVICE)
        aux_head = torch.nn.Linear(96, 16).to(DEVICE)
        model.aux_head = aux_head

        def aux_fn(m, xb):
            ctx = m.backbone.layers[0]._ctx_last
            B, C, D = ctx.shape
            onehot = F.one_hot(xb.clamp(max=16), 17)[..., :16].float()
            per_chunk = onehot.view(B, C, -1, 16).sum(2)
            seen = torch.cumsum(per_chunk, dim=1)
            seen = torch.cat([torch.zeros_like(seen[:, :1]),
                              seen[:, :-1]], 1).clamp(max=1.0)
            return 0.3 * F.binary_cross_entropy_with_logits(
                aux_head(ctx), seen)
        X.train("smoke:x_aux", model, xtr, ytr, xva, yva, 128, cap=1,
                aux_fn=aux_fn)
    check("x_aux_ctx", fn_aux)


if __name__ == "__main__":
    t0 = time.time()
    print(f"SMOKE START {time.strftime('%H:%M:%S')} device={DEVICE}\n")
    smoke_core()
    smoke_x()
    print("\n" + "=" * 60)
    if FAILS:
        print(f"ПРОВАЛЕНО {len(FAILS)}:")
        for name, last in FAILS:
            print(f"  FAIL {name}: {last}")
        print("НОЧЬ НЕ ЗАПУСКАТЬ — шли этот вывод, чиним.")
        sys.exit(1)
    print(f"ВСЕ PASS за {(time.time()-t0)/60:.1f} мин — ночь можно запускать:")
    print("  python wat_night.py 2>&1 | Tee-Object -FilePath night_console.log")
