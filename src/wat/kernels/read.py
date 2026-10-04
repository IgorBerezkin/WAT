import math
import time
import traceback

import torch
from torch.autograd.function import once_differentiable

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:
    HAVE_TRITON = False

TILE = 2048
OFFSETS = {}
CHECK_CASES = ((3, 2, 8), (2, 16, 24), (2, 64, 32), (4, 512, 104), (2, 2048, 168), (1, 8192, 104), (2, 1000, 256))
BENCH_CASES = ((32, 512, 104), (8, 2048, 104), (2, 8192, 104), (32, 512, 168), (32, 512, 256))


def available():
    return HAVE_TRITON and torch.cuda.is_available()


if HAVE_TRITON:
    @triton.jit
    def _read_fwd(Q, NODES, OFFS, G, E, OUT, K, R, D, n_low, n_levels,
                  BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr):
        t0 = tl.program_id(0) * BLOCK_T
        b = tl.program_id(1).to(tl.int64)
        t = t0 + tl.arange(0, BLOCK_T)
        d = tl.arange(0, BLOCK_D)
        tm = t < K
        dm = d < D
        pos = (b * K + t)[:, None] * D + d[None, :]
        q = tl.load(Q + pos, mask=tm[:, None] & dm[None, :], other=0.0).to(tl.float32)
        acc = tl.zeros((BLOCK_T, BLOCK_D), tl.float32)
        for level in range(0, n_low):
            base = tl.load(OFFS + level) + b * R
            g = tl.load(G + level * D + d, mask=dm, other=0.0)
            e = tl.load(E + level * D + d, mask=dm, other=0.0)
            node = (t >> level) - 1
            ok = (tm & (node >= 0))[:, None] & dm[None, :]
            n = tl.load(NODES + (base + node)[:, None] * D + d[None, :], mask=ok, other=0.0).to(tl.float32)
            s = tl.sigmoid(q + e[None, :])
            acc += (2.0 * s) * (n * g[None, :])
        for level in range(n_low, n_levels):
            base = tl.load(OFFS + level) + b * R
            g = tl.load(G + level * D + d, mask=dm, other=0.0)
            e = tl.load(E + level * D + d, mask=dm, other=0.0)
            node = (t0 >> level) - 1
            n = tl.load(NODES + (base + node) * D + d, mask=dm & (node >= 0), other=0.0).to(tl.float32)
            s = tl.sigmoid(q + e[None, :])
            acc += (2.0 * s) * (n * g)[None, :]
        tl.store(OUT + pos, acc, mask=tm[:, None] & dm[None, :])

    @triton.jit
    def _read_bwd(Q, NODES, OFFS, G, E, DOUT, DQ, DNODES, DG, DE, K, R, D, n_low, n_levels,
                  BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr):
        t0 = tl.program_id(0) * BLOCK_T
        b = tl.program_id(1).to(tl.int64)
        t = t0 + tl.arange(0, BLOCK_T)
        d = tl.arange(0, BLOCK_D)
        tm = t < K
        dm = d < D
        pos = (b * K + t)[:, None] * D + d[None, :]
        q = tl.load(Q + pos, mask=tm[:, None] & dm[None, :], other=0.0).to(tl.float32)
        dout = tl.load(DOUT + pos, mask=tm[:, None] & dm[None, :], other=0.0).to(tl.float32)
        dq = tl.zeros((BLOCK_T, BLOCK_D), tl.float32)
        for level in range(0, n_low):
            base = tl.load(OFFS + level) + b * R
            g = tl.load(G + level * D + d, mask=dm, other=0.0)
            e = tl.load(E + level * D + d, mask=dm, other=0.0)
            node = (t >> level) - 1
            ok = (tm & (node >= 0))[:, None] & dm[None, :]
            addr = (base + node)[:, None] * D + d[None, :]
            n = tl.load(NODES + addr, mask=ok, other=0.0).to(tl.float32)
            s = tl.sigmoid(q + e[None, :])
            w = dout * (2.0 * s)
            dn = w * g[None, :]
            dz = dn * (1.0 - s) * n
            dq += dz
            tl.atomic_add(DNODES + addr, dn, mask=ok, sem="relaxed")
            tl.atomic_add(DG + level * D + d, tl.sum(w * n, axis=0), mask=dm, sem="relaxed")
            tl.atomic_add(DE + level * D + d, tl.sum(dz, axis=0), mask=dm, sem="relaxed")
        for level in range(n_low, n_levels):
            base = tl.load(OFFS + level) + b * R
            g = tl.load(G + level * D + d, mask=dm, other=0.0)
            e = tl.load(E + level * D + d, mask=dm, other=0.0)
            node = (t0 >> level) - 1
            ok = dm & (node >= 0)
            addr = (base + node) * D + d
            n = tl.load(NODES + addr, mask=ok, other=0.0).to(tl.float32)
            s = tl.sigmoid(q + e[None, :])
            w = dout * (2.0 * s)
            u = w * (1.0 - s)
            gn = g * n
            dq += u * gn[None, :]
            sw = tl.sum(w, axis=0)
            tl.atomic_add(DNODES + addr, sw * g, mask=ok, sem="relaxed")
            tl.atomic_add(DG + level * D + d, sw * n, mask=ok, sem="relaxed")
            tl.atomic_add(DE + level * D + d, tl.sum(u, axis=0) * gn, mask=ok, sem="relaxed")
        tl.store(DQ + pos, dq.to(DQ.dtype.element_ty), mask=tm[:, None] & dm[None, :])


def _tiles(K, D):
    block_d = triton.next_power_of_2(D)
    block_t = min(K, max(2, TILE // block_d))
    return block_t, block_d, block_t.bit_length() - 1


def _offsets(sizes, device):
    key = (tuple(sizes), str(device))
    if key not in OFFSETS:
        OFFSETS[key] = torch.tensor([sum(sizes[:i]) for i in range(len(sizes))], dtype=torch.int64, device=device)
    return OFFSETS[key]


class ReadFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, flat, offsets, G, E, Lv, K):
        B, R, D = flat.shape
        if q.numel() != B * K * D:
            raise ValueError(f"read: q has {q.numel()} elements, expected {B * K * D}")
        if not torch.is_tensor(offsets):
            offsets = torch.tensor(list(offsets), dtype=torch.int64)
        offsets = offsets.to(device=q.device, dtype=torch.int64)
        block_t, block_d, log_t = _tiles(K, D)
        qc, fc = q.contiguous(), flat.contiguous()
        g32, e32 = G.float().contiguous(), E.float().contiguous()
        out = torch.empty(qc.shape, device=q.device, dtype=torch.float32)
        grid = (triton.cdiv(K, block_t), B)
        _read_fwd[grid](qc, fc, offsets, g32, e32, out, K, R, D, min(Lv, log_t), Lv,
                        BLOCK_T=block_t, BLOCK_D=block_d, num_warps=4)
        ctx.save_for_backward(qc, fc, g32, e32)
        ctx.offsets, ctx.Lv, ctx.K = offsets, Lv, K
        ctx.dtypes = (flat.dtype, G.dtype, E.dtype)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, gout):
        qc, fc, g32, e32 = ctx.saved_tensors
        B, R, D = fc.shape
        block_t, block_d, log_t = _tiles(ctx.K, D)
        dq = torch.empty_like(qc)
        dflat = torch.zeros(fc.shape, device=qc.device, dtype=torch.float32)
        dg = torch.zeros(g32.shape, device=qc.device, dtype=torch.float32)
        de = torch.zeros(e32.shape, device=qc.device, dtype=torch.float32)
        grid = (triton.cdiv(ctx.K, block_t), B)
        _read_bwd[grid](qc, fc, ctx.offsets, g32, e32, gout.float().contiguous(), dq, dflat, dg, de,
                        ctx.K, R, D, min(ctx.Lv, log_t), ctx.Lv, BLOCK_T=block_t, BLOCK_D=block_d, num_warps=4)
        fd, gd, ed = ctx.dtypes
        return dq, dflat.to(fd), None, dg.to(gd), de.to(ed), None, None


def fused_read(block, levels, x):
    Lv = len(levels) - 1
    if Lv == 0:
        return torch.zeros_like(levels[0])
    q = block.W_read(x)
    K, D = levels[0].shape[-2], levels[0].shape[-1]
    B = levels[0].numel() // (K * D)
    flat = torch.cat([lv.reshape(B, -1, D) for lv in levels[:-1]], dim=1)
    offsets = _offsets([lv.shape[-2] for lv in levels[:-1]], flat.device)
    out = ReadFn.apply(q, flat, offsets, block.level_gain[:Lv], block.level_emb[:Lv], Lv, K)
    dtype = q.dtype
    for t in list(levels[:-1]) + [block.level_gain, block.level_emb]:
        dtype = torch.promote_types(dtype, t.dtype)
    return out.reshape(levels[0].shape).to(dtype)


def error_text():
    return traceback.format_exc()[-4000:]


def _case(device, B, T, D, amp, seed=0):
    from wat.main.model import MainBlock
    torch.manual_seed(seed)
    block = MainBlock(D, max_len=T, mem=None).to(device)
    block.fused_read = False
    with torch.no_grad():
        block.level_gain.uniform_(0.5, 1.5)
        block.level_emb.normal_(0.0, 0.5)
    K = 1 << (T - 1).bit_length()
    grid = torch.randn(B, 1, K, D, device=device, requires_grad=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        tree = block.tree(grid)
    levels = [grid] + [lv.detach().float().requires_grad_() for lv in tree[1:]]
    return block, levels, torch.randn(B, 1, K, D, device=device)


def _compare(ref, new, tol):
    if ref is None or new is None:
        return {"abs": None, "rel": None, "tol": tol, "pass": ref is None and new is None}
    if ref.shape != new.shape:
        return {"abs": None, "rel": None, "tol": tol, "pass": False, "shapes": [list(ref.shape), list(new.shape)]}
    ref, new = ref.detach().double(), new.detach().double()
    err = (ref - new).abs().max().item()
    scale = ref.abs().max().item()
    rel = err / scale if scale > 0 else err
    return {"abs": err, "rel": rel, "ref_max": scale, "tol": tol, "pass": bool(rel < tol)}


def _check_case(device, B, T, D, amp):
    block, levels, gout = _case(device, B, T, D, amp)
    grid = levels[0]
    named = [("grid", grid)] + [(f"level{i}", lv) for i, lv in enumerate(levels) if i > 0]
    named += [("W_read.weight", block.W_read.weight), ("W_read.bias", block.W_read.bias),
              ("level_gain", block.level_gain), ("level_emb", block.level_emb)]
    outs, grads = [], []
    for fn in (block.read, lambda lv, x: fused_read(block, lv, x)):
        for _, tensor in named:
            tensor.grad = None
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            out = fn(levels, grid)
        out.backward(gout)
        outs.append(out.detach())
        grads.append({name: tensor.grad for name, tensor in named})
    loose = ("grid", "W_read.weight", "W_read.bias") if amp else ()
    errors = {"out": _compare(outs[0], outs[1], 1e-4)}
    for name, _ in named:
        errors[name] = _compare(grads[0][name], grads[1][name], 1e-2 if name in loose else 1e-4)
    return {"K": grid.shape[2], "levels": len(levels) - 1, "out_dtype": [str(o.dtype) for o in outs], "errors": errors,
            "pass": outs[0].dtype == outs[1].dtype and all(e["pass"] for e in errors.values())}


def check(device="cuda", cases=CHECK_CASES):
    records = []
    for B, T, D in cases:
        for amp in (True, False):
            rec = {"B": B, "T": T, "D": D, "amp": "fp16" if amp else "fp32"}
            try:
                rec.update(_check_case(device, B, T, D, amp))
            except Exception:
                rec.update({"pass": False, "error": error_text()})
            records.append(rec)
            torch.cuda.empty_cache()
    return records


def _timed(fn, min_time, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    n = 1
    while True:
        start = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if elapsed >= min_time:
            return elapsed * 1e3 / n, n
        n = max(2 * n, math.ceil(1.2 * n * min_time / max(elapsed, 1e-6)))


def _bench_case(device, B, T, D, amp, min_time, warmup):
    block, levels, gout = _case(device, B, T, D, amp)
    grid = levels[0]
    inputs = levels[:-1] + [block.W_read.weight, block.W_read.bias, block.level_gain, block.level_emb]
    rec = {"K": grid.shape[2], "levels": len(levels) - 1}
    for name, fn in (("ref", block.read), ("fused", lambda lv, x: fused_read(block, lv, x))):
        def fwd():
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                return fn(levels, grid)

        def fwdbwd():
            return torch.autograd.grad(fwd(), inputs, gout)

        rec[f"{name}_fwd_ms"], rec[f"{name}_fwd_iters"] = _timed(fwd, min_time, warmup)
        rec[f"{name}_fwdbwd_ms"], rec[f"{name}_fwdbwd_iters"] = _timed(fwdbwd, min_time, warmup)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
        start = torch.cuda.memory_allocated(device)
        fwdbwd()
        torch.cuda.synchronize()
        rec[f"{name}_peak_mb"] = round((torch.cuda.max_memory_allocated(device) - start) / 2 ** 20, 1)
    rec["fwd_speedup"] = rec["ref_fwd_ms"] / rec["fused_fwd_ms"]
    rec["fwdbwd_speedup"] = rec["ref_fwdbwd_ms"] / rec["fused_fwdbwd_ms"]
    return rec


def bench(device="cuda", cases=BENCH_CASES, min_time=0.3, warmup=5, amp=True):
    records = []
    for B, T, D in cases:
        rec = {"B": B, "T": T, "D": D, "amp": "fp16" if amp else "fp32"}
        try:
            rec.update(_bench_case(device, B, T, D, amp, min_time, warmup))
        except Exception:
            rec["error"] = error_text()
        records.append(rec)
        torch.cuda.empty_cache()
    return records
