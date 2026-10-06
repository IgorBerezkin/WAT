import time
import traceback

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:
    HAVE_TRITON = False


def available():
    return HAVE_TRITON and torch.cuda.is_available()


if HAVE_TRITON:
    @triton.jit
    def _logsig(x):
        return tl.minimum(x, 0.0) - tl.log(1.0 + tl.exp(-tl.abs(x)))

    @triton.jit
    def _beam_fwd(Q, S, U, BETA, OUT, IDX, LP, LSE, T, L, H, DH, NTOT, KP,
                  KW: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr):
        pid = tl.program_id(0)
        bh = tl.program_id(1)
        h = bh % H
        t = pid * BLOCK_T + tl.arange(0, BLOCK_T)
        tm = t < T
        d = tl.arange(0, BLOCK_D)
        dm = d < DH
        kk = tl.arange(0, KW)
        beta = tl.load(BETA + h)
        bh64 = bh.to(tl.int64)
        q = tl.load(Q + bh64 * T * DH + t[:, None] * DH + d[None, :], mask=tm[:, None] & dm[None, :], other=0.0).to(tl.float32)
        sbase = S + bh64 * NTOT * DH
        mx = tl.full((BLOCK_T,), float("-inf"), tl.float32)
        se = tl.zeros((BLOCK_T,), tl.float32)
        for m in range(0, L + 1):
            off = (2 << L) - (2 << (L - m))
            r = (t >> m) - 1
            ok = (((t >> m) & 1) == 1) & tm
            node = tl.load(sbase + (off + r)[:, None] * DH + d[None, :], mask=ok[:, None] & dm[None, :], other=0.0).to(tl.float32)
            logit = tl.where(ok, beta * tl.sum(q * node, axis=1), float("-inf"))
            nmx = tl.maximum(mx, logit)
            safe = tl.where(nmx == float("-inf"), 0.0, nmx)
            se = se * tl.exp(mx - safe) + tl.exp(logit - safe)
            mx = nmx
        lse = tl.where(mx == float("-inf"), float("-inf"), mx + tl.log(tl.where(se > 0.0, se, 1.0)))
        bidx = tl.zeros((BLOCK_T, KW), tl.int32)
        blp = tl.full((BLOCK_T, KW), float("-inf"), tl.float32)
        for i in range(0, L + 1):
            m = L - i
            off = (2 << L) - (2 << (L - m))
            r = (t >> m) - 1
            ok = (((t >> m) & 1) == 1) & tm
            node = tl.load(sbase + (off + r)[:, None] * DH + d[None, :], mask=ok[:, None] & dm[None, :], other=0.0).to(tl.float32)
            rlp = tl.where(ok, beta * tl.sum(q * node, axis=1) - lse, float("-inf"))
            live = blp > float("-inf")
            left = 2 * bidx
            lptr = sbase + (off + left)[:, :, None] * DH + d[None, None, :]
            lm = live[:, :, None] & dm[None, None, :]
            sl = tl.load(lptr, mask=lm, other=0.0).to(tl.float32)
            sr = tl.load(lptr + DH, mask=lm, other=0.0).to(tl.float32)
            z = beta * tl.sum(q[:, None, :] * (sr - sl), axis=2)
            lpl = tl.where(live, blp + _logsig(-z), float("-inf"))
            lpr = tl.where(live, blp + _logsig(z), float("-inf"))
            nidx = tl.zeros((BLOCK_T, KW), tl.int32)
            nlp = tl.full((BLOCK_T, KW), float("-inf"), tl.float32)
            for j in tl.static_range(KW):
                ml = tl.max(lpl, axis=1)
                al = tl.argmax(lpl, axis=1)
                mr = tl.max(lpr, axis=1)
                ar = tl.argmax(lpr, axis=1)
                take_root = (rlp >= ml) & (rlp >= mr)
                take_l = (rlp < ml) & (ml >= mr)
                take_r = (rlp < mr) & (mr > ml)
                val = tl.where(take_root, rlp, tl.where(take_l, ml, mr))
                pl = tl.sum(tl.where(kk[None, :] == al[:, None], left, 0), axis=1)
                pr = tl.sum(tl.where(kk[None, :] == ar[:, None], left + 1, 0), axis=1)
                cidx = tl.where(take_root, r, tl.where(take_l, pl, pr))
                nlp = tl.where(kk[None, :] == j, val[:, None], nlp)
                nidx = tl.where(kk[None, :] == j, cidx[:, None], nidx)
                rlp = tl.where(take_root, float("-inf"), rlp)
                lpl = tl.where(take_l[:, None] & (kk[None, :] == al[:, None]), float("-inf"), lpl)
                lpr = tl.where(take_r[:, None] & (kk[None, :] == ar[:, None]), float("-inf"), lpr)
            bidx = tl.where(nlp > float("-inf"), nidx, 0)
            blp = nlp
        found = blp > float("-inf")
        mxl = tl.max(blp, axis=1)
        safe = tl.where(mxl == float("-inf"), 0.0, mxl)
        e = tl.where(found, tl.exp(blp - safe[:, None]), 0.0)
        den = tl.sum(e, axis=1)
        w = e / tl.where(den > 0.0, den, 1.0)[:, None]
        ubase = U + bh64 * KP * DH
        ur = tl.load(ubase + bidx[:, :, None] * DH + d[None, None, :], mask=found[:, :, None] & dm[None, None, :], other=0.0).to(tl.float32)
        o = tl.sum(w[:, :, None] * ur, axis=1)
        tl.store(OUT + bh64 * T * DH + t[:, None] * DH + d[None, :], o, mask=tm[:, None] & dm[None, :])
        tl.store(IDX + bh64 * T * KW + t[:, None] * KW + kk[None, :], bidx, mask=tm[:, None])
        tl.store(LP + bh64 * T * KW + t[:, None] * KW + kk[None, :], blp, mask=tm[:, None])
        tl.store(LSE + bh64 * T + t, lse, mask=tm)

    @triton.jit
    def _beam_bwd(Q, S, U, BETA, IDX, LP, LSE, DOUT, DQ, DS, DD, DU, DBETA, T, L, H, DH, NTOT, KP,
                  KW: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_D: tl.constexpr):
        pid = tl.program_id(0)
        bh = tl.program_id(1)
        h = bh % H
        t = pid * BLOCK_T + tl.arange(0, BLOCK_T)
        tm = t < T
        d = tl.arange(0, BLOCK_D)
        dm = d < DH
        kk = tl.arange(0, KW)
        beta = tl.load(BETA + h)
        bh64 = bh.to(tl.int64)
        qd = tm[:, None] & dm[None, :]
        q = tl.load(Q + bh64 * T * DH + t[:, None] * DH + d[None, :], mask=qd, other=0.0).to(tl.float32)
        g = tl.load(DOUT + bh64 * T * DH + t[:, None] * DH + d[None, :], mask=qd, other=0.0).to(tl.float32)
        bidx = tl.load(IDX + bh64 * T * KW + t[:, None] * KW + kk[None, :], mask=tm[:, None], other=0)
        blp = tl.load(LP + bh64 * T * KW + t[:, None] * KW + kk[None, :], mask=tm[:, None], other=float("-inf"))
        lse = tl.load(LSE + bh64 * T + t, mask=tm, other=float("-inf"))
        found = (blp > float("-inf")) & tm[:, None]
        mxl = tl.max(blp, axis=1)
        safe = tl.where(mxl == float("-inf"), 0.0, mxl)
        e = tl.where(found, tl.exp(blp - safe[:, None]), 0.0)
        den = tl.sum(e, axis=1)
        w = e / tl.where(den > 0.0, den, 1.0)[:, None]
        fd = found[:, :, None] & dm[None, None, :]
        urow = bidx[:, :, None] * DH + d[None, None, :]
        ur = tl.load(U + bh64 * KP * DH + urow, mask=fd, other=0.0).to(tl.float32)
        dw = tl.sum(g[:, None, :] * ur, axis=2)
        dlp = w * (dw - tl.sum(w * dw, axis=1)[:, None])
        dlp = tl.where(found, dlp, 0.0)
        tl.atomic_add(DU + bh64 * KP * DH + urow, w[:, :, None] * g[:, None, :], mask=fd)
        x = t[:, None] ^ bidx
        top = tl.full((BLOCK_T, KW), -1, tl.int32)
        for m in range(0, L + 1):
            top = tl.where((x >> m) > 0, m, top)
        sbase = S + bh64 * NTOT * DH
        dsbase = DS + bh64 * NTOT * DH
        ddbase = DD + bh64 * NTOT * DH
        dq = tl.zeros((BLOCK_T, BLOCK_D), tl.float32)
        dbacc = tl.zeros((BLOCK_T,), tl.float32)
        for m in range(0, L):
            active = found & (m < top)
            p = bidx >> (m + 1)
            c = (bidx >> m) & 1
            off = (2 << L) - (2 << (L - m))
            offp = (2 << L) - (2 << (L - m - 1))
            row = (off + 2 * p)[:, :, None] * DH + d[None, None, :]
            am = active[:, :, None] & dm[None, None, :]
            sl = tl.load(sbase + row, mask=am, other=0.0).to(tl.float32)
            sr = tl.load(sbase + row + DH, mask=am, other=0.0).to(tl.float32)
            diff = sr - sl
            dot = tl.sum(q[:, None, :] * diff, axis=2)
            sig = tl.sigmoid(beta * dot)
            dz = tl.where(active, dlp * tl.where(c == 1, 1.0 - sig, -sig), 0.0)
            dq += tl.sum(dz[:, :, None] * beta * diff, axis=1)
            dbacc += tl.sum(dz * dot, axis=1)
            pk = tl.where(active, p, -1 - kk[None, :])
            dzs = tl.zeros((BLOCK_T, KW), tl.float32)
            first = active
            for k in tl.static_range(KW):
                pcol = tl.sum(tl.where(kk[None, :] == k, pk, 0), axis=1)
                zcol = tl.sum(tl.where(kk[None, :] == k, dz, 0.0), axis=1)
                same = pk == pcol[:, None]
                dzs += tl.where(same, zcol[:, None], 0.0)
                first = first & ~(same & (kk[None, :] > k))
            prow = (offp + p)[:, :, None] * DH + d[None, None, :]
            tl.atomic_add(ddbase + prow, dzs[:, :, None] * beta * q[:, None, :], mask=(first & active)[:, :, None] & dm[None, None, :])
        total = tl.sum(dlp, axis=1)
        for m in range(0, L + 1):
            off = (2 << L) - (2 << (L - m))
            r = (t >> m) - 1
            ok = (((t >> m) & 1) == 1) & tm
            okd = ok[:, None] & dm[None, :]
            nrow = (off + r)[:, None] * DH + d[None, :]
            node = tl.load(sbase + nrow, mask=okd, other=0.0).to(tl.float32)
            dot = tl.sum(q * node, axis=1)
            drl = tl.sum(tl.where(top == m, dlp, 0.0), axis=1)
            prob = tl.where(ok, tl.exp(beta * dot - lse), 0.0)
            dlogit = tl.where(ok, drl - prob * total, 0.0)
            dq += dlogit[:, None] * beta * node
            dbacc += dlogit * dot
            tl.atomic_add(dsbase + nrow, dlogit[:, None] * beta * q, mask=okd)
        tl.store(DQ + bh64 * T * DH + t[:, None] * DH + d[None, :], dq, mask=qd)
        tl.atomic_add(DBETA + h, tl.sum(dbacc, axis=0))


BLOCK_T = None


def _blocks(DP):
    if BLOCK_T:
        return BLOCK_T, BLOCK_T
    bt = 32 if DP <= 32 else 16 if DP <= 64 else 8 if DP <= 128 else 4
    return bt, bt


def _pad(x, width):
    x = x.contiguous()
    return x if x.shape[-1] == width else F.pad(x, (0, width - x.shape[-1])).contiguous()


class BeamFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, flat, u, beta, L, width):
        B, H, T, DH = q.shape
        DP = triton.next_power_of_2(DH)
        qp, fp, up = _pad(q, DP), _pad(flat, DP), _pad(u, DP)
        beta32 = beta.detach().float().contiguous()
        dev = q.device
        out = torch.empty(B, H, T, DP, device=dev, dtype=torch.float32)
        idx = torch.empty(B, H, T, width, device=dev, dtype=torch.int32)
        lp = torch.empty(B, H, T, width, device=dev, dtype=torch.float32)
        lse = torch.empty(B, H, T, device=dev, dtype=torch.float32)
        bt = _blocks(DP)[0]
        grid = (triton.cdiv(T, bt), B * H)
        _beam_fwd[grid](qp, fp, up, beta32, out, idx, lp, lse, T, L, H, DP, fp.shape[2], up.shape[2],
                        KW=width, BLOCK_T=bt, BLOCK_D=DP, num_warps=4)
        ctx.save_for_backward(qp, fp, up, beta32, idx, lp, lse)
        ctx.L, ctx.width, ctx.DH = L, width, DH
        ctx.dtypes = (q.dtype, flat.dtype, u.dtype, beta.dtype)
        return out[..., :DH]

    @staticmethod
    def backward(ctx, gout):
        qp, fp, up, beta32, idx, lp, lse = ctx.saved_tensors
        B, H, T, DP = qp.shape
        DH = ctx.DH
        dq = torch.empty(B, H, T, DP, device=qp.device, dtype=torch.float32)
        ds = torch.zeros(fp.shape, device=qp.device, dtype=torch.float32)
        dd = torch.zeros(fp.shape, device=qp.device, dtype=torch.float32)
        du = torch.zeros(up.shape, device=qp.device, dtype=torch.float32)
        dbeta = torch.zeros(H, device=qp.device, dtype=torch.float32)
        bt = _blocks(DP)[1]
        grid = (triton.cdiv(T, bt), B * H)
        _beam_bwd[grid](qp, fp, up, beta32, idx, lp, lse, _pad(gout.float(), DP), dq, ds, dd, du, dbeta,
                        T, ctx.L, H, DP, fp.shape[2], up.shape[2],
                        KW=ctx.width, BLOCK_T=bt, BLOCK_D=DP, num_warps=4)
        L, K = ctx.L, 1 << ctx.L
        for m in range(L):
            off, offp, n = 2 * K - (2 * K >> m), 2 * K - (K >> m), K >> m
            parent = dd[:, :, offp:offp + n // 2]
            ds[:, :, off:off + n:2] -= parent
            ds[:, :, off + 1:off + n:2] += parent
        qd, fd, ud, bd = ctx.dtypes
        return (dq[..., :DH].to(qd), ds[..., :DH].to(fd), du[..., :DH].to(ud), dbeta.to(bd), None, None)


def pyramid(k, u_next):
    B, H, T, DH = k.shape
    L = max(1, (T - 1).bit_length())
    K = 1 << L
    if K > T:
        k = torch.cat([k, k.new_zeros(B, H, K - T, DH)], dim=2)
        u_next = torch.cat([u_next, u_next.new_zeros(B, H, K - T, DH)], dim=2)
    sums = [k]
    while sums[-1].size(2) > 1:
        sums.append(torch.maximum(sums[-1][:, :, 0::2], sums[-1][:, :, 1::2]))
    return L, torch.cat(sums, dim=2), u_next


def fused_beam(mem, q, k, u_next, width):
    L, flat, u_pad = pyramid(k, u_next)
    return BeamFn.apply(q, flat, u_pad, mem.beta, L, width)


def error_text():
    return traceback.format_exc(limit=4)[-1500:]


def _inputs(B, H, T, DH, device, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    make = lambda: (torch.randn(B, H, T, DH, generator=g) * 0.7).to(device=device, dtype=torch.float16)
    return make(), make(), make()


def check(device="cuda", width=8):
    from wat.main.model import TreeSearch
    records = []
    for B, H, T, DH in ((2, 4, 16, 8), (2, 4, 100, 26), (4, 4, 512, 26), (2, 4, 2048, 42), (1, 4, 8192, 26), (2, 2, 300, 64), (2, 4, 512, 98)):
        rec = {"shape": [B, H, T, DH]}
        try:
            mem = TreeSearch(DH * H, "beam%d" % width, heads=H).to(device)
            with torch.no_grad():
                mem.beta.copy_(torch.linspace(0.6, 1.6, H))
            q16, k16, u16 = _inputs(B, H, T, DH, device, 7 + T)
            gout = torch.randn(B, H, T, DH, generator=torch.Generator().manual_seed(3)).to(device)
            ref_in = [x.float().requires_grad_() for x in (q16, k16, u16)]
            mem.beta.grad = None
            ref = mem._beam(ref_in[0], ref_in[1], ref_in[2], width).float()
            ref.backward(gout)
            ref_grads = [x.grad.float() for x in ref_in] + [mem.beta.grad.clone()]
            fus_in = [x.clone().requires_grad_() for x in (q16, k16, u16)]
            mem.beta.grad = None
            out = fused_beam(mem, fus_in[0], fus_in[1], fus_in[2], width)
            out.backward(gout)
            fus_grads = [x.grad.float() for x in fus_in] + [mem.beta.grad.clone()]
            diff = (out - ref).abs()
            scale = ref.abs().max().clamp(min=1e-6)
            rec["out_max_abs"] = float(diff.max())
            rec["out_max_rel"] = float(diff.max() / scale)
            rec["out_bad_positions"] = float(((diff > 1e-2 * scale).any(-1)).float().mean())
            for name, a, b in zip(("dq", "dk", "du", "dbeta"), fus_grads, ref_grads):
                err = (a - b).abs().max()
                rec[f"{name}_max_abs"] = float(err)
                rec[f"{name}_max_rel"] = float(err / b.abs().max().clamp(min=1e-6))
            rec["pass"] = bool(rec["out_bad_positions"] < 0.01 and all(rec[f"{n}_max_rel"] < 0.05 for n in ("dq", "dk", "du", "dbeta")))
        except Exception:
            rec["error"] = error_text()
        records.append(rec)
    return records


def _time(fn, device):
    for _ in range(3):
        fn()
    torch.cuda.synchronize(device)
    n, total = 3, 0.0
    while True:
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize(device)
        total = time.perf_counter() - t0
        if total >= 0.3 or n >= 1000:
            return round(total * 1e3 / n, 4)
        n *= 2


def bench(device="cuda", width=8, block_t=None):
    global BLOCK_T
    from wat.main.model import TreeSearch
    saved = BLOCK_T
    if block_t:
        BLOCK_T = block_t
    records = []
    for B, T, D in ((32, 512, 104), (8, 2048, 104), (2, 8192, 104), (32, 512, 168), (32, 512, 256)):
        H, DH = 4, D // 4
        rec = {"B": B, "T": T, "D": D}
        try:
            mem = TreeSearch(D, "beam%d" % width).to(device)
            q16, k16, u16 = _inputs(B, H, T, DH, device, 11)
            gout = torch.randn(B, H, T, DH, device=device)
            leaves = [x.clone().requires_grad_() for x in (q16, k16, u16)]

            def ref_fwd():
                with torch.no_grad():
                    mem._beam(q16, k16, u16, width)

            def fus_fwd():
                with torch.no_grad():
                    fused_beam(mem, q16, k16, u16, width)

            def ref_fb():
                for x in leaves:
                    x.grad = None
                mem._beam(*leaves, width).float().backward(gout)

            def fus_fb():
                for x in leaves:
                    x.grad = None
                fused_beam(mem, *leaves, width).backward(gout)

            rec["ref_fwd_ms"] = _time(ref_fwd, device)
            rec["fused_fwd_ms"] = _time(fus_fwd, device)
            rec["ref_fwdbwd_ms"] = _time(ref_fb, device)
            rec["fused_fwdbwd_ms"] = _time(fus_fb, device)
            rec["speedup_fwd"] = round(rec["ref_fwd_ms"] / rec["fused_fwd_ms"], 2)
            rec["speedup_fwdbwd"] = round(rec["ref_fwdbwd_ms"] / rec["fused_fwdbwd_ms"], 2)
        except Exception:
            rec["error"] = error_text()
        rec["block_t"] = BLOCK_T or "auto"
        records.append(rec)
        torch.cuda.empty_cache()
    BLOCK_T = saved
    return records
