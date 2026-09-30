import json, math, os, sys, time, traceback, types
sys.path.insert(0, ".")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from wat_lab import (WATBackboneX, WATBlockX, GLUMerge, CausalSelfAttention,
                     LMModel, make_copy)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K0 = 32
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "results_night")
os.makedirs(OUT, exist_ok=True)
RES_PATH = os.path.join(OUT, "results.json")


def _load():
    return json.load(open(RES_PATH, encoding="utf-8")) \
        if os.path.exists(RES_PATH) else {}


def _save(rid, payload):
    res = _load()
    res[rid] = payload
    with open(RES_PATH, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)


def rope(x, pos):
    D = x.size(-1)
    half = D // 2
    freqs = torch.exp(-math.log(10000.0) *
                      torch.arange(half, device=x.device) / half)
    ang = pos.unsqueeze(-1).float() * freqs
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., :half], x[..., half:2 * half]
    out = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos,
                     x[..., 2 * half:]], dim=-1)
    return out


class GainedLinear(nn.Module):
    def __init__(self, lin, alpha0=8.0):
        super().__init__()
        self.lin = lin
        self.alpha = nn.Parameter(torch.tensor(float(alpha0)))

    def forward(self, x):
        return self.alpha * self.lin(x)


class BaseXBlock(WATBlockX):
    def __init__(self, d, chunk_size=K0):
        super().__init__(d, chunk_size, ctx_mode="mean", intra=False)

    def _make_ctx(self, x_padded, summaries):
        B, Tp, D = x_padded.shape
        ctx = self._ctx_mean(summaries)
        return ctx.unsqueeze(2).expand(-1, -1, self.K, -1).reshape(B, Tp, D)

    def _inject(self, x_padded, ctx_pos):
        return x_padded + self.W_global(ctx_pos)

    def forward(self, x):
        B, T, D = x.shape
        K = self.K
        h = self.norm_conv(x)
        h = self.conv(h)
        h = h * torch.sigmoid(self.W_gate(h))
        x = x + h
        pad_len = (K - T % K) % K
        x_padded = x if pad_len == 0 else torch.cat(
            [x, x[:, -1:, :].expand(-1, pad_len, -1)], dim=1)
        chunks = x_padded.unfold(1, K, K).transpose(2, 3)
        summaries = self._tree_reduction_all(chunks)
        ctx_pos = self._make_ctx(x_padded, summaries)
        h_ctx = self._inject(x_padded, ctx_pos)
        h_ctx = h_ctx[:, :T, :]
        x = x + (h_ctx - x.detach()) * 0.5
        h = self.norm_ffn(x)
        x = x + self.ffn(h)
        return x


class MatrixBlock(BaseXBlock):
    def __init__(self, d, chunk_size=K0):
        super().__init__(d, chunk_size)
        self.Wk = nn.Linear(d, d)
        self.Wv = nn.Linear(d, d)
        self.Wq = nn.Linear(d, d)

    def _make_ctx(self, x_padded, s):
        B, Tp, D = x_padded.shape
        C = s.size(1)
        k = F.elu(self.Wk(s)) + 1.0
        v = self.Wv(s)
        kv = torch.einsum("bcd,bce->bcde", k, v)
        M = torch.cumsum(kv, dim=1)
        M = torch.cat([torch.zeros_like(M[:, :1]), M[:, :-1]], dim=1)
        z = torch.cumsum(k, dim=1)
        z = torch.cat([torch.zeros_like(z[:, :1]), z[:, :-1]], dim=1)
        q = F.elu(self.Wq(x_padded)) + 1.0
        outs = []
        for i in range(C):
            qi = q[:, i * self.K:(i + 1) * self.K]
            num = torch.einsum("bkd,bde->bke", qi, M[:, i])
            den = torch.einsum("bkd,bd->bk", qi, z[:, i]).unsqueeze(-1) + 1e-4
            outs.append(num / den)
        return torch.cat(outs, dim=1)


class KVRopeBlock(BaseXBlock):
    def __init__(self, d, chunk_size=K0):
        super().__init__(d, chunk_size)
        self.Wk = nn.Linear(d, d)
        self.Wv = nn.Linear(d, d)
        self.Wq = nn.Linear(d, d)

    def _make_ctx(self, x_padded, s):
        B, Tp, D = x_padded.shape
        K = self.K
        C = Tp // K
        pos = torch.arange(Tp, device=x_padded.device)
        k = rope(self.Wk(x_padded), pos) / math.sqrt(D)
        v = self.Wv(x_padded)
        kc = k.view(B, C, K, D)
        vc = v.view(B, C, K, D)
        Mc = torch.einsum("bckd,bcke->bcde", kc, vc)
        M = torch.cumsum(Mc, dim=1)
        M = torch.cat([torch.zeros_like(M[:, :1]), M[:, :-1]], dim=1)
        q = rope(self.Wq(x_padded), pos)
        outs = []
        for i in range(C):
            qi = q[:, i * K:(i + 1) * K]
            outs.append(torch.einsum("bkd,bde->bke", qi, M[:, i])
                        / max(1.0, (i * K) ** 0.5))
        return torch.cat(outs, dim=1)


class RotorBlock(BaseXBlock):
    def __init__(self, d, chunk_size=K0, max_len=1024):
        super().__init__(d, chunk_size)
        self.rotor = nn.Embedding(max_len, d)
        nn.init.zeros_(self.rotor.weight)

    def _inject(self, x_padded, ctx_pos):
        pos = torch.arange(x_padded.size(1), device=x_padded.device)
        return x_padded + self.W_global(ctx_pos) * (1.0 + self.rotor(pos))


class EMABlock(BaseXBlock):
    def __init__(self, d, chunk_size=K0):
        super().__init__(d, chunk_size)
        self.decay_logit = nn.Parameter(torch.full((d,), 2.0))
        self.We = nn.Linear(d, d)
        nn.init.normal_(self.We.weight, 0.0, 0.01)
        nn.init.zeros_(self.We.bias)

    def _make_ctx(self, x_padded, s):
        B, Tp, D = x_padded.shape
        C = s.size(1)
        base = self._ctx_mean(s)
        lam = torch.sigmoid(self.decay_logit)
        E = torch.zeros(B, D, device=s.device, dtype=s.dtype)
        es = [E]
        for i in range(C - 1):
            E = lam * E + (1 - lam) * s[:, i]
            es.append(E)
        ema = torch.stack(es, dim=1)
        ctx = base + self.We(ema)
        return ctx.unsqueeze(2).expand(-1, -1, self.K, -1).reshape(B, Tp, D)


class CheatBlock(BaseXBlock):
    def __init__(self, d, chunk_size=K0):
        super().__init__(d, chunk_size)
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.null_summary = nn.Parameter(torch.zeros(1, 1, d))
        self.u_head = nn.Linear(d, 1)
        nn.init.zeros_(self.u_head.weight)
        nn.init.constant_(self.u_head.bias, -2.0)

    def _make_ctx(self, x_padded, s):
        B, Tp, D = x_padded.shape
        C = s.size(1)
        base = super()._make_ctx(x_padded, s)
        kv = torch.cat([self.null_summary.expand(B, 1, D).to(s.dtype), s], 1)
        q = self.W_q(x_padded)
        k, v = self.W_k(kv), self.W_v(kv)
        att = torch.einsum("btd,bcd->btc", q, k) / math.sqrt(D)
        chunk_idx = torch.arange(Tp, device=x_padded.device) // self.K
        jj = torch.arange(C + 1, device=x_padded.device)
        valid = (jj.view(1, -1) == 0) | \
                ((jj.view(1, -1) - 1) < chunk_idx.view(-1, 1))
        att = att.masked_fill(~valid.unsqueeze(0), float("-inf"))
        r = torch.einsum("btc,bcd->btd", F.softmax(att, dim=-1), v)
        u = torch.sigmoid(self.u_head(x_padded))
        return base + u * r


class HybridBackbone(nn.Module):
    def __init__(self, vocab, d=96, max_len=512):
        super().__init__()
        self.embed_dim = d
        self.embedding = nn.Embedding(vocab, d)
        self.pos_encoding = nn.Embedding(max_len + K0, d)
        self.wat = WATBlockX(d, K0, ctx_mode="mean", intra=False)
        self.norm_a = nn.RMSNorm(d)
        self.attn = CausalSelfAttention(d, 4, dropout=0.0,
                                        max_len=max_len + K0)
        self.norm_f = nn.RMSNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, d * 4), nn.GELU(),
                                 nn.Linear(d * 4, d))
        self.output_norm = nn.RMSNorm(d)
        scale = 0.02 / 2.0
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, scale)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device)
        h = self.embedding(x) + self.pos_encoding(pos)
        h = self.wat(h)
        h = h + self.attn(self.norm_a(h))
        h = h + self.ffn(self.norm_f(h))
        return self.output_norm(h)


def make_backbone(kind, V, T):
    torch.manual_seed(42)
    d = 96

    def wrap(block):
        bb = WATBackboneX(V, d, n_layers=1, chunk_size=K0, max_len=T,
                          dropout=0.0, ctx_mode="mean", intra=False)
        bb.layers[0] = block
        scale = 0.02 / 2.0
        for m in block.modules():
            if isinstance(m, nn.Linear) and not isinstance(m, GainedLinear):
                nn.init.normal_(m.weight, 0.0, scale)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        return bb

    if kind == "x_matrix":
        return wrap(MatrixBlock(d))
    if kind == "x_kv_rope":
        return wrap(KVRopeBlock(d))
    if kind == "x_rotor":
        return wrap(RotorBlock(d, max_len=T + K0 + 8))
    if kind == "x_ema":
        return wrap(EMABlock(d))
    if kind == "x_cheatsheet":
        blk = CheatBlock(d)
        bb = wrap(blk)
        nn.init.zeros_(blk.u_head.weight)
        nn.init.constant_(blk.u_head.bias, -2.0)
        return bb
    if kind == "x_hybrid_attn":
        return HybridBackbone(V, d, max_len=T)
    if kind == "x_registers":
        bb = WATBackboneX(V, d, n_layers=1, chunk_size=K0, max_len=T,
                          dropout=0.0, ctx_mode="mean", intra=False)
        blk = bb.layers[0]
        blk.regs = nn.Parameter(torch.randn(1, 1, 6, d) * 0.02)

        def tree_regs(self, chunks):
            B, C = chunks.size(0), chunks.size(1)
            r = self.regs.expand(B, C, -1, -1)
            return WATBlockX._tree_reduction_all(
                self, torch.cat([r, chunks], dim=2))
        blk._tree_reduction_all = types.MethodType(tree_regs, blk)
        return bb
    if kind == "x_gate_bias":
        bb = WATBackboneX(V, d, n_layers=1, chunk_size=K0, max_len=T,
                          dropout=0.0, ctx_mode="mean", intra=False)
        with torch.no_grad():
            bb.layers[0].tree_merge.W_res.bias.fill_(-2.5)
        return bb
    if kind in ("x_ladder_anneal",):
        bb = WATBackboneX(V, d, n_layers=1, chunk_size=K0, max_len=T,
                          dropout=0.0, ctx_mode="prefix_tree", intra=False)
        blk = bb.layers[0]
        blk.ladder_w = 0.1

        def safe(self, s):
            w = float(self.ladder_w)
            return w * WATBlockX._ctx_prefix_tree(self, s) + \
                (1 - w) * WATBlockX._ctx_mean(self, s)
        blk._ctx_prefix_tree = types.MethodType(safe, blk)
        return bb
    bb = WATBackboneX(V, d, n_layers=1, chunk_size=K0, max_len=T,
                      dropout=0.0, ctx_mode="mean", intra=False)
    if kind in ("x_chunk_curr", "x_echo"):
        bb.layers[0].W_global = GainedLinear(bb.layers[0].W_global, 8.0)
    return bb


def verify(kind, V):
    bb = make_backbone(kind, V, 160)
    if kind == "x_chunk_curr":
        bb.layers[0].K = 32
    model = LMModel(bb, V).to(DEVICE)
    x = torch.randint(0, V, (2, 160), device=DEVICE)
    out = model(x)
    assert out.shape == (2, 160, V), out.shape
    out.sum().backward()
    model.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad():
        base = model(x[:1])
        for p in (12, 60, 120, 159):
            x2 = x[:1].clone()
            x2[0, p] = (x2[0, p] + 7) % V
            d = (model(x2) - base).abs().max(dim=-1).values[0]
            if p > 0 and d[:p].max().item() > 1e-4:
                return False, p
    return True, None


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


def train(rid, model, xtr, ytr, xva, yva, bs, cap, patience=70,
          epoch_hook=None, aux_fn=None):
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    tl = DataLoader(TensorDataset(xtr, ytr), batch_size=bs, shuffle=True,
                    generator=torch.Generator().manual_seed(42))
    best, best_ep, cross = 0.0, 0, None
    curve = []
    t0 = time.time()
    ep = 0
    for ep in range(1, cap + 1):
        if epoch_hook:
            epoch_hook(model, ep)
        model.train()
        for xb, yb in tl:
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = F.cross_entropy(out.reshape(-1, out.size(-1)),
                                   yb.reshape(-1), ignore_index=-100)
            if aux_fn is not None:
                loss = loss + aux_fn(model, xb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        acc = val_acc(model, xva, yva, bs)
        if acc > best + 1e-4:
            best, best_ep = acc, ep
        if cross is None and acc * 100 >= 15.0:
            cross = ep
        if ep % 10 == 0 or ep == 1 or cross == ep:
            curve.append([ep, round(acc * 100, 2)])
            note = "  <- ПЕРЕХОД" if cross == ep else ""
            print(f"  [{rid}] ep{ep:3d} loss={loss.item():.3f} "
                  f"val={acc*100:5.1f}%{note} ({time.time()-t0:.0f}s)",
                  flush=True)
        if cross is not None and ep - best_ep >= patience:
            print(f"  [{rid}] полка стабильна, выход ep{ep}", flush=True)
            break
    return {"best": round(best * 100, 2), "cross": cross, "epochs": ep,
            "sec": round(time.time() - t0), "curve": curve}


def guarded(rid, fn):
    res = _load()
    if rid in res and "error" not in res[rid] and "LEAK" not in res[rid]:
        print(f"[skip] {rid}: {res[rid].get('best')}%", flush=True)
        return
    print("=" * 70)
    print(f"RUN {rid}   ({time.strftime('%H:%M:%S')})")
    print("=" * 70)
    try:
        _save(rid, fn())
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        _save(rid, {"error": tb[-1200:]})
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def run_block_x():
    xtr, ytr, V = make_copy(4000, 512, 16, seed=42)
    xva, yva, _ = make_copy(1000, 512, 16, seed=43)
    xtr, ytr = xtr.to(DEVICE), ytr.to(DEVICE)
    xva, yva = xva.to(DEVICE), yva.to(DEVICE)

    simple = [("x_matrix", 200), ("x_kv_rope", 200), ("x_rotor", 250),
              ("x_registers", 200), ("x_gate_bias", 250), ("x_ema", 200),
              ("x_cheatsheet", 200), ("x_hybrid_attn", 150)]
    for kind, cap in simple:
        def fn(kind=kind, cap=cap):
            ok, pos = verify(kind, V)
            if not ok:
                print(f"  !!! {kind}: LEAK at {pos} — пропуск", flush=True)
                return {"LEAK": pos}
            m = LMModel(make_backbone(kind, V, 512), V).to(DEVICE)
            npar = sum(p.numel() for p in m.parameters())
            print(f"  params={npar:,}", flush=True)
            return train(kind, m, xtr, ytr, xva, yva, 128, cap)
        guarded(kind, fn)

    def fn_anneal():
        ok, pos = verify("x_ladder_anneal", V)
        if not ok:
            return {"LEAK": pos}
        m = LMModel(make_backbone("x_ladder_anneal", V, 512), V).to(DEVICE)

        def hook(model, ep):
            w = min(0.9, 0.1 + 0.8 * ep / 120.0)
            model.backbone.layers[0].ladder_w = w
        return train("x_ladder_anneal", m, xtr, ytr, xva, yva, 128, 250,
                     epoch_hook=hook)
    guarded("x_ladder_anneal", fn_anneal)

    def fn_chunk_curr():
        ok, pos = verify("x_chunk_curr", V)
        if not ok:
            return {"LEAK": pos}
        m = LMModel(make_backbone("x_chunk_curr", V, 512), V).to(DEVICE)
        total = {"phases": [], "sec": 0}
        for Kc, cap in ((128, 50), (64, 50), (32, 180)):
            m.backbone.layers[0].K = Kc
            r = train(f"x_chunk_curr K{Kc}", m, xtr, ytr, xva, yva, 128, cap,
                      patience=60)
            total["phases"].append({"K": Kc, "best": r["best"],
                                    "cross": r["cross"], "epochs": r["epochs"]})
            total["sec"] += r["sec"]
        total["best"] = total["phases"][-1]["best"]
        total["cross"] = total["phases"][-1]["cross"]
        return total
    guarded("x_chunk_curr", fn_chunk_curr)

    def fn_echo():
        ok, pos = verify("x_echo", V)
        if not ok:
            return {"LEAK": pos}
        m = LMModel(make_backbone("x_echo", V, 512), V).to(DEVICE)
        y_echo = torch.full_like(xtr, -100)
        y_echo[:, 64:] = xtr[:, :-64]
        r1 = train("x_echo фаза-эхо", m, xtr, y_echo, xva, yva, 128, 25,
                   patience=25)
        r2 = train("x_echo фаза-copy", m, xtr, ytr, xva, yva, 128, 200)
        r2["echo_phase"] = {"epochs": r1["epochs"], "sec": r1["sec"]}
        r2["sec"] += r1["sec"]
        return r2
    guarded("x_echo", fn_echo)

    def fn_aux():
        ok, pos = verify("x_aux_ctx", V)
        if not ok:
            return {"LEAK": pos}
        bb = make_backbone("x_aux_ctx", V, 512)
        blk = bb.layers[0]
        orig = WATBlockX._ctx_mean

        def stash(self, s):
            out = orig(self, s)
            self._ctx_last = out
            return out
        blk._ctx_mean = types.MethodType(stash, blk)
        model = LMModel(bb, V).to(DEVICE)
        aux_head = nn.Linear(96, 16).to(DEVICE)
        model.aux_head = aux_head

        def aux_fn(m, xb):
            ctx = m.backbone.layers[0]._ctx_last
            B, C, D = ctx.shape
            onehot = F.one_hot(xb.clamp(max=16), 17)[..., :16].float()
            per_chunk = onehot.view(B, C, -1, 16).sum(2)
            seen = torch.cumsum(per_chunk, dim=1)
            seen = torch.cat([torch.zeros_like(seen[:, :1]),
                              seen[:, :-1]], dim=1).clamp(max=1.0)
            return 0.3 * F.binary_cross_entropy_with_logits(
                aux_head(ctx), seen)
        return train("x_aux_ctx", model, xtr, ytr, xva, yva, 128, 200,
                     aux_fn=aux_fn)
    guarded("x_aux_ctx", fn_aux)


if __name__ == "__main__":
    t0 = time.time()
    print(f"БЛОК X СТАРТ {time.strftime('%H:%M:%S')} device={DEVICE}")
    run_block_x()
    print(f"Блок X готов за {(time.time()-t0)/3600:.1f} ч. "
          f"Результаты дописаны в results_night/results.json")
