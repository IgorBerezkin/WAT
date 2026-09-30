import sys, math
sys.path.insert(0, ".")
import torch
import torch.nn as nn
import torch.nn.functional as F

from wat_lab import (WATBackboneX, TransformerBackbone, LMModel, make_copy)

torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ED, K, T = 96, 32, 128
NTR, NTE, NDELTA = 4000, 1000, 512
NOISE_TOK = 16


def dataset():
    x, y, V = make_copy(NTR + NTE, T, 1, seed=42)
    body = T - 1
    pos = (x[:, :body] != NOISE_TOK).float().argmax(dim=1)
    lab = x[torch.arange(x.size(0)), pos]
    reachable = (pos < 96)
    return (x.to(DEVICE), lab.to(DEVICE), pos.to(DEVICE),
            reachable.to(DEVICE), V)


@torch.no_grad()
def trace(bb, x, pos):
    B = x.size(0)
    ar = torch.arange(B, device=x.device)
    blk = bb.layers[0]
    out = {}

    positions = torch.arange(x.size(1), device=x.device)
    h0 = bb.embedding(x) + bb.pos_encoding(positions)
    out["emb@token"] = h0[ar, pos]

    hh = blk.norm_conv(h0)
    hh = blk.conv(hh)
    hh = hh * torch.sigmoid(blk.W_gate(hh))
    x1 = h0 + hh
    out["conv@token"] = x1[ar, pos]

    chunks = x1.unfold(1, K, K).transpose(2, 3)
    chunk_idx = pos // K
    node_idx = pos % K
    curr, size, lvl = chunks, K, 0
    while size > 1:
        next_size = (size + 1) // 2
        if size % 2 != 0:
            curr = torch.cat([curr, curr[:, :, -1:, :]], dim=2)
        v = curr.view(B, curr.size(1), next_size, 2, curr.size(-1))
        curr = blk.tree_merge(v[:, :, :, 0, :], v[:, :, :, 1, :])
        size, lvl = next_size, lvl + 1
        node_idx = node_idx // 2
        out[f"tree_L{lvl} ({size} узл.)"] = curr[ar, chunk_idx, node_idx]
    summaries = curr.squeeze(2)

    ctx = blk._ctx_mean(summaries)
    out["ctx@reader"] = ctx[:, -1]
    inj = 0.5 * blk.W_global(ctx[:, -1])
    out["inject@reader"] = inj
    out["stream+inj@rdr"] = x1[:, -1] + inj
    out["hidden@reader"] = bb(x)[:, -1]
    return out


def stage_names():
    return (["emb@token", "conv@token"] +
            [f"tree_L{l} ({K >> l} узл.)" for l in range(1, 6)] +
            ["ctx@reader", "inject@reader", "stream+inj@rdr", "hidden@reader"])


def linear_probe(f_tr, y_tr, f_te, y_te, epochs=350):
    mu, sd = f_tr.mean(0), f_tr.std(0) + 1e-6
    f_tr, f_te = (f_tr - mu) / sd, (f_te - mu) / sd
    probe = nn.Linear(f_tr.size(1), 16).to(DEVICE)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-2, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(probe(f_tr), y_tr).backward()
        opt.step()
    with torch.no_grad():
        return (probe(f_te).argmax(-1) == y_te).float().mean().item()


def panel_a():
    x, lab, pos, reach, V = dataset()
    bb = WATBackboneX(V, ED, n_layers=1, chunk_size=K, max_len=T,
                      dropout=0.0, ctx_mode="mean", intra=False)
    bb = bb.to(DEVICE).eval()

    feats = {n: [] for n in stage_names()}
    for i in range(0, x.size(0), 256):
        tr = trace(bb, x[i:i + 256], pos[i:i + 256])
        for n in feats:
            feats[n].append(tr[n])
    feats = {n: torch.cat(v) for n, v in feats.items()}

    xs = x[:NDELTA].clone()
    ps = pos[:NDELTA]
    xw = xs.clone()
    xw[torch.arange(NDELTA, device=DEVICE), ps] = \
        (xw[torch.arange(NDELTA, device=DEVICE), ps] + 7) % 16
    t1 = trace(bb, xs, ps)
    t2 = trace(bb, xw, ps)

    reach_tr, reach_te = reach[:NTR], reach[NTR:]
    print("=" * 76)
    print("ПАНЕЛЬ A — ПУТЬ ДАННЫХ (v0, init, seed 42; шанс пробы = 6.25%)")
    print("=" * 76)
    print(f"доля сэмплов с токеном в чанке читателя (слепая зона ctx): "
          f"{(~reach).float().mean().item()*100:.1f}%  ->  потолок probe "
          f"по всем ≈ {(reach.float().mean().item()*100 + 6.25*(~reach).float().mean().item()):.1f}%")
    print(f"{'стадия':<20} {'rms':>9} {'dSig%':>8} {'probe%':>8} {'probe*%':>8}")
    for n in stage_names():
        f = feats[n]
        rms = f.pow(2).mean().sqrt().item()
        d = (t2[n] - t1[n]).norm(dim=1) / (t1[n].pow(2).mean(1).sqrt()
                                           * math.sqrt(f.size(1)) + 1e-9)
        dsig = d.mean().item() * 100
        p_all = linear_probe(f[:NTR], lab[:NTR], f[NTR:], lab[NTR:])
        p_sub = linear_probe(f[:NTR][reach_tr], lab[:NTR][reach_tr],
                             f[NTR:][reach_te], lab[NTR:][reach_te])
        print(f"{n:<20} {rms:>9.4f} {dsig:>7.1f}% {p_all*100:>7.1f}% "
              f"{p_sub*100:>7.1f}%", flush=True)
    print()


def grad_groups_wat(model):
    blk = model.backbone.layers[0]
    tm = blk.tree_merge
    return {
        "embedding": [model.backbone.embedding.weight],
        "conv":      [blk.conv.conv.weight],
        "tree_merge": [tm.W_val.weight, tm.W_gate.weight, tm.W_res.weight],
        "W_global":  [blk.W_global.weight],
        "ffn":       [blk.ffn[0].weight],
        "head":      [model.head.weight],
    }


def grad_groups_tr(model):
    blk = model.backbone.layers[0]
    return {
        "embedding": [model.backbone.embedding.weight],
        "attn_qkv":  [blk.attn.qkv.weight],
        "attn_proj": [blk.attn.proj.weight],
        "ffn":       [blk.ffn[0].weight],
        "head":      [model.head.weight],
    }


def gradient_snr(model, groups, x_all, y_all, bs, n_batches=12):
    model.train()
    grads = {g: [] for g in groups}
    for b in range(n_batches):
        xb = x_all[b * bs:(b + 1) * bs]
        yb = y_all[b * bs:(b + 1) * bs]
        model.zero_grad(set_to_none=True)
        out = model(xb)
        loss = F.cross_entropy(out.reshape(-1, out.size(-1)),
                               yb.reshape(-1), ignore_index=-100)
        loss.backward()
        for g, params in groups.items():
            grads[g].append(torch.cat(
                [p.grad.detach().flatten() for p in params]))
    snr = {}
    for g, gl in grads.items():
        G = torch.stack(gl)
        mean = G.mean(0)
        noise = (G - mean).pow(2).sum(1).mean()
        snr[g] = (mean.pow(2).sum() / (noise + 1e-12)).item()
    return snr


def panel_b():
    print("=" * 76)
    print("ПАНЕЛЬ B — ГРАДИЕНТНЫЙ SNR (12 независимых батчей на режим)")
    print("=" * 76)
    conds = []
    for bs, n_mem, tag in ((16, 1, "WAT bs16 nm1"),
                           (256, 1, "WAT bs256 nm1"),
                           (16, 16, "WAT bs16 nm16")):
        x, y, V = make_copy(bs * 12, T, n_mem, seed=100 + n_mem)
        torch.manual_seed(42)
        m = LMModel(WATBackboneX(V, ED, n_layers=1, chunk_size=K, max_len=T,
                                 dropout=0.0, ctx_mode="mean", intra=False),
                    V).to(DEVICE)
        snr = gradient_snr(m, grad_groups_wat(m), x.to(DEVICE), y.to(DEVICE),
                           bs)
        conds.append((tag, snr))
        del m
    x, y, V = make_copy(16 * 12, T, 1, seed=101)
    torch.manual_seed(42)
    m = LMModel(TransformerBackbone(V, ED, n_layers=1, max_len=T,
                                    dropout=0.0), V).to(DEVICE)
    snr = gradient_snr(m, grad_groups_tr(m), x.to(DEVICE), y.to(DEVICE), 16)
    conds.append(("TR  bs16 nm1", snr))

    all_groups = []
    for _, s in conds:
        for g in s:
            if g not in all_groups:
                all_groups.append(g)
    header = f"{'группа':<12}" + "".join(f"{tag:>16}" for tag, _ in conds)
    print(header)
    for g in all_groups:
        row = f"{g:<12}"
        for _, s in conds:
            row += f"{s.get(g, float('nan')):>16.4f}" if g in s \
                else f"{'—':>16}"
        print(row, flush=True)
    print()
    print("Чтение: SNR<0.1 — шаг почти чистый шум; ~1 — сигнал сопоставим с")
    print("шумом; >1 — батчи согласны. Сравнивай столбцы WAT между собой и")
    print("W_global/tree_merge WAT против attn_qkv трансформера.")


def smoke():
    x, lab, pos, reach, V = dataset()
    bb = WATBackboneX(V, ED, n_layers=1, chunk_size=K, max_len=T,
                      dropout=0.0, ctx_mode="mean", intra=False)
    bb = bb.to(DEVICE).eval()
    tr = trace(bb, x[:8], pos[:8])
    for n in stage_names():
        assert tr[n].shape == (8, ED), (n, tr[n].shape)
    print("shape-smoke OK\n", flush=True)


if __name__ == "__main__":
    smoke()
    panel_a()
    panel_b()
    print("Готово. Обе панели выше — шли целиком.")
