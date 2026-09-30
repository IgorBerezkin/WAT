import time, math, types
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from wat.lab import (WATBlockX, WATBackboneX, GLUMerge, LMModel, make_copy,
                     evaluate, n_params)

torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ED, LAYERS, K = 96, 2, 32


class GLUMergeSumRes(GLUMerge):
    def forward(self, left, right):
        combined = torch.cat([left, right], dim=-1)
        val = self.W_val(combined)
        gate = torch.sigmoid(self.W_gate(combined))
        merged = self.norm(val * gate)
        res_gate = torch.sigmoid(self.W_res(combined))
        residual = left + right
        return res_gate * merged + (1.0 - res_gate) * residual


def _tree_with_lane(self, chunks):
    s = WATBlockX._tree_reduction_all(self, chunks)
    g = torch.sigmoid(self.lane_g(chunks))
    v = self.lane_v(chunks)
    lane = (g * v).sum(dim=2) / math.sqrt(chunks.size(2))
    return s + lane


def build(config):
    bb = WATBackboneX(18, ED, n_layers=LAYERS, chunk_size=K, max_len=128,
                      dropout=0.1, ctx_mode="mean", intra=False)
    scale = 0.02 / (2 * LAYERS) ** 0.5
    for blk in bb.layers:
        if config in ("C1", "C2", "C4"):
            m = GLUMergeSumRes(bb.embed_dim)
            for mm in m.modules():
                if isinstance(mm, nn.Linear):
                    nn.init.normal_(mm.weight, 0.0, scale)
                    nn.init.zeros_(mm.bias)
            blk.tree_merge = m
        if config in ("C2", "C4"):
            with torch.no_grad():
                blk.W_global.weight.mul_(8.0)
        if config in ("C3", "C4"):
            blk.lane_g = nn.Linear(bb.embed_dim, 1)
            blk.lane_v = nn.Linear(bb.embed_dim, bb.embed_dim)
            nn.init.normal_(blk.lane_v.weight, 0.0, scale * 4)
            nn.init.zeros_(blk.lane_v.bias)
            nn.init.zeros_(blk.lane_g.weight)
            nn.init.zeros_(blk.lane_g.bias)
            blk._tree_reduction_all = types.MethodType(_tree_with_lane, blk)
    return bb


def train_config(config, n_mem, epochs=10):
    T = 128
    xtr, ytr, V = make_copy(3000, T, n_mem, seed=42)
    xva, yva, _ = make_copy(500, T, n_mem, seed=43)
    tl = DataLoader(TensorDataset(xtr, ytr), batch_size=32, shuffle=True)
    vl = DataLoader(TensorDataset(xva, yva), batch_size=64)
    model = LMModel(build(config), V).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    best = 0.0
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        for x, y in tl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out.reshape(-1, V), y.reshape(-1),
                                   ignore_index=-100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        va, vb = evaluate(model, vl, DEVICE, "lm")
        best = max(best, va)
        print(f"  [{config} n_mem={n_mem}] ep{ep+1:2d}  loss={loss.item():.3f}  "
              f"val_acc={va*100:5.1f}%  (шанс 6.25)  ({time.time()-t0:.0f}s)",
              flush=True)
    return best


def smoke():
    print("shape-smoke...", flush=True)
    for c in ("C0", "C1", "C2", "C3", "C4"):
        m = LMModel(build(c), 18).to(DEVICE)
        x = torch.randint(0, 18, (2, 128), device=DEVICE)
        out = m(x)
        assert out.shape == (2, 128, 18), (c, out.shape)
        loss = out.sum()
        loss.backward()
        print(f"  {c}: ok ({n_params(m):,} params)", flush=True)
    print("smoke OK\n", flush=True)


def main():
    smoke()
    print("=" * 70)
    print("КАНАРЕЙКА n_mem=1")
    print("=" * 70)
    results = {}
    for c in ("C0", "C1", "C2", "C3", "C4"):
        print(f"\n--- {c} ---")
        results[c] = train_config(c, n_mem=1)
    print("\n" + "=" * 70)
    print("ИТОГ n_mem=1: " + "  ".join(f"{c}={v*100:.1f}%"
                                       for c, v in results.items()))
    winners = [c for c, v in results.items() if v > 0.40]
    if winners:
        print("=" * 70)
        print(f"ПРОБИЛИ (>40%): {winners} -> проверка n_mem=4")
        print("=" * 70)
        for c in winners:
            print(f"\n--- {c} n_mem=4 ---")
            train_config(c, n_mem=4, epochs=12)
    else:
        print("Никто не пробил 40% — шлём вывод целиком, думаем дальше.")


if __name__ == "__main__":
    main()
