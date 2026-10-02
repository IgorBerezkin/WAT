import torch
import torch.nn as nn

from wat.common import GLUMerge
from wat.history.lab import WATBlockX, WATBackboneX

torch.manual_seed(42)
D, K, TRIALS = 128, 32, 24


class GLUMergeNoNorm(GLUMerge):
    def __init__(self, d):
        super().__init__(d)
        self.norm = nn.Identity()


class GLUMergeSumRes(GLUMerge):
    def forward(self, left, right):
        combined = torch.cat([left, right], dim=-1)
        val = self.W_val(combined)
        gate = torch.sigmoid(self.W_gate(combined))
        merged = self.norm(val * gate)
        res_gate = torch.sigmoid(self.W_res(combined))
        residual = left + right
        return res_gate * merged + (1.0 - res_gate) * residual


def scaled_init(module, n_layers=3, mult=1.0):
    scale = 0.02 / (2 * n_layers) ** 0.5 * mult
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, scale)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return module


@torch.no_grad()
def survival_curve(merge, sigma=1.0):
    levels = [[] for _ in range(5)]
    for t in range(TRIALS):
        g = torch.Generator().manual_seed(1000 + t)
        noise = torch.randn(1, K, D, generator=g) * sigma
        pos = t % K
        sig = noise.clone()
        sig[0, pos] = torch.randn(D, generator=g) * sigma
        d0 = (sig - noise).norm()
        cn, cs = noise, sig
        for lvl in range(5):
            cn = merge(cn[:, 0::2], cn[:, 1::2])
            cs = merge(cs[:, 0::2], cs[:, 1::2])
            levels[lvl].append(((cs - cn).norm() / d0).item())
    return [sum(v) / len(v) for v in levels]


@torch.no_grad()
def injection_ratio():
    blk = WATBlockX(D, K, ctx_mode="mean", intra=False).eval()
    scaled_init(blk)
    ratios = []
    for t in range(TRIALS):
        g = torch.Generator().manual_seed(2000 + t)
        x = torch.randn(1, 2 * K, D, generator=g) * 0.5
        h = blk.norm_conv(x)
        h = blk.conv(h)
        h = h * torch.sigmoid(blk.W_gate(h))
        x1 = x + h
        chunks = x1.unfold(1, K, K).transpose(2, 3)
        s = blk._tree_reduction_all(chunks)
        ctx = blk._ctx_mean(s)
        ctx_pos = ctx.unsqueeze(2).expand(-1, -1, K, -1).reshape(1, 2 * K, D)
        inj = 0.5 * blk.W_global(ctx_pos)
        reader = slice(K, 2 * K)
        r = inj[:, reader].pow(2).mean().sqrt() / \
            x1[:, reader].pow(2).mean().sqrt()
        ratios.append(r.item())
    return sum(ratios) / len(ratios)


@torch.no_grad()
def end_to_end():
    bb = WATBackboneX(65, D, n_layers=2, chunk_size=K, max_len=128,
                      dropout=0.0, ctx_mode="mean", intra=False).eval()
    rels = []
    for t in range(TRIALS):
        g = torch.Generator().manual_seed(3000 + t)
        x = torch.randint(0, 65, (1, 64), generator=g)
        x2 = x.clone()
        x2[0, t % K] = (x2[0, t % K] + 7) % 65
        h1, h2 = bb(x), bb(x2)
        reader = 40 + (t % 20)
        rel = (h2[0, reader] - h1[0, reader]).norm() / h1[0, reader].norm()
        rels.append(rel.item())
    return sum(rels) / len(rels)


def main():
    print("=" * 72)
    print("A. Выживание сигнала по уровням дерева (доля от исходного Δ)")
    print(f"   уровень:              16      8      4      2   корень")
    configs = [
        ("as-is      ", scaled_init(GLUMerge(D))),
        ("as-is x10  ", scaled_init(GLUMerge(D), mult=10.0)),
        ("no-norm    ", scaled_init(GLUMergeNoNorm(D))),
        ("sum-res    ", scaled_init(GLUMergeSumRes(D))),
        ("no-norm x10", scaled_init(GLUMergeNoNorm(D), mult=10.0)),
    ]
    for name, merge in configs:
        c = survival_curve(merge.eval())
        print(f"   {name}: " + "  ".join(f"{v*100:5.1f}%" for v in c))
    print()
    print("B. Инжекция: rms(0.5*W_global(ctx)) / rms(residual-потока)")
    print(f"   ratio = {injection_ratio():.5f}"
          f"   (1.0 = вклад сопоставим с потоком)")
    print()
    print("C. End-to-end чувствительность читателя (чанк 1) к токену чанка 0")
    print(f"   ||Δh||/||h|| = {end_to_end():.5f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
