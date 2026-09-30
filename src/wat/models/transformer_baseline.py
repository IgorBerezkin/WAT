"""
Transformer Baseline
====================
Causal (GPT-style) Transformer с идентичным интерфейсом WATDeepStackV1.
chunk_size принимается но игнорируется - только для совместимости.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.1, max_len: int = 2048):
        super().__init__()
        assert embed_dim % n_heads == 0, f"embed_dim {embed_dim} must be divisible by n_heads {n_heads}"

        self.n_heads  = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale    = self.head_dim ** -0.5

        self.qkv  = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)

        # Causal mask
        mask = torch.tril(torch.ones(max_len, max_len)).unsqueeze(0).unsqueeze(0)
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)                    # 3 x (B, T, D)
        q, k, v = [t.view(B, T, self.n_heads, self.head_dim)
                     .transpose(1, 2) for t in qkv]           # (B, H, T, hd)

        attn = (q @ k.transpose(-2, -1)) * self.scale         # (B, H, T, T)
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous()         # (B, T, H, hd)
        out = out.view(B, T, D)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.1, max_len: int = 2048):
        super().__init__()
        self.norm1 = nn.RMSNorm(embed_dim)
        self.attn  = CausalSelfAttention(embed_dim, n_heads, dropout, max_len)
        self.norm2 = nn.RMSNorm(embed_dim)
        self.ffn   = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class TransformerBaseline(nn.Module):
    """
    GPT-style Transformer.
    Интерфейс идентичен WATDeepStackV1 — chunk_size принимается но не используется.
    n_heads подбирается автоматически как наибольший делитель embed_dim <= 4.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim:  int,
        n_layers:   int   = 2,
        chunk_size: int   = 32,   # ignored, only for interface compatibility
        max_len:    int   = 2048,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_layers  = n_layers

        # Подбираем n_heads: максимальный делитель embed_dim из [1,2,4]
        n_heads = 1
        for h in [1, 2, 4]:
            if embed_dim % h == 0 and embed_dim // h >= 8:
                n_heads = h

        self.embedding    = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Embedding(max_len, embed_dim)
        self.input_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, dropout, max_len)
            for _ in range(n_layers)
        ])

        self.output_norm = nn.RMSNorm(embed_dim)
        self.predict     = nn.Linear(embed_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        scale = 0.02 / (2 * self.n_layers) ** 0.5
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=scale)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        positions = torch.arange(T, device=x.device)
        h = self.embedding(x) + self.pos_encoding(positions)
        h = self.input_dropout(h)
        for layer in self.layers:
            h = layer(h)
        h = self.output_norm(h)
        return self.predict(h)                                 # (B, T, vocab_size)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def find_embed_dim_transformer(
    vocab_size:    int,
    target_params: int,
    n_layers:      int = 2,
    max_ed:        int = 512,
) -> tuple[int, int]:
    """Подбирает embed_dim чтобы число параметров было близко к target_params."""
    best_ed, best_n = 8, float('inf')
    for ed in range(8, max_ed, 4):
        try:
            m = TransformerBaseline(vocab_size, ed, n_layers=n_layers)
            n = m.count_params()
            if abs(n - target_params) < abs(best_n - target_params):
                best_n, best_ed = n, ed
        except Exception:
            continue
    return best_ed, best_n


if __name__ == "__main__":
    vocab, B, T = 65, 2, 512
    ed, n = find_embed_dim_transformer(vocab, 50_000, n_layers=1)
    model = TransformerBaseline(vocab, ed, n_layers=1)
    dummy = torch.randint(0, vocab, (B, T))
    out   = model(dummy)
    print(f"embed_dim : {ed}")
    print(f"params    : {n:,}")
    print(f"output    : {out.shape}  expected (2, 512, {vocab})")
    assert out.shape == (B, T, vocab)
    print("[OK]")