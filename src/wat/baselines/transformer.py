import torch
import torch.nn as nn
import torch.nn.functional as F

from wat.common import RMSNorm, match_pointers


def rotary(x, base=10000.0):
    T, half = x.size(2), x.size(-1) // 2
    freq = base ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    angle = torch.arange(T, device=x.device, dtype=torch.float32)[:, None] * freq[None]
    cos, sin = angle.cos().to(x.dtype), angle.sin().to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class SwiGLU(nn.Module):
    def __init__(self, d):
        super().__init__()
        hidden = max(8, int(8 * d / 3) // 8 * 8)
        self.W_in = nn.Linear(d, 2 * hidden)
        self.W_out = nn.Linear(hidden, d)

    def forward(self, x):
        a, b = self.W_in(x).chunk(2, dim=-1)
        return self.W_out(F.silu(a) * b)


class CausalSelfAttention(nn.Module):
    def __init__(self, d, n_heads, dropout=0.1, max_len=2048, rope=False, sdpa=False):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, d // n_heads
        self.scale = self.head_dim ** -0.5
        self.rope, self.sdpa, self.p = rope, sdpa, dropout
        self.qkv = nn.Linear(d, d * 3)
        self.proj = nn.Linear(d, d)
        self.attn_dropout = nn.Dropout(dropout)
        if not sdpa:
            mask = torch.tril(torch.ones(max_len, max_len, dtype=torch.bool))
            self.register_buffer("mask", mask.view(1, 1, max_len, max_len),
                                 persistent=False)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = [t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
                   for t in self.qkv(x).chunk(3, dim=-1)]
        if self.rope:
            q, k = rotary(q), rotary(k)
        if self.sdpa:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                                 dropout_p=self.p if self.training else 0.0)
            return self.proj(out.transpose(1, 2).contiguous().view(B, T, D))
        att = (q @ k.transpose(-2, -1)) * self.scale
        att = att.masked_fill(~self.mask[:, :, :T, :T], float("-inf"))
        att = self.attn_dropout(F.softmax(att, dim=-1))
        out = (att @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.proj(out)


class TransformerBackbone(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_layers=3, max_len=2048,
                 dropout=0.1, ptr=None, pos="learned", ffn="gelu", attn="naive", **kw):
        super().__init__()
        self.embed_dim, self.vocab = embed_dim, vocab_size
        self.ptr = sorted(ptr) if ptr else []
        self.ptr_emb = nn.ModuleList([nn.Embedding(vocab_size + 1, embed_dim) for _ in self.ptr])
        n_heads = 1
        for h in (1, 2, 4):
            if embed_dim % h == 0 and embed_dim // h >= 8:
                n_heads = h
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Embedding(max_len, embed_dim) if pos == "learned" else None
        self.input_dropout = nn.Dropout(dropout)
        blocks = []
        for _ in range(n_layers):
            b = nn.Module()
            b.norm1 = RMSNorm(embed_dim)
            b.attn = CausalSelfAttention(embed_dim, n_heads, dropout, max_len,
                                         rope=pos == "rope", sdpa=attn == "sdpa")
            b.norm2 = RMSNorm(embed_dim)
            b.ffn = SwiGLU(embed_dim) if ffn == "swiglu" else nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4), nn.GELU(), nn.Linear(embed_dim * 4, embed_dim))
            b.drop = nn.Dropout(dropout)
            blocks.append(b)
        self.layers = nn.ModuleList(blocks)
        self.output_norm = RMSNorm(embed_dim)
        scale = 0.02 / (2 * max(1, n_layers)) ** 0.5
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, scale)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, x):
        h = self.embedding(x)
        if self.pos_encoding is not None:
            h = h + self.pos_encoding(torch.arange(x.size(1), device=x.device))
        for emb, cand in zip(self.ptr_emb, match_pointers(x, self.ptr, self.vocab)):
            h = h + emb(cand)
        h = self.input_dropout(h)
        for b in self.layers:
            h = h + b.drop(b.attn(b.norm1(h)))
            h = h + b.drop(b.ffn(b.norm2(h)))
        return self.output_norm(h)
