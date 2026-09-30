import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1d(nn.Module):
    def __init__(self, embed_dim: int, kernel_size: int = 3):
        super().__init__()
        self.padding = kernel_size - 1
        self.conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = F.pad(x, (self.padding, 0))
        x = self.conv(x)
        return x.transpose(1, 2)


class WATBlock(nn.Module):
    def __init__(self, embed_dim: int, chunk_size: int = 32):
        super().__init__()
        self.embed_dim  = embed_dim
        self.chunk_size = chunk_size

        self.conv   = CausalConv1d(embed_dim, kernel_size=3)
        self.W_gate = nn.Linear(embed_dim, embed_dim)

        self.W_merge_val  = nn.Linear(embed_dim * 2, embed_dim)
        self.W_merge_gate = nn.Linear(embed_dim * 2, embed_dim)
        self.W_res_gate   = nn.Linear(embed_dim * 2, embed_dim)
        self.tree_norm    = nn.RMSNorm(embed_dim)

        self.W_global = nn.Linear(embed_dim, embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        self.norm_conv = nn.RMSNorm(embed_dim)
        self.norm_tree = nn.RMSNorm(embed_dim)
        self.norm_ffn  = nn.RMSNorm(embed_dim)

    def _glu_merge(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([left, right], dim=-1)
        val      = self.W_merge_val(combined)
        gate     = torch.sigmoid(self.W_merge_gate(combined))
        merged   = self.tree_norm(val * gate)
        res_gate = torch.sigmoid(self.W_res_gate(combined))
        residual = (left + right) * 0.5
        return res_gate * merged + (1.0 - res_gate) * residual

    def _tree_reduction_all(self, chunks: torch.Tensor) -> torch.Tensor:
        B, C, K, D = chunks.shape
        curr = chunks
        size = K

        while size > 1:
            next_size = (size + 1) // 2
            if size % 2 != 0:
                curr = torch.cat([curr, curr[:, :, -1:, :]], dim=2)

            curr = curr.view(B, C, next_size, 2, D)
            left  = curr[:, :, :, 0, :]
            right = curr[:, :, :, 1, :]
            curr  = self._glu_merge(left, right)
            size  = next_size

        return curr.squeeze(2)

    def _build_global_ctx(self, summaries: torch.Tensor,
                          batch_size: int, embed_dim: int) -> torch.Tensor:
        n_chunks = summaries.size(1)
        device   = summaries.device

        if n_chunks == 1:
            return torch.zeros(batch_size, 1, embed_dim, device=device)

        cumsum = torch.cumsum(summaries, dim=1)
        counts = torch.arange(1, n_chunks + 1, device=device,
                               dtype=torch.float32).view(1, -1, 1)
        means  = cumsum / counts

        ctx = torch.cat([
            torch.zeros(batch_size, 1, embed_dim, device=device),
            means[:, :-1, :]
        ], dim=1)

        return ctx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        K       = self.chunk_size

        h = self.norm_conv(x)
        h = self.conv(h)
        h = h * torch.sigmoid(self.W_gate(h))
        x = x + h

        pad_len = (K - T % K) % K
        x_padded = x if pad_len == 0 else \
            torch.cat([x, x[:, -1:, :].expand(-1, pad_len, -1)], dim=1)

        T_padded = x_padded.size(1)
        n_chunks = T_padded // K

        chunks     = x_padded.unfold(1, K, K).transpose(2, 3)
        summaries  = self._tree_reduction_all(chunks)

        global_ctx = self._build_global_ctx(summaries, B, D)

        ctx_expanded = global_ctx.unsqueeze(2)\
                                  .expand(-1, -1, K, -1)\
                                  .reshape(B, T_padded, D)

        h_ctx = x_padded + self.W_global(ctx_expanded)
        h_ctx = h_ctx[:, :T, :]
        x = x + (h_ctx - x.detach()) * 0.5

        h = self.norm_ffn(x)
        x = x + self.ffn(h)

        return x


class WATDeepStackV1(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim:  int,
        n_layers:   int   = 2,
        chunk_size: int   = 32,
        max_len:    int   = 2048,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim  = embed_dim
        self.n_layers   = n_layers
        self.chunk_size = chunk_size

        self.embedding    = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Embedding(max_len, embed_dim)
        self.input_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            WATBlock(embed_dim, chunk_size)
            for _ in range(n_layers)
        ])

        self.layer_dropout = nn.Dropout(dropout)

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
            h = self.layer_dropout(h)

        h = self.output_norm(h)
        return self.predict(h)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_breakdown(self) -> dict:
        emb   = sum(p.numel() for p in self.embedding.parameters())
        pos   = sum(p.numel() for p in self.pos_encoding.parameters())
        layers_total = 0
        for i, layer in enumerate(self.layers):
            layers_total += sum(p.numel() for p in layer.parameters())
        out  = sum(p.numel() for p in self.predict.parameters())
        norm = sum(p.numel() for p in self.output_norm.parameters())
        return {
            "embedding":    emb,
            "pos_encoding": pos,
            "layers_total": layers_total,
            "per_layer":    layers_total // self.n_layers,
            "output":       out + norm,
            "total":        self.count_params(),
        }


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def find_embed_dim(vocab_size: int, target_params: int,
                   n_layers: int = 2, max_ed: int = 512) -> tuple[int, int]:
    best_ed, best_n = 8, float('inf')
    for ed in range(8, max_ed, 4):
        try:
            m = WATDeepStackV1(vocab_size, ed, n_layers=n_layers)
            n = count_params(m)
            if abs(n - target_params) < abs(best_n - target_params):
                best_n, best_ed = n, ed
        except Exception:
            continue
    return best_ed, best_n


def generate_text(
    model: nn.Module,
    prompt_tokens: list,
    idx_to_char: dict,
    device: torch.device,
    max_len: int   = 200,
    temperature: float = 0.8,
    top_k: int     = 40,
) -> str:
    model.eval()
    current = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_len):
            logits     = model(current)
            last       = logits[0, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(last, min(top_k, last.size(0)))
                last[last < v[-1]] = float('-inf')
            probs      = F.softmax(last, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            current    = torch.cat([current, next_token.unsqueeze(0)], dim=1)
            if current.size(1) > 1024:
                current = current[:, -1024:]

    return ''.join(idx_to_char.get(t, '?') for t in current[0].cpu().tolist())


if __name__ == "__main__":
    print("=" * 60)
    print("WAT DeepStack V1 — Architecture Check")
    print("=" * 60)

    VOCAB   = 65
    B, T, D = 2, 512, 0
    TARGET  = 500_000

    ed, n_params = find_embed_dim(VOCAB, TARGET, n_layers=2)
    model = WATDeepStackV1(vocab_size=VOCAB, embed_dim=ed, n_layers=2)

    print(f"\nEmbed dim: {ed}")
    print(f"Params: {n_params:,}  (target: {TARGET:,})")
    print(f"\nParam breakdown:")
    for k, v in model.param_breakdown().items():
        print(f"  {k:<20} {v:>10,}")

    dummy = torch.randint(0, VOCAB, (2, 512))
    out   = model(dummy)
    print(f"\nInput:  {dummy.shape}")
    print(f"Output: {out.shape}")
    print(f"Expected: (2, 512, {VOCAB})")
    assert out.shape == (2, 512, VOCAB), "Shape mismatch!"

    print("\n[OK] All checks passed")
    print("=" * 60)
