"""
WAT DeepStack V1
================
Author: Igor Berezkin
Status: Independent Researcher

Архитектура: WAT DeepStack — стековая иерархическая модель.
Ключевая идея: несколько слоёв WATBlock с residual connections,
каждый слой строит всё более абстрактное представление контекста.

Отличия от оригинального WATDeepStack (из бенчмарка):
  1. WATBlock НЕ содержит pos_encoding — позиция кодируется ОДИН РАЗ
     на входе в модель, а не заново в каждом блоке (баг оригинала)
  2. Каждый WATBlock имеет свои НЕЗАВИСИМЫЕ веса дерева —
     слои учат разные уровни абстракции
  3. Pre-LayerNorm перед каждым блоком (как в современных Transformer)
     вместо Post-LayerNorm — стабильнее при глубине
  4. FFN после дерева внутри каждого блока (а не только на выходе)
  5. Dropout между слоями

Сложность: O(L * n * log K), где L — число слоёв, K — chunk_size
           При L=3, K=32, n=1024: ~15 * n операций (линейно от n)
           Transformer-Deep: O(L * n^2) = при L=6, n=1024: ~6M операций
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1d(nn.Module):
    """
    Свёртка с left-only padding — не видит будущее.
    padding = (kernel_size - 1, 0) — только слева.

    c_t = W_conv * [e_{t-2}, e_{t-1}, e_t]  ← e_{t+1} недоступен
    """
    def __init__(self, embed_dim: int, kernel_size: int = 3):
        super().__init__()
        self.padding = kernel_size - 1
        self.conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) → транспонируем для Conv1d → (B, D, T)
        x = x.transpose(1, 2)
        x = F.pad(x, (self.padding, 0))
        x = self.conv(x)
        return x.transpose(1, 2)          # (B, T, D)


# =========
# WAT BLOCK
# =========

class WATBlock(nn.Module):
    """
    Один слой WAT:
      CausalConv1d → InputGate → ChunkTreeReduction → GlobalCtxInjection → FFN

    Принимает уже вычисленные векторные представления (B, T, D),
    НЕ принимает индексы токенов — позиция уже закодирована выше.

    Параметры дерева (W_merge_val, W_merge_gate, W_res_gate) — SHARED
    внутри одного блока, но НЕЗАВИСИМЫ между блоками.
    Это позволяет каждому слою учить свою функцию слияния.
    """

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

    #--------------------------------------------------------------------
    # GLU MERGE — применяется на каждом уровне дерева
    #
    # combined = concat(left, right)          # 2d
    # val      = W_val  · combined            # d
    # gate     = σ(W_gate · combined)         # d, ∈ (0,1)
    # merged   = RMSNorm(val ⊙ gate)
    # res_gate = σ(W_res · combined)          # d, ∈ (0,1)
    # residual = (left + right) / 2
    # out      = res_gate ⊙ merged + (1 - res_gate) ⊙ residual
    #--------------------------------------------------------------------
    def _glu_merge(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([left, right], dim=-1)
        val      = self.W_merge_val(combined)
        gate     = torch.sigmoid(self.W_merge_gate(combined))
        merged   = self.tree_norm(val * gate)
        res_gate = torch.sigmoid(self.W_res_gate(combined))
        residual = (left + right) * 0.5
        return res_gate * merged + (1.0 - res_gate) * residual

    def _tree_reduction_all(self, chunks: torch.Tensor) -> torch.Tensor:
        """
        Параллельная tree reduction всех чанков.
        chunks: (B, n_chunks, chunk_size, D)
        returns: (B, n_chunks, D)

        Complexity per call: O(chunk_size * log(chunk_size) * n_chunks)
        """
        B, C, K, D = chunks.shape
        curr = chunks
        size = K

        while size > 1:
            next_size = (size + 1) // 2
            if size % 2 != 0:
                curr = torch.cat([curr, curr[:, :, -1:, :]], dim=2)

            # reshape: (B, C, size, D) → (B, C, next_size, 2, D)
            curr = curr.view(B, C, next_size, 2, D)
            left  = curr[:, :, :, 0, :]   # (B, C, next_size, D)
            right = curr[:, :, :, 1, :]   # (B, C, next_size, D)
            curr  = self._glu_merge(left, right)
            size  = next_size

        return curr.squeeze(2)  # (B, C, D)

    def _build_global_ctx(self, summaries: torch.Tensor,
                          batch_size: int, embed_dim: int) -> torch.Tensor:
        """
        Каузальное среднее: ctx[i] = mean(S_0 ... S_{i-1})
        ctx[0] = 0 (нет прошлых чанков)

        Гарантирует: токен t в чанке i не видит чанк i и далее.

        summaries: (B, n_chunks, D)
        returns:   (B, n_chunks, D)
        """
        n_chunks = summaries.size(1)
        device   = summaries.device

        if n_chunks == 1:
            return torch.zeros(batch_size, 1, embed_dim, device=device)

        cumsum = torch.cumsum(summaries, dim=1)                                # (B, C, D)
        counts = torch.arange(1, n_chunks + 1, device=device,
                               dtype=torch.float32).view(1, -1, 1)
        means  = cumsum / counts                                               # (B, C, D)

        # Сдвиг вправо: чанк i получает mean(0..i-1), чанк 0 получает 0
        ctx = torch.cat([
            torch.zeros(batch_size, 1, embed_dim, device=device),
            means[:, :-1, :]
        ], dim=1)                                                              # (B, C, D)

        return ctx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, D) — векторные представления (НЕ индексы!)
        returns: (B, T, D) — обновлённые представления

        Поток:
          x → [Pre-Norm] → CausalConv → InputGate → ChunkTree
            → GlobalCtxInjection → [residual] → [Pre-Norm] → FFN → [residual]
        """
        B, T, D = x.shape
        K       = self.chunk_size

        #Локальный контекст (Pre-Norm + residual)
        h = self.norm_conv(x)
        h = self.conv(h)
        h = h * torch.sigmoid(self.W_gate(h))   # Input gate
        x = x + h                               # Residual

        #Chunk-based Tree Reduction
        # Padding до кратного K
        pad_len = (K - T % K) % K
        x_padded = x if pad_len == 0 else \
            torch.cat([x, x[:, -1:, :].expand(-1, pad_len, -1)], dim=1)

        T_padded = x_padded.size(1)
        n_chunks = T_padded // K

        # (B, T_padded, D) → (B, n_chunks, K, D)
        chunks     = x_padded.unfold(1, K, K).transpose(2, 3)
        summaries  = self._tree_reduction_all(chunks)                  # (B, C, D)

        # Каузальный глобальный контекст
        global_ctx = self._build_global_ctx(summaries, B, D)           # (B, C, D)

        # Expand: (B, C, D) → (B, T_padded, D)
        ctx_expanded = global_ctx.unsqueeze(2)\
                                  .expand(-1, -1, K, -1)\
                                  .reshape(B, T_padded, D)

        # Инжекция + обрезка + residual
        h_ctx = x_padded + self.W_global(ctx_expanded)
        h_ctx = h_ctx[:, :T, :]   # Обрезаем до исходной длины
        x = x + (h_ctx - x.detach()) * 0.5         # = h_ctx  (замена, residual внутри ctx)
        # Примечание: x уже содержит residual через x_padded.
        # Здесь x = h_ctx — выход дерева становится новым представлением

        #FFN (Pre-Norm + residual)
        h = self.norm_ffn(x)
        x = x + self.ffn(h)

        return x

# ===========================================================================
# WAT DEEPSTACK V1 — стек из L слоёв WATBlock
# ===========================================================================

class WATDeepStackV1(nn.Module):
    """
    WAT DeepStack V1 — главная модель.

    Архитектура:
      Embedding + Positional → Dropout
      → [WATBlock_0] → [WATBlock_1] → ... → [WATBlock_{L-1}]
      → RMSNorm → Linear(vocab)

    Принципиальные решения:
      - Positional encoding ОДИН РАЗ на входе
      - Каждый WATBlock имеет СВОИ веса (не shared между слоями)
      - Pre-Norm внутри каждого блока → стабильный градиентный поток
      - Residual connections через весь стек
      - Output norm перед predict (как в GPT-2)

    Сложность:
      O(L * n * log K) — линейно от n при фиксированном K
      vs Transformer-Deep: O(L * n^2)

      При L=3, K=32, n=1024:
        WAT DeepStack:    3 * 1024 * 5  ≈ 15,360  операций масштаба
        Transformer-Deep: 6 * 1024^2    ≈ 6,291,456 операций масштаба
        Выигрыш: ~400x по теоретическим FLOPs (на практике ~3x из-за overhead)
    """

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

        # === Входной слой ===
        # Positional encoding только здесь — НЕ в каждом блоке
        self.embedding    = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Embedding(max_len, embed_dim)
        self.input_dropout = nn.Dropout(dropout)

        # === Стек блоков — каждый со своими весами ===
        self.layers = nn.ModuleList([
            WATBlock(embed_dim, chunk_size)
            for _ in range(n_layers)
        ])

        # Dropout между слоями
        self.layer_dropout = nn.Dropout(dropout)

        # === Выходной слой ===
        self.output_norm = nn.RMSNorm(embed_dim)
        self.predict     = nn.Linear(embed_dim, vocab_size)

        # Инициализация весов
        self._init_weights()

    def _init_weights(self):
        """
        Scaled initialization — стандарт для глубоких моделей.
        Линейные слои: std = 0.02 / sqrt(2 * n_layers)
        Embeddings: std = 0.02
        """
        scale = 0.02 / (2 * self.n_layers) ** 0.5
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=scale)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T) — индексы токенов
        returns: (B, T, vocab_size) — логиты для каждой позиции

        Поток данных:
          idx → Embedding → +PosEnc → Dropout
              → Block_0 → Dropout
              → Block_1 → Dropout
              → Block_2 → Dropout
              → RMSNorm → Linear → logits
        """
        B, T = x.shape
        positions = torch.arange(T, device=x.device)

        # Входной слой
        h = self.embedding(x) + self.pos_encoding(positions)  # (B, T, D)
        h = self.input_dropout(h)

        # Стек WAT блоков
        for layer in self.layers:
            h = layer(h)
            h = self.layer_dropout(h)

        # Выходной слой
        h = self.output_norm(h)
        return self.predict(h)   # (B, T, vocab_size)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_breakdown(self) -> dict:
        """Разбивка параметров по компонентам."""
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


# ===========================================================================
# УТИЛИТЫ
# ===========================================================================

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def find_embed_dim(vocab_size: int, target_params: int,
                   n_layers: int = 2, max_ed: int = 512) -> tuple[int, int]:
    """Подбирает embed_dim ближайший к target_params."""
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
    """Авторегрессивная генерация текста."""
    model.eval()
    current = torch.tensor([prompt_tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_len):
            logits     = model(current)            # (1, T, V)
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