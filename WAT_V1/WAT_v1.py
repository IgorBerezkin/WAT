import numpy as np
import torch
import torch.nn as nn

np.random.seed(42)

class WATModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, max_len: int = 2048):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim  = embed_dim
        self.embedding    = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Embedding(max_len, embed_dim)
        self.conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1)
        self.W_gate1 = nn.Linear(embed_dim, embed_dim)
        self.W_merge_val  = nn.Linear(embed_dim * 2, embed_dim)
        self.W_merge_gate = nn.Linear(embed_dim * 2, embed_dim)
        self.W_res_gate   = nn.Linear(embed_dim * 2, embed_dim)
        self.rmsnorm      = nn.RMSNorm(embed_dim)
        self.predict = nn.Linear(embed_dim * 2, vocab_size)

    def tree_reduction(self, nodes: torch.Tensor) -> torch.Tensor:
        curr = nodes
        while curr.size(1) > 1:
            if curr.size(1) % 2 != 0:
                curr = torch.cat([curr, curr[:, -1:, :]], dim=1)
            left     = curr[:, 0::2, :]
            right    = curr[:, 1::2, :]
            combined = torch.cat([left, right], dim=-1)
            val    = self.W_merge_val(combined)
            gate   = torch.sigmoid(self.W_merge_gate(combined))
            merged = self.rmsnorm(val * gate)
            res_gate = torch.sigmoid(self.W_res_gate(combined))
            residual = (left + right) * 0.5
            curr     = res_gate * merged + (1 - res_gate) * residual
        return curr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len   = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        x = self.embedding(x) + self.pos_encoding(positions)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        nodes      = x * torch.sigmoid(self.W_gate1(x))
        nodes_past = nodes[:, :-1, :]
        curr = self.tree_reduction(nodes_past)
        root    = curr[:, 0, :]
        last    = nodes_past[:, -1, :]
        context = torch.cat([root, last], dim=-1)
        return self.predict(context)

class TransformerBaseline(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int,
                 n_heads: int = 4, max_len: int = 2048):
        super().__init__()
        self.embedding    = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Embedding(max_len, embed_dim)
        encoder_layer     = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=embed_dim * 4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.predict     = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len   = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = self.embedding(x) + self.pos_encoding(positions)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        x = self.transformer(x, mask=mask)
        return self.predict(x[:, -1, :])

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def find_embed_dim(model_class, vocab_size: int,
                   target_params: int, max_len: int = 2048) -> tuple:
    best_ed, best_n = None, float('inf')
    for ed in range(8, 256, 4):
        model = model_class(vocab_size, ed)
        n     = count_params(model)
        if abs(n - target_params) < abs(best_n - target_params):
            best_n  = n
            best_ed = ed
    return best_ed, best_n