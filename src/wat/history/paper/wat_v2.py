import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    def __init__(self, embed_dim, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size - 1
        self.conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=kernel_size)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = torch.nn.functional.pad(x, (self.padding, 0))
        x = self.conv(x)
        return x.transpose(1, 2)

class WATModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, max_len: int = 2048):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim  = embed_dim
        self.embedding    = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Embedding(max_len, embed_dim)
        self.conv = CausalConv1d(embed_dim, kernel_size=3)
        self.W_gate1 = nn.Linear(embed_dim, embed_dim)
        self.W_merge_val  = nn.Linear(embed_dim * 2, embed_dim)
        self.W_merge_gate = nn.Linear(embed_dim * 2, embed_dim)
        self.W_res_gate   = nn.Linear(embed_dim * 2, embed_dim)
        self.W_highway    = nn.Linear(embed_dim, embed_dim)
        self.rmsnorm      = nn.RMSNorm(embed_dim)
        self.predict = nn.Linear(embed_dim, vocab_size)

    def causal_scan(self, nodes: torch.Tensor) -> torch.Tensor:
        seq_len = nodes.size(1)
        curr = nodes
        step = 1
        while step < seq_len:
            left = curr[:, :-step, :]
            right = curr[:, step:, :]
            combined = torch.cat([left, right], dim=-1)
            val = self.W_merge_val(combined)
            gate = torch.sigmoid(self.W_merge_gate(combined))
            merged = self.rmsnorm(val * gate)
            highway_gate = torch.sigmoid(self.W_highway(right))
            merged = highway_gate * merged + (1 - highway_gate) * right
            res_gate = torch.sigmoid(self.W_res_gate(combined))
            residual = (left + right) * 0.5
            merged = res_gate * merged + (1 - res_gate) * residual
            new_curr = curr.clone()
            new_curr[:, step:, :] = merged
            curr = new_curr
            step *= 2
        return curr

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
        x = self.conv(x)
        nodes = x * torch.sigmoid(self.W_gate1(x))
        hidden = self.causal_scan(nodes)
        return self.predict(hidden)
