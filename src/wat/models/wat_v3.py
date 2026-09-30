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
    def __init__(self, vocab_size: int, embed_dim: int, max_len: int = 2048, chunk_size: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim  = embed_dim
        self.chunk_size = chunk_size
        self.embedding    = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = nn.Embedding(max_len, embed_dim)
        self.conv = CausalConv1d(embed_dim, kernel_size=3)
        self.W_gate1 = nn.Linear(embed_dim, embed_dim)
        self.W_merge_val  = nn.Linear(embed_dim * 2, embed_dim)
        self.W_merge_gate = nn.Linear(embed_dim * 2, embed_dim)
        self.W_res_gate   = nn.Linear(embed_dim * 2, embed_dim)
        self.rmsnorm      = nn.RMSNorm(embed_dim)
        self.W_global = nn.Linear(embed_dim, embed_dim)
        self.predict = nn.Linear(embed_dim, vocab_size)

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
        logits = self.chunk_based_forward(nodes)
        return logits

    def chunk_based_forward(self, nodes: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, embed_dim = nodes.shape
        chunk_size = self.chunk_size
        n_chunks = (seq_len + chunk_size - 1) // chunk_size
        if seq_len % chunk_size != 0:
            pad_len = chunk_size - (seq_len % chunk_size)
            nodes = torch.cat([nodes, nodes[:, -1:, :].expand(-1, pad_len, -1)], dim=1)
        chunks = nodes.unfold(1, chunk_size, chunk_size).transpose(2, 3)
        chunk_summaries = self.tree_reduction_all(chunks)
        if n_chunks == 1:
            global_ctx = torch.zeros(batch_size, 1, embed_dim, device=nodes.device)
        else:
            cumsum = torch.cumsum(chunk_summaries, dim=1)
            counts = torch.arange(1, n_chunks + 1, device=nodes.device, dtype=torch.float32)
            counts = counts.view(1, -1, 1)
            global_ctx = cumsum / counts.clamp(min=1)
            global_ctx = global_ctx[:, :-1, :]
            global_ctx = torch.cat([torch.zeros(batch_size, 1, embed_dim, device=nodes.device), global_ctx], dim=1)
        global_ctx_expanded = global_ctx.unsqueeze(2).expand(-1, -1, chunk_size, -1)
        global_ctx_expanded = global_ctx_expanded.reshape(batch_size, n_chunks * chunk_size, embed_dim)
        nodes_with_ctx = nodes + self.W_global(global_ctx_expanded)
        logits = self.predict(nodes_with_ctx)
        if logits.size(1) > seq_len:
            logits = logits[:, :seq_len, :]
        return logits

    def tree_reduction_all(self, chunks: torch.Tensor) -> torch.Tensor:
        batch_size, n_chunks, chunk_size, embed_dim = chunks.shape
        curr = chunks
        size = chunk_size
        while size > 1:
            next_size = size // 2
            if size % 2 != 0:
                next_size += 1
            curr = curr.view(batch_size, n_chunks, next_size, 2, embed_dim)
            left = curr[:, :, :, 0, :]
            right = curr[:, :, :, 1, :]
            combined = torch.cat([left, right], dim=-1)
            val = self.W_merge_val(combined)
            gate = torch.sigmoid(self.W_merge_gate(combined))
            merged = self.rmsnorm(val * gate)
            res_gate = torch.sigmoid(self.W_res_gate(combined))
            residual = (left + right) * 0.5
            curr = res_gate * merged + (1 - res_gate) * residual
            size = next_size
        return curr.squeeze(2)
