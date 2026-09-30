import numpy as np
import torch
import torch.nn as nn

np.random.seed(42)

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

    def init_cache(self, seq_len: int):
        self.cache = {
            'last_nodes': None,
            'seq_len': seq_len,
        }
    
    def forward_with_cache(self, x: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, 'cache') or self.cache is None:
            return self.forward(x)
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device)
        x = self.embedding(x) + self.pos_encoding(positions)
        x = self.conv(x)
        node = x * torch.sigmoid(self.W_gate1(x))
        last_nodes = self.cache.get('last_nodes')
        if last_nodes is None:
            last_nodes = [node]
            self.cache['last_nodes'] = last_nodes
        curr = node
        step = 1
        for i, cached_node in enumerate(last_nodes):
            if cached_node.size(1) >= step:
                left = cached_node[:, -step:, :]
                right = curr
                combined = torch.cat([left, right], dim=-1)
                val = self.W_merge_val(combined)
                gate = torch.sigmoid(self.W_merge_gate(combined))
                merged = self.rmsnorm(val * gate)
                highway_gate = torch.sigmoid(self.W_highway(right))
                merged = highway_gate * merged + (1 - highway_gate) * right
                res_gate = torch.sigmoid(self.W_res_gate(combined))
                residual = (left + right) * 0.5
                merged = res_gate * merged + (1 - res_gate) * residual
                curr = merged
                last_nodes[i] = torch.cat([cached_node, curr], dim=1)
        self.cache['last_nodes'] = last_nodes
        return self.predict(curr)

def generate_text(model, prompt_tokens, vocab_size, idx_to_char, device, 
                  max_len=200, temperature=0.8, top_k=40):
    import torch.nn.functional as F
    model.eval()
    current = torch.tensor([prompt_tokens], dtype=torch.long).to(device)
    with torch.no_grad():
        for _ in range(max_len):
            logits = model(current)
            last_logits = logits[0, -1, :]
            last_logits = last_logits / temperature
            if top_k > 0:
                v, _ = torch.topk(last_logits, min(top_k, last_logits.size(0)))
                last_logits[last_logits < v[-1]] = float('-inf')
            probs = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            current = torch.cat([current, next_token.unsqueeze(0)], dim=1)
            if current.size(1) > 512:
                current = current[:, -512:]
    result = ''.join([idx_to_char.get(t, '?') for t in current[0].cpu().tolist()])
    return result

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