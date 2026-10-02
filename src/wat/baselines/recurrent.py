import torch
import torch.nn as nn
import torch.nn.functional as F

from wat.common import RMSNorm


class LSTMBackbone(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_layers=3, dropout=0.1, **kw):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_dim, embed_dim, num_layers=n_layers,
                            batch_first=True,
                            dropout=dropout if n_layers > 1 else 0.0)
        self.output_norm = RMSNorm(embed_dim)

    def forward(self, x):
        h = self.input_dropout(self.embedding(x))
        h, _ = self.lstm(h)
        return self.output_norm(h)


def ssd_scan(x, dt, A_log, Bm, Cm, chunk=16):
    b, T, d = x.shape
    N = Bm.shape[-1]
    a = -torch.exp(A_log.float())
    dt = F.softplus(dt.float())
    logdec = dt * a
    xin = x.float() * dt
    Bm = Bm.float()
    Cm = Cm.float()
    y = torch.empty(b, T, d, device=x.device, dtype=torch.float32)
    S = torch.zeros(b, d, N, device=x.device, dtype=torch.float32)
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        Q = e - s
        P = logdec[:, s:e].cumsum(1)
        Bq, Cq, xq = Bm[:, s:e], Cm[:, s:e], xin[:, s:e]
        y_inter = torch.einsum("bqn,bdn->bqd", Cq, S) * P.exp()
        M = torch.einsum("bqn,bpn->bqp", Cq, Bq)
        dec = P.unsqueeze(2) - P.unsqueeze(1)
        tri = torch.ones(Q, Q, device=x.device, dtype=torch.bool).tril()
        dec = dec.masked_fill(~tri.view(1, Q, Q, 1), float("-inf")).exp()
        y[:, s:e] = y_inter + torch.einsum("bqp,bqpd,bpd->bqd", M, dec, xq)
        wdec = (P[:, -1].unsqueeze(1) - P).exp()
        S = P[:, -1].exp().unsqueeze(-1) * S + torch.einsum("bqd,bqn->bdn", wdec * xq, Bq)
    return y.to(x.dtype)


class MambaBlockMinimal(nn.Module):
    def __init__(self, embed_dim, d_state=16, expand=2, d_conv=4):
        super().__init__()
        d_inner = expand * embed_dim
        self.norm = RMSNorm(embed_dim)
        self.in_proj = nn.Linear(embed_dim, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, groups=d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1, bias=False)
        self.dt_proj = nn.Linear(1, d_inner, bias=True)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1).mean(dim=1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, embed_dim, bias=False)
        with torch.no_grad():
            u = torch.rand(d_inner) * (0.1 - 1e-3) + 1e-3
            self.dt_proj.bias.copy_(u + torch.log(-torch.expm1(-u)))

    def forward(self, x):
        res = x
        xs, z = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        T = xs.size(1)
        xs = F.silu(self.conv1d(xs.transpose(1, 2))[:, :, :T].transpose(1, 2))
        bcd = self.x_proj(xs)
        N = (bcd.shape[-1] - 1) // 2
        Bm, Cm, dt0 = bcd[..., :N], bcd[..., N:2 * N], bcd[..., 2 * N:]
        y = ssd_scan(xs, self.dt_proj(dt0), self.A_log, Bm, Cm)
        y = (y + self.D * xs) * F.silu(z)
        return res + self.out_proj(y)


class MambaBackbone(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_layers=2, dropout=0.1, **kw):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([MambaBlockMinimal(embed_dim) for _ in range(n_layers)])
        self.layer_dropout = nn.Dropout(dropout)
        self.output_norm = RMSNorm(embed_dim)

    def forward(self, x):
        h = self.input_dropout(self.embedding(x))
        for layer in self.layers:
            h = self.layer_dropout(layer(h))
        return self.output_norm(h)
