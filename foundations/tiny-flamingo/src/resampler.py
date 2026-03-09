"""Perceiver Resampler: compress vision tokens to fixed latents.

Learned latents cross-attend to vision features. Output (B, num_latents, d_latent).
Adapted from foundations/perceiver_arch.py.
"""
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """MHA with explicit Q and KV inputs."""
    def __init__(self, d_q: int, d_kv: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_q % n_heads == 0
        self.d_q = d_q
        self.d_kv = d_kv
        self.n_heads = n_heads
        self.head_dim = d_q // n_heads
        self.to_q = nn.Linear(d_q, d_q, bias=False)
        self.to_k = nn.Linear(d_kv, d_q, bias=False)
        self.to_v = nn.Linear(d_kv, d_q, bias=False)
        self.out = nn.Linear(d_q, d_q, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, q_in: torch.Tensor, kv_in: torch.Tensor, mask: torch.Tensor | None = None):
        B, Nq, _ = q_in.shape
        _, Nk, _ = kv_in.shape
        Q = self.to_q(q_in)
        K = self.to_k(kv_in)
        V = self.to_v(kv_in)
        Q = Q.view(B, Nq, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, Nk, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, Nk, self.n_heads, self.head_dim).transpose(1, 2)
        scale = self.head_dim ** -0.5
        attn_logits = (Q @ K.transpose(-2, -1)) * scale
        if mask is not None:
            attn_logits = attn_logits.masked_fill(~mask, float("-inf"))
        attn = self.drop(attn_logits.softmax(dim=-1))
        out = (attn @ V).transpose(1, 2).contiguous().view(B, Nq, self.d_q)
        return self.out(out)


class FeedForward(nn.Module):
    def __init__(self, d: int, d_hidden: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PerceiverBlock(nn.Module):
    """Cross-attn (latents query input) -> FF -> self-attn -> FF."""
    def __init__(self, d_latent: int, d_input: int, n_heads: int, ff_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.ln_latents_1 = nn.LayerNorm(d_latent)
        self.ln_input = nn.LayerNorm(d_input)
        self.cross_attn = MultiHeadAttention(d_q=d_latent, d_kv=d_input, n_heads=n_heads, dropout=dropout)
        self.ff1 = FeedForward(d_latent, ff_mult * d_latent, dropout)
        self.ln_latents_2 = nn.LayerNorm(d_latent)
        self.self_attn = MultiHeadAttention(d_q=d_latent, d_kv=d_latent, n_heads=n_heads, dropout=dropout)
        self.ff2 = FeedForward(d_latent, ff_mult * d_latent, dropout)

    def forward(self, latents: torch.Tensor, x: torch.Tensor, x_mask: torch.Tensor | None = None):
        attn_mask = x_mask[:, None, None, :] if x_mask is not None else None
        lat = self.ln_latents_1(latents)
        xin = self.ln_input(x)
        latents = latents + self.cross_attn(lat, xin, mask=attn_mask)
        latents = latents + self.ff1(latents)
        lat = self.ln_latents_2(latents)
        latents = latents + self.self_attn(lat, lat)
        latents = latents + self.ff2(latents)
        return latents


class PerceiverResampler(nn.Module):
    """Compress vision tokens (B, N, d_input) to fixed latents (B, num_latents, d_latent)."""
    def __init__(
        self,
        d_input: int,
        num_latents: int = 32,
        d_latent: int = 768,
        depth: int = 2,
        n_heads: int = 8,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, d_latent) * 0.02)
        self.blocks = nn.ModuleList([
            PerceiverBlock(d_latent=d_latent, d_input=d_input, n_heads=n_heads, ff_mult=ff_mult, dropout=dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(d_latent)

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, N, d_input) -> (B, num_latents, d_latent)"""
        B = x.shape[0]
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)
        for blk in self.blocks:
            latents = blk(latents, x, x_mask=x_mask)
        return self.norm(latents)
