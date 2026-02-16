import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Standard MHA but with explicit Q input and KV input.
    Q: (B, Nq, Dq)
    KV: (B, Nk, Dk)
    Output: (B, Nq, D_out) where D_out = Dq by default
    """
    def __init__(self, d_q: int, d_kv: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_q % n_heads == 0, "d_q must be divisible by n_heads"
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

        Q = self.to_q(q_in)  # (B, Nq, d_q)
        K = self.to_k(kv_in) # (B, Nk, d_q)
        V = self.to_v(kv_in) # (B, Nk, d_q)

        # Split heads
        Q = Q.view(B, Nq, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, Nq, Hd)
        K = K.view(B, Nk, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, Nk, Hd)
        V = V.view(B, Nk, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, Nk, Hd)

        # Attention
        scale = self.head_dim ** -0.5
        attn_logits = (Q @ K.transpose(-2, -1)) * scale  # (B, H, Nq, Nk)

        if mask is not None:
            # mask: (B, 1, 1, Nk) broadcastable
            attn_logits = attn_logits.masked_fill(~mask, float("-inf"))

        attn = attn_logits.softmax(dim=-1)
        attn = self.drop(attn)

        out = attn @ V  # (B, H, Nq, Hd)
        out = out.transpose(1, 2).contiguous().view(B, Nq, self.d_q)  # (B, Nq, d_q)
        out = self.out(out)
        return out


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

    def forward(self, x):
        return self.net(x)


class PerceiverBlock(nn.Module):
    """
    One Perceiver block:
      1) Cross-attn: latents query input
      2) FF on latents
      3) Self-attn on latents
      4) FF on latents
    """
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
        # Cross-attn
        lat = self.ln_latents_1(latents)
        xin = self.ln_input(x)
        latents = latents + self.cross_attn(lat, xin, mask=x_mask)
        latents = latents + self.ff1(latents)

        # Self-attn (on latents only)
        lat = self.ln_latents_2(latents)
        latents = latents + self.self_attn(lat, lat)
        latents = latents + self.ff2(latents)

        return latents


class PerceiverClassifier(nn.Module):
    def __init__(
        self,
        d_input: int,
        num_classes: int,
        num_latents: int = 128,
        d_latent: int = 256,
        depth: int = 4,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, d_latent) * 0.02)

        self.blocks = nn.ModuleList([
            PerceiverBlock(d_latent=d_latent, d_input=d_input, n_heads=n_heads, dropout=dropout)
            for _ in range(depth)
        ])

        self.head = nn.Sequential(
            nn.LayerNorm(d_latent),
            nn.Linear(d_latent, num_classes)
        )

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor | None = None):
        """
        x: (B, N, d_input)
        x_mask: optional boolean mask, shape (B, N), True where valid
        """
        B, N, _ = x.shape
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)  # (B, M, d_latent)

        attn_mask = None
        if x_mask is not None:
            # Make it broadcastable to (B, 1, 1, N)
            attn_mask = x_mask[:, None, None, :]

        for blk in self.blocks:
            latents = blk(latents, x, x_mask=attn_mask)

        # Pool latents (mean) -> classify
        pooled = latents.mean(dim=1)  # (B, d_latent)
        logits = self.head(pooled)    # (B, num_classes)
        return logits


def demo():
    torch.manual_seed(0)

    B = 8
    N = 2000      # big input length
    d_in = 64
    num_classes = 6

    x = torch.randn(B, N, d_in)
    y = torch.randint(0, num_classes, (B,))

    model = PerceiverClassifier(d_input=d_in, num_classes=num_classes, num_latents=128, d_latent=256, depth=3)
    logits = model(x)
    loss = F.cross_entropy(logits, y)

    print("logits:", logits.shape)  # (B, num_classes)
    print("loss:", float(loss))

    loss.backward()
    print("backward ok")


if __name__ == "__main__":
    demo()

