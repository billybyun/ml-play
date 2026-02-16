import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLMAdapter(nn.Module):
    """
    Per-session feature modulation (FiLM-style).
    X: (B, T, C, F)
    """
    def __init__(self, C: int, Fdim: int):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(C, Fdim))
        self.bias = nn.Parameter(torch.zeros(C, Fdim))

    def forward(self, X):
        scale = torch.exp(self.log_scale)[None, None, :, :]  # (1,1,C,F)
        bias  = self.bias[None, None, :, :]                  # (1,1,C,F)
        return X * scale + bias


class TinyECoGTransformer(nn.Module):
    """
    Tokens = (time, channel)
    Input X: (B, T, C, F)
    Output: logits (B, n_classes)
    """
    def __init__(
        self,
        C: int,
        Fdim: int,
        n_classes: int = 6,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        dropout: float = 0.1,
        max_T: int = 256,
    ):
        super().__init__()
        self.C = C
        self.Fdim = Fdim
        self.max_T = max_T

        # Per-session drift handling
        self.adapter = FiLMAdapter(C, Fdim)

        # Token embed: project per-channel features -> d_model
        self.in_proj = nn.Linear(Fdim, d_model)

        # Learnable embeddings for channel identity and time index
        self.chan_emb = nn.Embedding(C, d_model)
        self.time_emb = nn.Embedding(max_T, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # pre-norm
        )
        self.tr = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, X):
        """
        X: (B, T, C, F)
        """
        B, T, C, Fdim = X.shape
        assert C == self.C and Fdim == self.Fdim
        assert T <= self.max_T, f"T={T} exceeds max_T={self.max_T}"

        X = self.adapter(X)  # (B,T,C,F)

        # Build tokens: (B, T*C, d_model)
        x = self.in_proj(X)  # (B,T,C,d_model)

        # Add channel embedding
        chan_ids = torch.arange(C, device=X.device)              # (C,)
        chan_e = self.chan_emb(chan_ids)[None, None, :, :]       # (1,1,C,d)
        x = x + chan_e

        # Add time embedding
        time_ids = torch.arange(T, device=X.device)              # (T,)
        time_e = self.time_emb(time_ids)[None, :, None, :]       # (1,T,1,d)
        x = x + time_e

        # Flatten tokens (time-major): (B, T*C, d)
        x = x.reshape(B, T * C, -1)

        # Transformer encoder
        h = self.tr(x)  # (B, T*C, d)

        # Pool: mean pool tokens (or you can pool last time tokens only)
        pooled = h.mean(dim=1)  # (B,d)

        logits = self.head(pooled)  # (B,n_classes)
        return logits

