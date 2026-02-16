# conformer_arch.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardModule(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConformerConvModule(nn.Module):
    """
    Conformer conv module (simplified, widely used structure):
      LN -> pointwise conv -> GLU -> depthwise conv -> BN -> SiLU -> pointwise conv -> dropout
    """
    def __init__(self, dim: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size should be odd for 'same' padding"
        self.ln = nn.LayerNorm(dim)
        self.pw_conv1 = nn.Conv1d(dim, 2 * dim, kernel_size=1)
        self.dw_conv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim
        )
        self.bn = nn.BatchNorm1d(dim)
        self.pw_conv2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, D) -> conv wants (B, D, T)
        x = self.ln(x)
        x = x.transpose(1, 2)  # (B, D, T)

        x = self.pw_conv1(x)   # (B, 2D, T)
        x = F.glu(x, dim=1)    # (B, D, T)

        x = self.dw_conv(x)
        x = self.bn(x)
        x = F.silu(x)

        x = self.pw_conv2(x)
        x = self.dropout(x)

        return x.transpose(1, 2)  # (B, T, D)


class ConformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_hidden_dim: int,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.ffn1 = FeedForwardModule(dim, ffn_hidden_dim, dropout)
        self.attn_ln = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_drop = nn.Dropout(dropout)

        self.conv = ConformerConvModule(dim, kernel_size=conv_kernel_size, dropout=dropout)

        self.ffn2 = FeedForwardModule(dim, ffn_hidden_dim, dropout)
        self.final_ln = nn.LayerNorm(dim)

        # Macaron scaling commonly uses 0.5 on both FFNs
        self.ffn_scale = 0.5

    def forward(self, x, key_padding_mask=None):
        # x: (B, T, D)
        x = x + self.ffn_scale * self.ffn1(x)

        # MHSA
        a = self.attn_ln(x)
        a, _ = self.attn(a, a, a, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.attn_drop(a)

        # Conv module
        x = x + self.conv(x)

        x = x + self.ffn_scale * self.ffn2(x)
        return self.final_ln(x)


class ConformerEncoder(nn.Module):
    def __init__(
        self,
        dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        ffn_hidden_dim: int = 1024,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            ConformerBlock(dim, num_heads, ffn_hidden_dim, conv_kernel_size, dropout)
            for _ in range(depth)
        ])

    def forward(self, x, key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return x


class ConformerClassifier(nn.Module):
    """
    Minimal end-to-end classifier on (B,T,feat):
      proj -> conformer -> mean pool -> linear head
    """
    def __init__(self, input_dim: int, model_dim: int, num_classes: int, **enc_kwargs):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, model_dim)
        self.enc = ConformerEncoder(dim=model_dim, **enc_kwargs)
        self.head = nn.Linear(model_dim, num_classes)

    def forward(self, x, key_padding_mask=None):
        x = self.in_proj(x)
        x = self.enc(x, key_padding_mask=key_padding_mask)
        x = x.mean(dim=1)  # simple pooling
        return self.head(x)


if __name__ == "__main__":
    m = ConformerClassifier(input_dim=40, model_dim=256, num_classes=12, depth=2, num_heads=8, ffn_hidden_dim=512)
    x = torch.randn(4, 200, 40)  # (B,T,feat)
    y = m(x)
    print(y.shape)  # (4, 12)

