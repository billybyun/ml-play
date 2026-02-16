# vivit_arch.py
import torch
import torch.nn as nn


class TransformerStack(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, x):
        return self.enc(x)


class ViViT_Factorized(nn.Module):
    """
    Factorized ViViT:
      tubelet embed -> tokens shaped (B, T', S', D)
      spatial transformer over S' for each time
      temporal transformer over T' for pooled spatial tokens
      classifier head on CLS
    """
    def __init__(
        self,
        image_size: int = 224,
        frames: int = 16,
        in_chans: int = 3,
        num_classes: int = 400,
        tubelet_t: int = 2,
        patch_size: int = 16,
        embed_dim: int = 768,
        spatial_depth: int = 6,
        temporal_depth: int = 6,
        heads: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert image_size % patch_size == 0
        assert frames % tubelet_t == 0

        self.tubelet_t = tubelet_t
        self.patch_size = patch_size
        self.frames = frames

        t_tokens = frames // tubelet_t
        s_tokens = (image_size // patch_size) * (image_size // patch_size)
        self.t_tokens = t_tokens
        self.s_tokens = s_tokens

        # Tubelet embedding: (B,C,T,H,W) -> (B, D, T', H', W')
        self.tubelet_embed = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=(tubelet_t, patch_size, patch_size),
            stride=(tubelet_t, patch_size, patch_size),
        )

        # Positional embeddings (separate for space and time)
        self.spatial_pos = nn.Parameter(torch.zeros(1, s_tokens, embed_dim))
        self.temporal_pos = nn.Parameter(torch.zeros(1, t_tokens + 1, embed_dim))  # + CLS

        self.spatial_tr = TransformerStack(embed_dim, spatial_depth, heads, dropout=dropout)
        self.temporal_tr = TransformerStack(embed_dim, temporal_depth, heads, dropout=dropout)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.spatial_pos, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T, H, W)
        x = self.tubelet_embed(x)  # (B, D, T', H', W')
        B, D, Tp, Hp, Wp = x.shape
        x = x.permute(0, 2, 3, 4, 1)                 # (B, T', H', W', D)
        x = x.reshape(B, Tp, Hp * Wp, D)             # (B, T', S', D)

        # Spatial transformer per time step
        x = x.reshape(B * Tp, Hp * Wp, D)            # (B*T', S', D)
        x = x + self.spatial_pos[:, : x.size(1), :]
        x = self.spatial_tr(x)                       # (B*T', S', D)
        x = x.mean(dim=1)                            # pool spatial tokens -> (B*T', D)
        x = x.reshape(B, Tp, D)                      # (B, T', D)

        # Temporal transformer over time (add CLS)
        cls = self.cls_token.expand(B, -1, -1)       # (B,1,D)
        x = torch.cat([cls, x], dim=1)               # (B, 1+T', D)
        x = x + self.temporal_pos[:, : x.size(1), :]
        x = self.temporal_tr(x)

        out = self.norm(x[:, 0])
        return self.head(out)


if __name__ == "__main__":
    m = ViViT_Factorized(image_size=64, frames=8, tubelet_t=2, patch_size=16,
                         num_classes=10, embed_dim=256, spatial_depth=2, temporal_depth=2, heads=8)
    video = torch.randn(2, 3, 8, 64, 64)
    y = m(video)
    print(y.shape)  # (2, 10)

