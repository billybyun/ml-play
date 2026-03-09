"""Models for tiny-vit: pretrained ViT (timm), tiny ViT from scratch."""
import timm
import torch
import torch.nn as nn


class TinyViT(nn.Module):
    """Minimal ViT for CIFAR-10 (32x32), trained from scratch."""
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        num_classes: int = 10,
        embed_dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        B, N, D = x.shape
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, : (1 + N), :]
        x = self.pos_drop(x)
        x = self.encoder(x)
        x = self.norm(x[:, 0])
        return self.head(x)


def create_vit(config: dict) -> nn.Module:
    """Create ViT model from config (timm pretrained)."""
    model_name = config.get("model_name", "vit_base_patch16_224")
    pretrained = config.get("pretrained", True)
    num_classes = config.get("num_classes", 10)

    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model


def freeze_backbone_for_linear_probe(model: nn.Module) -> None:
    """Freeze all parameters except the classification head (for linear probe)."""
    for name, param in model.named_parameters():
        if "head" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def create_tiny_vit(config: dict) -> nn.Module:
    """Create tiny ViT from config (for training from scratch on CIFAR-10)."""
    return TinyViT(
        img_size=config.get("image_size", 32),
        patch_size=config.get("patch_size", 4),
        num_classes=config.get("num_classes", 10),
        embed_dim=config.get("embed_dim", 256),
        depth=config.get("depth", 4),
        num_heads=config.get("num_heads", 8),
        dropout=config.get("dropout", 0.1),
    )
