# models.py — Track A: CLIP wrapper; Custom dual encoder: ViT + text encoder + projection heads.
"""
Custom dual encoder: DualEncoderModel — timm ViT + DistilBERT + our projection heads.
Same structure as CLIP: two encoders, projections to shared dim, L2-normalized embeddings.
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

try:
    import timm
except ImportError:
    timm = None


class DualEncoderModel(nn.Module):
    """
    Custom dual encoder: Build from components. ViT (timm) + text encoder (DistilBERT) + our projection heads.
    Projections are randomly initialized; encoders are pretrained and frozen in B0.
    """
    def __init__(
        self,
        vision_model: str = "vit_base_patch16_224",
        text_model: str = "distilbert-base-uncased",
        projection_dim: int = 512,
        temperature_init: float = 0.07,
    ):
        super().__init__()
        if timm is None:
            raise ImportError("timm is required for custom dual encoder. pip install timm")

        # Vision encoder (timm ViT) — output 768
        self.vision_encoder = timm.create_model(vision_model, pretrained=True, num_classes=0)
        vision_dim = self.vision_encoder.num_features  # 768 for ViT-B

        # Text encoder (DistilBERT) — output 768
        self.text_encoder = AutoModel.from_pretrained(text_model)
        text_dim = self.text_encoder.config.hidden_size  # 768

        # Projection heads (random init for B0)
        self.image_projection = nn.Linear(vision_dim, projection_dim)
        self.text_projection = nn.Linear(text_dim, projection_dim)
        self._init_projection(self.image_projection)
        self._init_projection(self.text_projection)

        # Temperature (learnable)
        self.logit_scale = nn.Parameter(torch.tensor(1.0 / temperature_init).log())

    def _init_projection(self, proj: nn.Linear):
        nn.init.normal_(proj.weight, std=0.02)
        nn.init.zeros_(proj.bias)

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, projection_dim), L2-normalized."""
        feat = self.vision_encoder(pixel_values)  # (B, 768)
        feat = self.image_projection(feat)
        return feat / feat.norm(dim=-1, keepdim=True)

    def get_text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """(B, L) -> (B, projection_dim), L2-normalized. Uses mean pooling over non-padding tokens."""
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pooling: (B, L, 768) * mask -> (B, 768)
        hidden = out.last_hidden_state  # (B, L, 768)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = hidden.mean(dim=1)
        feat = self.text_projection(pooled)
        return feat / feat.norm(dim=-1, keepdim=True)

    @property
    def temperature(self) -> torch.Tensor:
        return self.logit_scale.exp()
