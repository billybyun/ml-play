# loss.py — Symmetric CLIP InfoNCE. Used from Stage A1 / B1 onward.
"""
Symmetric contrastive loss for dual encoder / CLIP.
sim = (z_img @ z_txt.T) / temperature; targets = diagonal; loss = (CE_i2t + CE_t2i) / 2
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def clip_symmetric_loss(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    temperature: torch.Tensor,
) -> torch.Tensor:
    """
    Symmetric InfoNCE loss for image-text contrastive learning.

    Args:
        image_embeds: (B, D) L2-normalized image embeddings
        text_embeds: (B, D) L2-normalized text embeddings
        temperature: scalar, typically from model.logit_scale.exp()

    Returns:
        loss: scalar, (loss_i2t + loss_t2i) / 2
    """
    logits = (image_embeds @ text_embeds.T) * temperature
    B = logits.size(0)
    targets = torch.arange(B, device=logits.device, dtype=torch.long)

    loss_i2t = F.cross_entropy(logits, targets)
    loss_t2i = F.cross_entropy(logits.T, targets)
    return (loss_i2t + loss_t2i) / 2
