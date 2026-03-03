"""Models for tiny-vit: pretrained ViT and (later) small ViT from scratch."""
import timm
import torch.nn as nn


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


# TODO: Small ViT from scratch (minimal patch size, depth, heads)
