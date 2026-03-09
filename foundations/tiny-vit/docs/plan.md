# tiny-vit Plan

## Goal

Explore Vision Transformers (ViT) in two phases:

1. **Pretrained ViT:** Use timm ViT (e.g. `vit_base_patch16_224`) for image classification. Establish baseline on CIFAR-10.
2. **Tiny ViT from scratch:** Build a minimal ViT (fewer layers, smaller patch size) and train from scratch.

## Phase 1: Pretrained ViT

- [x] Load CIFAR-10, resize to 224x224 (ImageNet normalization for pretrained)
- [x] Load pretrained ViT-B/16 from timm, replace head for 10 classes
- [x] Eval random head: `python -m src.eval --config configs/cifar10.yaml`
- [x] Linear probe: freeze backbone, train head only; `python -m src.train`
- [x] Record accuracy (run train + eval, add to README)

## Phase 2: Tiny ViT from scratch

- [x] Define minimal ViT (4 layers, 8 heads, patch 4, 32x32)
- [x] Train on CIFAR-10 from random init: `python -m src.train --model-type tiny_vit`
- [x] Run train + eval, add metrics to README
