# tiny-vit Plan

## Goal

Explore Vision Transformers (ViT) in two phases:

1. **Pretrained ViT:** Use timm ViT (e.g. `vit_base_patch16_224`) for image classification. Establish baseline on CIFAR-10.
2. **Small ViT from scratch:** Build a minimal ViT (fewer layers, smaller patch size) and train from scratch.

## Phase 1: Pretrained ViT

- [x] Load CIFAR-10, resize to 224x224 (ImageNet normalization for pretrained)
- [x] Load pretrained ViT-B/16 from timm, replace head for 10 classes
- [x] Eval zero-shot: `python -m src.eval --config configs/cifar10.yaml`
- [ ] Record accuracy (run eval and add to README)

## Phase 2: Small ViT from scratch

- [ ] Define minimal ViT (e.g. 4 layers, 4 heads, patch 16)
- [ ] Train on CIFAR-10 from random init
- [ ] Compare to pretrained baseline
