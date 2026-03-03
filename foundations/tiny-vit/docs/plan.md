# tiny-vit Plan

## Goal

Explore Vision Transformers (ViT) in two phases:

1. **Pretrained ViT:** Use timm ViT (e.g. `vit_base_patch16_224`) for image classification. Establish baseline on CIFAR-10.
2. **Small ViT from scratch:** Build a minimal ViT (fewer layers, smaller patch size) and train from scratch.

## Phase 1: Pretrained ViT

- [ ] Load CIFAR-10, resize to 224x224
- [ ] Load pretrained ViT-B/16 from timm, replace head for 10 classes
- [ ] Eval zero-shot (or minimal fine-tuning)
- [ ] Record accuracy

## Phase 2: Small ViT from scratch

- [ ] Define minimal ViT (e.g. 4 layers, 4 heads, patch 16)
- [ ] Train on CIFAR-10 from random init
- [ ] Compare to pretrained baseline
