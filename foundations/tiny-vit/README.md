# tiny-vit

Vision Transformer (ViT) explorations: pretrained ViT for image classification, then small ViT from scratch.

## Goal

1. **Pretrained ViT:** Use timm ViT (e.g. `vit_base_patch16_224`) for image classification on CIFAR-10 or similar.
2. **Small ViT from scratch:** Build a minimal ViT and train from scratch.

## Structure

```
foundations/tiny-vit/
  src/
    data.py
    models.py
    train.py
    eval.py
    utils.py
  configs/
    cifar10.yaml
  requirements.txt
  README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Commands

| Stage | Command |
|-------|---------|
| Print ViT shapes | `python -m src.eval --config configs/cifar10.yaml --print-shapes` |
| B0 eval (random head, no training) | `python -m src.eval --config configs/cifar10.yaml` |
| Save B0 metrics | `python -m src.eval --config configs/cifar10.yaml --output metrics_b0_random_head.json` |

## Status

- [x] Scaffold
- [x] Pretrained ViT eval on CIFAR-10
- [ ] Small ViT from scratch
