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
| Random head eval (no training) | `python -m src.eval --config configs/cifar10.yaml --output metrics_random_head.json` |
| **Linear probe train** | `python -m src.train --config configs/cifar10.yaml` |
| **Linear probe eval** | `python -m src.eval --config configs/cifar10.yaml --checkpoint checkpoints/linear_probe/final.pt --output metrics_linear_probe.json` |

## Results (CIFAR-10 test)

| Model | Top-1 | Top-5 | Notes |
|-------|-------|-------|-------|
| Random head | ~10% | ~40% | No training; baseline |
| Linear probe | — | — | Run train + eval to fill |

## Status

- [x] Scaffold
- [x] Pretrained ViT eval (random head)
- [x] Linear probe training
- [ ] Small ViT from scratch
