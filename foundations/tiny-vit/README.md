# tiny-vit

ViT on CIFAR-10: pretrained + linear probe, and tiny ViT from scratch. See [results](results/).

## Results

| Model | Top-1 | Top-5 |
|-------|-------|-------|
| Random head | ~10% | ~40% |
| Linear probe | 94.4% | 99.9% |
| Tiny ViT (from scratch) | 68.3% | 97.1% |

- Linear probe: [Samples](results/linear_probe/samples.png) · [Metrics](results/linear_probe/metrics.json)
- Tiny ViT: [Samples](results/tiny_vit/samples.png) · [Metrics](results/tiny_vit/metrics.json)

## Commands

```bash
pip install -r requirements.txt

# Linear probe (pretrained + trained head)
python -m src.train --model-type linear_probe
python -m src.eval --checkpoint checkpoints/linear_probe/final.pt --output results/linear_probe/metrics.json
python -m src.visualize --checkpoint checkpoints/linear_probe/final.pt

# Tiny ViT from scratch
python -m src.train --model-type tiny_vit
python -m src.eval --checkpoint checkpoints/tiny_vit/final.pt --output results/tiny_vit/metrics.json
python -m src.visualize --checkpoint checkpoints/tiny_vit/final.pt
```

## Status

- [x] Linear probe (pretrained ViT + trained head)
- [x] Tiny ViT from scratch
