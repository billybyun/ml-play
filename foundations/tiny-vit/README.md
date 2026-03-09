# tiny-vit

ViT on CIFAR-10: pretrained + linear probe, and small ViT from scratch. See [results](results/).

## Results

| Model | Top-1 | Top-5 |
|-------|-------|-------|
| Random head | ~10% | ~40% |
| Linear probe | 94.4% | 99.9% |
| Small ViT (from scratch) | — | — |

- Linear probe: [Samples](results/linear_probe/samples.png) · [Metrics](results/linear_probe/metrics.json)
- Small ViT: [Samples](results/small_vit/samples.png) · [Metrics](results/small_vit/metrics.json)

## Commands

```bash
pip install -r requirements.txt

# Linear probe (pretrained + trained head)
python -m src.train --model-type linear_probe
python -m src.eval --checkpoint checkpoints/linear_probe/final.pt --output results/linear_probe/metrics.json
python -m src.visualize --checkpoint checkpoints/linear_probe/final.pt

# Small ViT from scratch
python -m src.train --model-type small_vit
python -m src.eval --checkpoint checkpoints/small_vit/final.pt --output results/small_vit/metrics.json
python -m src.visualize --checkpoint checkpoints/small_vit/final.pt
```

## Status

- [x] Linear probe (pretrained ViT + trained head)
- [x] Small ViT from scratch
