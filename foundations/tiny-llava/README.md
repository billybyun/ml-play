# tiny-llava

LLaVA-style vision–language model: frozen CLIP + projection + frozen LLM. See [docs/plan.md](docs/plan.md) for the full plan.

## Tiny vs real LLaVA

We use a **tiny** setup for fast iteration on a laptop. Scale up once the pipeline works.

| Component | Ours (tiny) | Real LLaVA (paper) |
|-----------|-------------|--------------------|
| **Vision** | CLIP ViT-B/32 (512-d) | CLIP ViT-L/14 (768-d) |
| **LLM** | GPT-2 (~124M) | Vicuna-7B / LLaMA-7B |
| **Projection** | 2-layer MLP, hidden 1024 | 2-layer MLP, hidden 4096 |

## Status

- [ ] Phase 1: Scaffold & inference (no training)
- [ ] Phase 2: Data & training (projection only)
- [ ] Phase 3: Instruction tuning (optional)

## Setup

```bash
pip install -r requirements.txt
```

## Commands (planned)

```bash
# Inference (after Phase 1)
python -m src.inference --image path/to/image.jpg --prompt "Describe this image."

# Training (after Phase 2)
python -m src.train --config configs/llava.yaml
```

## Architecture

```
Image → [CLIP ViT] → [Projection MLP] → vision tokens
                                              ↓
Text  → [Tokenizer] → text tokens  ──────────→ [LLM] → generated text
```

- **Vision:** Frozen CLIP ViT-B/32
- **Projection:** 2-layer MLP (trained)
- **LLM:** Frozen GPT-2

## Next steps

1. **Phase 1:** Implement `LLaVAModel` in `src/models.py`, then `src/inference.py`. Run inference with random projection to validate the pipeline.
2. **Phase 2:** Implement `src/data.py` and `src/train.py`. Train the projection on image–caption data.
