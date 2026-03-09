# tiny-flamingo

Flamingo-style VLM: frozen ViT + Perceiver Resampler + frozen GPT-2. Designed for 12GB VRAM. See [docs/plan.md](docs/plan.md).

## Tiny vs real Flamingo

| Component | Ours | Real Flamingo |
|-----------|------|---------------|
| **Vision** | ViT-B/16 | NFNet / ViT |
| **Resampler** | 32 latents, 2 layers | 64 latents, 6 layers |
| **LM** | GPT-2 (~124M) | Chinchilla 70B |

## Status

- [x] Phase 1: Scaffold & inference (model + inference implemented)
- [ ] Phase 2: Data & training
- [ ] Phase 3: Cross-attention in LM (optional)

## Setup

```bash
pip install -r requirements.txt
```

## Commands (planned)

```bash
python -m src.inference --image path/to/image.jpg --prompt "Describe this image."
python -m src.train --config configs/flamingo.yaml
```

## Architecture

```
Image → [ViT-B/16] → [Perceiver Resampler] → visual tokens
                                              ↓
Text  → [Tokenizer] → text tokens ──────────→ [GPT-2] → generated text
```
