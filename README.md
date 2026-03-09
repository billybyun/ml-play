# ml-play

A playground for ML experiments and reference implementations.

## Structure

- **[foundations/](foundations/)** — Model architectures and small pipelines (ViT, Conformer, Perceiver, etc.).
  - **[tiny-clip](foundations/tiny-clip/)** — CLIP-style contrastive learning: modify pretrained CLIP (Track A) or build from components (Track B). Data sanity check, zero-shot eval, and retrieval on Flickr30k.
  - **[tiny-vit](foundations/tiny-vit/)** — ViT on CIFAR-10: linear probe (94.4% top-1) and tiny ViT from scratch (68.3% top-1).
  - **[tiny_transformer](foundations/tiny_transformer/)** — LLM representation (small transformer reference).
  - **BLIP2** (to be added) — Q-Former bridging frozen image encoder + frozen LLM.
  - **LLaVA** (to be added) — Simple projection + LLM for vision–language.
  - **SwinT, ViViT, Perceiver, Conformer, S4/Mamba** (to be added) — Vision, video, and SSM architectures.
  - **VLA** (to be added) — Vision–language–action for robotics.
- **[timeseries/](timeseries/)** — Time-series tools and notebooks (e.g. HFA envelope methods, ECoG).
- **[docs/](docs/)** — Project plans in [docs/plans/](docs/plans/) (e.g. [tiny-clip](docs/plans/tiny-clip.md)).
- **[notes/](notes/)** — Annotated references and scratch.

## Example: Tiny-CLIP custom images demo

Run pretrained CLIP on your own images — retrieve top-5 captions from the Flickr30k benchmark:

![Tiny-CLIP custom images demo](foundations/tiny-clip/demos/output/demo_results.png)

Add images to `foundations/tiny-clip/demos/custom_images/`, run `python demos/demo_custom_images.py`, and the figure is saved to `demos/output/`. See [tiny-clip](foundations/tiny-clip/) for setup and more.

## Example: Dual encoder analysis (B0 vs B1 vs CLIP)

We compose a pretrained ViT + pretrained DistilBERT, add projection heads, and train alignment from scratch. Random projections (B0) give near-random retrieval; training only the projections (B1) on ~29k images yields a ~340× improvement. B1 reaches ~43% of CLIP’s i2t R@1 with projection-only training.

| Model | i2t R@1 | t2i R@1 | Notes |
|-------|---------|---------|-------|
| **CLIP** (pretrained, 400M pairs) | 79.3% | 58.8% | Upper bound |
| **B1** (projections trained, encoders frozen) | 34.3% | 23.9% | ~340× over B0 |
| **B0** (random projections) | 0.1% | 0.08% | Near random |

Flickr30k 1k test set. See [tiny-clip](foundations/tiny-clip/) for details.

## Example: ViT on CIFAR-10

Linear probe (pretrained ViT-B/16 + trained head) reaches 94.4% top-1; tiny ViT trained from scratch on 32×32 reaches 68.3%. Pretrained features dominate; from-scratch learns useful representations but with a smaller model and no augmentation.

| Model | Top-1 | Top-5 | Notes |
|-------|-------|-------|-------|
| **Linear probe** | 94.4% | 99.9% | Pretrained ViT-B/16, frozen backbone |
| **Tiny ViT** | 68.3% | 97.1% | 4 layers, 8 heads, trained from scratch |
| **Random head** | ~10% | ~40% | Baseline |

See [tiny-vit](foundations/tiny-vit/) for setup and commands.

## Quick links

- [Tiny-CLIP](foundations/tiny-clip/) · [Tiny-ViT](foundations/tiny-vit/)
- [Repo roadmap](docs/plan.md)