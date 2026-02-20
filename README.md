# ml-play

A playground for ML experiments and reference implementations.

## Structure

- **[foundations/](foundations/)** — Model architectures and small pipelines (ViT, Conformer, Perceiver, etc.).
  - **[tiny-clip](foundations/tiny-clip/)** — CLIP-style contrastive learning: modify pretrained CLIP (Track A) or build from components (Track B). Data sanity check, zero-shot eval, and retrieval on Flickr30k.
  - **[tiny_transformer](foundations/tiny_transformer/)** — LLM representation (small transformer reference).
  - **VLA** ([foundations/vla/](foundations/vla/)) — Vision–language–action (scaffolding to be added).
- **[timeseries/](timeseries/)** — Time-series tools and notebooks (e.g. HFA envelope methods, ECoG).
- **[docs/](docs/)** — Project plans (e.g. [plan.md](docs/plan.md) for tiny-clip).
- **[notes/](notes/)** — Annotated references and scratch.

## Quick links

- [Tiny-CLIP setup, data sanity check, and sample results](foundations/tiny-clip/)
- [Project plan (Tiny-CLIP tracks and stages)](docs/plan.md)
