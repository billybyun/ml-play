# Tiny-CLIP Project Plan

## Goal

Demonstrate contrastive learning (InfoNCE) and retrieval in two ways:

1. **Track A — Modify pretrained CLIP:** Start from full CLIP, then reset/retrain parts (projections, optionally unfreeze encoders) to see controlled destruction and recovery of alignment.
2. **Track B — Build from components:** Compose a pretrained ViT + pretrained text encoder, add our own projection heads, and train alignment from (near) random.

Deliverables across both tracks:

- Reproducible training (config-driven)
- Retrieval metrics R@1/5/10 (both directions)
- Qualitative examples
- Ablation tables and clean README

---

## Repo Structure

Place under **foundations/tiny-clip/**:

```
foundations/tiny-clip/
  src/
    data.py
    models.py
    loss.py
    train.py
    eval_retrieval.py
    utils.py
  configs/
    flickr8k.yaml
    flickr30k.yaml
  notebooks/
    sanity_checks.ipynb
  demos/
    gradio_retrieval.py
  README.md
```

---

## Datasets

**Primary:** Flickr30k (~31k images, 5 captions each)

- Load via: `datasets.load_dataset("nlphuji/flickr30k")`
- Hugging Face caches images automatically; no manual download.

**Optional:** Flickr8k (~8k images) for faster iteration or debugging.

- Keep dataset choice configurable in `configs/*.yaml`.

---

## Evaluation (shared)

For both tracks:

- **Text → Image:** rank images by similarity to each caption → Recall@1, @5, @10
- **Image → Text:** rank captions by similarity to each image → Recall@1, @5, @10
- **Optional:** median rank, qualitative top-5 examples

---

## Track A: Modify Pretrained CLIP

**Model:** `openai/clip-vit-base-patch32` (Hugging Face). Includes vision encoder (ViT-B/32), text encoder, projection heads, and learnable temperature. We modify this structure.

### Stage A0 — Zero-shot baseline (no training)

- **Goal:** Establish strong baseline with pretrained CLIP.
- **Steps:** Load CLIP model + processor; run retrieval on Flickr30k; compute R@1/5/10 both directions.
- **Expected:** Strong zero-shot performance (not random).
- **Deliverables:** e.g. `eval_clip_zeroshot.py`, baseline metrics in README, qualitative examples.

### Stage A1 — Reset and retrain projection heads only

- **Goal:** Isolate the role of projections; break alignment then recover.
- **Modification:** Freeze vision and text encoders; reinitialize image and text projection layers (optionally keep temperature learnable).
- **Expected before training:** Retrieval drops to near random.
- **Training:** Train only projection layers (+ temperature). Loss: symmetric CLIP InfoNCE — `(CE(sim, targets) + CE(sim.T, targets)) / 2`.
- **Deliverables:** e.g. `train_projector.py`, training curve, before/after metrics, ablation (pretrained vs reset vs retrained).

### Stage A2 — Partial fine-tuning (unfreeze encoder)

- **Goal:** Measure gains from adapting part of the encoder.
- **Strategy (choose one first):**
  - **Option A:** Unfreeze last 1–2 ViT blocks; lower LR for ViT than for projections.
  - **Option B:** Unfreeze full text encoder; keep ViT frozen.
- **Training:** Separate param groups (projections vs encoder); lower LR for encoder; weight decay / dropout as needed.
- **Compare against:** Stage A0 baseline, Stage A1 projector-only.
- **Deliverables:** Ablation table, training stability notes, short analysis in README.

**Note on small data (Flickr):** On a dataset as small as Flickr30k, unfreezing often gives **marginal or no gain**, and can overfit. Most gains typically come from Stage A1 (projections). Stage A2 is still worth doing as an experiment: report whether unfreezing helped, hurt, or had no effect. To maximize chance of benefit: unfreeze only the last 1 block, use strong regularization and low encoder LR, few epochs, early stopping; LoRA (stretch) is more data-efficient.

---

## Track B: Build from Components

**Models:** Pretrained ViT (e.g. `vit_base_patch16_224` from timm) + pretrained text encoder (e.g. DistilBERT from Hugging Face). We add our own projection heads and learnable temperature.

### Stage B0 — Sanity check (no training)

- **Goal:** Validate data pipeline and metrics; establish baseline.
- **Steps:** Load pretrained ViT + text encoder; use **random** projection heads; run retrieval eval.
- **Expected:** Retrieval near random.
- **Deliverables:** Baseline metrics in README.

### Stage B1 — Train projections only

- **Goal:** Fast “it works” milestone.
- **Trainable:** image_proj, text_proj, temperature. **Frozen:** ViT, text encoder.
- **Loss:** Symmetric InfoNCE; L2-normalize embeddings; sim = (z_img @ z_txt^T) / tau.
- **Expected:** Retrieval improves noticeably over B0.
- **Deliverables:** Training curve, before/after retrieval examples.

### Stage B2 — Unfreeze part of encoder

- **Goal:** Show principled unfreezing.
- **Strategy (choose one first):**
  - **Option A:** Unfreeze last 1–2 ViT blocks (text frozen).
  - **Option B:** Unfreeze full text encoder (ViT frozen).
- **Training:** Lower LR for encoders, higher for projections; weight decay / dropout as needed.
- **Deliverables:** Ablation table: proj-only vs unfreeze ViT vs unfreeze text.

**Note on small data (Flickr):** Same as Track A: on Flickr-sized data, unfreezing (Stage B2) often gives little or no gain; main gains usually from Stage B1. Keep B2 as an exploratory step and report results honestly.

---

## Engineering

- Deterministic seeds
- Config-driven runs (YAML)
- Separate parameter groups for projections vs encoder blocks
- Save checkpoints and log metrics to JSON
- Optional: mixed precision (if GPU)
- Minimal CI: lint + import test
- README: setup, how to train, how to eval, sample results, lessons learned

---

## Stretch Goals

- Negative queue (MoCo-style) for small-batch training
- Compare pooling: CLS vs mean
- Compare projection: linear vs 2-layer MLP
- LoRA instead of full block unfreeze (more data-efficient)
- Gradio retrieval demo

---

## Narrative (optional)

This project demonstrates: (1) proper evaluation of pretrained CLIP; (2) controlled destruction and recovery of alignment (Track A); (3) building and aligning a dual-encoder system from components (Track B); (4) modular freeze/unfreeze strategies and ablations; (5) practical contrastive learning and retrieval evaluation.
