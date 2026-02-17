# Tiny-CLIP Project Plan

## Objective

Build a clean, research-style CLIP adaptation pipeline that demonstrates:

- Understanding of contrastive learning (InfoNCE)
- Retrieval evaluation (R@K both directions)
- Modular training (freeze / unfreeze strategies)
- Controlled ablations

We will NOT train CLIP from scratch.
We will start from pretrained CLIP and progressively modify it.

---

# Architecture Choice

Model: openai/clip-vit-base-patch32  
Framework: Hugging Face Transformers  

This model already contains:
- Vision encoder (ViT-B/32)
- Text encoder (Transformer)
- Projection heads
- Learnable temperature parameter

We will progressively modify this structure.

---

# Dataset

Primary dataset: Flickr30k (via Hugging Face datasets)

Reason:
- Medium size (~31k images, 5 captions each)
- Standard retrieval benchmark
- Feasible on a laptop GPU

Dataset access:
- Use `datasets.load_dataset("nlphuji/flickr30k")`
- Images cached automatically by HF

---

# Experimental Stages

---

## Stage 0 — Zero-Shot Baseline (No Training)

### Goal
Establish a clean baseline using pretrained CLIP.

### Steps
1. Load CLIP model + processor.
2. Run retrieval evaluation on Flickr30k.
3. Compute:
   - Text → Image: Recall@1/5/10
   - Image → Text: Recall@1/5/10

### Expected Result
Strong zero-shot performance (not random).

### Deliverables
- `eval_clip_zeroshot.py`
- Baseline metrics table in README
- Qualitative retrieval examples

---

## Stage 1 — Reset & Retrain Projection Heads

### Goal
Isolate and relearn alignment using frozen encoders.

### Modification
- Keep vision encoder frozen.
- Keep text encoder frozen.
- Reinitialize:
  - image projection layer
  - text projection layer
- Keep temperature learnable.

This intentionally breaks alignment.

### Expected Before Training
Retrieval drops significantly (near random).

### Training
Train only:
- projection layers
- temperature

Loss:
- Symmetric CLIP InfoNCE:
  loss = (CE(sim, targets) + CE(sim.T, targets)) / 2

### What This Demonstrates
- Role of projection in alignment
- Contrastive loss behavior
- Batch negatives importance

### Deliverables
- `train_projector.py`
- Training curve
- Before/after metrics table
- Ablation: pretrained vs reset vs retrained

---

## Stage 2 — Partial Fine-Tuning

### Goal
Measure gains from partial encoder adaptation.

### Strategy (choose one first)

Option A:
- Unfreeze last 1–2 ViT blocks
- Lower LR for ViT than projections

Option B:
- Unfreeze full text encoder
- Keep ViT frozen

### Compare Against
- Stage 0 baseline
- Stage 1 projector-only

### Deliverables
- Ablation table
- Training stability observations
- Short analysis section in README

---

# Evaluation Protocol

For retrieval:

For each image:
- Rank all captions by similarity

For each caption:
- Rank all images by similarity

Compute:
- Recall@1
- Recall@5
- Recall@10

Report both directions.

Optional:
- Median rank
- Qualitative top-5 visualizations

---

# Engineering Requirements

- Deterministic seeds
- Config-driven training (YAML)
- Separate parameter groups for:
  - projection
  - encoder blocks
- Mixed precision training (if GPU)
- Save checkpoints
- Log metrics to JSON

---

# Stretch Goals (Optional)

- Add negative queue for small batch training
- Compare CLS pooling vs mean pooling
- Add LoRA instead of full block unfreeze
- Gradio retrieval demo

---

# Final Interview Narrative

This project demonstrates:

1. Proper evaluation of pretrained CLIP.
2. Controlled destruction and recovery of alignment.
3. Modular fine-tuning strategies.
4. Practical understanding of contrastive learning dynamics.
5. Clean experimental design and ablation reporting.
