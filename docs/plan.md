# Tiny-CLIP Project Plan (ViT + Contrastive Learning)

## Goal
Build a small, working CLIP-style contrastive learning system:
- image encoder (ViT)
- text encoder (Transformer)
- projection heads into shared embedding space
- InfoNCE / CLIP symmetric loss
- retrieval evaluation + demo

Deliverables:
- reproducible training
- retrieval metrics (R@1/5/10)
- qualitative examples
- clean README + small demo script

---

## Repo Structure (suggested)
tiny-clip/
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
  plan.md
  README.md

---

## Datasets
### Option 1: Flickr8k (fastest iteration)
- ~8k images, 5 captions each
- Pros: quick to download/train, good for debugging
- Cons: easy to overfit, retrieval improvements may saturate

### Option 2: Flickr30k (recommended sweet spot)
- ~31k images, 5 captions each
- Pros: still manageable, more realistic retrieval learning
- Cons: heavier than Flickr8k

(Keep dataset choice configurable via configs/*.yaml)

---

## Model
### Image Encoder
- Start: pretrained ViT (e.g., ViT-B/16 from timm or HF)
- Output: pooled embedding (CLS token or mean pooling)

### Text Encoder
- Start: pretrained small text encoder (or lightweight Transformer encoder)
- Output: pooled embedding (CLS token or mean over tokens)

### Projections
- image_proj: Linear or 2-layer MLP -> shared dim (e.g., 256 or 512)
- text_proj: Linear or 2-layer MLP -> shared dim
- L2 normalize embeddings before similarity

### Similarity + Temperature
- sim = (z_img @ z_txt^T) / tau
- tau is learnable (initialize around 0.07 equivalent scale)

### Loss (CLIP symmetric InfoNCE)
- loss = (CE(sim, targets) + CE(sim^T, targets)) / 2
- targets = [0, 1, ..., B-1]

---

## Training Stages

### Stage 0: Zero-training sanity check
Purpose: validate data pipeline + metrics + shapes
Steps:
1) Load pretrained ViT + pretrained text encoder
2) Initialize random projection heads
3) Run eval_retrieval.py
Expected:
- retrieval ~ near random
- metrics establish a baseline
Artifacts:
- saved baseline metrics in README

---

### Stage 1: Train projections only (Option A)
Purpose: fastest “it works” milestone
Trainable:
- image_proj, text_proj, temperature
Frozen:
- ViT encoder, text encoder
Steps:
1) Train for a few epochs (start small)
2) Evaluate retrieval periodically
Expected:
- retrieval improves noticeably over Stage 0
Artifacts:
- training curve
- before/after retrieval examples

---

### Stage 2: Increase capacity (Option B-ish)
Purpose: show principled unfreezing
Two paths (choose one first):
A) Unfreeze last 1–2 ViT blocks (+ keep text frozen)
B) Unfreeze full text encoder (+ keep ViT frozen)
Trainable:
- projections + chosen unfrozen parts + temperature
Steps:
1) Lower LR for encoders, higher LR for projections
2) Add regularization (weight decay, dropout if needed)
Expected:
- incremental gains over Stage 1
Artifacts:
- ablation table: (proj-only) vs (unfreeze last ViT blocks) vs (unfreeze text)

---

## Evaluation
### Retrieval Metrics
Compute both directions:
- Text -> Image: Recall@1/5/10
- Image -> Text: Recall@1/5/10

### Qualitative
- show top-5 retrieved images for a caption
- show top-5 retrieved captions for an image

### Robustness (optional)
- evaluate retrieval under simple augmentations (crop/blur/brightness)

---

## Engineering Checklist
- Deterministic seeds
- Config-driven runs (yaml)
- Save checkpoints + metrics json
- Minimal CI: lint + import test
- README with:
  - setup
  - how to train
  - how to eval
  - sample results
  - lessons learned

---

## Stretch Goals (if time)
- Add a negative queue (MoCo-style) for small batch training
- Compare pooling (CLS vs mean)
- Compare projection (linear vs 2-layer MLP)
- Add Gradio demo for retrieval
