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
  requirements.txt
  notebooks/
    sanity_checks.ipynb
  demos/
    gradio_retrieval.py
  README.md
```

**Requirements:** Add `requirements.txt` in this folder with: `torch`, `transformers`, `datasets`, `Pillow`, `pyyaml`, and optionally `timm` (for Track B). Ensures reproducible env for tiny-clip.

---

## Datasets

**Primary:** Flickr30k (~31k images, 5 captions each)

- Load via: `datasets.load_dataset("nlphuji/flickr30k")`
- Hugging Face caches images automatically; no manual download.

**Optional:** Flickr8k (~8k images) for faster iteration or debugging.

- Keep dataset choice configurable in `configs/*.yaml`.

**Data pipeline:** Use a thin **Dataset class** (or wrapper) that:
- For HF: wraps `load_dataset("nlphuji/flickr30k")` (or dataset name from config); applies CLIP processor (Track A) or image transforms + tokenizer (Track B); returns `(pixel_values, input_ids, attention_mask)` or equivalent for the chosen model.
- Optionally supports a local dataset path from config (e.g. manual download) for the same interface.
- Single place for transforms/tokenization; easy to plug into `DataLoader` and to add augmentation later.

---

## Evaluation (shared)

**R@K definition:** Recall@K = fraction of queries whose ground-truth match appears in the top K ranked candidates (t2i: correct image; i2t: any of the image’s ground-truth captions).

For both tracks:

- **Text → Image:** rank images by similarity to each caption → Recall@1, @5, @10
- **Image → Text:** rank captions by similarity to each image → Recall@1, @5, @10
- **Optional:** median rank, qualitative top-5 examples

## Retrieval task definition (important)

This project uses **image-text retrieval**, not generation.

- We never reconstruct pixels or generate words.
- We embed images and captions into a shared space and rank by similarity.

Two tasks:

1) **Text → Image retrieval (t2i)**  
Given a caption, rank **all candidate images in the eval split** by similarity and check whether the correct image is in top-K.

2) **Image → Text retrieval (i2t)**  
Given an image, rank **all candidate captions in the eval split** by similarity and check whether any of the image’s ground-truth captions (typically 5) is in top-K.

### Training vs evaluation pools

- **Training (contrastive loss):** typically uses **in-batch negatives** (B×B logits matrix).
- **Evaluation (retrieval metrics):** uses the **full eval pool**, not the batch.

For Flickr30k test split:
- pool for t2i = #images in split
- pool for i2t = #captions in split (= 5 × #images)

**Standard benchmark:** Retrieval evaluation uses the **Flickr30k 1k test set** (same as in papers: 1,000 images, 5 captions each). Config: `eval_dataset_name: "nlphuji/flickr_1k_test_image_text_retrieval"`, `eval_split: "test"`. Use `get_dataloader(..., for_eval=True)` so the eval script loads this 1k set.

**Eval protocol note:** Report which split and pool you use; always log split name and counts (e.g. 1k test, full-pool ranking).


Do NOT report metrics computed only within a minibatch.




---

## Track A: Modify Pretrained CLIP

**Model:** `openai/clip-vit-base-patch32` (Hugging Face). Includes vision encoder (ViT-B/32), text encoder, projection heads, and learnable temperature. We modify this structure.

### Stage A0 — Zero-shot baseline (no training)

- **Goal:** Establish strong baseline with pretrained CLIP.
- **Steps:** Load CLIP model + processor; run retrieval on Flickr30k; compute R@1/5/10 both directions.
- **Expected:** Strong zero-shot performance (not random).
- **Deliverables:** e.g. `eval_clip_zeroshot.py`, baseline metrics in README, qualitative examples.

#### Implementation steps for Stage A0

Do in order so each step can be checked and committed.

1. **Placement**
   Create `foundations/tiny-clip/` and subdirs only: `src/`, `configs/`, `notebooks/`, `demos/`. No code yet. Commit.

2. **Scaffold**
   - Add minimal files: `src/data.py`, `models.py`, `loss.py`, `train.py`, `eval_retrieval.py`, `utils.py` (stubs or minimal docstrings; `train.py` and `loss.py` not used in A0).
   - Add `configs/flickr30k.yaml` (and optionally `flickr8k.yaml`) with keys: `dataset_name` (e.g. `nlphuji/flickr30k`), `split` (e.g. `test` or `val`), `batch_size`, and any paths if supporting local data.
   - Add `foundations/tiny-clip/requirements.txt`: `torch`, `transformers`, `datasets`, `Pillow`, `pyyaml`.
   Commit.

3. **Data**
   - In `src/data.py`, implement a Dataset that loads Flickr30k via `datasets.load_dataset(...)` (dataset name from config), applies `CLIPProcessor` (image + text), and returns batches compatible with CLIP (e.g. `pixel_values`, `input_ids`, `attention_mask`).
   - Add a small helper to build the DataLoader from config.
   Commit.

4. **Eval script**
   - Add `eval_retrieval.py` (or `eval_clip_zeroshot.py`) in `src/` or as a script under `foundations/tiny-clip/`: load config → load `CLIPModel` and `CLIPProcessor` from `openai/clip-vit-base-patch32` → load eval data via `get_dataloader(config, processor, for_eval=True)` (standard 1k test set) → compute image and text embeddings → similarity matrix → R@1, R@5, R@10 for text→image and image→text.
   - Optionally save a few qualitative examples (top-k retrievals).
   Commit.

5. **Run Stage A0**
   - Run the eval script (no training). Record baseline metrics (e.g. in a small table or JSON).
   - Confirm numbers are strong (not random), so the pipeline is correct.

6. **Demo: zero-shot on custom images**
   - Run pretrained CLIP on 1–3 **images the model has never seen** (e.g. profile picture, other sharable personal photos). No training or fine-tuning on these images.
   - For each image: show **top-k retrieved captions** (from the eval set) or similarity to a few hand-written captions. Save figure(s).
   - Add a **Results** or **Demo** section in `foundations/tiny-clip/README.md` that includes these examples (image + retrieved captions). Use only images you are comfortable sharing in the repo.

   6b. **Optional personal mini-retrieval demo (with self-written ground truth)**
   - Add 1–10 personal photos (only if comfortable sharing; otherwise keep local only).
   - For each photo, write 1–3 captions (self-authored ground truth).
   - Build a small eval pool: personal photos + N Flickr distractor images, and personal captions + M Flickr distractor captions.
   - Run both:
     - t2i: each personal caption should retrieve its paired personal photo in top-K
     - i2t: each personal photo should retrieve its caption(s) in top-K
   - Label this section clearly as **qualitative / mini-demo**, not an official benchmark.


7. **README**
   - In `foundations/tiny-clip/README.md`: how to install deps (`pip install -r requirements.txt`), how to run the zero-shot eval (e.g. `python eval_retrieval.py --config configs/flickr30k.yaml`), what the config keys mean, and where to paste the baseline metrics.
   Commit.

After this, Stage A0 is done. A1 will add resetting projections and training.

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

- **Goal:** Fast "it works" milestone.
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
