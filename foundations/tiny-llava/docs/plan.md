# tiny-llava Project Plan

## Goal

Implement a minimal LLaVA-style vision–language model: **frozen vision encoder → projection layer → LLM**. The projection maps image features into the LLM token space; vision tokens are prefixed to text for generation.

**Architecture (LLaVA-style):**
```
Image → [CLIP ViT] → [Projection MLP] → vision tokens
                                              ↓
Text  → [Tokenizer] → text tokens  ──────────→ [LLM] → generated text
```

- **Vision encoder:** Frozen CLIP ViT (e.g. `openai/clip-vit-base-patch32`)
- **Projection:** 2-layer MLP (vision_dim → hidden → llm_dim); trained
- **LLM:** Frozen (e.g. GPT-2, DistilGPT-2 for tiny; Vicuna/LLaMA for full scale)

---

## Repo Structure

```
foundations/tiny-llava/
  src/
    data.py         # Dataset, dataloader (COCO, VQA, or instruction data)
    models.py       # LLaVAModel: vision encoder + projection + LLM
    train.py        # Training loop (projection only, or projection + LLM)
    inference.py    # Generate captions / answers from image + prompt
    utils.py        # Config loading, etc.
  configs/
    llava.yaml      # Model names, projection dims, data, training
  demos/
    demo_inference.py   # Run on custom images
  docs/
    plan.md
  requirements.txt
  README.md
```

---

## Phase 1: Scaffold & Inference (No Training)

**Goal:** Load pretrained components, wire them together, run inference. Establish the pipeline.

### 1.1 Placement & scaffold

- [x] Create `foundations/tiny-llava/` with `src/`, `configs/`, `demos/`, `docs/`
- [x] Add `requirements.txt`: `torch`, `transformers`, `Pillow`, `pyyaml`, `datasets`
- [ ] Add minimal `src/models.py`: `LLaVAModel` with frozen CLIP + projection (random init) + frozen GPT-2
- [x] Add `configs/llava.yaml`: vision model name, LLM name, projection dims
- [x] Add `src/utils.py`: `load_config`

### 1.2 Inference script

- [ ] Add `src/inference.py`: load model, take image + text prompt, return generated text
- [ ] Support `--checkpoint` (optional; if none, use random projection for sanity check)
- [ ] Run: `python -m src.inference --image path/to/img.jpg --prompt "Describe this image."`
- [ ] Expected with random projection: gibberish or generic output (validates pipeline)

### 1.3 Demo

- [ ] Add `demos/demo_inference.py`: run on 1–3 images, save outputs to `demos/output/`
- [ ] README: setup, how to run inference

---

## Phase 2: Data & Training (Projection Only)

**Goal:** Train the projection layer on image–caption pairs. Vision encoder and LLM frozen.

### 2.1 Dataset

- [ ] Add `src/data.py`: dataset that returns `(image, caption)` or `(image, prompt, answer)`
- [ ] **Primary:** COCO captions (via `datasets` or Hugging Face)
- [ ] **Format:** Each sample = image + caption; prompt = `"Describe this image."` or similar; target = caption
- [ ] Config: `dataset_name`, `split`, `batch_size`, `max_length`
- [ ] Optional: VQA v2 for Q&A format

### 2.2 Training loop

- [ ] Add `src/train.py`: load data, model; train projection only (vision + LLM frozen)
- [ ] Loss: next-token prediction (cross-entropy on caption tokens)
- [ ] Input to LLM: `[vision_tokens] + [text_tokens]`; target = caption tokens only
- [ ] Config: `epochs`, `lr`, `output_dir`
- [ ] Save checkpoint: `{output_dir}/final.pt` (projection weights + config)

### 2.3 Eval & metrics

- [ ] Run training on COCO train split
- [ ] Eval: sample captions on val split; qualitative check (BLEU/ROUGE optional)
- [ ] Compare: random projection vs trained projection (qualitative)

---

## Phase 3: Instruction Tuning (Optional)

**Goal:** Fine-tune on instruction-style data (e.g. LLaVA-Instruct format) for chat/VQA.

- [ ] Add support for `(image, conversation)` format
- [ ] Train projection + optionally unfreeze last LLM layers (LoRA stretch)
- [ ] Eval: VQA accuracy or qualitative chat

---

## Model Choices (Tiny vs Full)

| Component | Tiny (fast iteration) | Full (paper-like) |
|-----------|----------------------|-------------------|
| Vision | CLIP ViT-B/32 (512-d) | CLIP ViT-L/14 (768-d) |
| Projection | 512 → 768 (1–2 layers) | 1024 → 4096 (2 layers) |
| LLM | GPT-2 small / DistilGPT-2 | Vicuna-7B / LLaMA-7B |

**Recommendation:** Start with **tiny** (CLIP ViT-B/32 + GPT-2 small) for fast iteration. Scale up once pipeline works.

---

## Config (`configs/llava.yaml`)

```yaml
# Model
vision_model: openai/clip-vit-base-patch32
llm_model: gpt2  # or distilgpt2
projection_hidden: 1024
projection_num_layers: 2

# Data
dataset_name: nlpconnect/vit-gpt2-image-captioning  # or COCO
split: train
batch_size: 8
max_length: 128

# Training
epochs: 3
lr: 1e-4
output_dir: checkpoints/llava
```

---

## Engineering

- Config-driven (YAML)
- Deterministic seeds
- Save checkpoints + train log JSON
- Optional: mixed precision (AMP)
- README: setup, train, inference, sample outputs

---

## Stretch

- LoRA on LLM instead of full freeze
- Multi-turn conversation
- BLIP2 next (Q-Former architecture)

---

## References

- [LLaVA paper](https://arxiv.org/abs/2304.08485)
- [LLaVA GitHub](https://github.com/haotian-liu/LLaVA)
- Hugging Face: `openai/clip-vit-base-patch32`, `gpt2`, `distilgpt2`
