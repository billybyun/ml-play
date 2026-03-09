# tiny-flamingo Project Plan

## Goal

Implement a minimal Flamingo-style vision–language model: **frozen ViT → Perceiver Resampler → GPT-2**. The Perceiver compresses many image tokens into a fixed number of visual tokens; these are prefixed to the LLM for generation.

**Target:** Trainable on 12GB VRAM (e.g. RTX 5070).

**Architecture:**
```
Image → [ViT-B/16] → (B, 196, 768) vision tokens
                          ↓
              [Perceiver Resampler]  ← learned latents cross-attend to vision
                          ↓
              (B, 32, 768) visual tokens
                          ↓
Text  → [Tokenizer] → text tokens ──→ [GPT-2] → generated text
```

---

## Tiny vs real Flamingo

| Component | Ours (tiny) | Real Flamingo |
|-----------|--------------|---------------|
| **Vision** | ViT-B/16 frozen | NFNet / ViT frozen |
| **Resampler** | Small Perceiver (32 latents, 2–4 layers) | Perceiver (64 latents, 6 layers) |
| **LM** | GPT-2 (~124M) | Chinchilla 70B |
| **Cross-attn in LM** | Optional (prefix concat first) | Gated cross-attention |

---

## Repo Structure

```
foundations/tiny-flamingo/
  src/
    models.py         # TinyFlamingo: ViT + PerceiverResampler + GPT-2
    resampler.py      # PerceiverResampler (adapt from foundations/perceiver_arch.py)
    data.py           # Image-caption dataset
    train.py          # Train resampler (ViT + GPT-2 frozen)
    inference.py      # Generate from image + prompt
    utils.py          # Config loading
  configs/
    flamingo.yaml     # Model dims, resampler, data, training
  demos/
    demo_inference.py
  docs/
    plan.md
  requirements.txt
  README.md
```

---

## Phase 1: Scaffold & Inference (No Training)

**Goal:** Wire ViT + Perceiver Resampler + GPT-2, run inference. Establish pipeline.

### 1.1 Placement & scaffold

- [x] Create `foundations/tiny-flamingo/` with `src/`, `configs/`, `demos/`, `docs/`
- [ ] Add `requirements.txt`, `configs/flamingo.yaml`, `src/utils.py`
- [x] Add `src/resampler.py`: `PerceiverResampler` — learned latents cross-attend to vision features; output (B, N_latent, d_latent)
- [x] Add `src/models.py`: `TinyFlamingo` — frozen ViT → resampler (random init) → concat as prefix to GPT-2
- [x] Add `src/inference.py`: load model, image + prompt → generated text

### 1.2 Inference

- [x] Run: `python -m src.inference --image path/to/img.jpg --prompt "Describe this image."`
- [ ] Expected with random resampler: gibberish (validates pipeline)

---

## Phase 2: Data & Training

**Goal:** Train the Perceiver Resampler on image–caption pairs. ViT and GPT-2 frozen.

### 2.1 Dataset

- [ ] Add `src/data.py`: COCO captions or `nlpconnect/vit-gpt2-image-captioning`
- [ ] Format: (image, caption); prompt = "Describe this image."; target = caption

### 2.2 Training loop

- [ ] Add `src/train.py`: train resampler only; loss = next-token prediction
- [ ] Input to GPT-2: `[visual_tokens] + [text_tokens]`
- [ ] Use mixed precision (AMP), gradient checkpointing if needed
- [ ] Batch size 4–8, max_length 128 for 12GB

### 2.3 Eval

- [ ] Qualitative: sample captions on val split
- [ ] Optional: BLEU/ROUGE

---

## Phase 3: Cross-Attention in LM (Optional)

**Goal:** Add cross-attention layers to GPT-2 so it attends to visual tokens (Flamingo-style).

- [ ] Modify GPT-2 blocks: add cross-attn (Q from text, K/V from visual tokens)
- [ ] Train cross-attn + resampler
- [ ] Compare: prefix concat vs cross-attn

---

## Config (`configs/flamingo.yaml`)

```yaml
# Model
vision_model: google/vit-base-patch16-224  # or timm vit_base_patch16_224
llm_model: gpt2
image_size: 224

# Perceiver Resampler
num_latents: 32
d_latent: 768        # match GPT-2 hidden
resampler_depth: 2
resampler_heads: 8

# Data
dataset_name: nlpconnect/vit-gpt2-image-captioning
batch_size: 4
max_length: 128

# Training (12GB-friendly)
epochs: 3
lr: 1e-4
use_amp: true
gradient_checkpointing: false  # enable if OOM
```

---

## Memory Budget (12GB)

| Component | Est. memory |
|-----------|-------------|
| ViT-B/16 (frozen) | ~350MB |
| Perceiver (32 latents, 2 layers) | ~50MB |
| GPT-2 (frozen) | ~500MB |
| Activations (batch 4, seq 160) | ~2–4GB |
| Gradients (resampler only) | ~100MB |
| **Total** | ~4–5GB |

Leaves headroom for AMP and longer sequences.

---

## References

- [Flamingo paper](https://arxiv.org/abs/2204.14198)
- [lucidrains/flamingo-pytorch](https://github.com/lucidrains/flamingo-pytorch)
- `foundations/perceiver_arch.py` — PerceiverBlock, MultiHeadAttention
