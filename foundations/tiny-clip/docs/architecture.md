# CLIP Architecture (for Tiny-CLIP)

Reference for understanding CLIP before and during Track B. See Hugging Face: `transformers.models.clip`.

---

## High-level structure

```
Image → Vision Encoder (ViT) → vision_projection → z_img (512-d)
Text  → Text Encoder          → text_projection  → z_txt (512-d)
Similarity = (z_img @ z_txt.T) * exp(logit_scale)
Loss = symmetric cross-entropy on similarity matrix (in-batch negatives)
```

---

## Key components (HF `openai/clip-vit-base-patch32`)

| Component | Location | Role |
|-----------|----------|------|
| **Vision encoder** | `CLIPVisionModel` | ViT: patch embedding → transformer blocks → pooler. Output: (B, 768). |
| **Text encoder** | `CLIPTextModel` | Token embed → transformer → pool (EOS token). Output: (B, 512). |
| **vision_projection** | `nn.Linear(768, 512)` | Map vision features to shared space. |
| **text_projection** | `nn.Linear(512, 512)` | Map text features to shared space. |
| **logit_scale** | `nn.Parameter` | Temperature: `exp(logit_scale)` ≈ 100. Init ~2.66. |

---

## Config (CLIPConfig)

- `projection_dim`: 512 (shared embedding size)
- `logit_scale_init_value`: 2.6592
- Vision: `image_size=224`, `patch_size=32`, `hidden_size=768`
- Text: `max_position_embeddings=77`, `hidden_size=512`

---

## Loss (InfoNCE / symmetric CE)

For batch of B image-text pairs:

1. `sim = (z_img @ z_txt.T) / temperature`  — (B, B) logits
2. Targets: `[0, 1, ..., B-1]` (diagonal = correct pairs)
3. `loss_i2t = CE(sim, targets)`  — image as query, text as target
4. `loss_t2i = CE(sim.T, targets)`  — text as query, image as target
5. `loss = (loss_i2t + loss_t2i) / 2`

---

## Where to look in code

- **HF modeling:** `transformers/models/clip/modeling_clip.py` — `CLIPModel`, `vision_projection`, `text_projection`, `logit_scale`
- **Our eval:** [foundations/tiny-clip/src/eval_retrieval.py](../foundations/tiny-clip/src/eval_retrieval.py) — `get_image_features`, `get_text_features`, `sim = image_embeds @ text_embeds.T`

---

## Track B vs CLIP

| | CLIP | Track B (DualEncoder) |
|---|------|------------------------|
| Vision | CLIP ViT (HF) | timm ViT-B/16 (768-d) |
| Text | CLIP text (HF) | DistilBERT (768-d) |
| Projections | 768→512, 512→512 | 768→512, 768→512 |
| Temperature | logit_scale (learned) | scalar tau (learned) |

Same idea: two encoders, projections to shared dim, contrastive loss.
