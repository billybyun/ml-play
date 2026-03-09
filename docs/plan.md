# ml-play Roadmap

A living plan for ML explorations in this repo. Same pattern as tiny-clip: small implementations, config-driven, reproducible, with ablations and clean READMEs.

---

## Vision & Video Architectures

| Project | Description | Status |
|---------|-------------|--------|
| **tiny-vit** | ViT: pretrained first, then tiny from scratch (CIFAR-10) | Done |
| **SwinT** | Swin Transformer — hierarchical, shifted windows, efficient for high-res | Planned |
| **ViViT** | Video Vision Transformer — factorized space-time attention | Planned |
| **Hybrid CNN–Transformer** | CNN stem + transformer (e.g. ResNet backbone → ViT head) | Planned |
| **Perceiver** | Cross-attention to latent bottleneck; modality-agnostic | Planned |
| **Conformer** | Convolution + transformer (speech/audio, or adapted for vision) | Planned |
| **S4 / Mamba** | State-space models — linear complexity, long context | Planned |

---

## Vision–Language & Multimodal

| Project | Description | Status |
|---------|-------------|--------|
| **tiny-clip** | CLIP-style dual encoder, B0/B1 ablations | Done |
| **BLIP2** | Q-Former bridging frozen image encoder + frozen LLM | Planned |
| **Qwen-VL** | Qwen vision–language model | Planned |
| **tiny-llava** | LLaVA-style: frozen CLIP + projection + LLM | Scaffold |
| **VLA** | Vision–language–action for robotics | Planned |

---

## Additional Explorations (suggested)

| Area | Ideas |
|------|-------|
| **Efficient attention** | Flash attention, linear attention, sparse attention |
| **Parameter-efficient tuning** | LoRA, QLoRA, adapter layers — compare to full fine-tuning |
| **Distillation & teacher–student** | Knowledge distillation, soft labels, feature matching, tiny student from large teacher |
| **Pruning & compression** | Structured/unstructured pruning, magnitude pruning, quantization (INT8, GPTQ) |
| **Diffusion** | Minimal diffusion (DDPM) or Stable Diffusion–style pipeline |
| **Audio** | Whisper, wav2vec, or Conformer for speech |
| **RAG / retrieval** | Dense retrieval + LLM for grounded generation |
| **Benchmarking** | Shared eval harness (same dataset, same metrics across models) |

---

## Principles

- **Small first:** Minimal implementations, then scale.
- **Pretrained first:** Use pretrained when available; build from scratch to understand.
- **Config-driven:** YAML configs, reproducible runs.
- **Ablations:** Before/after, random vs trained, frozen vs unfrozen.
- **Clean READMEs:** Setup, commands, results, lessons learned.
