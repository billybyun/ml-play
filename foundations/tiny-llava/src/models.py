"""LLaVA model: frozen vision encoder + projection + frozen LLM.

Architecture:
  Image -> [CLIP ViT] -> [Projection MLP] -> vision tokens
  Text  -> [Tokenizer] -> text tokens  -> [LLM] -> generated text

TODO: Implement LLaVAModel
- Load CLIP vision encoder (frozen)
- Projection MLP: vision_dim -> hidden -> llm_embed_dim
- Load GPT-2 or similar (frozen)
- forward(image, input_ids, attention_mask) -> logits
"""
