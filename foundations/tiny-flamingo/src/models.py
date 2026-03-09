"""TinyFlamingo: frozen ViT + Perceiver Resampler + frozen GPT-2."""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM

from src.resampler import PerceiverResampler


class TinyFlamingo(nn.Module):
    """Frozen ViT -> Perceiver Resampler -> concat as prefix -> frozen GPT-2."""

    def __init__(self, config: dict):
        super().__init__()
        vision_model = config.get("vision_model", "google/vit-base-patch16-224")
        llm_model = config.get("llm_model", "gpt2")
        num_latents = config.get("num_latents", 32)
        d_latent = config.get("d_latent", 768)
        resampler_depth = config.get("resampler_depth", 2)
        resampler_heads = config.get("resampler_heads", 8)

        self.vision_encoder = AutoModel.from_pretrained(vision_model)
        for p in self.vision_encoder.parameters():
            p.requires_grad = False

        vision_dim = self.vision_encoder.config.hidden_size  # 768 for ViT-B

        self.resampler = PerceiverResampler(
            d_input=vision_dim,
            num_latents=num_latents,
            d_latent=d_latent,
            depth=resampler_depth,
            n_heads=resampler_heads,
        )

        self.lm = AutoModelForCausalLM.from_pretrained(llm_model)
        for p in self.lm.parameters():
            p.requires_grad = False

        self.num_latents = num_latents
        self.config = config

    def get_vision_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, N, 768)"""
        out = self.vision_encoder(pixel_values)
        return out.last_hidden_state

    def get_visual_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, num_latents, 768)"""
        vision = self.get_vision_features(pixel_values)
        return self.resampler(vision)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Returns (logits, loss).
        logits: (B, num_latents + L, vocab_size). Loss is only on text tokens.
        """
        B = pixel_values.shape[0]
        visual_tokens = self.get_visual_tokens(pixel_values)  # (B, num_latents, 768)

        text_embeds = self.lm.transformer.wte(input_ids)  # (B, L, 768)
        combined = torch.cat([visual_tokens, text_embeds], dim=1)  # (B, num_latents + L, 768)

        seq_len = combined.shape[1]
        position_ids = torch.arange(seq_len, device=combined.device, dtype=torch.long)
        position_ids = position_ids.unsqueeze(0).expand(B, -1)

        outputs = self.lm.transformer(inputs_embeds=combined, position_ids=position_ids)
        hidden = outputs.last_hidden_state
        logits = self.lm.lm_head(hidden)

        loss = None
        if labels is not None:
            # Shift: predict next token. Labels for visual positions = -100
            shift_logits = logits[:, self.num_latents - 1 : -1, :].contiguous()
            shift_labels = labels.contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return logits, loss

    def generate(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        **kwargs,
    ) -> torch.Tensor:
        """Generate tokens given image + prompt."""
        visual_tokens = self.get_visual_tokens(pixel_values)
        text_embeds = self.lm.transformer.wte(input_ids)
        combined = torch.cat([visual_tokens, text_embeds], dim=1)
        B, seq_len, _ = combined.shape
        position_ids = torch.arange(seq_len, device=combined.device, dtype=torch.long).unsqueeze(0).expand(B, -1)

        return self.lm.generate(
            inputs_embeds=combined,
            position_ids=position_ids,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )


