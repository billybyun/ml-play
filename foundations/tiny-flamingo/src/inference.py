"""Inference: generate text from image + prompt."""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch
from PIL import Image

from src.models import TinyFlamingo
from src.utils import load_config


def get_image_transform(config: dict):
    """Build transform for ViT (224x224, ImageNet norm)."""
    from torchvision import transforms
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    size = config.get("image_size", 224)
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to image")
    parser.add_argument("--prompt", default="Describe this image.", help="Text prompt")
    parser.add_argument("--config", default="configs/flamingo.yaml")
    parser.add_argument("--checkpoint", default=None, help="Resampler checkpoint (optional)")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = TinyFlamingo(config).to(device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.resampler.load_state_dict(ckpt.get("resampler", ckpt), strict=False)
        print(f"Loaded: {args.checkpoint}")

    model.eval()
    transform = get_image_transform(config)
    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(config["llm_model"])

    img = Image.open(args.image).convert("RGB")
    pixel_values = transform(img).unsqueeze(0).to(device)
    enc = tokenizer(args.prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    with torch.no_grad():
        visual_tokens = model.get_visual_tokens(pixel_values)
        cur_embeds = model.lm.transformer.wte(input_ids)
        combined = torch.cat([visual_tokens, cur_embeds], dim=1)
        seq_len = combined.shape[1]
        generated_ids = list(input_ids[0].tolist())

        for _ in range(args.max_new_tokens):
            position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
            outputs = model.lm.transformer(inputs_embeds=combined, position_ids=position_ids)
            next_logits = model.lm.lm_head(outputs.last_hidden_state[:, -1, :])
            next_id = next_logits.argmax(dim=-1, keepdim=True)
            generated_ids.append(next_id.item())

            if next_id.item() == tokenizer.eos_token_id:
                break

            next_embed = model.lm.transformer.wte(next_id)
            combined = torch.cat([combined, next_embed], dim=1)
            seq_len += 1

    text = tokenizer.decode(generated_ids[len(input_ids[0]):], skip_special_tokens=True)
    print(text.strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
