# eval_retrieval.py — Load model, compute embeddings, similarity matrix, R@1/5/10 both directions.
"""
Zero-shot retrieval evaluation on the standard Flickr30k 1k test set.
Usage (from foundations/tiny-clip): python -m src.eval_retrieval --config configs/flickr30k.yaml
"""
import argparse
import json
import os
import sys

import torch
from transformers import CLIPModel, CLIPProcessor

from src.data import get_dataloader
from src.utils import load_config, set_seed


def compute_metrics(sim: torch.Tensor, n_images: int) -> dict:
    """
    sim: (N, 5*N) — similarity from images to captions (each image has 5 captions).
    Caption j belongs to image j // 5.
    """
    device = sim.device
    N = n_images
    # Image -> Text: for image i, correct captions are 5*i .. 5*i+4. R@k = hit if any in top-k.
    i2t_r1, i2t_r5, i2t_r10 = 0.0, 0.0, 0.0
    for i in range(N):
        scores = sim[i]  # (5*N,)
        topk = torch.topk(scores, 10, dim=0).indices  # (10,)
        correct = set(range(5 * i, 5 * i + 5))
        i2t_r1 += 1.0 if topk[0].item() in correct else 0.0
        i2t_r5 += 1.0 if any(t.item() in correct for t in topk[:5]) else 0.0
        i2t_r10 += 1.0 if any(t.item() in correct for t in topk[:10]) else 0.0
    i2t_r1 /= N
    i2t_r5 /= N
    i2t_r10 /= N

    # Text -> Image: for caption j, correct image is j // 5. R@k = hit if correct image in top-k.
    n_caps = 5 * N
    t2i_r1, t2i_r5, t2i_r10 = 0.0, 0.0, 0.0
    for j in range(n_caps):
        scores = sim[:, j]  # (N,)
        topk = torch.topk(scores, 10, dim=0).indices
        correct_img = j // 5
        t2i_r1 += 1.0 if topk[0].item() == correct_img else 0.0
        t2i_r5 += 1.0 if correct_img in topk[:5].tolist() else 0.0
        t2i_r10 += 1.0 if correct_img in topk[:10].tolist() else 0.0
    t2i_r1 /= n_caps
    t2i_r5 /= n_caps
    t2i_r10 /= n_caps

    return {
        "i2t_R@1": round(i2t_r1, 4),
        "i2t_R@5": round(i2t_r5, 4),
        "i2t_R@10": round(i2t_r10, 4),
        "t2i_R@1": round(t2i_r1, 4),
        "t2i_R@5": round(t2i_r5, 4),
        "t2i_R@10": round(t2i_r10, 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Zero-shot retrieval eval (Flickr30k 1k test set)")
    parser.add_argument("--config", type=str, default="configs/flickr30k.yaml", help="Path to config YAML")
    parser.add_argument("--output", type=str, default=None, help="Path to save metrics JSON (default: print only)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    os.chdir(root)

    config_path = args.config if os.path.isabs(args.config) else os.path.join(root, args.config)
    config = load_config(config_path)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    loader = get_dataloader(config, processor, for_eval=True)
    n_images = len(loader.dataset)

    image_embeds_list = []
    text_embeds_list = []

    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)   # (B, 5, L)
            attention_mask = batch["attention_mask"].to(device)

            B = pixel_values.size(0)
            image_embeds = model.get_image_features(pixel_values)  # (B, D)
            input_ids_flat = input_ids.view(B * 5, -1)
            attention_mask_flat = attention_mask.view(B * 5, -1)
            text_embeds = model.get_text_features(input_ids=input_ids_flat, attention_mask=attention_mask_flat)  # (B*5, D)

            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            image_embeds_list.append(image_embeds.cpu())
            text_embeds_list.append(text_embeds.cpu())

    image_embeds = torch.cat(image_embeds_list, dim=0)  # (N, D)
    text_embeds = torch.cat(text_embeds_list, dim=0)    # (5*N, D)
    sim = image_embeds @ text_embeds.T  # (N, 5*N)

    metrics = compute_metrics(sim, n_images)
    metrics["n_images"] = n_images
    metrics["split"] = "1k test (standard benchmark)"

    print("Retrieval metrics (zero-shot CLIP, Flickr30k 1k test set):")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    if args.output:
        out_path = args.output if os.path.isabs(args.output) else os.path.join(root, args.output)
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to {out_path}")


if __name__ == "__main__":
    main()
