"""Evaluation for tiny-vit (accuracy on test set)."""
import argparse
import os
import sys

# Add project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch

from src.data import get_dataloader
from src.models import create_vit, create_tiny_vit
from src.utils import load_config


def evaluate(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: torch.device, top_k: tuple = (1, 5)):
    """Compute top-1 and top-k accuracy."""
    model.eval()
    correct = {k: 0 for k in top_k}
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            _, pred = outputs.topk(max(top_k), 1, largest=True, sorted=True)
            pred = pred.t()

            for k in top_k:
                correct[k] += (pred[:k] == labels.unsqueeze(0)).any(dim=0).sum().item()
            total += labels.size(0)

    acc = {k: 100.0 * correct[k] / total for k in top_k}
    return acc


def main():
    parser = argparse.ArgumentParser(description="Evaluate ViT on CIFAR-10.")
    parser.add_argument("--config", default=None, help="Config path (default: from checkpoint or cifar10.yaml)")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path (optional; uses pretrained if not set)")
    parser.add_argument("--output", default=None, help="Save metrics to JSON")
    parser.add_argument("--print-shapes", action="store_true", help="Print input/output dimensions and exit")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        config = ckpt.get("config") or load_config(args.config or "configs/cifar10.yaml")
        model_type = config.get("model_type") or ("tiny_vit" if ("patch_size" in config or "embed_dim" in config) else "timm")
        # Support legacy small_vit checkpoints
        if model_type in ("tiny_vit", "small_vit"):
            model = create_tiny_vit(config).to(device)
        else:
            model = create_vit(config).to(device)
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    else:
        config = load_config(args.config or "configs/cifar10.yaml")
        model = create_vit(config).to(device)

    if args.print_shapes:
        dataloader = get_dataloader(config, split="test")
        images, labels = next(iter(dataloader))
        images = images.to(device)
        with torch.no_grad():
            logits = model(images)
            print("--- ViT shape check ---")
            print(f"  Input (images):  {tuple(images.shape)}")
            print(f"  Output (logits): {tuple(logits.shape)}")
            if hasattr(model, "forward_features"):
                features = model.forward_features(images)
                print(f"  Backbone (CLS): {tuple(features.shape)}  <- 768-d per sample before head")
            print("---")
        return 0

    dataloader = get_dataloader(config, split="test")

    acc = evaluate(model, dataloader, device)

    print(f"Top-1 accuracy: {acc[1]:.2f}%")
    print(f"Top-5 accuracy: {acc[5]:.2f}%")

    if args.output:
        import json
        if not args.checkpoint:
            note = "random head, no training"
        elif config.get("model_type") in ("tiny_vit", "small_vit"):
            note = "tiny ViT from scratch"
        else:
            note = "linear probe (trained head)"
        result = {"note": note, "device": str(device), "top1": acc[1], "top5": acc[5]}
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
