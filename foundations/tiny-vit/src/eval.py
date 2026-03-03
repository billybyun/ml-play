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
from src.models import create_vit
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
    parser.add_argument("--config", default="configs/cifar10.yaml", help="Config path")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path (optional; uses pretrained if not set)")
    parser.add_argument("--output", default=None, help="Save metrics to JSON")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_vit(config).to(device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)

    dataloader = get_dataloader(config, split="test")

    acc = evaluate(model, dataloader, device)

    print(f"Top-1 accuracy: {acc[1]:.2f}%")
    print(f"Top-5 accuracy: {acc[5]:.2f}%")

    if args.output:
        import json
        with open(args.output, "w") as f:
            json.dump(acc, f, indent=2)
        print(f"Saved: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
