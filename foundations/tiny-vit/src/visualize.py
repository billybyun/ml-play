"""Visualize ViT predictions: sample images with predicted vs ground truth labels."""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch
import matplotlib.pyplot as plt

from src.data import get_dataloader, IMAGENET_MEAN, IMAGENET_STD, CIFAR10_MEAN, CIFAR10_STD
from src.models import create_vit, create_small_vit
from src.utils import load_config

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def denormalize(tensor, mean, std):
    """Convert normalized tensor to displayable image (0-1)."""
    x = tensor.clone()
    for c in range(3):
        x[c] = x[c] * std[c] + mean[c]
    return x.clamp(0, 1)


def main():
    parser = argparse.ArgumentParser(description="Visualize ViT predictions.")
    parser.add_argument("--config", default=None, help="Config path (default: from checkpoint)")
    parser.add_argument("--checkpoint", default="checkpoints/linear_probe/final.pt", help="Checkpoint path")
    parser.add_argument("--n", type=int, default=12, help="Number of samples to show")
    parser.add_argument("--output", default=None, help="Output figure path")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    config = ckpt.get("config") or load_config(args.config or "configs/cifar10.yaml")

    if config.get("model_type") == "small_vit" or "patch_size" in config or "embed_dim" in config:
        model = create_small_vit(config).to(device)
        mean, std = CIFAR10_MEAN, CIFAR10_STD
        title = "Small ViT from scratch"
    else:
        model = create_vit(config).to(device)
        mean, std = IMAGENET_MEAN, IMAGENET_STD
        title = "Linear probe"
    model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    model.eval()

    if args.output is None:
        args.output = "results/small_vit/samples.png" if (config.get("model_type") == "small_vit" or "patch_size" in config) else "results/linear_probe/samples.png"

    dataloader = get_dataloader(config, split="test")
    images, labels = next(iter(dataloader))
    images, labels = images.to(device), labels.cpu()

    with torch.no_grad():
        logits = model(images)
        preds = logits.argmax(dim=1).cpu()

    n = min(args.n, images.size(0))
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = axes.flatten()

    for i in range(n):
        img = denormalize(images[i].cpu(), mean, std).permute(1, 2, 0).numpy()
        gt = labels[i].item()
        pred = preds[i].item()
        correct = gt == pred

        ax = axes[i]
        ax.imshow(img)
        ax.axis("off")
        color = "green" if correct else "red"
        ax.set_title(f"Pred: {CIFAR10_CLASSES[pred]}\nTrue: {CIFAR10_CLASSES[gt]}", fontsize=9, color=color)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"{title} on CIFAR-10 (green=correct, red=wrong)", fontsize=11)
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    plt.savefig(args.output, dpi=120, bbox_inches="tight")
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
