"""Training loop for tiny-vit: linear probe or tiny ViT from scratch."""
import argparse
import json
import os
import sys

# Add project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch
import torch.nn.functional as F

from src.data import get_dataloader
from src.models import create_vit, create_tiny_vit, freeze_backbone_for_linear_probe
from src.utils import load_config


def main():
    parser = argparse.ArgumentParser(description="Train ViT: linear probe or tiny ViT from scratch.")
    parser.add_argument("--model-type", choices=["linear_probe", "tiny_vit"], default="linear_probe")
    parser.add_argument("--config", default=None, help="Config path (default: cifar10.yaml or tiny_vit.yaml)")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs from config")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate from config")
    parser.add_argument("--output-dir", default=None, help="Checkpoint directory")
    args = parser.parse_args()

    if args.config is None:
        args.config = "configs/tiny_vit.yaml" if args.model_type == "tiny_vit" else "configs/cifar10.yaml"
    if args.output_dir is None:
        args.output_dir = "checkpoints/tiny_vit" if args.model_type == "tiny_vit" else "checkpoints/linear_probe"

    config = load_config(args.config)
    epochs = args.epochs if args.epochs is not None else config.get("epochs", 10)
    lr = args.lr if args.lr is not None else config.get("lr", 1e-3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Model type: {args.model_type}")

    torch.manual_seed(42)
    if args.model_type == "tiny_vit":
        model = create_tiny_vit(config).to(device)
    else:
        model = create_vit(config).to(device)
        freeze_backbone_for_linear_probe(model)

    train_loader = get_dataloader(config, split="train")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    os.makedirs(args.output_dir, exist_ok=True)
    train_log = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        train_log.append({"epoch": epoch + 1, "loss": avg_loss, "train_acc": train_acc})
        print(f"Epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}  train_acc={train_acc:.2f}%")

    ckpt_path = os.path.join(args.output_dir, "final.pt")
    config_to_save = {**config, "model_type": args.model_type}
    torch.save({"model": model.state_dict(), "config": config_to_save}, ckpt_path)
    print(f"Saved: {ckpt_path}")

    log_path = os.path.join(args.output_dir, "train_log.json")
    log_data = {"device": str(device), "model_type": args.model_type, "epochs": epochs, "history": train_log}
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"Saved: {log_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
