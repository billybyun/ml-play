"""Training loop for tiny-vit: linear probe (freeze backbone, train head only)."""
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
from src.models import create_vit, freeze_backbone_for_linear_probe
from src.utils import load_config


def main():
    parser = argparse.ArgumentParser(description="Train linear probe on ViT (freeze backbone, train head only).")
    parser.add_argument("--config", default="configs/cifar10.yaml", help="Config path")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output-dir", default="checkpoints/linear_probe", help="Checkpoint directory")
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.manual_seed(42)
    model = create_vit(config).to(device)
    freeze_backbone_for_linear_probe(model)

    train_loader = get_dataloader(config, split="train")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.output_dir, exist_ok=True)
    train_log = []

    for epoch in range(args.epochs):
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
        print(f"Epoch {epoch + 1}/{args.epochs}  loss={avg_loss:.4f}  train_acc={train_acc:.2f}%")

    ckpt_path = os.path.join(args.output_dir, "final.pt")
    torch.save({"model": model.state_dict(), "config": config}, ckpt_path)
    print(f"Saved: {ckpt_path}")

    log_path = os.path.join(args.output_dir, "train_log.json")
    log_data = {"device": str(device), "epochs": args.epochs, "history": train_log}
    with open(log_path, "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"Saved: {log_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
