"""Training loop for tiny-flamingo (Perceiver Resampler only)."""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch

from src.data import get_dataloader
from src.models import TinyFlamingo
from src.utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/flamingo.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    epochs = args.epochs or config.get("epochs", 3)
    lr = args.lr or config.get("lr", 1e-4)
    output_dir = args.output_dir or config.get("output_dir", "checkpoints/flamingo")
    use_amp = config.get("use_amp", True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.manual_seed(42)
    model = TinyFlamingo(config).to(device)
    # gradient_checkpointing: can add torch.utils.checkpoint in resampler if OOM

    train_loader = get_dataloader(config, split="train")
    optimizer = torch.optim.Adam(model.resampler.parameters(), lr=lr)
    scaler = torch.amp.GradScaler("cuda") if use_amp and device.type == "cuda" else None

    os.makedirs(output_dir, exist_ok=True)
    train_log = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            if use_amp and device.type == "cuda" and scaler is not None:
                with torch.amp.autocast("cuda"):
                    _, loss = model(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                _, loss = model(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        train_log.append({"epoch": epoch + 1, "loss": avg_loss})
        print(f"Epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}")

    ckpt_path = os.path.join(output_dir, "final.pt")
    torch.save(
        {
            "resampler": model.resampler.state_dict(),
            "config": {**config, "model_type": "tiny_flamingo"},
        },
        ckpt_path,
    )
    print(f"Saved: {ckpt_path}")

    log_path = os.path.join(output_dir, "train_log.json")
    with open(log_path, "w") as f:
        json.dump(
            {"device": str(device), "epochs": epochs, "history": train_log},
            f,
            indent=2,
        )
    print(f"Saved: {log_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
