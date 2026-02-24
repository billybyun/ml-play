# train.py — Config-driven training loop. Used from Stage A1 / B1 onward.
"""
Train custom dual encoder (Stage B1): projections + temperature only.
Frozen: ViT, text encoder. Loss: symmetric InfoNCE.
"""
import argparse
import json
import os
import sys

import torch
from transformers import AutoTokenizer

from src.data import get_dataloader_dual_encoder, get_dual_encoder_image_transform
from src.loss import clip_symmetric_loss
from src.models import DualEncoderModel
from src.utils import load_config, set_seed


def main():
    parser = argparse.ArgumentParser(description="Train custom dual encoder (projections only)")
    parser.add_argument("--config", type=str, default="configs/custom_dual_encoder.yaml")
    parser.add_argument("--output-dir", type=str, default="checkpoints/dual_encoder")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=500, help="Steps between eval (0 = no eval)")
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
    os.makedirs(args.output_dir, exist_ok=True)

    # Model: freeze encoders, train projections + temperature
    model = DualEncoderModel(
        vision_model=config.get("vision_model", "vit_base_patch16_224"),
        text_model=config.get("text_model", "distilbert-base-uncased"),
        projection_dim=config.get("projection_dim", 512),
    ).to(device)
    for p in model.vision_encoder.parameters():
        p.requires_grad = False
    for p in model.text_encoder.parameters():
        p.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(config.get("text_model", "distilbert-base-uncased"))
    image_transform = get_dual_encoder_image_transform()
    loader = get_dataloader_dual_encoder(
        config, tokenizer, image_transform,
        split=config.get("train_split", "train"),
        shuffle=True,
        for_eval=False,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )

    model.train()
    step = 0
    log_losses = []

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        n_batches = 0
        for batch in loader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)   # (B, 5, L)
            attention_mask = batch["attention_mask"].to(device)

            # Use first caption per image for contrastive (B pairs -> BxB sim)
            B = pixel_values.size(0)
            input_ids_1 = input_ids[:, 0, :]       # (B, L)
            attention_mask_1 = attention_mask[:, 0, :]

            image_embeds = model.get_image_features(pixel_values)
            text_embeds = model.get_text_features(input_ids=input_ids_1, attention_mask=attention_mask_1)

            loss = clip_symmetric_loss(
                image_embeds, text_embeds,
                temperature=model.temperature,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            epoch_loss += loss.item()
            n_batches += 1
            log_losses.append({"step": step, "loss": loss.item()})

            if step % 100 == 0:
                print(f"Epoch {epoch+1}/{args.epochs} step {step} loss={loss.item():.4f}")

        avg_loss = epoch_loss / n_batches
        print(f"Epoch {epoch+1}/{args.epochs} avg_loss={avg_loss:.4f}")

    # Save checkpoint
    ckpt_path = os.path.join(args.output_dir, "final.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "epochs": args.epochs,
    }, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

    log_path = os.path.join(args.output_dir, "train_log.json")
    with open(log_path, "w") as f:
        json.dump(log_losses, f, indent=2)
    print(f"Saved train log to {log_path}")


if __name__ == "__main__":
    main()
