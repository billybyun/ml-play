# data.py — Dataset and DataLoader for Flickr30k / Flickr8k (HF or local).
# Stage A0: wraps load_dataset + CLIPProcessor.
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


class Flickr30kCLIPDataset(Dataset):
    """
    Flickr30k with CLIP preprocessing. One item per image; each image has 5 captions.
    Returns pixel_values (from CLIPProcessor) and tokenized captions (input_ids, attention_mask)
    for the 5 captions, so batch shapes are (B, C, H, W) and (B, 5, L).
    Uses parquet revision to avoid "dataset scripts are no longer supported" (HF deprecation).
    """
    def __init__(self, dataset_name: str, split: str, processor, max_length: int = 77, revision: str | None = "refs/convert/parquet"):
        self.processor = processor
        self.max_length = max_length
        # Load from parquet conversion to avoid legacy script (HF deprecated dataset .py scripts)
        self.hf_ds = load_dataset(dataset_name, split=split, revision=revision)
        # nlphuji/flickr30k: each row has "image" (PIL) and "caption" (list of 5 strings)
        self._check_columns()

    def _check_columns(self):
        features = self.hf_ds.features
        if "image" not in self.hf_ds.column_names:
            raise ValueError(f"Expected 'image' column; got {self.hf_ds.column_names}")
        if "caption" not in self.hf_ds.column_names:
            raise ValueError(f"Expected 'caption' column; got {self.hf_ds.column_names}")

    def __len__(self) -> int:
        return len(self.hf_ds)

    def __getitem__(self, idx: int) -> dict:
        row = self.hf_ds[idx]
        image = row["image"]
        captions = row["caption"]
        if not isinstance(captions, list):
            captions = [captions]
        # Ensure we have 5 captions (some rows might have fewer)
        captions = (captions * 5)[:5]

        # Process image: processor expects PIL, returns dict with pixel_values
        inputs = self.processor(
            images=image,
            text=captions,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }


def get_dataloader(config: dict, processor, split: str | None = None, shuffle: bool = False) -> DataLoader:
    """Build a DataLoader from config and CLIP processor."""
    dataset_name = config["dataset_name"]
    split = split or config.get("split", "test")
    batch_size = config.get("batch_size", 32)
    revision = config.get("revision", "refs/convert/parquet")
    dataset = Flickr30kCLIPDataset(dataset_name, split, processor, revision=revision)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=_collate_fn,
    )


def _collate_fn(batch: list[dict]) -> dict:
    """Stack batch; pixel_values (B, C, H, W), input_ids (B, 5, L), attention_mask (B, 5, L)."""
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


if __name__ == "__main__":
    # Sanity check: load config, dataset, one batch; print shapes.
    # Run from foundations/tiny-clip:  python -m src.data
    import sys
    import os
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from transformers import CLIPProcessor
    from src.utils import load_config

    config_path = os.path.join(_root, "configs", "flickr30k.yaml")
    config = load_config(config_path)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    dataset = Flickr30kCLIPDataset(
        config["dataset_name"],
        config["split"],
        processor,
    )
    loader = get_dataloader(config, processor, shuffle=False)

    print("Dataset size:", len(dataset))
    batch = next(iter(loader))
    print("Batch keys:", batch.keys())
    print("pixel_values shape:", batch["pixel_values"].shape)
    print("input_ids shape:", batch["input_ids"].shape)
    print("attention_mask shape:", batch["attention_mask"].shape)
    print("Sanity check passed.")
