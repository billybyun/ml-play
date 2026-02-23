# data.py — Dataset and DataLoader for Flickr30k / Flickr8k (HF or local).
# Stage A0: wraps load_dataset + CLIPProcessor.
import hashlib
import os

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
        # Load; revision avoids "dataset scripts are no longer supported" when set (e.g. parquet)
        if revision is not None:
            self.hf_ds = load_dataset(dataset_name, split=split, revision=revision)
        else:
            self.hf_ds = load_dataset(dataset_name, split=split)
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


def get_dataloader(config: dict, processor, split: str | None = None, shuffle: bool = False, for_eval: bool = False) -> DataLoader:
    """Build a DataLoader from config and CLIP processor.
    If for_eval=True and config has eval_dataset_name, use the standard 1k test set for retrieval benchmark.
    """
    batch_size = config.get("batch_size", 32)
    if for_eval and config.get("eval_dataset_name"):
        dataset_name = config["eval_dataset_name"]
        split = split or config.get("eval_split", "test")
        revision = config.get("eval_revision")
    else:
        dataset_name = config["dataset_name"]
        split = split or config.get("split", "test")
        revision = config.get("revision", "refs/convert/parquet")
    dataset = Flickr30kCLIPDataset(dataset_name, split, processor, revision=revision)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=_collate_fn,
    )


def _image_id(row: dict) -> str:
    """Extract a stable identifier for an image (hash of bytes)."""
    img = row["image"]
    return hashlib.sha256(img.tobytes()).hexdigest()


def check_split_disjointness(
    dataset_name: str = "nlphuji/flickr30k",
    eval_dataset_name: str = "nlphuji/flickr_1k_test_image_text_retrieval",
    revision: str = "refs/convert/parquet",
    eval_revision: str | None = None,
) -> bool:
    """
    Verify that train and test splits are disjoint, and that 30k test == 1k benchmark.
    Returns True if checks pass; raises AssertionError otherwise.
    """
    train_ds = load_dataset(dataset_name, split="train", revision=revision)
    test_30k_ds = load_dataset(dataset_name, split="test", revision=revision)
    rev_1k = eval_revision or revision
    test_1k_ds = load_dataset(eval_dataset_name, split="test", revision=rev_1k)

    train_ids = {_image_id(train_ds[i]) for i in range(len(train_ds))}
    test_30k_ids = {_image_id(test_30k_ds[i]) for i in range(len(test_30k_ds))}
    test_1k_ids = {_image_id(test_1k_ds[i]) for i in range(len(test_1k_ds))}

    overlap = train_ids & test_30k_ids
    assert len(overlap) == 0, f"Train and test overlap: {len(overlap)} images. Do not use split='train' for training."
    assert test_30k_ids == test_1k_ids, (
        f"30k test and 1k benchmark differ: 30k has {len(test_30k_ids)}, 1k has {len(test_1k_ids)}, "
        f"symmetric diff size {len(test_30k_ids ^ test_1k_ids)}."
    )
    return True


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


def _run_sanity_check(dataset: Flickr30kCLIPDataset, loader, label: str, out_filename: str, root: str):
    """Load one batch, print shapes, visualize samples, save figure."""
    print(f"\n--- {label} ---")
    print("Dataset size:", len(dataset))
    batch = next(iter(loader))
    print("Batch keys:", batch.keys())
    print("pixel_values shape:", batch["pixel_values"].shape)
    print("input_ids shape:", batch["input_ids"].shape)
    print("attention_mask shape:", batch["attention_mask"].shape)

    import matplotlib.pyplot as plt
    num_show = 3
    fig, axes = plt.subplots(num_show, 1, figsize=(8, 4 * num_show))
    if num_show == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        row = dataset.hf_ds[i]
        img = row["image"]
        caps = row["caption"]
        if not isinstance(caps, list):
            caps = [caps]
        caps = (caps * 5)[:5]
        ax.imshow(img)
        ax.axis("off")
        cap_text = "\n".join(f"  {j+1}. {c}" for j, c in enumerate(caps))
        ax.set_title(f"Sample {i} — captions:\n{cap_text}", fontsize=8, loc="left")
    plt.tight_layout()
    out_path = os.path.join(root, out_filename)
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    print("Saved visualization to", out_path)
    plt.close()


if __name__ == "__main__":
    # Sanity check: load both 30k (train/test) and 1k benchmark; print shapes; visualize.
    # Run from foundations/tiny-clip:  python -m src.data  [--check-splits]
    import argparse
    import sys

    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from transformers import CLIPProcessor
    from src.utils import load_config

    parser = argparse.ArgumentParser(description="Data sanity check")
    parser.add_argument("--check-splits", action="store_true", help="Verify train/test disjointness and 30k test == 1k benchmark")
    args = parser.parse_args()

    config_path = os.path.join(_root, "configs", "flickr30k.yaml")
    config = load_config(config_path)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    if args.check_splits:
        print("Running split disjointness check...")
        check_split_disjointness(
            dataset_name=config["dataset_name"],
            eval_dataset_name=config.get("eval_dataset_name", "nlphuji/flickr_1k_test_image_text_retrieval"),
            revision=config.get("revision", "refs/convert/parquet"),
            eval_revision=config.get("eval_revision"),
        )
        print("Verified: nlphuji/flickr30k uses Karpathy splits; train and 1k test are disjoint.")
        print("Split disjointness check passed.\n")

    # 1. Flickr30k (full dataset, split from config)
    dataset_30k = Flickr30kCLIPDataset(
        config["dataset_name"],
        config["split"],
        processor,
        revision=config.get("revision", "refs/convert/parquet"),
    )
    loader_30k = get_dataloader(config, processor, shuffle=False, for_eval=False)
    _run_sanity_check(
        dataset_30k, loader_30k,
        "Flickr30k (dataset_name, split)",
        "sanity_check_samples.png",
        _root,
    )

    # 2. Flickr30k 1k benchmark (standard retrieval eval set)
    dataset_1k = Flickr30kCLIPDataset(
        config["eval_dataset_name"],
        config.get("eval_split", "test"),
        processor,
        revision=config.get("eval_revision", "refs/convert/parquet"),
    )
    loader_1k = get_dataloader(config, processor, shuffle=False, for_eval=True)
    _run_sanity_check(
        dataset_1k, loader_1k,
        "Flickr30k 1k benchmark (eval_dataset_name)",
        "sanity_check_samples_1k.png",
        _root,
    )

    print("\nSanity check passed (both 30k and 1k).")
