"""Data loading for tiny-flamingo (image-caption pairs)."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# ImageNet norm for ViT
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_PROMPT = "Describe this image."


def get_image_transform(image_size: int = 224):
    """ViT preprocessing: resize, normalize."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class ImageCaptionDataset(Dataset):
    """Dataset that returns (pixel_values, input_ids, attention_mask, labels)."""

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        tokenizer=None,
        image_size: int = 224,
        max_length: int = 128,
        prompt: str = DEFAULT_PROMPT,
        revision: str | None = None,
        subset_split: str | None = None,
    ):
        from datasets import load_dataset

        self.tokenizer = tokenizer
        self.image_size = image_size
        self.max_length = max_length
        self.prompt = prompt
        self.transform = get_image_transform(image_size)

        # Load dataset
        if "coco" in dataset_name.lower():
            ds = load_dataset("HuggingFaceM4/COCO", "2017", split=split)
            self.data = ds
            self.image_key = "image"
            self.caption_key = "sentences"
        else:
            # nlphuji/flickr30k parquet: HF only exposes split="test"; filter by internal "split" column
            ds = load_dataset(dataset_name, split="test", revision=revision or "refs/convert/parquet")
            if subset_split is not None and "split" in ds.column_names:
                ds = ds.filter(lambda x: x["split"] == subset_split)
            self.data = ds
            self.image_key = "image"
            self.caption_key = "caption" if "caption" in ds.column_names else "sentence"

    def __len__(self):
        return len(self.data)

    def _get_caption(self, idx):
        row = self.data[idx]
        cap = row.get(self.caption_key)
        if cap is None:
            return ""
        if isinstance(cap, list):
            cap = cap[0] if isinstance(cap[0], str) else (cap[0]["caption"] if isinstance(cap[0], dict) else cap[0])
        return str(cap).strip()

    def __getitem__(self, idx):
        row = self.data[idx]
        image = row[self.image_key]
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB") if isinstance(image, (str, bytes)) else image
        caption = self._get_caption(idx)

        pixel_values = self.transform(image)

        # Format: prompt + " " + caption
        text = f"{self.prompt} {caption}"
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        # Labels: next-token prediction. labels[i] = input_ids[i+1]. Use -100 for prompt.
        prompt_enc = self.tokenizer(
            self.prompt,
            add_special_tokens=False,
            return_tensors="pt",
        )
        prompt_len = prompt_enc["input_ids"].shape[1]
        labels = torch.full_like(input_ids, -100)
        labels[:-1] = input_ids[1:].clone()
        labels[: prompt_len - 1] = -100

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def get_dataloader(config: dict, split: str = "train"):
    """Build DataLoader from config.
    For nlphuji/flickr30k parquet: loads HF split 'test', filters by internal split (train/val/test).
    """
    from transformers import AutoTokenizer

    dataset_name = config.get("dataset_name", "nlphuji/flickr30k")
    revision = config.get("revision", "refs/convert/parquet")
    batch_size = config.get("batch_size", 4)
    max_length = config.get("max_length", 128)
    image_size = config.get("image_size", 224)
    llm_model = config.get("llm_model", "gpt2")

    # Parquet: HF only has "test"; use subset_split to get train/val
    subset_split = split if ("flickr30k" in dataset_name.lower() and revision) else None

    tokenizer = AutoTokenizer.from_pretrained(llm_model)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = ImageCaptionDataset(
        dataset_name=dataset_name,
        split=split,
        tokenizer=tokenizer,
        image_size=image_size,
        max_length=max_length,
        revision=revision,
        subset_split=subset_split,
    )

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=0,
    )
