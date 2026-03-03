"""Data loading for tiny-vit (CIFAR-10, etc.)."""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_cifar10_transform(image_size: int = 224):
    """Standard transform for ViT on CIFAR-10 (resize to 224x224)."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])


def get_dataloader(config: dict, split: str = "train") -> DataLoader:
    """Build DataLoader from config."""
    dataset_name = config.get("dataset_name", "cifar10")
    batch_size = config.get("batch_size", 32)
    image_size = config.get("image_size", 224)

    transform = get_cifar10_transform(image_size)

    if dataset_name == "cifar10":
        train = split == "train"
        dataset = datasets.CIFAR10(
            root="./data",
            train=train,
            download=True,
            transform=transform,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=0)
