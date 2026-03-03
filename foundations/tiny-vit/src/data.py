"""Data loading for tiny-vit (CIFAR-10, etc.)."""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ImageNet stats (for pretrained ViT from timm)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# CIFAR-10 stats (for training from scratch on CIFAR-10)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def get_cifar10_transform(image_size: int = 224, normalization: str = "imagenet"):
    """Transform for ViT on CIFAR-10 (resize to 224x224). Use 'imagenet' for pretrained ViT."""
    if normalization == "imagenet":
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    else:
        mean, std = CIFAR10_MEAN, CIFAR10_STD
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def get_dataloader(config: dict, split: str = "train") -> DataLoader:
    """Build DataLoader from config."""
    dataset_name = config.get("dataset_name", "cifar10")
    batch_size = config.get("batch_size", 32)
    image_size = config.get("image_size", 224)
    normalization = config.get("normalization", "imagenet")

    transform = get_cifar10_transform(image_size, normalization)

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
