#!/usr/bin/env python3
"""
Step 6A: Run pretrained CLIP on custom images — retrieve top-k captions from Flickr30k 1k test set.
Usage (from foundations/tiny-clip): python demos/demo_custom_images.py [--debug]
"""
import argparse
import os
import sys

# Add project root (needed for both debug and main)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

# Config (no heavy imports yet — debug mode uses only stdlib)
CUSTOM_IMAGES_DIR = os.path.join(ROOT, "demos", "custom_images")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG", ".WEBP"}
TOP_K = 5
OUTPUT_DIR = os.path.join(ROOT, "demos", "output")
OUTPUT_FIG = os.path.join(OUTPUT_DIR, "demo_results.png")


def find_custom_images(dirpath: str) -> list[tuple[str, str]]:
    """Return [(full_path, basename), ...] for image files, sorted by name."""
    if not os.path.isdir(dirpath):
        return []
    pairs = []
    for f in os.listdir(dirpath):
        ext = os.path.splitext(f)[1]
        if ext in IMAGE_EXTENSIONS:
            full = os.path.join(dirpath, f)
            if os.path.isfile(full):
                pairs.append((full, f))
    return sorted(pairs, key=lambda x: x[1])


def debug_discovery(dirpath: str) -> None:
    """Print diagnostic info for image discovery (used with --debug or when no images found)."""
    print("--- Image discovery debug ---")
    print(f"  ROOT (script parent): {ROOT}")
    print(f"  CUSTOM_IMAGES_DIR:    {dirpath}")
    print(f"  dir exists:           {os.path.exists(dirpath)}")
    print(f"  dir isdir:            {os.path.isdir(dirpath)}")
    if os.path.isdir(dirpath):
        all_files = os.listdir(dirpath)
        print(f"  all files ({len(all_files)}): {sorted(all_files)}")
        for f in sorted(all_files):
            ext = os.path.splitext(f)[1]
            full = os.path.join(dirpath, f)
            is_img = ext in IMAGE_EXTENSIONS
            is_file = os.path.isfile(full)
            status = "OK" if (is_img and is_file) else ("skip: not image" if not is_img else "skip: not file")
            print(f"    {f}: ext={ext!r} -> {status}")
    print("  Supported extensions:", sorted(IMAGE_EXTENSIONS))
    print("---")


def load_caption_pool(revision: str = "refs/convert/parquet") -> list[str]:
    """Load all captions from Flickr30k 1k test set (5000 captions)."""
    from datasets import load_dataset
    ds = load_dataset("nlphuji/flickr_1k_test_image_text_retrieval", split="test", revision=revision)
    captions = []
    for row in ds:
        caps = row["caption"]
        if isinstance(caps, str):
            caps = [caps]
        captions.extend((caps * 5)[:5])
    return captions


def main():
    parser = argparse.ArgumentParser(description="Run CLIP on custom images, retrieve top-k captions.")
    parser.add_argument("--debug", action="store_true", help="Print image discovery diagnostics and exit.")
    args = parser.parse_args()

    if args.debug:
        debug_discovery(CUSTOM_IMAGES_DIR)
        return 0

    # Heavy imports only when not in debug mode
    import matplotlib.pyplot as plt
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    images = find_custom_images(CUSTOM_IMAGES_DIR)
    if not images:
        print(f"No images found in {CUSTOM_IMAGES_DIR}")
        debug_discovery(CUSTOM_IMAGES_DIR)
        print("Add 1–3 jpg/png images, then run again.")
        print("Example: photo1.jpg, photo2.png")
        return 1

    print(f"Found {len(images)} image(s): {[b for _, b in images]}")
    print("Loading CLIP and caption pool...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    captions = load_caption_pool()
    print(f"Caption pool: {len(captions)} captions from Flickr30k 1k test set")

    # Encode all captions
    with torch.no_grad():
        cap_inputs = processor(text=captions, return_tensors="pt", padding=True, truncation=True, max_length=77)
        cap_inputs = {k: v.to(device) for k, v in cap_inputs.items()}
        cap_embeds = model.get_text_features(**cap_inputs)
        cap_embeds = cap_embeds / cap_embeds.norm(dim=-1, keepdim=True)

    n_images = len(images)
    ncols = min(n_images, 3)
    nrows = (n_images + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    if n_images == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for idx, (img_path, basename) in enumerate(images):
        pil_img = Image.open(img_path).convert("RGB")
        inputs = processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            img_embed = model.get_image_features(**inputs)
            img_embed = img_embed / img_embed.norm(dim=-1, keepdim=True)
            sim = (img_embed @ cap_embeds.T).squeeze(0)
            topk_indices = torch.topk(sim, TOP_K).indices.cpu().tolist()

        top_captions = [captions[i] for i in topk_indices]
        ax = axes[idx]
        ax.imshow(pil_img)
        ax.axis("off")
        ax.set_title(basename, fontsize=10)
        caption_lines = []
        for i, c in enumerate(top_captions):
            s = f"{i+1}. {c[:55]}..." if len(c) > 58 else f"{i+1}. {c}"
            caption_lines.append(s)
        ax.text(0.5, -0.05, f"Top-{TOP_K} retrieved:\n" + "\n".join(caption_lines),
                transform=ax.transAxes, fontsize=7, verticalalignment="top", horizontalalignment="center")

    for j in range(len(images), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(OUTPUT_FIG, dpi=120, bbox_inches="tight")
    print(f"Saved: {OUTPUT_FIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
