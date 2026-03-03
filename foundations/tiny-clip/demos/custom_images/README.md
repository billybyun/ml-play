# Custom images for zero-shot demo (Step 6)

Put 1–3 images here for the zero-shot retrieval demo. Use only images you're comfortable sharing in the repo.

## What you need

### Option A: Images only (retrieve captions from Flickr30k)

- **Input:** Image files (`.jpg`, `.png`, etc.)
- **Output:** For each image, top-k retrieved captions from the 1k test set
- **Captions needed?** No. The demo uses the Flickr30k 1k benchmark as the caption pool.

### Option B: Images + hand-written captions (test similarity)

- **Input:** Image files + a text file with your own captions to test against
- **Output:** Similarity scores between each image and each caption
- **Captions needed?** Yes. Plain text, one caption per line, or in a simple format (e.g. `image_name: caption`).

### Option 6b: Personal mini-retrieval demo

- **Input:** 1–10 images + 1–3 captions per image (your ground truth)
- **Format:** Either:
  - A folder of images + a JSON/text file mapping image filename → list of captions, or
  - One caption per line in a `.txt` file named after the image (e.g. `photo1.txt` with 3 lines = 3 captions for `photo1.jpg`)

## Suggested layout (Step 6A)

```
demos/custom_images/
  photo1.jpg            # or photo1.png — any name works
  photo2.jpg
  photo3.png
  captions.txt          # optional: hand-written captions for Option B (one per line)
  # For 6b: photo1_captions.txt with one caption per line, etc.
```

**Filenames:** Use any descriptive names (e.g. `beach.jpg`, `dog.png`). The script lists images alphabetically.

## Caption format (when needed)

- **Plain text**, UTF-8
- One caption per line, or one caption per file
- No special formatting; CLIP will tokenize them
