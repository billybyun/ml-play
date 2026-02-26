# Verification: nlphuji/flickr30k parquet "split" column

## Summary: **Verified ✓**

The suggestion to filter by the internal `split` column is **correct**. The nlphuji/flickr30k parquet dataset has a `split` column with train/val/test values.

## Actual results (from verify_split.py)

- **Columns**: image, caption, sentids, split, img_id, filename
- **Has split**: True
- **Split value counts**: train=29000, test=1000, val=1014
- **Sample values**: ['train', 'train', 'train', ...]

## What verify_split.py checks vs --check-splits

| Check | verify_split.py | --check-splits |
|-------|-----------------|----------------|
| Split column exists | ✓ | — |
| Train/val/test counts | ✓ | — |
| Train ∩ test = ∅ (disjoint) | — | ✓ |
| 30k test == 1k benchmark | — | ✓ |

**verify_split.py** confirms the data structure. **--check-splits** (in data.py) verifies integrity (disjointness, benchmark alignment) — but it must be updated to use the filter approach first.

## Local verification

Run from project root or `foundations/tiny-clip` (requires `conda activate ml-play`):

```bash
python foundations/tiny-clip/verify_split.py
```

## Implementation (done)

- **check_split_disjointness:** Loads HF `"test"`, filters by internal `split` to get train/test, verifies disjointness and 1k benchmark alignment.
- **Flickr30kCLIPDataset / Flickr30kDualEncoderDataset:** Accept `subset_split`; when provided, filter loaded data by `row["split"] == subset_split`.
- **get_dataloader / get_dataloader_dual_encoder:** For nlphuji/flickr30k parquet, pass `split="test"` (HF) and `subset_split=config["train_split"]` (internal) for training.
