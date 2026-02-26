#!/usr/bin/env python3
"""Verify nlphuji/flickr30k parquet has 'split' column and its values."""
from datasets import load_dataset
from collections import Counter

try:
    print("Loading nlphuji/flickr30k (parquet)...")
    ds = load_dataset("nlphuji/flickr30k", split="test", revision="refs/convert/parquet")

    lines = []
    lines.append("Columns: " + str(ds.column_names))
    lines.append("Has split: " + str("split" in ds.column_names))
    if "split" in ds.column_names:
        vals = ds["split"]
        counts = Counter(vals)
        lines.append("Split value counts: " + str(dict(counts)))
        lines.append("Sample values: " + str(vals[:10]))
    else:
        lines.append("No split column found")

    result = "\n".join(lines)
    print(result)

    with open("verify_split_result.txt", "w") as f:
        f.write(result)

    print("\n✓ Verification complete. Results saved to verify_split_result.txt")
except Exception as e:
    print(f"\n✗ Verification failed: {e}")
    raise
