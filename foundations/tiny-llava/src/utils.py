"""Utilities for tiny-llava."""
import yaml


def load_config(path: str) -> dict:
    """Load YAML config."""
    with open(path) as f:
        return yaml.safe_load(f)
