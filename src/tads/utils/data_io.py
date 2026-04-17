"""Data I/O utilities shared across pipeline steps."""

import json
import os

import torch
from datasets import load_dataset


def load_data(file_path: str) -> list:
    """Load data from .json, .parquet, or .pt files.

    Returns a list of dicts (one per sample).
    """
    print(f"Loading data from {file_path}...")
    ext = os.path.splitext(file_path)[-1].lower()

    if ext == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif ext == ".parquet":
        dataset = load_dataset("parquet", data_files=file_path)["train"]
        data = dataset.to_list()
    elif ext == ".pt":
        data = torch.load(file_path, weights_only=False)
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        if isinstance(data, torch.Tensor):
            data = data.tolist()
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    print(f"Loaded {len(data)} items.")
    return data


def convert_alpaca_to_string(sample: dict) -> str:
    """Convert an Alpaca-format sample (with 'messages' field) to a flat string."""
    return "\n".join(f"{msg['role']}: {msg['content']}" for msg in sample["messages"])


def save_samples(samples: list, output_path: str, metadata: dict = None):
    """Save samples as .pt with optional JSON metadata sidecar."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(samples, output_path)

    if metadata:
        metadata_path = output_path.replace(".pt", "_metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(samples)} samples to: {output_path}")
