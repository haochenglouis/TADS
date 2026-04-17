"""Step 2.2: Pre-compute and cache BGE-M3 embeddings for the training set (§4.2)."""

import json
import os

import torch
from tqdm import tqdm

from tads.utils.data_io import convert_alpaca_to_string, load_data
from tads.utils.embeddings import encode_texts, load_embedding_model
from tads.utils.tags import (
    analyze_all_items_from_data,
    load_and_restore_tags_from_ids,
    normalize_tag,
)


def run(config):
    """Encode training data and tag vocabularies, save to cache directory."""
    cluster_ids_json = config.cluster_ids_json
    tag_mapping_json = config.tag_id_mapping_json
    train_data_path = config.train_data_parquet
    save_dir = config.train_embeds_dir
    emb_cfg = config.embedding

    os.makedirs(save_dir, exist_ok=True)

    # Load tag statistics from target set
    dedup_data = load_and_restore_tags_from_ids(cluster_ids_json, tag_mapping_json)
    all_items = analyze_all_items_from_data(dedup_data)

    # Load training data
    data_train = load_data(train_data_path)

    # Load embedding model
    tokenizer, model, device = load_embedding_model(emb_cfg.name)

    # Encode tag vocabularies per field
    field_tag_texts = {}
    field_tag_embeddings = {}
    for field in ["Topic", "Style", "Audience", "Task"]:
        tags = list(all_items[field].keys())
        print(f"Encoding {field}: {len(tags)} tags")
        embeds = encode_texts(tags, tokenizer, model, device, max_length=emb_cfg.max_length)
        field_tag_texts[field] = tags
        field_tag_embeddings[field] = embeds

    # Build text samples (one per field per training sample)
    text_samples = []
    for sample_idx, sample in enumerate(data_train):
        if "messages" not in sample:
            continue
        text_input = convert_alpaca_to_string(sample)
        for field in ["Topic", "Style", "Audience", "Task"]:
            text_samples.append((text_input, field))

    texts_only = [text for text, _ in text_samples]

    # Deduplicate before encoding
    print(f"\nOriginal text count: {len(texts_only)}")
    unique_texts = []
    text_to_unique_idx = {}
    original_to_unique_idx = []
    for text in texts_only:
        if text not in text_to_unique_idx:
            text_to_unique_idx[text] = len(unique_texts)
            unique_texts.append(text)
        original_to_unique_idx.append(text_to_unique_idx[text])

    dedup_ratio = (1 - len(unique_texts) / len(texts_only)) * 100
    print(f"Unique text count: {len(unique_texts)} (dedup ratio: {dedup_ratio:.2f}%)")

    unique_embeddings = encode_texts(
        unique_texts, tokenizer, model, device,
        batch_size=emb_cfg.batch_size, max_length=emb_cfg.max_length,
    )
    text_embeddings = unique_embeddings[original_to_unique_idx]
    print(f"Final embeddings shape: {text_embeddings.shape}")

    # Save all artifacts
    torch.save(text_embeddings.cpu(), os.path.join(save_dir, "text_embeddings.pt"))
    with open(os.path.join(save_dir, "text_samples.json"), "w", encoding="utf-8") as f:
        json.dump(text_samples, f, ensure_ascii=False, indent=2)

    for field in ["Topic", "Style", "Audience", "Task"]:
        torch.save(
            field_tag_embeddings[field].cpu(),
            os.path.join(save_dir, f"{field.lower()}_tag_embeddings.pt"),
        )
        with open(os.path.join(save_dir, f"{field.lower()}_tag_texts.json"), "w", encoding="utf-8") as f:
            json.dump(field_tag_texts[field], f, ensure_ascii=False, indent=2)

    print(f"\nAll embeddings saved to {save_dir}")
