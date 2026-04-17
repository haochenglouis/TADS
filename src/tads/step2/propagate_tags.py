"""Step 2.3: Propagate clustered tags to training set via semantic similarity (§4.2)."""

import copy
import json
import os
from collections import defaultdict

import torch
from tqdm import tqdm

from tads.utils.data_io import convert_alpaca_to_string, load_data
from tads.utils.embeddings import encode_texts, load_embedding_model, match_to_tags_topk
from tads.utils.tags import (
    analyze_all_items_from_data,
    load_and_restore_tags_from_ids,
    normalize_tag,
)


def run(config):
    """Assign top-k cluster tags to each training sample using cached embeddings."""
    cluster_ids_json = config.cluster_ids_json
    tag_mapping_json = config.tag_id_mapping_json
    train_data_path = config.train_data_parquet
    output_file = config.train_with_tags_pt
    embeds_dir = config.train_embeds_dir
    emb_cfg = config.embedding
    top_k = config.clustering.top_k
    fields = ["Topic", "Style", "Audience", "Task"]

    # Load mappings
    with open(tag_mapping_json, "r", encoding="utf-8") as f:
        tag_id_to_text = json.load(f)
    text_to_tag_id = {normalize_tag(v): k for k, v in tag_id_to_text.items()}

    dedup_data = load_and_restore_tags_from_ids(cluster_ids_json, tag_mapping_json)
    all_items = analyze_all_items_from_data(dedup_data)

    # Load training data and cached embeddings
    data_train = load_data(train_data_path)
    text_embeddings = torch.load(
        os.path.join(embeds_dir, "text_embeddings.pt"), weights_only=False
    ).to("cuda")

    # Build sample-to-field index
    text_samples = []
    sample_info = []
    for sample_idx, sample in enumerate(data_train):
        if "messages" not in sample:
            continue
        text_input = convert_alpaca_to_string(sample)
        for field in fields:
            text_samples.append((text_input, field))
            sample_info.append((sample_idx, field))

    assert len(text_samples) == len(text_embeddings), \
        f"Mismatch: {len(text_samples)} samples vs {len(text_embeddings)} embeddings"

    # Encode tags
    tokenizer, model, device = load_embedding_model(emb_cfg.name)
    field_tag_texts, field_tag_embeddings = {}, {}
    for field in fields:
        tags = list(all_items[field].keys())
        print(f"Encoding {field}: {len(tags)} tags")
        embeddings = encode_texts(tags, tokenizer, model, device, max_length=emb_cfg.max_length)
        field_tag_texts[field] = tags
        field_tag_embeddings[field] = embeddings.to(device)

    # Match top-k tags per field
    field_to_indices = defaultdict(list)
    for i, (_, field) in enumerate(text_samples):
        field_to_indices[field].append(i)

    best_tags_all = [None] * len(text_samples)
    for field in fields:
        indices = field_to_indices[field]
        if not indices:
            continue
        field_text_embeds = text_embeddings[indices]
        tags_k, _ = match_to_tags_topk(
            field_text_embeds, field_tag_embeddings[field],
            field_tag_texts[field], top_k=top_k,
        )
        for idx, tags in zip(indices, tags_k):
            best_tags_all[idx] = tags

    # Map to tag IDs and merge into samples
    sample_mapped = [{} for _ in range(len(data_train))]
    for (sample_idx, field), tag_list in zip(sample_info, best_tags_all):
        if tag_list is None:
            continue
        for tag in tag_list:
            tag_norm = normalize_tag(tag)
            tag_id = text_to_tag_id.get(tag_norm, f"{field}_UNK_{tag_norm}")
            sample_mapped[sample_idx].setdefault(field, []).append(tag_id)

    final_output = []
    for idx, mapping in enumerate(sample_mapped):
        sample = copy.deepcopy(data_train[idx])
        sample.setdefault("generated_content", {})
        for field in fields:
            if field in sample["generated_content"]:
                sample["generated_content"][f"raw_{field}"] = sample["generated_content"][field]
            if field in mapping:
                # Task: single value (top-1), others: list (top-k)
                sample["generated_content"][field] = mapping[field][0] if field == "Task" else mapping[field]
        final_output.append(sample)

    # Save
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    torch.save(final_output, output_file)
    print(f"Saved {len(final_output)} samples with propagated tags to {output_file}")
