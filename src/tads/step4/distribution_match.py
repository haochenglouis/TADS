"""Step 4.1: Incremental Sampling for Mitigating Domain Shifts (§4.4, Algorithm 1)."""

import json
import os
import random
from collections import Counter, defaultdict

import numpy as np
import torch
from tqdm import tqdm

from tads.utils.data_io import save_samples
from tads.utils.tags import analyze_all_items_from_data


def run(config):
    """Select 10k samples per field via distribution-matching incremental sampling."""
    random.seed(config.random_seed)
    np.random.seed(config.random_seed)

    reference_file = config.cluster_ids_json
    mapping_file = config.extracted_keywords_json
    tag_id_mapping_file = config.tag_id_mapping_json
    filtered_dir = config.filtered_dir
    output_dir = config.selected_dir
    target_count = config.selection.target_count
    fields = config.selection.fields

    # Load reference distribution
    with open(reference_file, "r", encoding="utf-8") as f:
        reference_data = json.load(f)
    print(f"Reference data: {len(reference_data)} samples")

    # Load mappings for tag conversion
    with open(mapping_file, "r", encoding="utf-8") as f:
        keyword_mapping = json.load(f)
    with open(tag_id_mapping_file, "r", encoding="utf-8") as f:
        tag_id_mapping = json.load(f)
    reverse_mapping = _create_reverse_mapping(keyword_mapping, tag_id_mapping)

    # Process each filtered threshold file
    os.makedirs(output_dir, exist_ok=True)
    for threshold in config.selection.score_thresholds:
        file_path = config.filtered_pt(threshold)
        if not os.path.exists(file_path):
            continue
        data = torch.load(file_path, weights_only=False)
        if len(data) <= target_count:
            continue
        print(f"\nThreshold {threshold}: {len(data)} samples")

        for field in fields:
            converted = [_convert_tags(s, reverse_mapping, field) for s in data]
            selected = _sample_to_match_distribution(converted, reference_data, field, target_count)
            out_path = config.distribution_pt(field, threshold)
            metadata = {
                "reference_file": reference_file,
                "source_count": len(data),
                "field": field,
                "threshold": threshold,
                "selected_count": len(selected),
            }
            save_samples(selected, out_path, metadata)

        # Also save a random baseline
        random_samples = random.sample(data, min(target_count, len(data)))
        random_path = os.path.join(output_dir, f"random_10k_gte_{threshold}.pt")
        save_samples(random_samples, random_path, {"threshold": threshold, "method": "random"})


def _sample_to_match_distribution(data, reference_data, field, target_total=10000):
    """Algorithm 1: Distribution Matching via Incremental Sampling."""
    target_counter = analyze_all_items_from_data(reference_data)[field]
    total_tags = sum(target_counter.values())
    if total_tags == 0:
        return random.sample(data, min(target_total, len(data)))
    target_ratio = {tag: cnt / total_tags for tag, cnt in target_counter.items()}

    # Build tag -> candidate index mapping
    idx_to_tags = {}
    tag_to_candidates = defaultdict(list)
    for idx, item in enumerate(data):
        tags = item.get("generated_content", {}).get(field, [])
        if isinstance(tags, str):
            tags = [tags]
        tags = [t for t in tags if t in target_ratio]
        if tags:
            idx_to_tags[idx] = tags
            for tag in tags:
                tag_to_candidates[tag].append(idx)

    selected, selected_set = [], set()
    selected_counter = Counter()

    while len(selected) < target_total:
        current_total = sum(selected_counter.values()) or 1
        current_ratio = {tag: selected_counter[tag] / current_total for tag in target_ratio}

        # Find tag with largest gap
        gap_tag, max_gap = None, -1
        for tag in target_ratio:
            gap = target_ratio[tag] - current_ratio.get(tag, 0)
            if gap > max_gap and tag_to_candidates[tag]:
                gap_tag, max_gap = tag, gap
        if gap_tag is None:
            break

        candidates = [i for i in tag_to_candidates[gap_tag] if i not in selected_set]
        if not candidates:
            del tag_to_candidates[gap_tag]
            continue

        # Prefer samples with fewer tags to minimize pollution
        chosen = min(candidates, key=lambda i: len(idx_to_tags[i]))
        selected_set.add(chosen)
        selected.append(data[chosen])
        for tag in idx_to_tags[chosen]:
            selected_counter[tag] += 1

    print(f"  {field}: selected {len(selected)} samples")
    return selected


def _create_reverse_mapping(keyword_mapping, tag_id_mapping):
    reverse = {}
    for abstract_tag, long_text in tag_id_mapping.items():
        reverse[long_text] = abstract_tag
        if abstract_tag in keyword_mapping:
            reverse[keyword_mapping[abstract_tag]] = abstract_tag
    return reverse


def _convert_tags(sample, reverse_mapping, field):
    content = sample.get("generated_content", {})
    if field not in content:
        return sample
    original = content[field]
    if not original:
        return sample
    if isinstance(original, str):
        original = [original]
    converted = []
    for tag in original:
        if isinstance(tag, str) and tag in reverse_mapping:
            converted.append(reverse_mapping[tag])
        else:
            converted.append(tag)
    out = sample.copy()
    out["generated_content"] = content.copy()
    out["generated_content"][field] = converted
    return out
