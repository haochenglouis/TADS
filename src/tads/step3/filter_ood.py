"""Step 3.3: Filter out-of-distribution samples by score thresholds (§4.3)."""

import os

import torch


def run(config):
    """Filter scored samples at each threshold, save to filtered/ directory."""
    input_file = config.scored_data_pt
    output_dir = config.filtered_dir
    thresholds = config.selection.score_thresholds

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading scored data from {input_file}")
    data = torch.load(input_file, weights_only=False)
    print(f"Total samples: {len(data)}")

    print("=" * 60)
    for threshold in thresholds:
        filtered = _filter_all_scores_gte(data, threshold)
        out_path = config.filtered_pt(threshold)
        torch.save(filtered, out_path)
        pct = len(filtered) / len(data) * 100 if data else 0
        print(f"  Threshold {threshold:2d}: {len(filtered):>7,} samples ({pct:5.1f}%) -> {out_path}")
    print("=" * 60)


def _filter_all_scores_gte(data: list, threshold: int) -> list:
    """Keep samples where ALL score values >= threshold."""
    filtered = []
    for sample in data:
        for key in sample:
            if "score" not in key.lower():
                continue
            scores = sample[key]
            if not isinstance(scores, dict):
                continue
            vals = [v for k, v in scores.items() if isinstance(v, (int, float)) and not k.startswith("_")]
            if vals and all(v >= threshold for v in vals):
                filtered.append(sample)
                break
    return filtered
