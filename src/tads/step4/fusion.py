"""Step 4.2: Multi-Domain Joint Matching via equal-weight voting (Appendix D.7)."""

import json
import os
from typing import List

import torch


def run(config):
    """Fuse per-field distribution-matched selections via voting."""
    selected_dir = config.selected_dir
    fields = config.selection.fields
    target_count = config.selection.target_count

    for threshold in config.selection.score_thresholds:
        # Collect IDs from each field's distribution-matched file
        method_results = []
        for field in fields:
            path = config.distribution_pt(field, threshold)
            if not os.path.exists(path):
                print(f"Skipping {field} gte_{threshold} (file not found)")
                continue
            data = torch.load(path, weights_only=False)
            ids = [item["id"] for item in data if "id" in item]
            method_results.append(ids)
            print(f"Loaded {len(ids)} IDs from {field} gte_{threshold}")

        if not method_results:
            continue

        fused_ids = _equal_weight_fusion(*method_results, target_count=target_count)
        output_path = config.fused_json(threshold)
        output_data = {
            "fusion_config": {
                "dimensions": fields,
                "gte_threshold": threshold,
                "total_methods": len(method_results),
                "final_count": len(fused_ids),
            },
            "fused_ids": fused_ids,
        }
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"Fused {len(fused_ids)} IDs -> {output_path}")


def _equal_weight_fusion(*method_results: List[str], target_count: int = 10000) -> List[str]:
    """Equal-weight voting: each method gets 1 vote per item, take top items by votes."""
    item_votes = {}
    for ids in method_results:
        for item_id in ids:
            item_votes[item_id] = item_votes.get(item_id, 0) + 1

    # Sort by votes descending, then ID ascending for deterministic tie-breaking
    voted = sorted(item_votes.items(), key=lambda x: (-x[1], x[0]))

    vote_dist = {}
    for _, v in voted:
        vote_dist[v] = vote_dist.get(v, 0) + 1
    print(f"Vote distribution: {vote_dist}")

    return [item_id for item_id, _ in voted[:target_count]]
