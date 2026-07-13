"""Materialize a TADS selection into a finetuning train file.

Step 4 of the selection pipeline writes data/selected/fused_10k_gte_<t>.json,
which contains only the selected sample IDs ("fused_ids"). This script joins
those IDs back against the training parquet and emits a JSON list of
{"dataset", "id", "messages"} records, the format consumed by finetune.py
via --train_file.

Usage (from model_finetune/):
    python export_train_file.py --threshold 5
    python export_train_file.py --fused-json ../data/selected/fused_10k_gte_5.json \
        --train-parquet ../data/train-00000-of-00001.parquet \
        --output selected_data/tads_10k_gte_5_dataset.json
"""

import argparse
import json
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Export selected samples for finetuning")
    parser.add_argument("--threshold", type=int, default=None,
                        help="Score threshold t; shorthand for --fused-json <data-dir>/selected/fused_10k_gte_<t>.json")
    parser.add_argument("--data-dir", default="../data",
                        help="TADS data directory (used with --threshold)")
    parser.add_argument("--fused-json", default=None,
                        help="Path to a fused_10k_gte_<t>.json produced by step 4")
    parser.add_argument("--train-parquet", default=None,
                        help="Training pool parquet (default: <data-dir>/train-00000-of-00001.parquet)")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: selected_data/tads_10k_gte_<t>_dataset.json)")
    args = parser.parse_args()

    if args.fused_json is None:
        if args.threshold is None:
            parser.error("provide either --threshold or --fused-json")
        args.fused_json = os.path.join(args.data_dir, "selected", f"fused_10k_gte_{args.threshold}.json")
    if args.train_parquet is None:
        args.train_parquet = os.path.join(args.data_dir, "train-00000-of-00001.parquet")
    if args.output is None:
        tag = f"gte_{args.threshold}" if args.threshold is not None else \
            os.path.splitext(os.path.basename(args.fused_json))[0].replace("fused_10k_", "")
        args.output = os.path.join("selected_data", f"tads_10k_{tag}_dataset.json")

    with open(args.fused_json, encoding="utf-8") as f:
        fused_ids = json.load(f)["fused_ids"]
    print(f"Loaded {len(fused_ids)} selected IDs from {args.fused_json}")

    df = pd.read_parquet(args.train_parquet, columns=["dataset", "id", "messages"])
    print(f"Loaded training pool with {len(df)} samples from {args.train_parquet}")

    # First occurrence wins for duplicated IDs in the pool.
    pool = {}
    for dataset, sample_id, messages in df.itertuples(index=False):
        pool.setdefault(sample_id, (dataset, messages))

    missing = [i for i in fused_ids if i not in pool]
    if missing:
        print(f"WARNING: {len(missing)} selected IDs not found in the training pool "
              f"(first few: {missing[:5]})")

    records = []
    for sample_id in fused_ids:
        if sample_id in pool:
            dataset, messages = pool[sample_id]
            records.append({
                "dataset": dataset,
                "id": sample_id,
                "messages": [dict(m) for m in messages],
            })

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)
    print(f"Wrote {len(records)} samples to {args.output}")


if __name__ == "__main__":
    main()
