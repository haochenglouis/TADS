"""Step 1: Generating Proxy-Labels by LLM (§4.1, §5.1).

Usage:
    python -m tads.step1                          # run all substeps
    python -m tads.step1 --substep split          # split eval datasets
    python -m tads.step1 --substep merge          # merge target sets
    python -m tads.step1 --substep annotate       # annotate with LLM
"""

import argparse

from tads.config import load_config
from tads.step1 import annotate, merge_targets, split_datasets

SUBSTEPS = {
    "split": split_datasets.run,
    "merge": merge_targets.run,
    "annotate": annotate.run,
}


def main():
    parser = argparse.ArgumentParser(description="Step 1: Generate proxy labels")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--substep", choices=[*SUBSTEPS, "all"], default="all")
    parser.add_argument("--data-dir", default=None, help="Override data directory")
    args = parser.parse_args()

    config = load_config(args.config, data_dir=args.data_dir)

    if args.substep == "all":
        for name, fn in SUBSTEPS.items():
            print(f"\n{'='*60}\nStep 1 — {name}\n{'='*60}")
            fn(config)
    else:
        SUBSTEPS[args.substep](config)


if __name__ == "__main__":
    main()
