"""Step 3: Filtering OOD Samples with Label Noise (§4.3).

Usage:
    python -m tads.step3                           # run all substeps
    python -m tads.step3 --substep extract         # extract keywords from anchors
    python -m tads.step3 --substep score           # score training samples
    python -m tads.step3 --substep filter          # filter OOD samples
"""

import argparse

from tads.config import load_config
from tads.step3 import extract_keywords, filter_ood, score_samples

SUBSTEPS = {
    "extract": extract_keywords.run,
    "score": score_samples.run,
    "filter": filter_ood.run,
}


def main():
    parser = argparse.ArgumentParser(description="Step 3: OOD filtering")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--substep", choices=[*SUBSTEPS, "all"], default="all")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config, data_dir=args.data_dir)
    if args.batch_size:
        config.keyword_extraction.batch_size = args.batch_size
        config.scoring.batch_size = args.batch_size

    if args.substep == "all":
        for name, fn in SUBSTEPS.items():
            print(f"\n{'='*60}\nStep 3 — {name}\n{'='*60}")
            fn(config)
    else:
        SUBSTEPS[args.substep](config)


if __name__ == "__main__":
    main()
