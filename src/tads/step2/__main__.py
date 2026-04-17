"""Step 2: Proxy-Label Clustering and Propagation (§4.2).

Usage:
    python -m tads.step2                           # run all substeps
    python -m tads.step2 --substep cluster         # cluster target tags
    python -m tads.step2 --substep cache           # cache training set embeddings
    python -m tads.step2 --substep propagate       # propagate tags to training set
"""

import argparse

from tads.config import load_config
from tads.step2 import cache_embeddings, cluster_tags, propagate_tags

SUBSTEPS = {
    "cluster": cluster_tags.run,
    "cache": cache_embeddings.run,
    "propagate": propagate_tags.run,
}


def main():
    parser = argparse.ArgumentParser(description="Step 2: Clustering and propagation")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--substep", choices=[*SUBSTEPS, "all"], default="all")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config, data_dir=args.data_dir)

    if args.substep == "all":
        for name, fn in SUBSTEPS.items():
            print(f"\n{'='*60}\nStep 2 — {name}\n{'='*60}")
            fn(config)
    else:
        SUBSTEPS[args.substep](config)


if __name__ == "__main__":
    main()
