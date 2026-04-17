"""Step 4: Incremental Sampling for Mitigating Domain Shifts (§4.4, Appendix D.7).

Usage:
    python -m tads.step4                       # run all substeps
    python -m tads.step4 --substep match       # distribution matching
    python -m tads.step4 --substep fuse        # multi-domain fusion
"""

import argparse

from tads.config import load_config
from tads.step4 import distribution_match, fusion

SUBSTEPS = {
    "match": distribution_match.run,
    "fuse": fusion.run,
}


def main():
    parser = argparse.ArgumentParser(description="Step 4: Selection and fusion")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--substep", choices=[*SUBSTEPS, "all"], default="all")
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config, data_dir=args.data_dir)

    if args.substep == "all":
        for name, fn in SUBSTEPS.items():
            print(f"\n{'='*60}\nStep 4 — {name}\n{'='*60}")
            fn(config)
    else:
        SUBSTEPS[args.substep](config)


if __name__ == "__main__":
    main()
