# Task-Aware Data Selection via Proxy-Label Enhanced Distribution Matching for LLM Finetuning

This code is a PyTorch implementation of our ICLR'26 paper "Task-Aware Data Selection via Proxy-Label Enhanced Distribution Matching for LLM Finetuning". [[paper]](https://openreview.net/pdf?id=R40WoYbYab)

## Overview

The pipeline consists of 4 main steps:
1. **Step 1** (§4.1, §5.1): Dataset splitting and target annotation with proxy labels
2. **Step 2** (§4.2): Proxy-label clustering and propagation to source data
3. **Step 3** (§4.3): Keyword extraction, sample scoring, and OOD filtering
4. **Step 4** (§4.4, Appendix D.7): Incremental sampling for distribution matching and multi-domain fusion

## Installation

```bash
pip install -e .
```

## Dataset Preparation

> **Note:** All scripts and commands use paths relative to the repository root, and `data_dir` is resolved against the current working directory. Always run the data-preparation scripts and pipeline commands from the repository root directory.

```bash
# Download evaluation data
bash data/prepare_eval_data.sh

# Download training data
bash data/prepare_train_data.sh
```

## Running the Pipeline

### Full pipeline

```bash
bash scripts/run_pipeline.sh
```

### Step-by-step

```bash
# Step 1: Generate Proxy-Labels
python -m tads.step1                          # run all substeps
python -m tads.step1 --substep split          # split eval datasets (20:80)
python -m tads.step1 --substep merge          # merge target sets into parquet
python -m tads.step1 --substep annotate       # annotate with Qwen2.5-7B-Instruct

# Step 2: Clustering and Propagation
python -m tads.step2                          # run all substeps
python -m tads.step2 --substep cluster        # K-means clustering on tags
python -m tads.step2 --substep cache          # cache BGE-M3 embeddings
python -m tads.step2 --substep propagate      # propagate tags to training set

# Step 3: OOD Filtering
python -m tads.step3                          # run all substeps
python -m tads.step3 --substep extract        # extract keywords from anchors
python -m tads.step3 --substep score          # LLM-based quality scoring
python -m tads.step3 --substep filter         # threshold-based OOD filtering

# Step 4: Selection and Fusion
python -m tads.step4                          # run all substeps
python -m tads.step4 --substep match          # incremental distribution matching (Algorithm 1)
python -m tads.step4 --substep fuse           # multi-domain equal-weight voting fusion
```

### Custom configuration

```bash
python -m tads.step1 --config configs/default.yaml --data-dir /path/to/data
python -m tads.step3 --substep score --batch-size 128
```

## File Structure

```
TADS/
├── configs/
│   └── default.yaml                 # Centralized pipeline configuration
├── src/
│   └── tads/
│       ├── config.py                # Configuration dataclasses and loader
│       ├── vllm_engine.py           # Unified vLLM initialization
│       ├── utils/
│       │   ├── data_io.py           # Data loading/saving (json, parquet, pt)
│       │   ├── tags.py              # Tag normalization, analysis, long-tail detection
│       │   └── embeddings.py        # BGE-M3 encoding and top-k matching
│       ├── step1/                   # §4.1, §5.1: Proxy-label generation
│       │   ├── split_datasets.py    # Split eval datasets 20:80
│       │   ├── merge_targets.py     # Merge target sets to Alpaca format
│       │   └── annotate.py          # LLM annotation (Task/Style/Topic/Audience)
│       ├── step2/                   # §4.2: Clustering and propagation
│       │   ├── cluster_tags.py      # K-means clustering with FAISS
│       │   ├── cache_embeddings.py  # Pre-compute BGE-M3 embeddings
│       │   └── propagate_tags.py    # Assign anchors to training samples
│       ├── step3/                   # §4.3: OOD filtering
│       │   ├── extract_keywords.py  # Extract representative keywords per anchor
│       │   ├── score_samples.py     # LLM-based quality scoring
│       │   └── filter_ood.py        # Threshold filtering
│       └── step4/                   # §4.4, Appendix D.7: Selection
│           ├── distribution_match.py # Incremental sampling (Algorithm 1)
│           └── fusion.py            # Multi-domain voting fusion
├── scripts/
│   └── run_pipeline.sh             # Run full pipeline
├── data/                           # Data directory
│   ├── eval/                       # Evaluation datasets
│   └── prepare_*_data.sh           # Download scripts
├── consistency_precision_result/   # Annotation quality evaluation results
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Finetune & Eval

The finetune & eval of TADS is based on [open-instruct](https://github.com/allenai/open-instruct).

## Citation

```bibtex
@inproceedings{cheng2026tads,
  title={Task-Aware Data Selection via Proxy-Label Enhanced Distribution Matching for LLM Finetuning},
  author={Cheng, Hao and Zhang, Rui and Li, Ling and Di, Na and Wei, Jiaheng and Zhu, Zhaowei and Han, Bo},
  booktitle={International Conference on Learning Representations},
  year={2026}
}
```
