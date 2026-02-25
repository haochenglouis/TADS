# Task-Aware Data Selection via Proxy-Label Enhanced Distribution Matching for LLM Finetuning

This code is a PyTorch implementation of our ICLR'26 paper "Task-Aware Data Selection via Proxy-Label Enhanced Distribution Matching for LLM Finetuning". [[paper]](https://openreview.net/pdf?id=R40WoYbYab)

## Overview

The pipeline consists of 4 main steps:
1. **Step 1**: Dataset splitting and target annotation
2. **Step 2**: Tag clustering and propagation 
3. **Step 3**: Keyword extraction and data scoring
4. **Step 4**: Quality-based task-oriented selection

## Dataset preparation

```bash
# eval data
bash data/prepare_eval_data.sh

# train data
bash data/prepare_train_data.sh
```

## Step 1: Generate Labels

### 1.1 Dataset Splitting
Split evaluation datasets into target (20%) and evaluation (80%) sets.

```bash
cd step1_generate_labels
python 1dataset_splitter.py
```


### 1.2 Target Split Merge
Merge all target datasets into a unified parquet format.

```bash
python 2target_split_merge.py
```


### 1.3 Target Annotation
Annotate target dataset with tags using Qwen2.5-7B-Instruct.

```bash
python 3target_annotation.py
```


## Step 2: Clustering and Propagating

### 2.1 Testset Tag Cluster Merge
Cluster and deduplicate tags from target annotation.

```bash
cd ../step2_clustering_and_propagating
python 1testset_tag_cluster_merge.py
```


### 2.2 Training Set Content Embedding Cache
Generate embeddings for training set content using BGE-M3 model.

```bash
python 2training_set_content_embedding_cache.py
```


### 2.3 Propagating Tags Using Cached Embedding
Propagate clustered tags to training set using semantic similarity.

```bash
python 3propagating_tags_using_cached_embedding.py
```


## Step 3: Tag Clustering and Label Training Set

### 3.1 Keyword Extraction
Extract keywords from clustered tags using vLLM and Qwen2.5-7B-Instruct.

```bash
cd ../step3_tag_clustering_label_training_set
python 1keyword_extraction_vllm.py
```


### 3.2 Score Based on Anchors
Score training data using keyword mapping and Qwen2.5-7B-Instruct.

```bash
python 2score_based_on_anchors.py
```


### 3.3 Filter OOD
Filter out-of-distribution samples based on score thresholds.

```bash
python 3filter_ood.py
```


## Step 4: Quality Task-Oriented Selection

### 4.1 Mitigating Domain Shift
Select high-quality samples using distribution-based sampling.

```bash
cd ../step4_quality_task_orient
python 1mitigating_domain_shift.py
python 2joint_filter_fusion.py
```



## File Structure

```
/
├── step1_generate_labels/
│   ├── 1dataset_splitter.py          # Split evaluation datasets
│   ├── 2target_split_merge.py        # Merge target datasets
│   └── 3target_annotation.py         # Annotate target data
├── step2_clustering_and_propagating/
│   ├── 1testset_tag_cluster_merge.py # Cluster and deduplicate tags
│   ├── 2training_set_content_embedding_cache.py # Cache embeddings
│   └── 3propagating_tags_using_cached_embedding.py # Propagate tags
├── step3_tag_clustering_label_training_set/
│   ├── 1keyword_extraction_vllm.py   # Extract keywords
│   ├── 2score_based_on_anchors.py    # Score data
│   └── 3filter_ood.py                # Filter OOD samples
├── step4_quality_task_orient/
│   └── 1mitigating_domain_shift.py   # Quality-based selection
|   └── 2joint_filter_fusion.py       # match multiple label domain
├── data/                             # Data directory
|   ├── eval/                         # Evaluation datasets
|   ├── train_embeds_and_tags/        # Cached embeddings
|   ├── prepare_*_data.sh             # Prepare datasets
|   └── *.pt, *.json                  # Processed data files
└── consistency_precision_reuslt/     # consistency and precision
    ├── consistency_gpt_eval.xlsx
    ├── consistency_human_eval.xlsx
    ├── precision_gpt_eval.xlsx
    └── precision_human_eval.xlsx

```

## Finetune & eval
The finetune & eval of TADS is based on [open-instruct](https://github.com/allenai/open-instruct).


