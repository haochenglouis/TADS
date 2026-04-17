"""Step 2.1: Proxy-label clustering using K-means on tag embeddings (§4.2)."""

import copy
import json
import os
from collections import Counter, OrderedDict, defaultdict

import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def run(config):
    """Cluster tags from annotated target data and create tag-ID mappings."""
    input_file = config.target_annotation_pt
    output_cluster_ids = config.cluster_ids_json
    output_tag_mapping = config.tag_id_mapping_json
    num_clusters = config.clustering.num_clusters
    emb_model = config.embedding.name
    batch_size = config.embedding.batch_size

    data = torch.load(input_file, weights_only=False)
    print(f"Loaded {len(data)} annotated samples from {input_file}")

    # Extract unique tags per category
    raw_tags = {cat: set() for cat in ["Topic", "Style", "Audience", "Task"]}
    for item in data:
        content = item.get("generated_content", {})
        for cat in raw_tags:
            val = content.get(cat, [])
            if isinstance(val, str):
                raw_tags[cat].add(val)
            elif isinstance(val, list):
                raw_tags[cat].update(val)

    # Cluster each category
    tag_mappings = {}
    cluster_to_tags_all = {}
    for cat in raw_tags:
        tag_list = sorted(raw_tags[cat])
        print(f"\n{cat}: {len(tag_list)} unique tags -> {num_clusters} clusters")
        feature_map, cluster_map = _cluster_tags(tag_list, emb_model, num_clusters, batch_size)
        tag_mappings[cat] = feature_map
        cluster_to_tags_all[cat] = cluster_map

    # Build tag_id -> representative text mapping
    tag_id_mappings = {}
    cluster_representatives = {}
    for cat in cluster_to_tags_all:
        rep_map = {}
        for cluster_id, tags in cluster_to_tags_all[cat].items():
            english_tags = [t for t in tags if _is_english(t)]
            rep = "_".join(english_tags if english_tags else tags)
            rep_map[cluster_id] = rep
            tag_id_mappings[f"{cat}_{cluster_id}"] = rep
        cluster_representatives[cat] = rep_map

    # Replace tags with cluster IDs
    processed_data = copy.deepcopy(data)
    for item in tqdm(processed_data, desc="Replacing tags with cluster IDs"):
        content = item.get("generated_content", {})
        for cat in cluster_representatives:
            tags = content.get(cat, [])
            if isinstance(tags, str):
                tags = [tags]
            tags = list(OrderedDict.fromkeys(tags))  # deduplicate preserving order
            new_ids = []
            for tag in tags:
                cluster_id = tag_mappings[cat].get(tag)
                if cluster_id is None:
                    raise ValueError(f"Missing tag '{tag}' in mapping for '{cat}'")
                new_ids.append(f"{cat}_{cluster_id}")
            content[cat] = new_ids if isinstance(content.get(cat), list) else new_ids[0]

    # Save
    os.makedirs(os.path.dirname(output_cluster_ids) or ".", exist_ok=True)
    with open(output_cluster_ids, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)
    with open(output_tag_mapping, "w", encoding="utf-8") as f:
        json.dump(tag_id_mappings, f, ensure_ascii=False, indent=4)

    print(f"Saved cluster IDs to {output_cluster_ids}")
    print(f"Saved tag mapping to {output_tag_mapping}")


def _is_english(text: str) -> bool:
    ascii_count = sum(1 for c in text if ord(c) < 128)
    return ascii_count / max(1, len(text)) > 0.8


def _cluster_tags(tags, emb_model, num_clusters, batch_size):
    """Cluster a list of tags and return (feature_to_cluster, cluster_to_tags) dicts."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(emb_model)
    model = AutoModel.from_pretrained(emb_model).to(device)
    model.eval()

    all_emb = []
    with torch.no_grad():
        for i in tqdm(range(0, len(tags), batch_size), desc="Encoding tags"):
            inputs = tokenizer(
                tags[i : i + batch_size], padding=True, truncation=True,
                return_tensors="pt", max_length=64,
            ).to(device)
            emb = model(**inputs).last_hidden_state.mean(dim=1)
            all_emb.append(emb.detach().cpu().numpy())

    embeddings = np.concatenate(all_emb, axis=0)
    d = embeddings.shape[1]
    kmeans = faiss.Kmeans(d, num_clusters, niter=100, verbose=True, gpu=torch.cuda.is_available())
    kmeans.train(embeddings)
    _, labels = kmeans.index.search(embeddings, 1)

    feature_to_cluster = {tags[i]: int(labels[i][0]) for i in range(len(tags))}
    cluster_to_tags = defaultdict(list)
    for i, tag in enumerate(tags):
        cluster_to_tags[int(labels[i][0])].append(tag)

    return feature_to_cluster, {str(k): v for k, v in cluster_to_tags.items()}
