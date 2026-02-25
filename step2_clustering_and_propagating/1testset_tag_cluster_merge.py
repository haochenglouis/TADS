import json
import os
from collections import defaultdict
import torch
import faiss
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import copy
from collections import Counter


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
dataset_name = "ds-target-split"
DEDUPED_TAGS_DIR = os.path.join(BASE_DIR, "deduped_tags", dataset_name)

for dir_path in [DATA_DIR, DEDUPED_TAGS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

input_file = os.path.join(DATA_DIR, "step1_target_annotation_Qwen_Qwen2.5-7B-Instruct.pt")


def is_english(text):
    ascii_count = sum(1 for c in text if ord(c) < 128)
    return ascii_count / max(1, len(text)) > 0.8


def cluster_tags(file_path, embedding_model, num_clusters, batch_size, output_dir):
    with open(file_path, "r", encoding="utf-8") as f:
        features = json.load(f)

    if len(features) < 2:
        print(f"Skipping clustering for {file_path} - not enough tags")
        return None

    print(f"Clustering {len(features)} tags from {file_path} into {num_clusters} clusters")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(embedding_model)
    model = AutoModel.from_pretrained(embedding_model).to(device)
    model.eval()

    all_emb = []
    with torch.no_grad():
        for i in tqdm(range(0, len(features), batch_size), desc=f"Encoding {os.path.basename(file_path)}"):
            tokenized_inputs = tokenizer(features[i:i + batch_size], padding=True, truncation=True, return_tensors="pt", max_length=64).to(device)
            text_embeddings = model(**tokenized_inputs).last_hidden_state.mean(dim=1)
            all_emb.append(text_embeddings.detach().cpu().numpy())

    embeddings = np.concatenate(all_emb, axis=0)

    d = embeddings.shape[1]
    kmeans = faiss.Kmeans(d, num_clusters, niter=100, verbose=True, gpu=torch.cuda.is_available())
    kmeans.train(embeddings)
    _, cluster_labels = kmeans.index.search(embeddings, 1)

    feature_to_cluster_map = {features[i]: int(cluster_labels[i][0]) for i in range(len(features))}

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    with open(os.path.join(output_dir, f"feature_cluster_map_{base_name}.json"), "w", encoding="utf-8") as f:
        json.dump(feature_to_cluster_map, f, ensure_ascii=False, indent=4)

    cluster_to_tags = defaultdict(list)
    for i, feature in enumerate(features):
        cluster_index = int(cluster_labels[i][0])
        cluster_to_tags[cluster_index].append(feature)

    with open(os.path.join(output_dir, f"cluster_tag_map_{base_name}.json"), "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in cluster_to_tags.items()}, f, ensure_ascii=False, indent=4)

    return feature_to_cluster_map

from collections import OrderedDict
def deduplicate_list_preserve_order(seq):
    """Remove duplicates while preserving order"""
    return list(OrderedDict.fromkeys(seq))



if __name__ == "__main__":
    cluster_numbers = {
        "Topic": 100,
        "Style": 100,
        "Audience": 100,
        "Task": 100
    }
    output_dir = DEDUPED_TAGS_DIR


    os.makedirs(output_dir, exist_ok=True)
    data = torch.load(input_file)

    raw_tags_per_category = {"Topic": set(), "Style": set(), "Audience": set(), "Task": set()}
    for item in tqdm(data):
        content = item.get("generated_content", {})
        for category in raw_tags_per_category:
            val = content.get(category, [])
            if isinstance(val, str):
                raw_tags_per_category[category].add(val)
            elif isinstance(val, list):
                raw_tags_per_category[category].update(val)

    tag_files = {}
    for category, tag_set in raw_tags_per_category.items():
        tag_list = sorted(tag_set)
        tag_file = os.path.join(output_dir, f"all_tags_{category.lower()}_{dataset_name}.json")
        with open(tag_file, "w", encoding="utf-8") as f:
            json.dump(tag_list, f, ensure_ascii=False, indent=4)
        tag_files[category] = tag_file
        print(f"Saved {len(tag_list)} tags for {category}")

    tag_mappings = {}
    for category in tag_files:
        mapping = cluster_tags(tag_files[category], "BAAI/bge-m3", cluster_numbers.get(category, 100), 2048, output_dir)
        tag_mappings[category] = mapping

    tag_id_mappings = {}
    cluster_representatives = {}

    for category in tag_mappings:
        cluster_map_file = os.path.join(output_dir, f"cluster_tag_map_all_tags_{category.lower()}_{dataset_name}.json")
        with open(cluster_map_file, "r", encoding="utf-8") as f:
            cluster_to_tags = json.load(f)

        rep_map = {}
        for cluster_id, tags in cluster_to_tags.items():
            english_tags = [t for t in tags if is_english(t)]
            rep = "_".join(english_tags if english_tags else tags)
            rep_map[cluster_id] = rep
            tag_id_mappings[f"{category}_{cluster_id}"] = rep
        cluster_representatives[category] = rep_map

    def is_cluster_tag(tag):
        # Check if tag is in format like "Topic_72"
        return isinstance(tag, str) and "_" in tag and tag.split("_")[0] in cluster_representatives
    # Deep copy data
    processed_data = copy.deepcopy(data)

    # Counter for replacement statistics
    tag_counter = Counter()

    for item in tqdm(processed_data, desc="Replacing tags with cluster IDs"):
        content = item.get("generated_content", {})
        for category in cluster_representatives:
            tags = content.get(category, [])

            if isinstance(tags, str):
                tags = [tags]

            tags = deduplicate_list_preserve_order(tags)

            new_ids = []
            for tag in tags:
                if is_cluster_tag(tag):
                    new_ids.append(tag)  # Already in cluster_id format, skip processing
                    tag_counter[tag] += 1
                    continue

                cluster_id = tag_mappings[category].get(tag)
                if cluster_id is None:
                    raise ValueError(f"Missing tag '{tag}' in cluster mapping for category '{category}'")
                new_id = f"{category}_{cluster_id}"
                new_ids.append(new_id)
                tag_counter[new_id] += 1

            content[category] = new_ids if isinstance(content.get(category), list) else new_ids[0]

    print("Tag replacement summary:")
    for tag_id, count in tag_counter.most_common():
        print(f"{tag_id}: {count} samples")

    # Save processed data
    processed_output_file = os.path.join(DATA_DIR, f"processed_with_cluster_ids_{dataset_name}.json")
    with open(processed_output_file, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=4)

    print(f"✅ Processed data saved to {processed_output_file}")

    # Save tag_id_mappings mapping file
    tag_id_map_output_path = os.path.join(DATA_DIR, f"tag_id_to_text_mapping_{dataset_name}.json")
    with open(tag_id_map_output_path, "w", encoding="utf-8") as f:
        json.dump(tag_id_mappings, f, ensure_ascii=False, indent=4)

    print(f"✅ Tag ID to text mapping saved to {tag_id_map_output_path}")