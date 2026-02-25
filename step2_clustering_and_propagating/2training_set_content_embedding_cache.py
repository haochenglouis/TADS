import json
import os
import copy
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import re


def load_and_restore_tags_from_ids(deduped_file: str, mapping_file: str):
    print(f"Loading deduplicated data from: {deduped_file}")
    with open(deduped_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loading tag_id -> tag_text mapping from: {mapping_file}")
    with open(mapping_file, "r", encoding="utf-8") as f:
        tag_id_to_text = json.load(f)

    print("Restoring tags from tag_ids...")
    for item in tqdm(data):
        content = item.get("generated_content", {})
        for category in ["Topic", "Style", "Audience", "Task"]:
            value = content.get(category, [])
            if isinstance(value, list):
                content[category] = [tag_id_to_text.get(tag_id, tag_id) for tag_id in value]
            elif isinstance(value, str):
                content[category] = tag_id_to_text.get(value, value)

    print("Tag restoration complete. Returning restored data.")
    return data


def normalize_tag(text):
    text = text.lower()
    text = re.sub(r"[-/]", " ", text)
    text = re.sub(r"\band\b", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_data(file_path):
    print(f"Loading data from {file_path}...")
    ext = os.path.splitext(file_path)[-1].lower()

    if ext == ".json":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif ext == ".parquet":
        dataset = load_dataset("parquet", data_files=file_path)["train"]
        data = dataset.to_list()
    elif ext == ".pt":
        data = torch.load(file_path)
        if isinstance(data, dict) and "data" in data:
            data = data["data"]  # Extract if standard structure
        if isinstance(data, torch.Tensor):
            data = data.tolist()  # Convert tensor to list
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    print(f"Loaded {len(data)} items.")
    return data


def convert_alpaca_to_string(sample):
    return "\n".join(f"{msg['role']}: {msg['content']}" for msg in sample['messages'])


def encode_texts(texts, tokenizer, model, device, batch_size=8):
    all_embeddings = []
    model.eval()
    torch.cuda.empty_cache()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
            outputs = model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0]  # CLS token
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu())
            torch.cuda.empty_cache()
    return torch.cat(all_embeddings, dim=0)


def analyze_all_items_from_data(data, output_dir, dataset_name, ii=10):
    all_items = {
        "Topic": Counter(),
        "Style": Counter(),
        "Audience": Counter(),
        "Task": Counter()
    }

    print("\nAnalyzing ALL items in each category...")
    for item in tqdm(data):
        content = item.get('generated_content', {})
        if not content:
            continue

        for category in ["Topic", "Style", "Audience", "Task"]:
            category_value = content.get(category)

            if isinstance(category_value, list) and category_value:
                for tag in category_value:
                    all_items[category][tag] += 1
            elif isinstance(category_value, str):
                all_items[category][category_value] += 1

    # save_detailed_distribution(all_items, output_dir, dataset_name, ii=ii, mode="all_item")
    return all_items


def identify_long_tail_tags(counter, freq_threshold=10, cumulative_percent_threshold=90):
    """
    Identify long tail tags based on frequency and cumulative distribution.

    Args:
        counter: Counter object (tag -> count)
        freq_threshold: Tags with count <= this are considered long tail
        cumulative_percent_threshold: Cumulative percentage after which tags are considered long tail

    Returns:
        Dict of {tag: count} for long tail tags
    """
    total_count = sum(counter.values())
    sorted_items = counter.most_common()

    long_tail_dict = {}
    cumulative_percentage = 0

    for tag, count in sorted_items:
        cumulative_percentage += (count / total_count) * 100

        if count <= freq_threshold and cumulative_percentage > cumulative_percent_threshold:
            long_tail_dict[tag] = count

    return long_tail_dict


def match_to_tags_topk(text_embeds, tag_embeds, tag_texts, top_k=3):
    sims = torch.matmul(text_embeds, tag_embeds.T)
    best_scores, best_indices = torch.topk(sims, k=top_k, dim=1)
    best_tags_all = []
    best_scores_all = []
    for indices, scores in zip(best_indices, best_scores):
        tags = [tag_texts[i] for i in indices.tolist()]
        score_vals = scores.tolist()
        best_tags_all.append(tags)
        best_scores_all.append(score_vals)
    return best_tags_all, best_scores_all


def print_top50_formatted(items, top_n=50, mode="first_item"):
    print(f"\n{'=' * 80}")
    print(f"Top {top_n} Items - {mode.replace('_', ' ').title()}")
    print(f"{'=' * 80}\n")

    for category, counter in items.items():
        print(f"\n{category} - Top {top_n}:")

        top_items = counter.most_common(top_n)
        total_count = sum(counter.values())

        print(f"{'Rank':<5}\t{'Item':<30}\t{'Count':>8}\t{'Percentage':>10}")
        print("-" * 70)

        for idx, (item, count) in enumerate(top_items, 1):
            percentage = (count / total_count) * 100
            item_display = item[:30] + ("..." if len(item) > 30 else "")
            print(f"{idx:<5}\t{item_display:<30}\t{count:>8}\t{percentage:>9.2f}%")

        print("-" * 70)


if __name__ == "__main__":
    # ===== Configuration =====
    dataset_name = "ds-target-split"
    input_file = f"../data/processed_with_cluster_ids_{dataset_name}.json"
    output_dir = f"../deduped_tags/{dataset_name}"
    ii = 100

    data = load_and_restore_tags_from_ids(input_file,
                                          f"../data/tag_id_to_text_mapping_{dataset_name}.json")
    all_items = analyze_all_items_from_data(data, output_dir, dataset_name, ii=ii)
    print_top50_formatted(all_items, top_n=ii, mode="all_item")
    long_tail_all = {cat: identify_long_tail_tags(counter) for cat, counter in all_items.items()}

    print("\n✅ All analysis, plotting and long tail identification completed!")

    dataset_name = "tulu-300k"
    input_file = "../data/train-00000-of-00001.parquet"
    output_dir = f"../deduped_tags/{dataset_name}_test"
    ii = 100

    data_train = load_data(input_file)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "BAAI/bge-m3"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    mapping_file = f"../data/tag_id_to_text_mapping_ds-target-split.json"
    with open(mapping_file, "r", encoding="utf-8") as f:
        tag_id_to_text = json.load(f)
    text_to_tag_id = {normalize_tag(v): k for k, v in tag_id_to_text.items()}

    if torch.cuda.device_count() > 1:
        print(f"✅ Detected {torch.cuda.device_count()} GPUs. Using DataParallel...")
        model = torch.nn.DataParallel(model)
    # model = torch.compile(model)

    print(f"✅ Loaded {model_name} on {device}")

    field_tag_texts = {}
    field_tag_embeddings = {}
    for field in ['Topic', 'Style', 'Audience', 'Task']:
        tags = list(all_items[field].keys())
        print(f"Encoding {field}: {len(tags)} tags")
        embeds = encode_texts(tags, tokenizer, model, device)
        field_tag_texts[field] = tags
        field_tag_embeddings[field] = embeds

    print(f"✅ All fields encoded separately.")

    text_samples = []
    sample_info = []
    err_sample = []
    for sample_idx, sample in enumerate(data_train):
        if "messages" not in sample:
            err_sample.append(sample_idx)
            continue
        text_input = convert_alpaca_to_string(sample)
        for field in ['Topic', 'Style', 'Audience', 'Task']:
            text_samples.append((text_input, field))
            sample_info.append((sample_idx, field))

    texts_only = [text for text, _ in text_samples]
    
    # Deduplicate: only encode unique texts, then restore to original structure
    print(f"\n📊 Original text count: {len(texts_only)}")
    unique_texts = []
    text_to_unique_idx = {}
    original_to_unique_idx = []
    
    for text in texts_only:
        if text not in text_to_unique_idx:
            unique_idx = len(unique_texts)
            text_to_unique_idx[text] = unique_idx
            unique_texts.append(text)
        original_to_unique_idx.append(text_to_unique_idx[text])
    
    dedup_ratio = (1 - len(unique_texts)/len(texts_only))*100
    print(f"📊 Unique text count: {len(unique_texts)} (deduplication ratio: {dedup_ratio:.2f}%)")
    
    # Only encode unique texts
    unique_text_embeddings = encode_texts(unique_texts, tokenizer, model, device)
    
    # Restore to original structure
    print("🔄 Restoring original text structure...")
    text_embeddings = unique_text_embeddings[original_to_unique_idx]
    print(f"✅ Restoration complete, final embeddings shape: {text_embeddings.shape}")


    # === Save text_embeddings, text_samples and field_tag_embeddings, field_tag_texts ===
    print("\n📦 Saving text embeddings, text samples and tag embeddings...")

    save_dir = "../data/train_embeds_and_tags"
    os.makedirs(save_dir, exist_ok=True)

    # 1. Save text_embeddings
    text_embed_path = os.path.join(save_dir, "text_embeddings.pt")
    torch.save(text_embeddings.cpu(), text_embed_path)
    print(f"✅ Saved text embeddings to {text_embed_path}")

    # 2. Save text_samples
    text_sample_info_path = os.path.join(save_dir, "text_samples.json")
    with open(text_sample_info_path, "w", encoding="utf-8") as f:
        json.dump(text_samples, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved text sample info to {text_sample_info_path}")

    # 3. Save tag embeddings and tag texts for each field
    for field in ['Topic', 'Style', 'Audience', 'Task']:
        # Save embeddings
        field_embed_path = os.path.join(save_dir, f"{field.lower()}_tag_embeddings.pt")
        torch.save(field_tag_embeddings[field].cpu(), field_embed_path)
        print(f"✅ Saved {field} tag embeddings to {field_embed_path}")

        # Save tag texts
        field_tags_path = os.path.join(save_dir, f"{field.lower()}_tag_texts.json")
        with open(field_tags_path, "w", encoding="utf-8") as f:
            json.dump(field_tag_texts[field], f, ensure_ascii=False, indent=2)
        print(f"✅ Saved {field} tag texts to {field_tags_path}")

    print("\n🎯 All embeddings and auxiliary data saved successfully!")