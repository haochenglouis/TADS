"""Tag processing utilities shared across pipeline steps."""

import json
import re
from collections import Counter

from tqdm import tqdm


def normalize_tag(text: str) -> str:
    """Normalize tag text: lowercase, remove hyphens/slashes/'and', collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[-/]", " ", text)
    text = re.sub(r"\band\b", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_and_restore_tags_from_ids(deduped_file: str, mapping_file: str) -> list:
    """Load cluster-id data and restore human-readable tag text using a mapping file.

    Args:
        deduped_file: JSON file where tags are stored as cluster IDs (e.g. "Topic_42").
        mapping_file: JSON file mapping tag IDs to their representative text.

    Returns:
        List of sample dicts with tag IDs replaced by text.
    """
    print(f"Loading deduplicated data from: {deduped_file}")
    with open(deduped_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Loading tag_id -> tag_text mapping from: {mapping_file}")
    with open(mapping_file, "r", encoding="utf-8") as f:
        tag_id_to_text = json.load(f)

    print("Restoring tags from tag_ids...")
    for item in tqdm(data, desc="Restoring tags"):
        content = item.get("generated_content", {})
        for category in ["Topic", "Style", "Audience", "Task"]:
            value = content.get(category, [])
            if isinstance(value, list):
                content[category] = [tag_id_to_text.get(tag_id, tag_id) for tag_id in value]
            elif isinstance(value, str):
                content[category] = tag_id_to_text.get(value, value)

    return data


def analyze_all_items_from_data(data: list) -> dict:
    """Count tag frequencies per category across all samples.

    Returns:
        Dict mapping category name -> Counter of tag frequencies.
    """
    all_items = {
        "Topic": Counter(),
        "Style": Counter(),
        "Audience": Counter(),
        "Task": Counter(),
    }

    for item in tqdm(data, desc="Analyzing tag distribution"):
        content = item.get("generated_content", {})
        if not content:
            continue
        for category in all_items:
            category_value = content.get(category)
            if isinstance(category_value, list) and category_value:
                for tag in category_value:
                    all_items[category][tag] += 1
            elif isinstance(category_value, str):
                all_items[category][category_value] += 1

    return all_items


def identify_long_tail_tags(
    counter: Counter,
    freq_threshold: int = 10,
    cumulative_percent_threshold: float = 90,
) -> dict:
    """Identify long-tail tags by frequency and cumulative distribution.

    Returns:
        Dict of {tag: count} for tags that are both low-frequency and past
        the cumulative percentage threshold.
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
