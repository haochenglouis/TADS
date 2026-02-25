#!/usr/bin/env python3
"""
Select 10k samples from filtered datasets using distribution-based sampling
Reference target set distribution for sampling
Updated version: Use processed_with_cluster_ids_ds-target-split.json as reference distribution
"""

import torch
import os
import random
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm
import json

def load_data_safe(file_path):
    """Safely load torch data file"""
    try:
        return torch.load(file_path, weights_only=False)
    except Exception as e:
        print(f"❌ Failed to load file {file_path}: {e}")
        return None

def load_keyword_mapping(mapping_file):
    """Load keyword mapping file"""
    try:
        with open(mapping_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Failed to load mapping file {mapping_file}: {e}")
        return None

def create_reverse_mapping(keyword_mapping, tag_id_mapping):
    """Create reverse mapping from specific long text tags to abstract tags"""
    reverse_mapping = {}
    
    # Directly use tag_id_mapping to create reverse mapping
    for abstract_tag, long_text in tag_id_mapping.items():
        reverse_mapping[long_text] = abstract_tag
        
        # Also handle underscore-separated keyword format
        if abstract_tag in keyword_mapping:
            keywords_str = keyword_mapping[abstract_tag]
            reverse_mapping[keywords_str] = abstract_tag
    
    return reverse_mapping

def convert_tags_to_abstract(sample, reverse_mapping, field):
    """Convert specific tags in sample to abstract tags"""
    try:
        content = sample.get('generated_content', {})
        if field not in content:
            return sample
        
        original_tags = content[field]
        if not original_tags:
            return sample
        
        # Ensure tags are in list format
        if isinstance(original_tags, str):
            original_tags = [original_tags]
        
        converted_tags = []
        for tag in original_tags:
            if isinstance(tag, str):
                # Direct lookup in reverse mapping for exact match
                if tag in reverse_mapping:
                    converted_tags.append(reverse_mapping[tag])
                else:
                    # If no exact match found, try fuzzy matching
                    found = False
                    for long_text, abstract_tag in reverse_mapping.items():
                        if tag in long_text or long_text in tag:
                            converted_tags.append(abstract_tag)
                            found = True
                            break
                    
                    if not found:
                        print(f"⚠️ Cannot find tag mapping: {tag[:100]}...")
                        converted_tags.append(tag)
            else:
                converted_tags.append(tag)
        
        # Create converted sample copy
        converted_sample = sample.copy()
        converted_sample['generated_content'] = content.copy()
        converted_sample['generated_content'][field] = converted_tags
        
        return converted_sample
    
    except Exception as e:
        print(f"⚠️ Tag conversion error: {e}")
        return sample

def analyze_all_items_from_data(data):
    """
    Analyze tag distribution in data
    Adapted from task_orient_select_with_quality.py
    """
    all_items = {
        "Topic": Counter(),
        "Style": Counter(),
        "Audience": Counter(),
        "Task": Counter()
    }
    for item in tqdm(data, desc="Analyzing tag distribution"):
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
    return all_items

def sample_to_match_distribution_incremental(data1, data2, field, target_total=10000):
    """
    Incremental sampling to match target distribution
    Adapted from task_orient_select_with_quality.py
    
    Args:
        data1: Source candidate data
        data2: Target distribution reference data
        field: Tag field name (e.g. 'Topic')
        target_total: Target sampling total
        
    Returns:
        List of sampled results
    """
    print(f"\n🚀 Incremental sampling to match field [{field}] distribution...")

    target_counter = analyze_all_items_from_data(data2)[field]
    target_total_tags = sum(target_counter.values())
    if target_total_tags == 0:
        print(f"⚠️ Field {field} has no valid tags, skipping")
        return random.sample(data1, min(target_total, len(data1)))
        
    target_ratio = {tag: cnt / target_total_tags for tag, cnt in target_counter.items()}

    all_indices = list(range(len(data1)))
    random.shuffle(all_indices)

    selected_samples = []
    selected_indices = set()
    selected_counter = Counter()

    tag_to_candidate_indices = defaultdict(list)
    idx_to_tags = {}

    for idx, item in enumerate(data1):
        tags = item.get("generated_content", {}).get(field, [])
        if isinstance(tags, str):
            tags = [tags]
        tags = [t for t in tags if t in target_ratio]
        if tags:
            idx_to_tags[idx] = tags
            for tag in tags:
                tag_to_candidate_indices[tag].append(idx)

    while len(selected_samples) < target_total:
        current_total = sum(selected_counter.values()) or 1
        current_ratio = {tag: selected_counter[tag] / current_total for tag in target_ratio}

        # Find tag with largest gap (target ratio - current ratio) that still has samples
        gap_tag = None
        max_gap = -1
        for tag in target_ratio:
            gap = target_ratio[tag] - current_ratio.get(tag, 0)
            if gap > max_gap and tag_to_candidate_indices[tag]:
                gap_tag = tag
                max_gap = gap

        if gap_tag is None:
            print("⚠️ No more useful candidate samples, stopping sampling")
            break

        # Select from unselected candidates
        candidates = [idx for idx in tag_to_candidate_indices[gap_tag] if idx not in selected_indices]
        if not candidates:
            del tag_to_candidate_indices[gap_tag]
            continue

        # Prefer samples with fewer tags to minimize "other tag pollution"
        chosen_idx = min(candidates, key=lambda i: len(idx_to_tags[i]))
        selected_indices.add(chosen_idx)
        selected_samples.append(data1[chosen_idx])
        for tag in idx_to_tags[chosen_idx]:
            selected_counter[tag] += 1

    print(f"✅ Sampling completed: selected {len(selected_samples)} samples")
    return selected_samples

def check_files_with_10k_plus():
    """Check which files have >10k samples"""
    large_files = []
    
    # Check all_scores files only
    all_scores_dir = '../data/filtered_samples_all_scores'
    if os.path.exists(all_scores_dir):
        for i in range(1, 11):
            file_path = f'{all_scores_dir}/samples_all_scores_gte_{i}.pt'
            if os.path.exists(file_path):
                data = load_data_safe(file_path)
                if data and len(data) > 10000:
                    large_files.append({
                        'path': file_path,
                        'type': 'all_scores',
                        'threshold': i,
                        'count': len(data),
                        'data': data
                    })
    
    return large_files

def select_random_10k(data):
    """Randomly select 10k samples"""
    if len(data) <= 10000:
        return data
    return random.sample(data, 10000)

def save_samples(samples, output_path, metadata=None):
    """Save samples to file with metadata"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save samples
    torch.save(samples, output_path)
    
    # Save metadata
    if metadata:
        metadata_path = output_path.replace('.pt', '_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(samples)} samples to: {output_path}")

def main():
    """Main function"""
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    print("=" * 80)
    print("Select 10k samples from filtered datasets using target set distribution as reference")
    print("Updated version: Use processed_with_cluster_ids_ds-target-split.json as reference distribution")
    print("=" * 80)
    
    # File paths - updated to use relative paths and new naming conventions
    reference_file = "../data/processed_with_cluster_ids_ds-target-split.json"
    mapping_file = "../data/extracted_keywords_ds-target-split.json"
    tag_id_mapping_file = "../data/tag_id_to_text_mapping_ds-target-split.json"
    
    # Check required files
    if not os.path.exists(reference_file):
        print(f"❌ Reference file does not exist: {reference_file}")
        return
    
    if not os.path.exists(mapping_file):
        print(f"❌ Mapping file does not exist: {mapping_file}")
        return
    
    if not os.path.exists(tag_id_mapping_file):
        print(f"❌ Tag ID mapping file does not exist: {tag_id_mapping_file}")
        return
    
    # Load reference data
    print("Loading reference data...")
    try:
        with open(reference_file, 'r', encoding='utf-8') as f:
            reference_data = json.load(f)
        print(f"Reference data loaded: {len(reference_data):,} samples (target set)")
    except Exception as e:
        print(f"❌ Failed to load reference data: {e}")
        return
    
    # Load mapping files
    print("Loading tag mappings...")
    keyword_mapping = load_keyword_mapping(mapping_file)
    if not keyword_mapping:
        return
    
    tag_id_mapping = load_keyword_mapping(tag_id_mapping_file)
    if not tag_id_mapping:
        return
    
    # Create reverse mapping
    print("Creating reverse mapping...")
    reverse_mapping = create_reverse_mapping(keyword_mapping, tag_id_mapping)
    print(f"Reverse mapping created: {len(reverse_mapping):,} mapping relationships")
    
    # Find files with >10k samples
    large_files = check_files_with_10k_plus()
    
    if not large_files:
        print("No files found with >10k samples!")
        return
    
    print(f"\nFound {len(large_files)} files with >10k samples:")
    for file_info in large_files:
        print(f"  {file_info['type']} threshold {file_info['threshold']}: {file_info['count']:,} samples")
    
    print("\n" + "=" * 80)
    print("Starting file processing...")
    print("=" * 80)
    
    # Create output directories
    distribution_output_dir = "../data/selected_10k_distribution_target"
    random_output_dir = "../data/selected_10k_random_target"
    
    # Distribution fields
    distribution_fields = ['Topic', 'Style', 'Audience', 'Task']
    
    for file_info in large_files:
        print(f"\nProcessing: {file_info['path']}")
        print(f"Original count: {file_info['count']:,}")
        
        data = file_info['data']
        file_type = file_info['type']
        threshold = file_info['threshold']
        
        # Distribution-based selection (for each field)
        for field in distribution_fields:
            print(f"  Distribution-based selection for {field}...")
            print(f"    Converting {field} tags to abstract format...")
            
            try:
                # Convert tags to abstract format
                converted_data = []
                for sample in tqdm(data, desc=f"Converting {field} tags"):
                    converted_sample = convert_tags_to_abstract(sample, reverse_mapping, field)
                    converted_data.append(converted_sample)
                
                print(f"    Tag conversion completed, starting distribution sampling...")
                
                # Use target set as reference distribution
                distribution_samples = sample_to_match_distribution_incremental(
                    converted_data, reference_data, field, target_total=10000
                )
                
                distribution_filename = f"distribution_10k_{field}_{file_type}_gte_{threshold}_target.pt"
                distribution_path = os.path.join(distribution_output_dir, distribution_filename)
                
                distribution_metadata = {
                    'source_file': file_info['path'],
                    'reference_file': reference_file,
                    'source_count': file_info['count'],
                    'reference_count': len(reference_data),
                    'selection_method': 'reference_distribution_based',
                    'distribution_field': field,
                    'selection_type': file_type,
                    'threshold': threshold,
                    'selected_count': len(distribution_samples),
                    'reference_source': 'processed_with_cluster_ids_ds-target-split',
                    'random_seed': 42
                }
                
                save_samples(distribution_samples, distribution_path, distribution_metadata)
                
            except Exception as e:
                print(f"    ❌ {field} distribution sampling error: {e}")
        
        # Random selection (once per file)
        print(f"  Random selection...")
        random_samples = select_random_10k(data)
        
        random_filename = f"random_10k_{file_type}_gte_{threshold}.pt"
        random_path = os.path.join(random_output_dir, random_filename)
        
        random_metadata = {
            'source_file': file_info['path'],
            'source_count': file_info['count'],
            'selection_method': 'random',
            'selection_type': file_type,
            'threshold': threshold,
            'selected_count': len(random_samples),
            'random_seed': 42
        }
        
        save_samples(random_samples, random_path, random_metadata)
    
    print("\n" + "=" * 80)
    print("Selection completed!")
    print("=" * 80)
    
    # Summary
    print(f"\nOutput directories:")
    print(f"  Distribution-based selection: {distribution_output_dir}/")
    print(f"  Random selection: {random_output_dir}/")
    
    # List generated files
    if os.path.exists(distribution_output_dir):
        dist_files = [f for f in os.listdir(distribution_output_dir) if f.endswith('.pt')]
        print(f"\nDistribution-based files ({len(dist_files)} files):")
        for f in sorted(dist_files):
            print(f"  {f}")
    
    if os.path.exists(random_output_dir):
        random_files = [f for f in os.listdir(random_output_dir) if f.endswith('.pt')]
        print(f"\nRandom selection files ({len(random_files)} files):")
        for f in sorted(random_files):
            print(f"  {f}")

if __name__ == "__main__":
    main()