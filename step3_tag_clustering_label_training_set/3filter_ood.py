#!/usr/bin/env python3
"""
Filter samples from scored_data_with_keywords.pt based on score thresholds
Creates sample sets for thresholds 1-10 where ALL score fields >= threshold
"""

import torch
import os
from collections import defaultdict

def load_scored_data(file_path):
    """
    Load the scored data file
    
    Args:
        file_path: Path to the scored data file
        
    Returns:
        List of data samples
    """
    print(f"Loading scored data from: {file_path}")
    data = torch.load(file_path)
    print(f"Total samples loaded: {len(data)}")
    return data

def filter_samples_all_scores_gte(data, threshold):
    """
    Filter samples where ALL score fields >= threshold
    
    Args:
        data: List of data samples
        threshold: Minimum score threshold for all fields
        
    Returns:
        Tuple: (List of filtered samples, field_name used)
    """
    filtered_samples = []
    field_used = None
    
    for sample in data:
        # Look for scoring fields
        score_fields = [key for key in sample.keys() if 'score' in key.lower()]
        
        sample_passes = False
        for field in score_fields:
            scores = sample.get(field, {})
            if isinstance(scores, dict):
                # Get all score values (excluding metadata fields starting with '_')
                score_values = [v for k, v in scores.items() 
                               if isinstance(v, (int, float)) and not k.startswith('_')]
                
                # Check if ALL scores >= threshold
                if score_values and all(score >= threshold for score in score_values):
                    sample_passes = True
                    if field_used is None:
                        field_used = field
                    break
        
        if sample_passes:
            filtered_samples.append(sample)
    
    return filtered_samples, field_used

def analyze_score_distribution(data, dataset_name):
    """
    Analyze the score distribution of a dataset
    
    Args:
        data: List of data samples
        dataset_name: Name of the dataset for display
    """
    if not data:
        print(f"{dataset_name}: No samples found")
        return
    
    scores = defaultdict(list)
    
    for sample in data:
        score_fields = [key for key in sample.keys() if 'score' in key.lower()]
        
        for field in score_fields:
            sample_scores = sample.get(field, {})
            if isinstance(sample_scores, dict):
                for metric, value in sample_scores.items():
                    if isinstance(value, (int, float)) and not metric.startswith('_'):
                        scores[metric].append(value)
    
    print(f"\n{dataset_name}:")
    print(f"  Total samples: {len(data)}")
    
    for metric, values in scores.items():
        if values:
            mean_score = sum(values) / len(values)
            min_score = min(values)
            max_score = max(values)
            print(f"  {metric}: mean={mean_score:.2f}, min={min_score}, max={max_score}")

def main():
    """Main function"""
    # Input file path - updated to use relative path and new naming convention
    input_file = "../data/scored_data_with_keywords_ds-target-split.pt"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist")
        return
    
    # Load data
    data = load_scored_data(input_file)
    
    # Analyze available score fields
    if data:
        sample_score_fields = []
        for key in data[0].keys():
            if 'score' in key.lower():
                sample_score_fields.append(key)
        print(f"Available score fields in data: {sample_score_fields}")
        
        # Show sample score structure
        if sample_score_fields:
            sample_scores = data[0].get(sample_score_fields[0], {})
            if isinstance(sample_scores, dict):
                score_metrics = [k for k in sample_scores.keys() if not k.startswith('_')]
                print(f"Score metrics in '{sample_score_fields[0]}': {score_metrics}")
    
    # Define thresholds from 1 to 10
    thresholds = list(range(1, 11))
    
    print("=" * 80)
    print("Filtering samples by score thresholds (1-10)")
    print("=" * 80)
    
    # Create output directory
    all_scores_dir = "../data/filtered_samples_all_scores"
    os.makedirs(all_scores_dir, exist_ok=True)
    
    # Store results for summary
    all_scores_results = {}
    
    for threshold in thresholds:
        print(f"\nProcessing threshold: {threshold}")
        
        # Filter samples - ALL scores >= threshold
        print(f"  Filtering: ALL scores >= {threshold}")
        all_scores_filtered, all_scores_field = filter_samples_all_scores_gte(data, threshold)
        all_scores_file = os.path.join(all_scores_dir, f"samples_all_scores_gte_{threshold}.pt")
        torch.save(all_scores_filtered, all_scores_file)
        all_scores_results[threshold] = len(all_scores_filtered)
        print(f"    Using score field: {all_scores_field}")
        print(f"    Saved {len(all_scores_filtered)} samples to: {all_scores_file}")
    
    print("\n" + "=" * 80)
    print("Sample filtering completed!")
    print("=" * 80)
    
    # Summary
    print(f"\nSUMMARY:")
    print(f"Original dataset: {len(data):,} samples")
    print()
    
    print("ALL SCORES >= threshold:")
    for threshold in thresholds:
        count = all_scores_results[threshold]
        percentage = (count / len(data)) * 100 if data else 0
        print(f"  Threshold {threshold:2d}: {count:7,} samples ({percentage:5.1f}%)")

if __name__ == "__main__":
    main()