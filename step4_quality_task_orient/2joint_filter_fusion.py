#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Joint filtering and equal-weight voting fusion tool
Used to fuse data from archive_first_version_data/selected_10k_distribution_refDS2_json folder
"""

import json
import os
import random
from typing import List, Dict, Set
from pathlib import Path


def load_dataset_ids(file_path: str) -> List[str]:
    """
    Load all data item IDs from a JSON file
    
    Args:
    - file_path: Path to the JSON file
    
    Returns:
    - ids: List of data item IDs
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        ids = [item['id'] for item in data if 'id' in item]
        print(f"Loaded {len(ids)} IDs from {os.path.basename(file_path)}")
        return ids
    
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return []


def find_dataset_files(base_dir: str, dimensions: List[str], gte_threshold: int, 
                      score_type: str = "overall_score") -> Dict[str, str]:
    """
    Find dataset files for specified dimensions and threshold
    
    Args:
    - base_dir: Path to the dataset folder
    - dimensions: List of dimensions ['Audience', 'Style', 'Task', 'Topic']
    - gte_threshold: Threshold value (gte_1, gte_2, ..., gte_8)
    - score_type: Score type ("overall_score" or "all_scores")
    
    Returns:
    - files: Mapping from dimension to file path
    """
    files = {}
    
    for dimension in dimensions:
        filename = f"distribution_10k_{dimension}_{score_type}_gte_{gte_threshold}_refDS2.json"
        file_path = os.path.join(base_dir, filename)
        
        if os.path.exists(file_path):
            files[dimension] = file_path
            print(f"Found file: {filename}")
        else:
            print(f"Warning: File not found {filename}")
    
    return files


def equal_weight_fusion(*method_results: List[str], random_tie_breaking: bool = False, 
                        random_seed: int = None) -> List[str]:
    """
    Equal-weight voting fusion: Each method has the same weight, selected items get 1 vote
    
    Args:
    - *method_results: Variable number of method results, each is a list of item IDs
    - random_tie_breaking: Whether to use random sorting when votes are equal (default False, sort by ID)
    - random_seed: Random seed for reproducibility (only effective when random_tie_breaking=True)
    
    Returns:
    - final_top10k: Final selected 10K item IDs (returns all if total is less than 10K)
    """
    if not method_results:
        return []
    
    # Step 1: Count votes for each item (how many methods selected it)
    item_votes = {}
    
    for i, method_result in enumerate(method_results):
        print(f"Method {i+1} contributed {len(method_result)} items")
        for item_id in method_result:
            item_votes[item_id] = item_votes.get(item_id, 0) + 1
    
    # Step 2: Sort items by vote count
    voted_items = list(item_votes.items())
    
    if random_tie_breaking:
        # Set random seed for reproducibility
        if random_seed is not None:
            random.seed(random_seed)
        
        # Sort by vote count descending, random order for ties
        voted_items.sort(key=lambda x: (-x[1], random.random()))
        print(f"Using random sorting for ties (random seed: {random_seed})")
    else:
        # Sort by vote count descending, then by item ID ascending for stability
        voted_items.sort(key=lambda x: (-x[1], x[0]))
        print("Using ID sorting for ties")
    
    # Count vote distribution
    vote_distribution = {}
    for _, votes in voted_items:
        vote_distribution[votes] = vote_distribution.get(votes, 0) + 1
    
    print(f"Vote distribution: {vote_distribution}")
    
    # Step 3: Take top 10K items
    final_top10k = [item_id for item_id, votes in voted_items[:10000]]
    
    print(f"Fusion selected {len(final_top10k)} items in total")
    
    return final_top10k


def joint_filtering_fusion(base_dir: str, dimensions: List[str], gte_threshold: int,
                          score_type: str = "overall_score", output_file: str = None,
                          random_tie_breaking: bool = False, random_seed: int = None) -> List[str]:
    """
    Perform joint filtering and fusion for specified dimensions
    
    Args:
    - base_dir: Path to the dataset folder
    - dimensions: List of dimensions to fuse
    - gte_threshold: Threshold value
    - score_type: Score type
    - output_file: Output file path (optional)
    - random_tie_breaking: Whether to use random sorting when votes are equal
    - random_seed: Random seed
    
    Returns:
    - final_results: List of fused result IDs
    """
    print(f"Starting joint filtering fusion...")
    print(f"Dimensions: {dimensions}")
    print(f"Threshold: gte_{gte_threshold}")
    print(f"Score type: {score_type}")
    print("-" * 50)
    
    # Find corresponding data files
    files = find_dataset_files(base_dir, dimensions, gte_threshold, score_type)
    
    if len(files) != len(dimensions):
        missing = set(dimensions) - set(files.keys())
        print(f"Warning: Missing files for the following dimensions: {missing}")
    
    # Load data IDs for each dimension
    method_results = []
    for dimension in dimensions:
        if dimension in files:
            ids = load_dataset_ids(files[dimension])
            method_results.append(ids)
        else:
            print(f"Skipping dimension {dimension} (file not found)")
    
    if not method_results:
        print("Error: Failed to load any data files")
        return []
    
    print("-" * 50)
    
    # Perform equal-weight fusion
    final_results = equal_weight_fusion(*method_results, 
                                      random_tie_breaking=random_tie_breaking,
                                      random_seed=random_seed)
    
    # Save results to file (if output file is specified)
    if output_file:
        try:
            output_data = {
                "fusion_config": {
                    "dimensions": dimensions,
                    "gte_threshold": gte_threshold,
                    "score_type": score_type,
                    "total_methods": len(method_results),
                    "final_count": len(final_results)
                },
                "fused_ids": final_results
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            
            print(f"Results saved to: {output_file}")
        
        except Exception as e:
            print(f"Error saving results: {e}")
    
    return final_results


def main():
    """Main function: Demonstrates how to use joint filtering fusion"""
    
    # Configuration parameters
    base_dir = "selected_10k_distribution_refDS2_json"
    
    # Example 1: Fuse all four dimensions, threshold 7, using overall_score
    dimensions = ["Audience", "Style", "Task", "Topic"]
    gte_threshold = 7
    score_type = "overall_score"
    
    output_file = f"fused_results_{score_type}_gte_{gte_threshold}.json"
    
    print("=" * 60)
    print("Joint Filtering Fusion - Example Run")
    print("=" * 60)
    
    final_results = joint_filtering_fusion(
        base_dir=base_dir,
        dimensions=dimensions,
        gte_threshold=gte_threshold,
        score_type=score_type,
        output_file=output_file
    )
    
    print(f"\nFinal fusion result: {len(final_results)} items")
    print(f"First 10 IDs: {final_results[:10]}")
    
    # # Example 2: Fuse only three dimensions
    # print("\n" + "=" * 60)
    # print("Example 2: Fuse three dimensions (Audience, Style, Task)")
    # print("=" * 60)
    
    # dimensions_3 = ["Audience", "Style", "Task"]
    # output_file_3 = f"fused_results_3dims_{score_type}_gte_{gte_threshold}.json"
    
    # final_results_3 = joint_filtering_fusion(
    #     base_dir=base_dir,
    #     dimensions=dimensions_3,
    #     gte_threshold=gte_threshold,
    #     score_type=score_type,
    #     output_file=output_file_3
    # )
    
    # print(f"\nThree-dimension fusion result: {len(final_results_3)} items")


if __name__ == "__main__":
    main()
