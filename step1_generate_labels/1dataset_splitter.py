"""
Unified Dataset Splitter for Multiple Evaluation Datasets
This script splits various evaluation datasets (BBH, GSM, MMLU, TruthfulQA, TyDiQA) 
into target and evaluation sets with a 20:80 ratio.
"""

import os
import json
import random
import shutil
import pandas as pd
from tqdm import tqdm


def print_section_header(section_name):
    """Print a formatted section header for better readability"""
    print(f"\n{'='*60}")
    print(f"🔄 Processing {section_name}")
    print(f"{'='*60}")


def split_bbh_dataset():
    """
    Split BBH (Big Bench Hard) dataset
    - Input: Multiple JSON files in bbh/ directory
    - Output: 20% for target set, 80% for evaluation
    """
    print_section_header("BBH Dataset")
    
    # Define input and output paths
    input_root = "../data/eval/bbh"
    input_bbh_dir = os.path.join(input_root, "bbh")
    output_train_root = "../data/eval/split_target/bbh"
    output_eval_root = "../data/eval/split_eval/bbh"
    
    output_train_bbh = os.path.join(output_train_root, "bbh")
    output_eval_bbh = os.path.join(output_eval_root, "bbh")
    
    # Create output directories
    os.makedirs(output_train_bbh, exist_ok=True)
    os.makedirs(output_eval_bbh, exist_ok=True)
    print(f"📁 Created output directories: {output_train_bbh}, {output_eval_bbh}")
    
    total_count = 0
    
    # Process each JSON file in the BBH directory
    task_files = [f for f in os.listdir(input_bbh_dir) if f.endswith(".json")]
    print(f"📊 Found {len(task_files)} BBH task files to process")
    
    for file in tqdm(task_files, desc="Splitting BBH tasks"):
        path = os.path.join(input_bbh_dir, file)
        
        # Load the JSON data
        with open(path, "r") as f:
            raw = json.load(f)
        
        examples = raw["examples"]
        
        # Set random seed for reproducible splits
        random.seed(42)
        total_indices = list(range(len(examples)))
        target_indices = set(random.sample(total_indices, int(len(examples) * 0.2)))
        
        # Split examples into target and eval sets
        target_examples = [examples[i] for i in range(len(examples)) if i in target_indices]
        eval_examples = [examples[i] for i in range(len(examples)) if i not in target_indices]
        
        total_count += len(total_indices)
        
        # Save split datasets
        with open(os.path.join(output_train_bbh, file), "w") as f:
            json.dump({"examples": target_examples}, f, indent=2, ensure_ascii=False)
        with open(os.path.join(output_eval_bbh, file), "w") as f:
            json.dump({"examples": eval_examples}, f, indent=2, ensure_ascii=False)
        
        print(f"✅ {file}: {len(target_examples)} target / {len(eval_examples)} eval")
    
    # Copy other files and directories (excluding bbh/ subdirectory)
    print(f"📂 Copying additional files from {input_root}")
    for name in os.listdir(input_root):
        src_path = os.path.join(input_root, name)
        for target_root in [output_train_root, output_eval_root]:
            dst_path = os.path.join(target_root, name)
            if name == "bbh":
                continue  # Already processed
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            elif os.path.isfile(src_path):
                shutil.copy2(src_path, dst_path)
            print(f"📂 Copied {name} → {target_root}")
    
    print(f"🎯 BBH Dataset Summary: Total {total_count} examples processed")


def split_gsm_dataset():
    """
    Split GSM (Grade School Math) dataset
    - Input: Single JSONL file
    - Output: 20% for target set, 80% for evaluation
    """
    print_section_header("GSM Dataset")
    
    # Define paths
    input_path = "../data/eval/gsm/test.jsonl"
    output_train_dir = "../data/eval/split_target/gsm"
    output_eval_dir = "../data/eval/split_eval/gsm"
    
    # Create output directories
    os.makedirs(output_train_dir, exist_ok=True)
    os.makedirs(output_eval_dir, exist_ok=True)
    print(f"📁 Created output directories: {output_train_dir}, {output_eval_dir}")
    
    # Read the original JSONL file
    print(f"📖 Reading GSM dataset from: {input_path}")
    with open(input_path, "r") as f:
        lines = f.readlines()
    
    print(f"📊 Total lines in GSM dataset: {len(lines)}")
    
    # Create random split indices
    total_indices = list(range(len(lines)))
    random.seed(42)
    target_indices = set(random.sample(total_indices, int(len(lines) * 0.2)))
    
    # Split lines while maintaining original order
    target_lines = [line for idx, line in enumerate(lines) if idx in target_indices]
    eval_lines = [line for idx, line in enumerate(lines) if idx not in target_indices]
    
    # Save split datasets
    target_output_path = os.path.join(output_train_dir, "test.jsonl")
    eval_output_path = os.path.join(output_eval_dir, "test.jsonl")
    
    with open(target_output_path, "w") as f:
        f.writelines(target_lines)
    with open(eval_output_path, "w") as f:
        f.writelines(eval_lines)
    
    print(f"💾 Saved target set to: {target_output_path}")
    print(f"💾 Saved evaluation set to: {eval_output_path}")
    print(f"✅ GSM split completed: {len(target_lines)} target / {len(eval_lines)} eval")


def split_mmlu_dataset():
    """
    Split MMLU (Massive Multitask Language Understanding) dataset
    - Input: Multiple CSV files for different subjects
    - Output: 20% for target set, 80% for evaluation
    """
    print_section_header("MMLU Dataset")
    
    # Define paths
    input_mmlu_dir = "../data/eval/mmlu"
    input_test_dir = os.path.join(input_mmlu_dir, "test")
    
    output_train_root = "../data/eval/split_target/mmlu"
    output_eval_root = "../data/eval/split_eval/mmlu"
    output_train_test = os.path.join(output_train_root, "test")
    output_eval_test = os.path.join(output_eval_root, "test")
    
    # Create output directories
    os.makedirs(output_train_test, exist_ok=True)
    os.makedirs(output_eval_test, exist_ok=True)
    print(f"📁 Created output directories: {output_train_test}, {output_eval_test}")
    
    # Get all subject test files
    subjects = sorted([
        f.split("_test.csv")[0]
        for f in os.listdir(input_test_dir)
        if f.endswith("_test.csv")
    ])
    
    print(f"📊 Found {len(subjects)} MMLU subjects to process")
    total_count = 0
    
    # Process each subject
    for subject in subjects:
        path = os.path.join(input_test_dir, f"{subject}_test.csv")
        print(f"📖 Processing subject: {subject}")
        
        # Load CSV data
        df = pd.read_csv(path, header=None)
        indices = list(range(len(df)))
        total_count += len(indices)
        
        # Create random split
        random.seed(42)
        target_indices = set(random.sample(indices, int(len(df) * 0.2)))
        
        # Split dataframe
        df_target = df[[i in target_indices for i in df.index]]
        df_eval = df[[i not in target_indices for i in df.index]]
        
        # Save split datasets
        target_path = os.path.join(output_train_test, f"{subject}_test.csv")
        eval_path = os.path.join(output_eval_test, f"{subject}_test.csv")
        
        df_target.to_csv(target_path, index=False, header=False)
        df_eval.to_csv(eval_path, index=False, header=False)
        
        print(f"✅ {subject}: {len(df_target)} target / {len(df_eval)} eval")
    
    # Copy other files (dev, categories.py, etc.) excluding test directory
    print(f"📂 Copying additional MMLU files from {input_mmlu_dir}")
    for item in os.listdir(input_mmlu_dir):
        src_path = os.path.join(input_mmlu_dir, item)
        if item == "test":
            continue  # Already processed
        
        for target_root in [output_train_root, output_eval_root]:
            dst_path = os.path.join(target_root, item)
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            else:
                shutil.copy2(src_path, dst_path)
            print(f"📂 Copied {item} → {target_root}")
    
    print(f"🎯 MMLU Dataset Summary: Total {total_count} examples across {len(subjects)} subjects")


def split_truthfulqa_dataset():
    """
    Split TruthfulQA dataset
    - Input: Single CSV file
    - Output: 20% for target set, 80% for evaluation
    """
    print_section_header("TruthfulQA Dataset")
    
    # Define paths
    input_path = "../data/eval/truthfulqa/TruthfulQA.csv"
    output_train_dir = "../data/eval/split_target/truthfulqa"
    output_eval_dir = "../data/eval/split_eval/truthfulqa"
    
    # Create output directories
    os.makedirs(output_train_dir, exist_ok=True)
    os.makedirs(output_eval_dir, exist_ok=True)
    print(f"📁 Created output directories: {output_train_dir}, {output_eval_dir}")
    
    # Read the original CSV
    print(f"📖 Reading TruthfulQA dataset from: {input_path}")
    df = pd.read_csv(input_path)
    total_indices = list(range(len(df)))
    print(f"📊 Total rows in TruthfulQA dataset: {len(df)}")
    
    # Create random split (20% for target set)
    random.seed(42)
    target_indices = set(random.sample(total_indices, int(len(df) * 0.2)))
    
    # Split dataframe while maintaining original order
    df_target = df[[i in target_indices for i in df.index]]
    df_eval = df[[i not in target_indices for i in df.index]]
    
    # Save split datasets
    target_output_path = os.path.join(output_train_dir, "TruthfulQA.csv")
    eval_output_path = os.path.join(output_eval_dir, "TruthfulQA.csv")
    
    df_target.to_csv(target_output_path, index=False)
    df_eval.to_csv(eval_output_path, index=False)
    
    print(f"💾 Saved target set to: {target_output_path}")
    print(f"💾 Saved evaluation set to: {eval_output_path}")
    print(f"✅ TruthfulQA split completed: {len(df_target)} target / {len(df_eval)} eval")


def split_tydiqa_dataset():
    """
    Split TyDiQA (Typologically Diverse Question Answering) dataset
    - Input: Two JSON files (train and dev)
    - Output: 20% for target set, 80% for evaluation for each file
    """
    print_section_header("TyDiQA Dataset")
    
    # Define paths
    input_dir = "../data/eval/tydiqa"
    output_train_dir = "../data/eval/split_target/tydiqa"
    output_eval_dir = "../data/eval/split_eval/tydiqa"
    
    # Create output directories
    os.makedirs(output_train_dir, exist_ok=True)
    os.makedirs(output_eval_dir, exist_ok=True)
    print(f"📁 Created output directories: {output_train_dir}, {output_eval_dir}")
    
    def split_and_save_tydiqa_file(filename):
        """Helper function to split and save a single TyDiQA file"""
        print(f"📖 Processing TyDiQA file: {filename}")
        
        path = os.path.join(input_dir, filename)
        with open(path, "r") as f:
            raw = json.load(f)
        
        version = raw.get("version", "")
        data = raw["data"]
        total_indices = list(range(len(data)))
        
        print(f"📊 Total examples in {filename}: {len(data)}")
        
        # Create random split
        random.seed(42)
        target_indices = set(random.sample(total_indices, int(len(data) * 0.2)))
        
        # Split data while maintaining original order
        data_target = [data[i] for i in range(len(data)) if i in target_indices]
        data_eval = [data[i] for i in range(len(data)) if i not in target_indices]
        
        # Save split datasets
        target_output_path = os.path.join(output_train_dir, filename)
        eval_output_path = os.path.join(output_eval_dir, filename)
        
        with open(target_output_path, "w") as f:
            json.dump({"version": version, "data": data_target}, f, ensure_ascii=False, indent=2)
        with open(eval_output_path, "w") as f:
            json.dump({"version": version, "data": data_eval}, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Saved target set to: {target_output_path}")
        print(f"💾 Saved evaluation set to: {eval_output_path}")
        print(f"✅ {filename}: {len(data_target)} target / {len(data_eval)} eval")
        
        return len(data)
    
    # Process both TyDiQA files
    files_to_process = [
        "tydiqa-goldp-v1.1-train.json",
        "tydiqa-goldp-v1.1-dev.json"
    ]
    
    total_examples = 0
    for filename in files_to_process:
        total_examples += split_and_save_tydiqa_file(filename)
    
    print(f"🎯 TyDiQA Dataset Summary: Total {total_examples} examples processed across {len(files_to_process)} files")


def main():
    """
    Main function to execute all dataset splitting operations
    """
    print("🚀 Starting Unified Dataset Splitting Process")
    print("This script will split multiple evaluation datasets into 20:80 target/eval splits")
    print(f"Random seed: 42 (for reproducible results)")
    
    try:
        # Process each dataset
        split_bbh_dataset()
        split_gsm_dataset()
        split_mmlu_dataset()
        split_truthfulqa_dataset()
        split_tydiqa_dataset()
        
        print("\n" + "="*60)
        print("🎉 All datasets have been successfully split!")
        print("✨ Target sets (20%) and evaluation sets (80%) are ready")
        print("📍 Output locations:")
        print("   - Target sets: ../data/eval/split_target/")
        print("   - Evaluation sets: ../data/eval/split_eval/")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error occurred during processing: {str(e)}")
        print("Please check the input file paths and try again.")
        raise


if __name__ == "__main__":
    main()
