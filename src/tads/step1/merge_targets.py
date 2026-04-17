"""Step 1.2: Merge all target datasets into unified Alpaca-format parquet (§5.1)."""

import glob
import json
import os
import re

import pandas as pd
from datasets import Dataset, DatasetDict


def run(config):
    """Merge target splits into a single parquet file."""
    root_dir = config.split_target_dir
    output_path = config.combined_target_parquet

    dataset_dict = _combine_split_target_datasets(root_dir)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    dataset_dict["train"].to_parquet(output_path)
    print(f"Saved combined target dataset ({len(dataset_dict['train'])} samples) to {output_path}")


def _combine_split_target_datasets(root_dir: str) -> DatasetDict:
    combined = {"dataset": [], "id": [], "messages": []}

    # TruthfulQA
    path = os.path.join(root_dir, "truthfulqa", "TruthfulQA.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        for idx, row in df.iterrows():
            combined["dataset"].append("truthfulqa")
            combined["id"].append(f"truthfulqa_{idx}")
            combined["messages"].append([
                {"role": "user", "content": row["Question"].strip()},
                {"role": "assistant", "content": row["Best Answer"].strip()},
            ])

    # GSM
    path = os.path.join(root_dir, "gsm", "test.jsonl")
    if os.path.exists(path):
        with open(path) as f:
            for idx, line in enumerate(f):
                ex = json.loads(line)
                answer = re.sub(r"(\d),(\d)", r"\1\2", ex["answer"].strip())
                combined["dataset"].append("gsm")
                combined["id"].append(f"gsm_{idx}")
                combined["messages"].append([
                    {"role": "user", "content": ex["question"].strip()},
                    {"role": "assistant", "content": answer},
                ])

    # TyDiQA
    tydiqa_dir = os.path.join(root_dir, "tydiqa")
    if os.path.exists(tydiqa_dir):
        for filename in os.listdir(tydiqa_dir):
            if not filename.endswith(".json"):
                continue
            with open(os.path.join(tydiqa_dir, filename)) as f:
                data = json.load(f)
            for article in data["data"]:
                for para in article["paragraphs"]:
                    for qa in para["qas"]:
                        for ans in qa["answers"]:
                            combined["dataset"].append("tydiqa")
                            combined["id"].append(qa["id"])
                            combined["messages"].append([
                                {"role": "user", "content": f"Context: {para['context']}\nQuestion: {qa['question']}"},
                                {"role": "assistant", "content": ans["text"]},
                            ])

    # MMLU
    mmlu_test = os.path.join(root_dir, "mmlu", "test")
    choices = ["A", "B", "C", "D"]
    if os.path.exists(mmlu_test):
        for fname in os.listdir(mmlu_test):
            if not fname.endswith("_test.csv"):
                continue
            subject = fname.split("_test.csv")[0]
            df = pd.read_csv(os.path.join(mmlu_test, fname), header=None)
            for i in range(df.shape[0]):
                question = df.iloc[i, 0]
                options = "\n".join(f"{ch}. {df.iloc[i, j + 1]}" for j, ch in enumerate(choices))
                combined["dataset"].append("mmlu")
                combined["id"].append(f"mmlu_{subject}_{i}")
                combined["messages"].append([
                    {"role": "user", "content": f"Subject: {subject.replace('_', ' ')}\nQuestion: {question}\n{options}"},
                    {"role": "assistant", "content": str(df.iloc[i, -1])},
                ])

    # BBH
    bbh_dir = os.path.join(root_dir, "bbh", "bbh")
    cot_dir = os.path.join(root_dir, "bbh", "cot-prompts")
    if os.path.exists(bbh_dir) and os.path.exists(cot_dir):
        for fname in os.listdir(bbh_dir):
            task = fname.replace(".json", "")
            prompt_file = os.path.join(cot_dir, f"{task}.txt")
            if not os.path.exists(prompt_file):
                continue
            with open(os.path.join(bbh_dir, fname)) as f:
                examples = json.load(f)["examples"]
            with open(prompt_file) as f:
                task_prompt = "".join(f.readlines()[2:])
            # Remove CoT reasoning
            parts = task_prompt.split("\n\n")
            new_parts = []
            for p in parts:
                if p.startswith("Q:") and "So the answer is" in p:
                    answer = p.split("So the answer is")[-1].strip()
                    question = p.split("\nA:")[0].strip()
                    new_parts.append(question + "\nA: " + answer)
                else:
                    new_parts.append(p)
            prompt_str = "\n\n".join(new_parts)
            for i, ex in enumerate(examples):
                combined["dataset"].append("bbh")
                combined["id"].append(f"bbh_{task}_{i}")
                combined["messages"].append([
                    {"role": "user", "content": f"{prompt_str}\n\nQ: {ex['input']}"},
                    {"role": "assistant", "content": ex["target"]},
                ])

    return DatasetDict({"train": Dataset.from_dict(combined)})
