"""Step 1.1: Split evaluation datasets into target (20%) and eval (80%) sets (§5.1)."""

import json
import os
import random
import shutil

import pandas as pd
from tqdm import tqdm


def run(config):
    """Split BBH, GSM, MMLU, TruthfulQA, TyDiQA datasets."""
    seed = config.random_seed
    ratio = config.split_ratio
    eval_dir = config.eval_dir
    target_dir = config.split_target_dir
    eval_out_dir = config.split_eval_dir

    _split_bbh(eval_dir, target_dir, eval_out_dir, ratio, seed)
    _split_gsm(eval_dir, target_dir, eval_out_dir, ratio, seed)
    _split_mmlu(eval_dir, target_dir, eval_out_dir, ratio, seed)
    _split_truthfulqa(eval_dir, target_dir, eval_out_dir, ratio, seed)
    _split_tydiqa(eval_dir, target_dir, eval_out_dir, ratio, seed)

    print("\nAll datasets split successfully.")
    print(f"  Target sets: {target_dir}")
    print(f"  Eval sets:   {eval_out_dir}")


# ---------------------------------------------------------------------------
# Per-dataset helpers
# ---------------------------------------------------------------------------

def _split_bbh(eval_dir, target_dir, eval_out_dir, ratio, seed):
    input_root = os.path.join(eval_dir, "bbh")
    input_bbh = os.path.join(input_root, "bbh")
    out_target = os.path.join(target_dir, "bbh", "bbh")
    out_eval = os.path.join(eval_out_dir, "bbh", "bbh")
    os.makedirs(out_target, exist_ok=True)
    os.makedirs(out_eval, exist_ok=True)

    task_files = [f for f in os.listdir(input_bbh) if f.endswith(".json")]
    for file in tqdm(task_files, desc="Splitting BBH"):
        with open(os.path.join(input_bbh, file)) as f:
            examples = json.load(f)["examples"]
        random.seed(seed)
        indices = set(random.sample(range(len(examples)), int(len(examples) * ratio)))
        target = [examples[i] for i in range(len(examples)) if i in indices]
        evl = [examples[i] for i in range(len(examples)) if i not in indices]
        with open(os.path.join(out_target, file), "w") as f:
            json.dump({"examples": target}, f, indent=2, ensure_ascii=False)
        with open(os.path.join(out_eval, file), "w") as f:
            json.dump({"examples": evl}, f, indent=2, ensure_ascii=False)

    # Copy auxiliary files (cot-prompts etc.)
    for name in os.listdir(input_root):
        if name == "bbh":
            continue
        src = os.path.join(input_root, name)
        for dst_root in [os.path.join(target_dir, "bbh"), os.path.join(eval_out_dir, "bbh")]:
            dst = os.path.join(dst_root, name)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            elif os.path.isfile(src):
                shutil.copy2(src, dst)


def _split_gsm(eval_dir, target_dir, eval_out_dir, ratio, seed):
    input_path = os.path.join(eval_dir, "gsm", "test.jsonl")
    for out_dir in [os.path.join(target_dir, "gsm"), os.path.join(eval_out_dir, "gsm")]:
        os.makedirs(out_dir, exist_ok=True)

    with open(input_path) as f:
        lines = f.readlines()

    random.seed(seed)
    target_idx = set(random.sample(range(len(lines)), int(len(lines) * ratio)))
    target = [l for i, l in enumerate(lines) if i in target_idx]
    evl = [l for i, l in enumerate(lines) if i not in target_idx]

    with open(os.path.join(target_dir, "gsm", "test.jsonl"), "w") as f:
        f.writelines(target)
    with open(os.path.join(eval_out_dir, "gsm", "test.jsonl"), "w") as f:
        f.writelines(evl)
    print(f"GSM: {len(target)} target / {len(evl)} eval")


def _split_mmlu(eval_dir, target_dir, eval_out_dir, ratio, seed):
    input_test = os.path.join(eval_dir, "mmlu", "test")
    out_target_test = os.path.join(target_dir, "mmlu", "test")
    out_eval_test = os.path.join(eval_out_dir, "mmlu", "test")
    os.makedirs(out_target_test, exist_ok=True)
    os.makedirs(out_eval_test, exist_ok=True)

    subjects = sorted(f.split("_test.csv")[0] for f in os.listdir(input_test) if f.endswith("_test.csv"))
    for subject in subjects:
        df = pd.read_csv(os.path.join(input_test, f"{subject}_test.csv"), header=None)
        random.seed(seed)
        target_idx = set(random.sample(range(len(df)), int(len(df) * ratio)))
        df_t = df[[i in target_idx for i in df.index]]
        df_e = df[[i not in target_idx for i in df.index]]
        df_t.to_csv(os.path.join(out_target_test, f"{subject}_test.csv"), index=False, header=False)
        df_e.to_csv(os.path.join(out_eval_test, f"{subject}_test.csv"), index=False, header=False)

    # Copy dev, categories etc.
    mmlu_root = os.path.join(eval_dir, "mmlu")
    for item in os.listdir(mmlu_root):
        if item == "test":
            continue
        src = os.path.join(mmlu_root, item)
        for dst_root in [os.path.join(target_dir, "mmlu"), os.path.join(eval_out_dir, "mmlu")]:
            dst = os.path.join(dst_root, item)
            if os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)


def _split_truthfulqa(eval_dir, target_dir, eval_out_dir, ratio, seed):
    input_path = os.path.join(eval_dir, "truthfulqa", "TruthfulQA.csv")
    for d in [os.path.join(target_dir, "truthfulqa"), os.path.join(eval_out_dir, "truthfulqa")]:
        os.makedirs(d, exist_ok=True)

    df = pd.read_csv(input_path)
    random.seed(seed)
    target_idx = set(random.sample(range(len(df)), int(len(df) * ratio)))
    df_t = df[[i in target_idx for i in df.index]]
    df_e = df[[i not in target_idx for i in df.index]]
    df_t.to_csv(os.path.join(target_dir, "truthfulqa", "TruthfulQA.csv"), index=False)
    df_e.to_csv(os.path.join(eval_out_dir, "truthfulqa", "TruthfulQA.csv"), index=False)
    print(f"TruthfulQA: {len(df_t)} target / {len(df_e)} eval")


def _split_tydiqa(eval_dir, target_dir, eval_out_dir, ratio, seed):
    input_dir = os.path.join(eval_dir, "tydiqa")
    for d in [os.path.join(target_dir, "tydiqa"), os.path.join(eval_out_dir, "tydiqa")]:
        os.makedirs(d, exist_ok=True)

    for filename in ["tydiqa-goldp-v1.1-train.json", "tydiqa-goldp-v1.1-dev.json"]:
        path = os.path.join(input_dir, filename)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            raw = json.load(f)
        data = raw["data"]
        random.seed(seed)
        target_idx = set(random.sample(range(len(data)), int(len(data) * ratio)))
        data_t = [data[i] for i in range(len(data)) if i in target_idx]
        data_e = [data[i] for i in range(len(data)) if i not in target_idx]
        for out_dir, subset in [(os.path.join(target_dir, "tydiqa"), data_t),
                                (os.path.join(eval_out_dir, "tydiqa"), data_e)]:
            with open(os.path.join(out_dir, filename), "w") as f:
                json.dump({"version": raw.get("version", ""), "data": subset}, f, ensure_ascii=False, indent=2)
        print(f"TyDiQA {filename}: {len(data_t)} target / {len(data_e)} eval")
