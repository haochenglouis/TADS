from datasets import Dataset, DatasetDict, load_dataset
import pandas as pd
import os
import json
import re
import random
import glob
from types import SimpleNamespace


def process_truthfulqa():
    # Load TruthfulQA data
    questions = pd.read_csv(os.path.join("../data/eval/truthfulqa", "TruthfulQA.csv"))

    data = {
        'dataset': [],
        'id': [],
        'messages': []
    }

    # Format prompts and create messages
    for idx in questions.index:
        question = questions.loc[idx, 'Question'].strip()
        best_answer = questions.loc[idx, 'Best Answer'].strip()

        messages = [
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': best_answer}
        ]

        data['dataset'].append('truthfulqa')
        data['id'].append(f'truthfulqa_{idx}')
        data['messages'].append(messages)

    return data


def process_tydiqa():
    # Load TydiQA data
    test_data = []
    with open(os.path.join("../data/eval/tydiqa", "tydiqa-goldp-v1.1-dev.json")) as fin:
        dev_data = json.load(fin)
        for article in dev_data["data"]:
            for paragraph in article["paragraphs"]:
                for qa in paragraph["qas"]:
                    example = {
                        "id": qa["id"],
                        "lang": qa["id"].split("-")[0],
                        "context": paragraph["context"],
                        "question": qa["question"],
                        "answers": qa["answers"]
                    }
                    test_data.append(example)

    data = {
        'dataset': [],
        'id': [],
        'messages': []
    }

    # Create messages for each example
    for example in test_data:
        for ans in example["answers"]:
            # Format the prompt with context and question
            user_content = f"Context: {example['context']}\nQuestion: {example['question']}"
            assistant_content = ans["text"]

            messages = [
                {'role': 'user', 'content': user_content},
                {'role': 'assistant', 'content': assistant_content}
            ]

            data['dataset'].append('tydiqa')
            data['id'].append(f"{example['id']}")
            data['messages'].append(messages)

    return data


def process_mmlu():
    # Get all subjects
    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join("../data/eval/mmlu", "test"))
            if "_test.csv" in f
        ]
    )

    choices = ["A", "B", "C", "D"]

    data = {
        'dataset': [],
        'id': [],
        'messages': []
    }

    for subject in subjects:
        # Load dev and test data
        dev_df = pd.read_csv(
            os.path.join("../data/eval/mmlu", "dev", subject + "_dev.csv"),
            header=None
        )[: 5]

        test_df = pd.read_csv(
            os.path.join("../data/eval/mmlu", "test", subject + "_test.csv"),
            header=None
        )

        for i in range(test_df.shape[0]):
            # Format the question with choices
            question = test_df.iloc[i, 0]
            options = ""
            for j in range(len(choices)):
                options += f"\n{choices[j]}. {test_df.iloc[i, j + 1]}"

            user_content = f"Subject: {subject.replace('_', ' ')}\nQuestion: {question}{options}"

            # Get the answer (which is in the last column)
            answer_idx = test_df.iloc[i, -1]
            answer = f"{answer_idx}"

            messages = [
                {'role': 'user', 'content': user_content},
                {'role': 'assistant', 'content': answer}
            ]

            data['dataset'].append('mmlu')
            data['id'].append(f'mmlu_{subject}_{i}')
            data['messages'].append(messages)

    return data


def process_gsm():
    # Load GSM data
    test_data = []
    with open(os.path.join("../data/eval/gsm", "test.jsonl")) as fin:
        for line in fin:
            example = json.loads(line)
            test_data.append({
                "question": example["question"],
                "answer": example["answer"]
            })

    # Clean up answers (remove commas in numbers)
    for example in test_data:
        example["answer"] = re.sub(r"(\d),(\d)", r"\1\2", example["answer"])

    data = {
        'dataset': [],
        'id': [],
        'messages': []
    }

    # Create messages for each example
    for idx, example in enumerate(test_data):
        question = example["question"].strip()
        answer = example["answer"].strip()

        messages = [
            {'role': 'user', 'content': question},
            {'role': 'assistant', 'content': answer}
        ]

        data['dataset'].append('gsm')
        data['id'].append(f'gsm_{idx}')
        data['messages'].append(messages)

    return data


def process_bbh():
    print("Processing BBH dataset...")

    # Create SimpleNamespace for arguments
    args = SimpleNamespace(
        data_dir="../data/eval/bbh",
        no_cot=True,
        max_num_examples_per_task=5
    )

    # Load BBH tasks and prompts
    all_tasks = {}
    task_files = glob.glob(os.path.join(args.data_dir, "bbh", "*.json"))
    for task_file in task_files:
        with open(task_file, "r") as f:
            task_name = os.path.basename(task_file).split(".")[0]
            all_tasks[task_name] = json.load(f)["examples"]
            if args.max_num_examples_per_task:
                all_tasks[task_name] = random.sample(
                    all_tasks[task_name],
                    args.max_num_examples_per_task
                )

    # Load task prompts
    all_prompts = {}
    cot_prompt_files = glob.glob(os.path.join(args.data_dir, "cot-prompts", "*.txt"))
    for cot_prompt_file in cot_prompt_files:
        with open(cot_prompt_file, "r") as f:
            task_name = os.path.basename(cot_prompt_file).split(".")[0]
            task_prompt = "".join(f.readlines()[2:])  # Skip first two lines

            # If not using CoT, simplify the prompts
            if args.no_cot:
                prompt_fields = task_prompt.split("\n\n")
                new_prompt_fields = []
                for prompt_field in prompt_fields:
                    if prompt_field.startswith("Q:"):
                        answer = prompt_field.split("So the answer is")[-1].strip()
                        question = prompt_field.split("\nA:")[0].strip()
                        new_prompt_fields.append(question + "\nA: " + answer)
                    else:
                        new_prompt_fields.append(prompt_field)
                task_prompt = "\n\n".join(new_prompt_fields)

            all_prompts[task_name] = task_prompt

    # Ensure task names match between data and prompts
    assert set(all_tasks.keys()) == set(all_prompts.keys()), \
        "Task names in task data and task prompts are not the same."

    data = {
        'dataset': [],
        'id': [],
        'messages': []
    }

    # Process each task
    for task_name in all_tasks.keys():
        task_examples = all_tasks[task_name]
        task_prompt = all_prompts[task_name]

        for i, example in enumerate(task_examples):
            # Format the user question
            user_content = task_prompt.strip() + "\n\nQ: " + example["input"]

            # Get the target answer
            assistant_content = example["target"]

            messages = [
                {'role': 'user', 'content': user_content},
                {'role': 'assistant', 'content': assistant_content}
            ]

            data['dataset'].append('bbh')
            data['id'].append(f'bbh_{task_name}_{i}')
            data['messages'].append(messages)

    return data


def combine_datasets():
    truthfulqa_data = process_truthfulqa()
    tydiqa_data = process_tydiqa()
    mmlu_data = process_mmlu()
    gsm_data = process_gsm()
    bbh_data = process_bbh()

    # Combine all data
    combined_data = {
        'dataset': [],
        'id': [],
        'messages': []
    }

    # Add data from each dataset
    for dataset_data in [truthfulqa_data, tydiqa_data, mmlu_data, gsm_data, bbh_data]:
        combined_data['dataset'].extend(dataset_data['dataset'])
        combined_data['id'].extend(dataset_data['id'])
        combined_data['messages'].extend(dataset_data['messages'])

    # Create dataset
    combined_dataset = Dataset.from_dict(combined_data)

    # Create DatasetDict
    dataset_dict = DatasetDict({
        'train': combined_dataset
    })

    return dataset_dict


def main():
    # Process and combine all datasets
    dataset_dict = combine_datasets()

    # Save the combined dataset
    save_dir = 'combined_datasets_alpaca_format'
    dataset_dict.save_to_disk(save_dir)

    # Save as parquet file
    dataset_dict['train'].to_parquet('combined_datasets_alpaca_format.parquet')

    # Print dataset info
    print("\nCombined Dataset Info:")
    print(dataset_dict)

    # Print counts by dataset
    dataset_counts = {}
    for dataset_name in dataset_dict['train']['dataset']:
        if dataset_name in dataset_counts:
            dataset_counts[dataset_name] += 1
        else:
            dataset_counts[dataset_name] = 1

    print("\nSample counts by dataset:")
    for dataset_name, count in dataset_counts.items():
        print(f"{dataset_name}: {count} samples")

    # Print a few examples
    print("\nSample examples:")
    for i in range(min(3, len(dataset_dict['train']))):
        print(f"\nExample {i + 1} from {dataset_dict['train']['dataset'][i]}:")
        print(f"ID: {dataset_dict['train']['id'][i]}")
        for message in dataset_dict['train']['messages'][i]:
            print(f"{message['role']}: {message['content'][:100]}...")


def combine_split_target_datasets(root_dir="../data/eval/split_target"):
    combined = {
        'dataset': [],
        'id': [],
        'messages': []
    }

    # 1. TruthfulQA
    truthful_path = os.path.join(root_dir, "truthfulqa", "TruthfulQA.csv")
    if os.path.exists(truthful_path):
        df = pd.read_csv(truthful_path)
        for idx, row in df.iterrows():
            q = row["Question"].strip()
            a = row["Best Answer"].strip()
            messages = [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a}
            ]
            combined['dataset'].append("truthfulqa")
            combined['id'].append(f"truthfulqa_{idx}")
            combined['messages'].append(messages)

    # 2. GSM
    gsm_path = os.path.join(root_dir, "gsm", "test.jsonl")
    if os.path.exists(gsm_path):
        with open(gsm_path) as f:
            for idx, line in enumerate(f):
                ex = json.loads(line)
                answer = re.sub(r"(\d),(\d)", r"\1\2", ex["answer"].strip())
                messages = [
                    {"role": "user", "content": ex["question"].strip()},
                    {"role": "assistant", "content": answer}
                ]
                combined['dataset'].append("gsm")
                combined['id'].append(f"gsm_{idx}")
                combined['messages'].append(messages)

    # 3. TydiQA
    tydiqa_dir = os.path.join(root_dir, "tydiqa")
    for filename in os.listdir(tydiqa_dir):
        if filename.endswith(".json"):
            data = json.load(open(os.path.join(tydiqa_dir, filename)))
            for article in data["data"]:
                for para in article["paragraphs"]:
                    for qa in para["qas"]:
                        context = para["context"]
                        question = qa["question"]
                        for ans in qa["answers"]:
                            user_content = f"Context: {context}\nQuestion: {question}"
                            messages = [
                                {"role": "user", "content": user_content},
                                {"role": "assistant", "content": ans["text"]}
                            ]
                            combined['dataset'].append("tydiqa")
                            combined['id'].append(qa["id"])
                            combined['messages'].append(messages)

    # 4. MMLU
    mmlu_test_dir = os.path.join(root_dir, "mmlu", "test")
    choices = ["A", "B", "C", "D"]
    for fname in os.listdir(mmlu_test_dir):
        if not fname.endswith("_test.csv"):
            continue
        subject = fname.split("_test.csv")[0]
        df = pd.read_csv(os.path.join(mmlu_test_dir, fname), header=None)
        for i in range(df.shape[0]):
            question = df.iloc[i, 0]
            options = "\n".join([f"{ch}. {df.iloc[i, j+1]}" for j, ch in enumerate(choices)])
            answer = df.iloc[i, -1]
            messages = [
                {"role": "user", "content": f"Subject: {subject.replace('_', ' ')}\nQuestion: {question}\n{options}"},
                {"role": "assistant", "content": answer}
            ]
            combined['dataset'].append("mmlu")
            combined['id'].append(f"mmlu_{subject}_{i}")
            combined['messages'].append(messages)

    # 5. BBH
    bbh_dir = os.path.join(root_dir, "bbh", "bbh")
    cot_dir = os.path.join(root_dir, "bbh", "cot-prompts")
    if os.path.exists(bbh_dir):
        for fname in os.listdir(bbh_dir):
            task = fname.replace(".json", "")
            prompt_file = os.path.join(cot_dir, f"{task}.txt")
            if not os.path.exists(prompt_file):
                continue
            with open(os.path.join(bbh_dir, fname)) as f:
                examples = json.load(f)["examples"]
            with open(prompt_file) as f:
                task_prompt = "".join(f.readlines()[2:])  # Skip first two lines
            prompt_fields = task_prompt.split("\n\n")
            new_prompt = []
            for pf in prompt_fields:
                if pf.startswith("Q:") and "So the answer is" in pf:
                    answer = pf.split("So the answer is")[-1].strip()
                    question = pf.split("\nA:")[0].strip()
                    new_prompt.append(question + "\nA: " + answer)
                else:
                    new_prompt.append(pf)
            prompt_str = "\n\n".join(new_prompt)

            for i, ex in enumerate(examples):
                user_msg = f"{prompt_str}\n\nQ: {ex['input']}"
                messages = [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": ex["target"]}
                ]
                combined["dataset"].append("bbh")
                combined["id"].append(f"bbh_{task}_{i}")
                combined["messages"].append(messages)

    return DatasetDict({"train": Dataset.from_dict(combined)})



if __name__ == "__main__":
    dataset_dict = combine_split_target_datasets("../data/eval/split_target")

    save_dir = 'step1_combined_target'
    dataset_dict.save_to_disk(save_dir)
    dataset_dict['train'].to_parquet('step1_combined_target_format.parquet')


