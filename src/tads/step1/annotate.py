"""Step 1.3: Annotate target dataset with proxy labels using LLM (§4.1)."""

import json
import os
import re

import json_repair
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from tads.utils.data_io import convert_alpaca_to_string
from tads.vllm_engine import create_sampling_params, create_vllm_engine


def run(config):
    """Generate Task/Style/Topic/Audience annotations for the combined target set."""
    input_path = config.combined_target_parquet
    output_pt = config.target_annotation_pt
    output_json = config.target_annotation_json
    model_cfg = config.annotation

    llm = create_vllm_engine(
        model_name=model_cfg.name,
        tensor_parallel_size=model_cfg.tensor_parallel_size,
        gpu_memory_utilization=model_cfg.gpu_memory_utilization,
        max_num_batched_tokens=32768 * 2,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.name)

    raw_dataset = load_dataset("parquet", data_files=input_path)
    dataset = {"train": raw_dataset["train"]}
    print(f"Processing {len(dataset['train'])} samples from {input_path}")

    processed = _batch_process(dataset, llm, tokenizer, model_cfg)

    # Retry failed samples
    failed = [item for item in processed if item["needs_regeneration"]]
    retry_count = 0
    max_retries = 999
    while failed and retry_count < max_retries:
        retry_count += 1
        print(f"Retry {retry_count}, {len(failed)} samples remaining")
        retry_dataset = {"train": [
            {"dataset": it["dataset"], "id": it["id"], "messages": it["messages"]}
            for it in failed
        ]}
        regenerated = _batch_process(retry_dataset, llm, tokenizer, model_cfg)
        id_to_item = {it["id"]: it for it in regenerated}
        new_failed = []
        for idx, item in enumerate(processed):
            if item["id"] in id_to_item:
                new_item = id_to_item[item["id"]]
                if new_item["needs_regeneration"]:
                    new_failed.append(new_item)
                else:
                    processed[idx] = new_item
        failed = new_failed

    os.makedirs(os.path.dirname(output_pt) or ".", exist_ok=True)
    torch.save(processed, output_pt)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=4)
    print(f"Saved {len(processed)} annotated samples to {output_pt}")


# ---------------------------------------------------------------------------

def _is_valid_summary_format(obj) -> bool:
    if not isinstance(obj, dict):
        return False
    if not {"Task", "Style", "Topic", "Audience"}.issubset(obj.keys()):
        return False
    if not isinstance(obj["Task"], str):
        return False
    for field in ["Style", "Topic", "Audience"]:
        if not isinstance(obj[field], list) or not all(isinstance(x, str) for x in obj[field]):
            return False
    return True


def _parse_json_data(text: str):
    try:
        json_pattern = re.compile(r"```(?:json)?\s*([\s\S]*?)```")
        match = json_pattern.search(text)
        json_str = match.group(1).strip() if match else text.strip()
        return json_repair.repair_json(json_str, return_objects=True)
    except Exception as e:
        print(f"JSON parsing error: {e}")
        return None


def _generate_prompts(dataset, tokenizer) -> list:
    prompts = []
    max_chars = 60000
    for sample in dataset["train"]:
        formatted = convert_alpaca_to_string(sample)
        if len(formatted) > max_chars:
            formatted = formatted[:max_chars]
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"""【Background Information】\n{formatted}

Please strictly output in JSON format and generate rich, high-quality summaries suitable for LLM.
Summarize the entire conversation **as a single summary**, not per message or per role.

Requirements:

1. Task: Describe the single most appropriate task type from the background as a short sentence or detailed multi-phrase descriptor (e.g., "Answering open-domain factual questions based on user input", "Editing and correcting grammatical errors in user-submitted text").
2. Style: List the top 3 most relevant styles or tones as detailed phrases or brief sentences (e.g., "Uses a formal and professional tone suitable for academic writing", "Employs a casual and conversational style with friendly expressions").
3. Topic: List 3 key topics covered in the background, each expressed as a short descriptive sentence or multi-part phrase (e.g., "Evaluation of neural network models for logical reasoning tasks", "Analysis of historical trends in machine translation datasets").
4. Audience: List the top 3 intended audiences using sentence-level or multi-phrase descriptors (e.g., "Graduate students and researchers in natural language processing", "General public interested in AI-generated content").

Only return a single JSON object as shown below, using full descriptive phrases:

```json
{{
    "Task": "<short sentence or descriptive phrase>",
    "Style": [
        "<descriptive phrase or sentence>",
        "<descriptive phrase or sentence>",
        "<descriptive phrase or sentence>"
    ],
    "Topic": [
        "<short sentence or compound phrase>",
        "<short sentence or compound phrase>",
        "<short sentence or compound phrase>"
    ],
    "Audience": [
        "<short sentence or descriptive phrase>",
        "<short sentence or descriptive phrase>",
        "<short sentence or descriptive phrase>"
    ],
}}```"""},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
    return prompts


def _batch_process(dataset, llm, tokenizer, model_cfg) -> list:
    prompts = _generate_prompts(dataset, tokenizer)
    sampling = create_sampling_params(
        temperature=model_cfg.sampling.temperature,
        top_p=model_cfg.sampling.top_p,
        max_tokens=model_cfg.sampling.max_tokens,
        repetition_penalty=model_cfg.sampling.repetition_penalty,
        stop=["<|im_end|>"],
    )
    processed = []
    batch_size = model_cfg.batch_size

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating annotations"):
        batch = prompts[i : i + batch_size]
        outputs = llm.generate(batch, sampling)
        for j, output in enumerate(outputs):
            response = output.outputs[0].text.strip()
            parsed = _parse_json_data(response)
            needs_regen = not _is_valid_summary_format(parsed)
            processed.append({
                "dataset": dataset["train"][i + j]["dataset"],
                "id": dataset["train"][i + j]["id"],
                "messages": dataset["train"][i + j]["messages"],
                "generated_content": parsed,
                "response": response,
                "needs_regeneration": needs_regen,
            })
    return processed
