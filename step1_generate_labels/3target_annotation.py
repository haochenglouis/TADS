import json
import os
import json_repair
import re
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import torch
from vllm import LLM, SamplingParams


def convert_alpaca_to_string(sample):
    return "\n".join(f"{msg['role']}: {msg['content']}" for msg in sample['messages'])

def is_valid_summary_format(obj):
    if not isinstance(obj, dict):
        return False

    required_keys = {"Task", "Style", "Topic", "Audience"}
    if not required_keys.issubset(obj.keys()):
        return False

    if not isinstance(obj["Task"], str):
        return False
    for field in ["Style", "Topic", "Audience"]:
        if not isinstance(obj[field], list) or not all(isinstance(x, str) for x in obj[field]):
            return False

    return True

def parse_json_data(text):
    try:
        json_pattern = re.compile(r'```(?:json)?\s*([\s\S]*?)```')
        match = json_pattern.search(text)
        json_str = match.group(1).strip() if match else text.strip()
        return json_repair.repair_json(json_str, return_objects=True)
    except Exception as e:
        print(f"JSON parsing error: {e}")
        return None

def generate_prompts_with_chat_template(dataset, tokenizer):
    prompts = []
    max_chars = 60000
    for sample in dataset['train']:
        formatted_message = convert_alpaca_to_string(sample)
        if len(formatted_message) > max_chars:
            formatted_message = formatted_message[:max_chars]

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f'''【Background Information】\n{formatted_message}

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
}}```'''}
        ]

        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
    return prompts

# L40S batch_size=256, A5000*2 batch_size=32
def batch_process(dataset, llm, tokenizer, batch_size=32):
    prompts = generate_prompts_with_chat_template(dataset, tokenizer)
    processed_samples = []
    failed_count = 0

    sampling_params = SamplingParams(
        max_tokens=2048,
        repetition_penalty=1.2,
        temperature=1.0,
        top_p=0.95,
        stop=["<|im_end|>"]
    )

    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating summaries"):
        batch_prompts = prompts[i:i + batch_size]
        outputs = llm.generate(batch_prompts, sampling_params)

        for j, output in enumerate(outputs):
            response = output.outputs[0].text.strip()
            parsed = parse_json_data(response)
            needs_regeneration = not is_valid_summary_format(parsed)

            sample_data = {
                'dataset': dataset['train'][i + j]['dataset'],
                'id': dataset['train'][i + j]['id'],
                'messages': dataset['train'][i + j]['messages'],
                'generated_content': parsed,
                'response': response,
                'needs_regeneration': needs_regeneration
            }

            if needs_regeneration:
                failed_count += 1
                print(f"\n❌ Failed Sample ID: {sample_data['id']}\nResponse: {response}\n")

            processed_samples.append(sample_data)

    print(f"\nSummary: {len(prompts)} samples processed, {failed_count} failed.")
    return processed_samples

def main():
    input_files = ["step1_combined_target_format.parquet"]
    output_dir = "../data"
    model_path = "Qwen/Qwen2.5-7B-Instruct"
    model_name = model_path.replace("/", "_")


    llm = LLM(
        model=model_path,
        tensor_parallel_size=2,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        max_num_batched_tokens=32768 * 2
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    max_retries = 999

    for input_path in input_files:
        raw_dataset = load_dataset('parquet', data_files=input_path)
        dataset = {"train": raw_dataset["train"]}
        print(f"\n📄 Processing: {input_path}, {len(dataset['train'])} samples")

        processed = batch_process(dataset, llm, tokenizer)
        failed = [item for item in processed if item['needs_regeneration']]
        retry_count = 0

        while failed and retry_count < max_retries:
            retry_count += 1
            print(f"\n🔁 Retry {retry_count}, {len(failed)} samples")
            retry_dataset = {"train": [
                {
                    'dataset': item['dataset'],
                    'id': item['id'],
                    'messages': item['messages']
                } for item in failed
            ]}
            regenerated = batch_process(retry_dataset, llm, tokenizer)
            id_to_item = {item['id']: item for item in regenerated}

            new_failed = []
            for idx, item in enumerate(processed):
                if item['id'] in id_to_item:
                    new_item = id_to_item[item['id']]
                    if new_item['needs_regeneration']:
                        new_failed.append(new_item)
                    else:
                        processed[idx] = new_item
            failed = new_failed

        base_name = os.path.splitext(os.path.basename(input_path))[0]
        json_out = os.path.join(output_dir, f"step1_target_annotation_{model_name}.json")
        pt_out = os.path.join(output_dir, f"step1_target_annotation_{model_name}.pt")

        with open(json_out, 'w', encoding='utf-8') as f:
            json.dump(processed, f, ensure_ascii=False, indent=4)
        torch.save(processed, pt_out)

        print(f"✅ JSON saved to {json_out}")
        print(f"✅ PT saved to {pt_out}")

if __name__ == "__main__":
    main()
