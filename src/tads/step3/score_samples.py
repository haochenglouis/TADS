"""Step 3.2: Score training samples based on anchor alignment (§4.3, Appendix B)."""

import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Tuple

import torch
from tqdm import tqdm

from tads.vllm_engine import create_sampling_params, create_vllm_engine


def setup_logger(log_file: str = None) -> logging.Logger:
    if log_file is None:
        os.makedirs("logs", exist_ok=True)
        log_file = f"logs/scoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = logging.getLogger("DataScorer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(ch)
    logger.info(f"Log file: {log_file}")
    return logger


class DataScorer:
    """Score training samples by their semantic alignment with assigned anchors."""

    def __init__(self, model_cfg, logger=None):
        self.logger = logger or logging.getLogger("DataScorer")
        self.llm = create_vllm_engine(
            model_name=model_cfg.name,
            tensor_parallel_size=model_cfg.tensor_parallel_size,
            gpu_memory_utilization=model_cfg.gpu_memory_utilization,
            max_model_len=model_cfg.max_model_len,
            enforce_eager=False,
        )
        self.sampling_params = create_sampling_params(
            temperature=model_cfg.sampling.temperature,
            top_p=model_cfg.sampling.top_p,
            max_tokens=model_cfg.sampling.max_tokens,
            stop=["</s>"],
        )
        self.max_model_len = model_cfg.max_model_len or 32768

    def create_scoring_prompt(self, instruction, input_text, response,
                              task_tag, style_tags, topic_tags, audience_tags):
        fixed_tokens = 500
        tag_tokens = (len(task_tag[:200]) + sum(len(t[:200]) for t in style_tags[:3])
                      + sum(len(t[:200]) for t in topic_tags[:3])
                      + sum(len(t[:200]) for t in audience_tags[:3])) // 4
        available = self.max_model_len - fixed_tokens - tag_tokens - 300

        instruction = self._truncate(instruction, available // 6)
        input_text = self._truncate(input_text, available // 2 + available // 6)
        response = self._truncate(response, available // 3)

        if instruction.strip():
            sample_fmt = f"Instruction: {instruction}\n\nInput: {input_text}\n\nResponse: {response}"
        else:
            sample_fmt = f"Input: {input_text}\n\nResponse: {response}"

        def fmt_tag(t):
            return f"• {t[:200]}{'...' if len(t)>200 else ''}"

        prompt = f"""You are an expert evaluator. Please evaluate the following sample based on these criteria:
- Completeness (1-10): How complete is the response?
- Information Richness (1-10): How much useful information does it contain?
- Rarity (1-10): How unique or rare is this type of content?
- Complexity (1-10): How complex is the task/content?

Sample to Evaluate:
{sample_fmt}

Associated Tags:
- Task: {task_tag[:200]}
- Style Tags:
  {chr(10).join('  ' + fmt_tag(t) for t in style_tags[:3])}
- Topic Tags:
  {chr(10).join('  ' + fmt_tag(t) for t in topic_tags[:3])}
- Audience Tags:
  {chr(10).join('  ' + fmt_tag(t) for t in audience_tags[:3])}

Use these tags as reference points for your scoring decisions. Consider ALL the tags in each category when evaluating.

Please respond with ONLY a JSON object in this exact format:
{{
    "Completeness": "X",
    "Information Richness": "X",
    "Rarity": "X",
    "Complexity": "X",
    "Overall Score": "X"
}}

Replace X with numbers 1-10. JSON:"""
        return prompt

    def parse_score_response(self, response: str) -> Dict[str, Any]:
        if not response.strip():
            return {"_needs_retry": True}
        try:
            m = re.search(r"\{[^}]*\}", response, re.DOTALL)
            if m:
                scores = json.loads(m.group())
                result = {}
                for k, v in scores.items():
                    if isinstance(v, str):
                        nm = re.search(r"\d+", v)
                        result[k] = int(nm.group()) if nm else 5
                    else:
                        result[k] = int(v)
                required = ["Completeness", "Information Richness", "Rarity", "Complexity", "Overall Score"]
                if all(f in result for f in required):
                    return result
            # Fallback: extract numbers
            numbers = re.findall(r"\d+", response)
            if len(numbers) >= 5:
                return {
                    "Completeness": int(numbers[0]),
                    "Information Richness": int(numbers[1]),
                    "Rarity": int(numbers[2]),
                    "Complexity": int(numbers[3]),
                    "Overall Score": int(numbers[4]),
                }
        except Exception:
            pass
        return {"_needs_retry": True}

    def score_batch(self, prompts: List[str], max_retries: int = 2) -> List[Dict[str, int]]:
        results = [None] * len(prompts)
        retry_idx = list(range(len(prompts)))
        retry_prompts = prompts.copy()
        default = {"Completeness": 5, "Information Richness": 5, "Rarity": 5, "Complexity": 5, "Overall Score": 5}

        for rnd in range(max_retries + 1):
            if not retry_prompts:
                break
            outputs = self.llm.generate(retry_prompts, self.sampling_params)
            new_idx, new_prompts = [], []
            for i, output in enumerate(outputs):
                oi = retry_idx[i]
                scores = self.parse_score_response(output.outputs[0].text.strip())
                if scores.get("_needs_retry") and rnd < max_retries:
                    new_idx.append(oi)
                    new_prompts.append(retry_prompts[i])
                elif scores.get("_needs_retry"):
                    results[oi] = default.copy()
                else:
                    results[oi] = {k: v for k, v in scores.items() if not k.startswith("_")}
            retry_idx, retry_prompts = new_idx, new_prompts

        for i in range(len(results)):
            if results[i] is None:
                results[i] = default.copy()
        return results

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        t = text[:max_chars]
        end = max(t.rfind("."), t.rfind("?"), t.rfind("!"), t.rfind("\n"))
        if end > max_chars * 0.8:
            return t[: end + 1]
        sp = t.rfind(" ")
        return (t[:sp] + "...") if sp > max_chars * 0.7 else t + "..."


def _extract_messages_content(messages):
    instruction, input_text, response = "", "", ""
    sys_msgs = [m for m in messages if m.get("role") == "system"]
    if sys_msgs:
        instruction = sys_msgs[0].get("content", "")
    if len(messages) <= 2:
        for m in messages:
            if m["role"] == "user":
                input_text = m["content"]
            elif m["role"] == "assistant":
                response = m["content"]
    else:
        for m in messages[-2:]:
            if m["role"] == "user":
                input_text = m["content"]
            elif m["role"] == "assistant":
                response = m["content"]
        history = [f"Previous {m['role'].title()}: {m['content']}"
                   for m in messages[:-2] if m["role"] != "system" and m["content"].strip()]
        if history:
            h = "\n".join(history)
            instruction = f"{instruction}\n\nConversation History:\n{h}" if instruction.strip() else f"Conversation History:\n{h}"
    return instruction, input_text, response


def run(config):
    logger = setup_logger()
    model_cfg = config.scoring

    # Load data and mapping
    data_file = config.train_with_tags_pt
    mapping_file = config.extracted_keywords_json
    output_file = config.scored_data_pt

    logger.info(f"Loading data: {data_file}")
    data = torch.load(data_file, weights_only=False)
    with open(mapping_file, "r", encoding="utf-8") as f:
        tag_id_to_text = json.load(f)

    # Apply keyword mapping to tags
    for item in tqdm(data, desc="Mapping tags to keywords"):
        content = item.get("generated_content", {})
        for cat in ["Topic", "Style", "Audience", "Task"]:
            val = content.get(cat, [])
            if isinstance(val, list):
                content[cat] = [tag_id_to_text.get(tid, tid) for tid in val]
            elif isinstance(val, str):
                content[cat] = tag_id_to_text.get(val, val)

    scorer = DataScorer(model_cfg, logger)
    batch_size = model_cfg.batch_size

    for i in tqdm(range(0, len(data), batch_size), desc="Scoring batches"):
        batch = data[i : i + batch_size]
        prompts = []
        for item in batch:
            content = item.get("generated_content", {})
            task = content.get("Task", "Unknown")
            if isinstance(task, list):
                task = task[0] if task else "Unknown"
            style = content.get("Style", ["Unknown"])
            if isinstance(style, str):
                style = [style]
            topic = content.get("Topic", ["Unknown"])
            if isinstance(topic, str):
                topic = [topic]
            audience = content.get("Audience", ["Unknown"])
            if isinstance(audience, str):
                audience = [audience]
            instruction, input_text, response = _extract_messages_content(item.get("messages", []))
            prompts.append(scorer.create_scoring_prompt(instruction, input_text, response, task, style, topic, audience))

        scores = scorer.score_batch(prompts)
        for j, s in enumerate(scores):
            data[i + j]["qwen_scores"] = s

    torch.save(data, output_file)
    logger.info(f"Saved scored data ({len(data)} samples) to {output_file}")
