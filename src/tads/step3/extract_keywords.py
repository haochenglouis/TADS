"""Step 3.1: Extract representative keywords from cluster anchors (§4.3, Appendix B)."""

import json
import re
from typing import List

from tqdm import tqdm

from tads.vllm_engine import create_vllm_engine, create_sampling_params


class KeywordExtractor:
    """Extract 20 representative keywords from long cluster tag strings using an LLM."""

    def __init__(self, model_cfg):
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
            repetition_penalty=model_cfg.sampling.repetition_penalty,
            stop=["</s>", "\n\n"],
        )

    def extract_keywords_batch(self, tag_texts: List[str]) -> List[str]:
        prompts = [self._create_prompt(text) for text in tag_texts]
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [self._clean_keywords(o.outputs[0].text.strip()) for o in outputs]

    def process_json_file(self, input_file: str, output_file: str, batch_size: int = 32):
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Processing {len(data)} tag entries from {input_file}")

        keys = list(data.keys())
        values = list(data.values())
        result = {}
        for i in tqdm(range(0, len(values), batch_size), desc="Extracting keywords"):
            batch_keys = keys[i : i + batch_size]
            batch_vals = values[i : i + batch_size]
            batch_results = self.extract_keywords_batch(batch_vals)
            for k, v in zip(batch_keys, batch_results):
                result[k] = v

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Saved extracted keywords to {output_file}")
        return result

    @staticmethod
    def _create_prompt(tag_text: str) -> str:
        return f"""You are an expert at analyzing and extracting key concepts from text. Given a long text that consists of multiple topics and concepts joined by underscores, your task is to identify and extract exactly 20 of the most important and representative keywords or key phrases.

Input text (topics joined by underscores):
{tag_text}

Instructions:
1. Analyze the entire text and identify the most important concepts, themes, and topics
2. Extract exactly 20 diverse keywords/phrases that best represent the core themes
3. Prioritize more general and important concepts over very specific details
4. Ensure keywords are diverse and cover different aspects of the content - NO DUPLICATES
5. Keywords can be single words or short phrases (1-4 words each)
6. Within each keyword/phrase, use proper spacing and hyphens as needed
7. Only use underscores to separate different keywords, NOT within keywords
8. Avoid repetition - each keyword should be unique and distinct
9. Do not include explanations, just provide the keywords joined by underscores

Output format: keyword1_keyword2_keyword3_..._keyword20

Keywords:"""

    @staticmethod
    def _clean_keywords(text: str) -> str:
        text = text.strip()
        if "\n" in text:
            text = text.split("\n")[0]
        text = re.sub(r'["""\'\'`]', "", text)
        text = re.sub(r"[^\w\s_-]", "", text)
        keywords = [kw.strip() for kw in text.split("_") if kw.strip()]
        seen = []
        for kw in keywords:
            cleaned = re.sub(r"\s+", " ", kw.strip())
            if cleaned and cleaned not in seen:
                seen.append(cleaned)
        return "_".join(seen[:20])


def run(config):
    extractor = KeywordExtractor(config.keyword_extraction)
    extractor.process_json_file(
        input_file=config.tag_id_mapping_json,
        output_file=config.extracted_keywords_json,
        batch_size=config.keyword_extraction.batch_size,
    )
