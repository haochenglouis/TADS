#!/usr/bin/env python3
"""
Data scoring using keyword mapping and Qwen2.5
"""

import torch
import json
import re
import logging
import os
from datetime import datetime
from tqdm import tqdm
from vllm import LLM, SamplingParams
import argparse
from typing import Dict, List, Any, Tuple

def setup_logger(log_file: str = None) -> logging.Logger:
    """
    Setup logging system
    
    Args:
        log_file: Log file path, if None then auto-generate
        
    Returns:
        Configured logger
    """
    if log_file is None:
        # Create logs directory
        os.makedirs("logs", exist_ok=True)
        # Generate timestamped log filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/scoring_{timestamp}.log"
    
    # Create logger
    logger = logging.getLogger("DataScorer")
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging system initialized, log file: {log_file}")
    return logger

def normalize_tag(text):
    """Normalize tag text"""
    text = text.lower()
    text = re.sub(r"[-/]", " ", text)
    text = re.sub(r"\band\b", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

class DataScorer:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", tensor_parallel_size: int = 4, logger: logging.Logger = None):
        """
        Initialize data scorer
        
        Args:
            model_name: Model name
            tensor_parallel_size: Tensor parallel size
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger("DataScorer")
        
        self.logger.info(f"Loading scoring model: {model_name}")
        self.logger.info(f"Configuration: tensor_parallel_size={tensor_parallel_size}, gpu_memory_utilization=0.95, max_model_len=32768")
        
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=0.95,  # Further increase to 95%
            max_model_len=32768,  # Use 32K context length
            trust_remote_code=True,
            enforce_eager=False,
        )
        
        self.logger.info("Model loading completed")
        
        # Set sampling parameters
        self.sampling_params = SamplingParams(
            temperature=0.3,  # Increase temperature to ensure output
            top_p=0.9,
            max_tokens=300,  # Increase max token count
            stop=["</s>"],  # Simplify stop tokens
            skip_special_tokens=True,
        )
        
        print("Scoring model loading completed!")

    def _truncate_text(self, text: str, max_chars: int) -> str:
        """
        Truncate text to fit token limits
        
        Args:
            text: Original text
            max_chars: Maximum character count (rough estimate of token count)
            
        Returns:
            Truncated text
        """
        if len(text) <= max_chars:
            return text
        
        # Truncate text, try to truncate at sentence boundaries
        truncated = text[:max_chars]
        
        # Try to truncate at sentence endings
        last_period = truncated.rfind('.')
        last_question = truncated.rfind('?')
        last_exclamation = truncated.rfind('!')
        last_newline = truncated.rfind('\n')
        
        # Find the last sentence ending
        sentence_end = max(last_period, last_question, last_exclamation, last_newline)
        
        if sentence_end > max_chars * 0.8:  # If truncation point is after 80%, use it
            return truncated[:sentence_end + 1]
        else:
            # Otherwise truncate at word boundary
            last_space = truncated.rfind(' ')
            if last_space > max_chars * 0.7:
                return truncated[:last_space] + "..."
            else:
                return truncated + "..."

    def create_scoring_prompt(self, instruction: str, input_text: str, response: str, 
                            task_tag: str, style_tags: list, topic_tags: list, audience_tags: list) -> tuple:
        """
        Create scoring prompt, based on LaTeX template conversion, adaptive format adjustment
        
        Args:
            instruction: Instruction (may be empty)
            input_text: Input text
            response: Response text
            task_tag: Task tag (single)
            style_tags: Style tags list
            topic_tags: Topic tags list
            audience_tags: Audience tags list
            
        Returns:
            tuple: (formatted scoring prompt, whether truncation occurred)
        """
        # Estimate fixed part token count (prompt template + tags)
        fixed_template_tokens = 500  # Conservative estimate of fixed template token count
        # Estimate all tags token count: Task(100 chars) + max 3 subtags per category × 200 chars
        task_tokens = len(task_tag[:200]) // 4
        style_tokens = sum(len(tag[:200]) for tag in style_tags[:3]) // 4
        topic_tokens = sum(len(tag[:200]) for tag in topic_tags[:3]) // 4
        audience_tokens = sum(len(tag[:200]) for tag in audience_tags[:3]) // 4
        tag_tokens = task_tokens + style_tokens + topic_tokens + audience_tokens
        available_tokens = 32768 - fixed_template_tokens - tag_tokens - 300  # Reserve 300 token safety margin, use 32K context
        
        # Truncate content to fit token limits
        # instruction_truncated = self._truncate_text(instruction, max_chars=available_tokens // 6)
        # input_text_truncated = self._truncate_text(input_text, max_chars=available_tokens // 2)
        # response_truncated = self._truncate_text(response, max_chars=available_tokens // 3)

        instruction_truncated = self._truncate_text(instruction, max_chars=available_tokens // 6)
        input_text_truncated = self._truncate_text(input_text, max_chars=available_tokens // 2 + available_tokens // 6)
        response_truncated = self._truncate_text(response, max_chars=available_tokens // 3)
        
        # Detailed logging: show truncation info and content preview
        truncation_occurred = False
        
        if len(instruction) != len(instruction_truncated):
            truncation_occurred = True
            self.logger.warning(f"Truncated Instruction: {len(instruction)} -> {len(instruction_truncated)} chars")
            self.logger.info(f"Instruction original content preview: {instruction[:100]}...")
            self.logger.info(f"Instruction truncated preview: {instruction_truncated[:100]}...")
            
        if len(input_text) != len(input_text_truncated):
            truncation_occurred = True
            self.logger.warning(f"Truncated Input: {len(input_text)} -> {len(input_text_truncated)} chars")
            self.logger.info(f"Input original content preview: {input_text[:100]}...")
            self.logger.info(f"Input truncated preview: {input_text_truncated[:100]}...")
            
        if len(response) != len(response_truncated):
            truncation_occurred = True
            self.logger.warning(f"Truncated Response: {len(response)} -> {len(response_truncated)} chars")
            self.logger.info(f"Response original content preview: {response[:100]}...")
            self.logger.info(f"Response truncated preview: {response_truncated[:100]}...")
            
        if truncation_occurred:
            self.logger.warning(f"Sample truncated - available tokens: {available_tokens}, Task tag: {task_tag[:50]}...")
        
        # Adaptively adjust format based on whether instruction exists
        if instruction_truncated.strip():
            sample_format = f"""Instruction: {instruction_truncated}

Input: {input_text_truncated}

Response: {response_truncated}"""
        else:
            sample_format = f"""Input: {input_text_truncated}

Response: {response_truncated}"""
        
        # Format tag display, max 200 chars per subtag
        def format_tag(tag):
            if len(tag) > 200:
                return f"• {tag[:200]}..."
            else:
                return f"• {tag}"
        
        style_display = "\n  ".join([format_tag(tag) for tag in style_tags[:3]])  # Max display 3
        topic_display = "\n  ".join([format_tag(tag) for tag in topic_tags[:3]])  # Max display 3
        audience_display = "\n  ".join([format_tag(tag) for tag in audience_tags[:3]])  # Max display 3
        
        prompt = f"""You are an expert evaluator. Please evaluate the following sample based on these criteria:
- Completeness (1-10): How complete is the response?
- Information Richness (1-10): How much useful information does it contain?
- Rarity (1-10): How unique or rare is this type of content?
- Complexity (1-10): How complex is the task/content?

Sample to Evaluate:
{sample_format}

Associated Tags:
- Task: {task_tag[:200]}
- Style Tags:
  {style_display}
- Topic Tags:
  {topic_display}
- Audience Tags:
  {audience_display}

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
        
        return prompt, truncation_occurred

    def parse_score_response(self, response: str) -> Dict[str, Any]:
        """
        Parse scoring response, extract JSON format scores
        
        Args:
            response: Model response
            
        Returns:
            Dictionary containing scores, includes retry flag if parsing fails
        """
        try:
            self.logger.debug(f"Original response: '{response}'")
            
            # If response is empty, mark for retry
            if not response.strip():
                self.logger.warning("Empty response, marking for retry")
                return {"_needs_retry": True, "_error": "empty_response"}
            
            # Try to extract JSON part, use more lenient regex
            json_match = re.search(r'\{[^}]*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                self.logger.debug(f"Extracted JSON string: {json_str}")
                scores = json.loads(json_str)
                
                # Convert to integer scores
                result = {}
                for key, value in scores.items():
                    if isinstance(value, str):
                        # Extract numbers
                        num_match = re.search(r'\d+', value)
                        if num_match:
                            result[key] = int(num_match.group())
                        else:
                            result[key] = 5  # Default score
                    else:
                        result[key] = int(value)
                
                # Check if required fields exist
                required_fields = ["Completeness", "Information Richness", "Rarity", "Complexity", "Overall Score"]
                if all(field in result for field in required_fields):
                    self.logger.debug(f"Parsing successful: {result}")
                    return result
                else:
                    self.logger.warning(f"Missing required fields, marking for retry")
                    return {"_needs_retry": True, "_error": "missing_fields"}
            else:
                self.logger.debug("No JSON format found, trying to extract numbers from text")
                # Try to extract numbers as fallback
                numbers = re.findall(r'\d+', response)
                if len(numbers) >= 5:
                    result = {
                        "Completeness": int(numbers[0]),
                        "Information Richness": int(numbers[1]),
                        "Rarity": int(numbers[2]),
                        "Complexity": int(numbers[3]),
                        "Overall Score": int(numbers[4])
                    }
                    self.logger.debug(f"Number extraction successful: {result}")
                    return result
                else:
                    self.logger.warning("Cannot extract enough numbers, marking for retry")
                    return {"_needs_retry": True, "_error": "insufficient_numbers"}
        except Exception as e:
            self.logger.error(f"Error parsing scoring response: {e}")
            self.logger.debug(f"Original response: {response}")
            return {"_needs_retry": True, "_error": f"exception: {str(e)}"}

    def score_batch(self, prompts: List[str], max_retries: int = 2) -> List[Dict[str, int]]:
        """
        Batch scoring with retry logic
        
        Args:
            prompts: List of scoring prompts
            max_retries: Maximum retry count
            
        Returns:
            List of scoring results
        """
        # Initialize result list, maintain same order as input
        results = [None] * len(prompts)
        retry_indices = list(range(len(prompts)))
        retry_prompts = prompts.copy()
        
        for retry_round in range(max_retries + 1):
            if not retry_prompts:
                break
                
            if retry_round > 0:
                self.logger.info(f"Retry round {retry_round}, processing {len(retry_prompts)} samples")
            else:
                self.logger.info(f"Starting batch scoring, processing {len(retry_prompts)} samples")
            
            # Execute current batch
            outputs = self.llm.generate(retry_prompts, self.sampling_params)
            
            # Process results and collect samples that need retry
            new_retry_indices = []
            new_retry_prompts = []
            
            for i, output in enumerate(outputs):
                original_index = retry_indices[i]
                generated_text = output.outputs[0].text.strip()
                scores = self.parse_score_response(generated_text)
                
                if scores.get("_needs_retry", False):
                    # Need retry
                    if retry_round < max_retries:
                        new_retry_indices.append(original_index)
                        new_retry_prompts.append(retry_prompts[i])
                        print(f"[Retry] Sample {original_index} parsing failed: {scores.get('_error', 'unknown')}")
                    else:
                        # Reached max retry count, use default values
                        print(f"[Retry] Sample {original_index} reached max retry count, using default values")
                        results[original_index] = {
                            "Completeness": 5,
                            "Information Richness": 5,
                            "Rarity": 5,
                            "Complexity": 5,
                            "Overall Score": 5
                        }
                else:
                    # Parsing successful
                    # Remove retry flag fields
                    clean_scores = {k: v for k, v in scores.items() if not k.startswith('_')}
                    results[original_index] = clean_scores
            
            # Update retry list
            retry_indices = new_retry_indices
            retry_prompts = new_retry_prompts
        
        # Ensure all positions have results
        for i, result in enumerate(results):
            if result is None:
                print(f"[Warning] Sample {i} has no result, using default values")
                results[i] = {
                    "Completeness": 5,
                    "Information Richness": 5,
                    "Rarity": 5,
                    "Complexity": 5,
                    "Overall Score": 5
                }
        
        return results

def extract_messages_content(messages: List[Dict[str, str]]) -> Tuple[str, str, str]:
    """
    Extract instruction, input_text, response from messages field
    
    Args:
        messages: Message list, format [{'role': 'xxx', 'content': 'xxx'}, ...]
        
    Returns:
        (instruction, input_text, response) tuple
    """
    if not messages:
        return "", "", ""
    
    instruction = ""
    input_text = ""
    response = ""
    
    # Process system messages
    system_messages = [msg for msg in messages if msg.get('role') == 'system']
    if system_messages:
        instruction = system_messages[0].get('content', '')
    
    if len(messages) <= 2:
        # Simple case: directly extract last user and assistant
        for message in messages:
            role = message.get('role', '')
            content = message.get('content', '')
            
            if role == 'user':
                input_text = content
            elif role == 'assistant':
                response = content
    else:
        # Complex case: last 2 rounds as input and response, previous as instruction supplement
        last_two = messages[-2:]
        history = messages[:-2]
        
        # Extract last round's user and assistant
        for message in last_two:
            role = message.get('role', '')
            content = message.get('content', '')
            
            if role == 'user':
                input_text = content
            elif role == 'assistant':
                response = content
        
        # Use conversation history (except system) as instruction supplement
        history_parts = []
        for message in history:
            role = message.get('role', '')
            content = message.get('content', '')
            
            if role != 'system' and content.strip():  # Exclude system messages to avoid duplication
                if role == 'user':
                    history_parts.append(f"Previous User: {content}")
                elif role == 'assistant':
                    history_parts.append(f"Previous Assistant: {content}")
        
        # Combine instruction
        if history_parts:
            history_text = "\n".join(history_parts)
            if instruction.strip():
                instruction = f"{instruction}\n\nConversation History:\n{history_text}"
            else:
                instruction = f"Conversation History:\n{history_text}"
    
    return instruction, input_text, response

def load_and_map_data(data_file: str, mapping_file: str) -> List[Dict[str, Any]]:
    """
    Load data and apply keyword mapping
    
    Args:
        data_file: Data file path
        mapping_file: Mapping file path
        
    Returns:
        Mapped data list
    """
    logger = logging.getLogger("DataScorer")
    logger.info(f"Loading data file: {data_file}")
    data = torch.load(data_file)
    
    logger.info(f"Loading mapping file: {mapping_file}")
    with open(mapping_file, "r", encoding="utf-8") as f:
        tag_id_to_text = json.load(f)
    
    logger.info("Applying keyword mapping...")
    for item in tqdm(data, desc="Mapping tags"):
        content = item.get("generated_content", {})
        for category in ["Topic", "Style", "Audience", "Task"]:
            value = content.get(category, [])
            if isinstance(value, list):
                content[category] = [tag_id_to_text.get(tag_id, tag_id) for tag_id in value]
            elif isinstance(value, str):
                content[category] = tag_id_to_text.get(value, value)
    
    return data

def main():
    # 设置日志系统
    logger = setup_logger()
    
    parser = argparse.ArgumentParser(description="Data scoring using keyword mapping and Qwen2.5")
    parser.add_argument("--data-file", "-d",
                       default="../data/train_result_with_tag_ids_top3_split.pt",
                       help="Input data file path")
    parser.add_argument("--mapping-file", "-m",
                       default="../data/extracted_keywords_ds-target-split.json",
                       help="Keyword mapping file path")
    parser.add_argument("--output-file", "-o",
                       default="../data/scored_data_with_mapping.pt",
                       help="Output file path")
    parser.add_argument("--model", 
                       default="Qwen/Qwen2.5-7B-Instruct",
                       help="Scoring model name")
    parser.add_argument("--tensor-parallel-size", "-t",
                       type=int, default=4,
                       help="Tensor parallel size")
    parser.add_argument("--batch-size", "-b",
                       type=int, default=64,
                       help="Batch size")
    parser.add_argument("--max-samples", 
                       type=int, default=None,
                       help="Maximum number of samples to process (for testing)")
    
    args = parser.parse_args()
    
    # Record startup information
    logger.info("=" * 60)
    logger.info("Data scoring task started")
    logger.info(f"Data file: {args.data_file}")
    logger.info(f"Mapping file: {args.mapping_file}")
    logger.info(f"Output file: {args.output_file}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Tensor parallel size: {args.tensor_parallel_size}")
    logger.info("=" * 60)
    
    # Load and map data
    logger.info("Starting to load and map data...")
    data = load_and_map_data(args.data_file, args.mapping_file)
    
    if args.max_samples:
        data = data[:args.max_samples]
        logger.info(f"Limited processing samples to: {args.max_samples}")
    
    logger.info(f"Total samples to score: {len(data)}")
    
    # Initialize scorer
    logger.info("Initializing scorer...")
    scorer = DataScorer(model_name=args.model, tensor_parallel_size=args.tensor_parallel_size, logger=logger)
    
    # Batch scoring
    logger.info("Starting batch scoring...")
    total_truncated_samples = 0
    
    for i in tqdm(range(0, len(data), args.batch_size), desc="Scoring batches"):
        batch_data = data[i:i+args.batch_size]
        batch_prompts = []
        batch_start_idx = i
        
        logger.info(f"Processing batch {i//args.batch_size + 1}: samples {i+1}-{min(i+args.batch_size, len(data))}")
        
        # Prepare batch prompts
        for item in batch_data:
            content = item.get("generated_content", {})
            
            # Extract tag information - correctly handle single and multiple tags
            # Task: single tag
            task_tag = content.get("Task", "Unknown")
            if isinstance(task_tag, list):
                task_tag = task_tag[0] if task_tag else "Unknown"
            
            # Style: multiple tags, keep as list
            style_tags = content.get("Style", ["Unknown"])
            if isinstance(style_tags, str):
                style_tags = [style_tags]
            elif not style_tags:
                style_tags = ["Unknown"]
            
            # Topic: multiple tags, keep as list
            topic_tags = content.get("Topic", ["Unknown"])
            if isinstance(topic_tags, str):
                topic_tags = [topic_tags]
            elif not topic_tags:
                topic_tags = ["Unknown"]
            
            # Audience: multiple tags, keep as list
            audience_tags = content.get("Audience", ["Unknown"])
            if isinstance(audience_tags, str):
                audience_tags = [audience_tags]
            elif not audience_tags:
                audience_tags = ["Unknown"]
            
            # Extract content from messages field
            messages = item.get("messages", [])
            instruction, input_text, response = extract_messages_content(messages)
            
            # Create scoring prompt
            prompt, was_truncated = scorer.create_scoring_prompt(
                instruction=instruction,
                input_text=input_text,
                response=response,
                task_tag=task_tag,
                style_tags=style_tags,
                topic_tags=topic_tags,
                audience_tags=audience_tags
            )
            batch_prompts.append(prompt)
            
            # Count truncated samples
            if was_truncated:
                total_truncated_samples += 1
        
        # Batch scoring
        batch_scores = scorer.score_batch(batch_prompts)
        
        # Add scoring results to data
        for j, scores in enumerate(batch_scores):
            data[i+j]["qwen_scores"] = scores
    
    # Save results
    logger.info(f"Saving scoring results to: {args.output_file}")
    torch.save(data, args.output_file)
    
    # Statistics
    logger.info("Scoring task completed! Statistics:")
    logger.info(f"Total samples: {len(data)}")
    logger.info(f"Truncated samples: {total_truncated_samples}")
    logger.info(f"Truncation ratio: {total_truncated_samples/len(data)*100:.1f}%")
    
    if data:
        sample_scores = data[0].get("qwen_scores", {})
        for metric in sample_scores.keys():
            scores = [item.get("qwen_scores", {}).get(metric, 0) for item in data]
            avg_score = sum(scores) / len(scores) if scores else 0
            logger.info(f"Average {metric}: {avg_score:.2f}")
    
    logger.info("=" * 60)
    logger.info("Data scoring task completed successfully")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
