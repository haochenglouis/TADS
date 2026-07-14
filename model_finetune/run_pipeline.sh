#!/bin/bash
# TADS finetune & eval pipeline.
# Adapted from DS2 (https://github.com/UCSC-REAL/DS2, Apache-2.0),
# which builds on open-instruct (https://github.com/allenai/open-instruct, Apache-2.0).
#
# Flow: export selected data -> LoRA finetune -> merge LoRA -> evaluate on 5 benchmarks -> aggregate.
# Hyperparameters below follow Table 14 of the paper.
#
# Run from this directory (model_finetune/) after the selection pipeline has
# produced ../data/selected/fused_10k_gte_<THRESHOLD>.json.
#
# NOTE: meta-llama models are gated on Hugging Face; request access and set:
#   export HF_TOKEN=<your_token>

set -e

echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] TADS Finetune & Eval Start ==="

# =================== Configuration ===================
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F',' '{print NF}')
export NCCL_P2P_LEVEL=NVL
export WANDB_DISABLED=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OPENAI_API_KEY=${OPENAI_API_KEY:-fake}  # only needed for OpenAI-judged metrics

SEED=42
THRESHOLD=${THRESHOLD:-5}                # score threshold t of the fused selection to train on
BASE_MODEL=${BASE_MODEL:-meta-llama/Meta-Llama-3.1-8B}   # paper: Llama-3.1-8B (App. D.1: Mistral-7B-v0.3)
EPOCHS=${EPOCHS:-3}                      # paper Table 14
TOTAL_BATCH_SIZE=${TOTAL_BATCH_SIZE:-128}  # paper Table 14
BATCH_SIZE_PER_GPU=${BATCH_SIZE_PER_GPU:-1}
MAX_SEQ_LENGTH=${MAX_SEQ_LENGTH:-2048}
USE_FLASH_ATTN=${USE_FLASH_ATTN:-1}      # set 0 if flash-attn is not installed (falls back to SDPA)
GRADIENT_ACC_STEPS=$((TOTAL_BATCH_SIZE / NUM_GPUS / BATCH_SIZE_PER_GPU))
FLASH_ATTN_FLAG=""
[ "$USE_FLASH_ATTN" = "1" ] && FLASH_ATTN_FLAG="--use_flash_attn"

TRAIN_FILE=selected_data/tads_10k_gte_${THRESHOLD}_dataset.json
EVAL_DATA_ROOT=../data/eval              # or ../data/eval/split_eval for the held-out 80% splits
OUTPUT_ROOT=model_output
LORA_DIR=$OUTPUT_ROOT/models/lora_gte_${THRESHOLD}
MERGED_DIR=$OUTPUT_ROOT/models/lora_merged_gte_${THRESHOLD}
RESULTS_ROOT=$OUTPUT_ROOT/results

mkdir -p "$OUTPUT_ROOT"

# =================== Step 0: Export selected data ===================
if [ ! -f "$TRAIN_FILE" ]; then
    echo "###### Exporting selected samples (threshold=$THRESHOLD)"
    python export_train_file.py --threshold "$THRESHOLD" --data-dir ../data --output "$TRAIN_FILE"
fi

# =================== Step 1: LoRA finetuning ===================
echo "###### Training $BASE_MODEL on $TRAIN_FILE"
echo "GPUs: $NUM_GPUS | per-GPU batch: $BATCH_SIZE_PER_GPU | grad-acc: $GRADIENT_ACC_STEPS | max_seq_len: $MAX_SEQ_LENGTH"

accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes "$NUM_GPUS" \
    finetune.py \
    --model_name_or_path "$BASE_MODEL" \
    --use_lora \
    --lora_rank 64 \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --seed $SEED \
    --tokenizer_name "$BASE_MODEL" \
    --use_slow_tokenizer \
    --train_file "$TRAIN_FILE" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --preprocessing_num_workers 16 \
    --checkpointing_steps epoch \
    --gradient_checkpointing \
    $FLASH_ATTN_FLAG \
    --per_device_train_batch_size "$BATCH_SIZE_PER_GPU" \
    --gradient_accumulation_steps "$GRADIENT_ACC_STEPS" \
    --learning_rate 1e-4 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs "$EPOCHS" \
    --output_dir "$LORA_DIR" \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1

# =================== Step 2: Merge LoRA ===================
echo "###### Merging LoRA into base model"
python merge_lora.py \
    --base_model_name_or_path "$BASE_MODEL" \
    --lora_model_name_or_path "$LORA_DIR" \
    --output_dir "$MERGED_DIR" \
    --tokenizer_name_or_path "$BASE_MODEL" \
    --save_tokenizer

# =================== Step 3: Evaluation ===================
# Benchmarks and flags follow the paper setup. Evals run sequentially, each pinned
# to a SINGLE GPU (the first of CUDA_VISIBLE_DEVICES). This matches the paper's
# one-card-per-eval setup and is required for correctness: the HF scoring/generation
# paths shard the model with device_map="balanced_low_0" when more than one GPU is
# visible, but feed inputs to cuda:0, which crashes when the embedding layer lands on
# another shard. With >=4 GPUs the five evals can instead be run in parallel, one card
# each. Do NOT unset CUDA_VISIBLE_DEVICES here.
MODEL=$MERGED_DIR
TAG=gte_${THRESHOLD}
EVAL_GPU=$(echo "$CUDA_VISIBLE_DEVICES" | cut -d',' -f1)

echo "###### Evaluating MMLU (0-shot) on GPU $EVAL_GPU"
CUDA_VISIBLE_DEVICES=$EVAL_GPU python -m eval.mmlu.run_eval \
    --ntrain 0 --data_dir "$EVAL_DATA_ROOT/mmlu" --save_dir "$RESULTS_ROOT/mmlu/$TAG" \
    --model_name_or_path "$MODEL" --tokenizer_name_or_path "$MODEL" --eval_batch_size 8

echo "###### Evaluating GSM8K (8-shot, vLLM) on GPU $EVAL_GPU"
CUDA_VISIBLE_DEVICES=$EVAL_GPU python -m eval.gsm.run_eval \
    --data_dir "$EVAL_DATA_ROOT/gsm/" --max_num_examples 200 --save_dir "$RESULTS_ROOT/gsm/$TAG" \
    --model_name_or_path "$MODEL" --tokenizer_name_or_path "$MODEL" --n_shot 8 --use_vllm

echo "###### Evaluating BBH (vLLM) on GPU $EVAL_GPU"
CUDA_VISIBLE_DEVICES=$EVAL_GPU python -m eval.bbh.run_eval \
    --data_dir "$EVAL_DATA_ROOT/bbh/" --save_dir "$RESULTS_ROOT/bbh/$TAG" \
    --model_name_or_path "$MODEL" --tokenizer_name_or_path "$MODEL" --max_num_examples_per_task 40 --use_vllm

echo "###### Evaluating TruthfulQA on GPU $EVAL_GPU"
# truth/info metrics use the allenai judge models below (~2x7B download);
# drop them and keep "--metrics mc" for a lighter run.
CUDA_VISIBLE_DEVICES=$EVAL_GPU python -m eval.truthfulqa.run_eval \
    --data_dir "$EVAL_DATA_ROOT/truthfulqa" --save_dir "$RESULTS_ROOT/truthfulqa/$TAG" \
    --model_name_or_path "$MODEL" --tokenizer_name_or_path "$MODEL" \
    --metrics truth info mc --preset qa \
    --hf_truth_model_name_or_path allenai/truthfulqa-truth-judge-llama2-7B \
    --hf_info_model_name_or_path allenai/truthfulqa-info-judge-llama2-7B \
    --eval_batch_size 40 --load_in_8bit

echo "###### Evaluating TyDiQA (1-shot) on GPU $EVAL_GPU"
CUDA_VISIBLE_DEVICES=$EVAL_GPU python -m eval.tydiqa.run_eval \
    --data_dir "$EVAL_DATA_ROOT/tydiqa/" --n_shot 1 --max_num_examples_per_lang 100 --max_context_length 512 \
    --save_dir "$RESULTS_ROOT/tydiqa/$TAG" \
    --model_name_or_path "$MODEL" --tokenizer_name_or_path "$MODEL" --eval_batch_size 40 --load_in_8bit

# =================== Step 4: Aggregate results ===================
python read_results.py --root_result_path "$RESULTS_ROOT" --baseline_tag "$TAG"

echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] Done. Results under $RESULTS_ROOT ==="
