#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
MEASURED_SCANPATH_FILE="${MEASURED_SCANPATH_FILE:-/home/tomi/tesis_LIAA/reading-et/aligned_output/aligned_scanpaths.jsonl}"
MLM_TRAIN_FILE="${MLM_TRAIN_FILE:-/home/tomi/tesis_LIAA/reading-et/mlm_dataset_limpio_train_test/train.jsonl}"
MLM_EVAL_FILE="${MLM_EVAL_FILE:-/home/tomi/tesis_LIAA/reading-et/mlm_dataset_limpio_train_test/test.jsonl}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-dccuchile/bert-base-spanish-wwm-cased}"

BASELINE_OUTPUT_DIR="${BASELINE_OUTPUT_DIR:-Pasos/paso_7_clean_sentence_split_beto_baseline_full}"
SCANPATH_OUTPUT_DIR="${SCANPATH_OUTPUT_DIR:-Pasos/paso_7_clean_sentence_split_full_scanpath}"

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-8}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-128}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-4}"
LEARNING_RATE="${LEARNING_RATE:-5e-05}"
MAX_MASKED_POSITIONS="${MAX_MASKED_POSITIONS:-3}"
SEED="${SEED:-13}"
AUX_WEIGHT="${AUX_WEIGHT:-0.1}"

MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:--1}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:--1}"
SPLIT_STRATEGY="${SPLIT_STRATEGY:-precomputed_files}"
EVAL_SENTENCE_POSITION_MOD="${EVAL_SENTENCE_POSITION_MOD:-10}"
EVAL_SENTENCE_POSITION_REMAINDER="${EVAL_SENTENCE_POSITION_REMAINDER:-5}"
DRY_RUN_SPLIT_ONLY="${DRY_RUN_SPLIT_ONLY:-False}"

build_split_args() {
  if [[ "$SPLIT_STRATEGY" == "precomputed_files" ]]; then
    printf '%s\n' --train_file "$MLM_TRAIN_FILE" --eval_file "$MLM_EVAL_FILE"
  else
    printf '%s\n' --measured_scanpath_file "$MEASURED_SCANPATH_FILE"
  fi
}

run_baseline() {
  mapfile -t SPLIT_ARGS < <(build_split_args)
  "$PYTHON_BIN" train_mlm_beto_baseline_step7_pretrain.py \
    "${SPLIT_ARGS[@]}" \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --measured_text_field text \
    --measured_word_id_field word_id \
    --split train \
    --output_dir "$BASELINE_OUTPUT_DIR" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --max_train_samples "$MAX_TRAIN_SAMPLES" \
    --max_eval_samples "$MAX_EVAL_SAMPLES" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --max_masked_positions "$MAX_MASKED_POSITIONS" \
    --split_strategy "$SPLIT_STRATEGY" \
    --eval_sentence_position_mod "$EVAL_SENTENCE_POSITION_MOD" \
    --eval_sentence_position_remainder "$EVAL_SENTENCE_POSITION_REMAINDER" \
    --split_report_path "$BASELINE_OUTPUT_DIR/split_report.json" \
    --dry_run_split_only "$DRY_RUN_SPLIT_ONLY" \
    --save_every_epoch True \
    --seed "$SEED"

  if [[ "$DRY_RUN_SPLIT_ONLY" != "True" && "$DRY_RUN_SPLIT_ONLY" != "true" && "$DRY_RUN_SPLIT_ONLY" != "1" ]]; then
    "$PYTHON_BIN" scripts/plot_loss_curves.py --output_dir "$BASELINE_OUTPUT_DIR"
  fi
}

run_scanpath() {
  mapfile -t SPLIT_ARGS < <(build_split_args)
  "$PYTHON_BIN" train_mlm_combined_step7.py \
    "${SPLIT_ARGS[@]}" \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --measured_text_field text \
    --measured_word_id_field word_id \
    --split train \
    --output_dir "$SCANPATH_OUTPUT_DIR" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --max_train_samples "$MAX_TRAIN_SAMPLES" \
    --max_eval_samples "$MAX_EVAL_SAMPLES" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --max_masked_positions "$MAX_MASKED_POSITIONS" \
    --split_strategy "$SPLIT_STRATEGY" \
    --eval_sentence_position_mod "$EVAL_SENTENCE_POSITION_MOD" \
    --eval_sentence_position_remainder "$EVAL_SENTENCE_POSITION_REMAINDER" \
    --split_report_path "$SCANPATH_OUTPUT_DIR/split_report.json" \
    --dry_run_split_only "$DRY_RUN_SPLIT_ONLY" \
    --aux_weight "$AUX_WEIGHT" \
    --save_every_epoch True \
    --seed "$SEED"

  if [[ "$DRY_RUN_SPLIT_ONLY" != "True" && "$DRY_RUN_SPLIT_ONLY" != "true" && "$DRY_RUN_SPLIT_ONLY" != "1" ]]; then
    "$PYTHON_BIN" scripts/plot_loss_curves.py --output_dir "$SCANPATH_OUTPUT_DIR"
  fi
}

echo "Running Paso 7 clean sentence split baseline: $BASELINE_OUTPUT_DIR"
run_baseline

echo "Running Paso 7 clean sentence split scanpath: $SCANPATH_OUTPUT_DIR"
run_scanpath

echo "Done."
echo "Baseline artifacts: $BASELINE_OUTPUT_DIR"
echo "Scanpath artifacts: $SCANPATH_OUTPUT_DIR"
