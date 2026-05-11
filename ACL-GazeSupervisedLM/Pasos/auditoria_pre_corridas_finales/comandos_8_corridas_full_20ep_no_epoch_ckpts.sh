#!/usr/bin/env bash
set -euo pipefail

# Corrida full-data de 8 pasos/modelos con pretraining a 20 epochs.
# Protegido para no ejecutarse accidentalmente:
# RUN_EXPERIMENTS=1 bash Pasos/auditoria_pre_corridas_finales/comandos_8_corridas_full_20ep_no_epoch_ckpts.sh
#
# Guarda best_checkpoint, checkpoint_final, metricas y plots.
# No guarda checkpoint_epoch_* en pretraining.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

if [[ "${RUN_EXPERIMENTS:-0}" != "1" ]]; then
  echo "Modo seguro: no se ejecutaron corridas full 20ep."
  echo "Para ejecutar realmente: RUN_EXPERIMENTS=1 bash Pasos/auditoria_pre_corridas_finales/comandos_8_corridas_full_20ep_no_epoch_ckpts.sh"
  exit 0
fi

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
MLM_TRAIN_FILE="${MLM_TRAIN_FILE:-/home/tomi/tesis_LIAA/reading-et/mlm_dataset_limpio_train_test/train.jsonl}"
MLM_EVAL_FILE="${MLM_EVAL_FILE:-/home/tomi/tesis_LIAA/reading-et/mlm_dataset_limpio_train_test/test.jsonl}"
BETO_BASE_MODEL="${BETO_BASE_MODEL:-dccuchile/bert-base-spanish-wwm-cased}"

SEED_PRETRAINING="${SEED_PRETRAINING:-13}"
SEED_DOWNSTREAM="${SEED_DOWNSTREAM:-111}"
DOWNSTREAM_TRAIN_SAMPLES="${DOWNSTREAM_TRAIN_SAMPLES:-1000}"
DOWNSTREAM_EVAL_SAMPLES="${DOWNSTREAM_EVAL_SAMPLES:-1000}"

OUTPUT_ROOT="${OUTPUT_ROOT:-Pasos/corridas_finales_full_20ep}"
BASELINE_PRETRAIN_OUT="${BASELINE_PRETRAIN_OUT:-$OUTPUT_ROOT/pretraining_beto_mlm_limpio}"
SCANPATH_PRETRAIN_OUT="${SCANPATH_PRETRAIN_OUT:-$OUTPUT_ROOT/pretraining_beto_scanpath_lambda_limpio}"
BASELINE_BEST_CKPT="$BASELINE_PRETRAIN_OUT/best_checkpoint"
SCANPATH_BEST_CKPT="$SCANPATH_PRETRAIN_OUT/best_checkpoint"

BASELINE_XNLI_OUT="$OUTPUT_ROOT/downstream/beto_pretrained/xnli_es/k${DOWNSTREAM_TRAIN_SAMPLES}/seed_${SEED_DOWNSTREAM}"
BASELINE_RIO_OUT="$OUTPUT_ROOT/downstream/beto_pretrained/rioplatense_hate_binary/k${DOWNSTREAM_TRAIN_SAMPLES}/seed_${SEED_DOWNSTREAM}"
SCANPATH_XNLI_OUT="$OUTPUT_ROOT/downstream/scanpath_pretrained/xnli_es/k${DOWNSTREAM_TRAIN_SAMPLES}/seed_${SEED_DOWNSTREAM}"
SCANPATH_RIO_OUT="$OUTPUT_ROOT/downstream/scanpath_pretrained/rioplatense_hate_binary/k${DOWNSTREAM_TRAIN_SAMPLES}/seed_${SEED_DOWNSTREAM}"
BETO_BASE_XNLI_OUT="$OUTPUT_ROOT/downstream/beto_base/xnli_es/k${DOWNSTREAM_TRAIN_SAMPLES}/seed_${SEED_DOWNSTREAM}"
BETO_BASE_RIO_OUT="$OUTPUT_ROOT/downstream/beto_base/rioplatense_hate_binary/k${DOWNSTREAM_TRAIN_SAMPLES}/seed_${SEED_DOWNSTREAM}"

plot_pretraining_losses() {
  local output_dir="$1"
  if [[ -f "$output_dir/loss_curves.csv" ]]; then
    "$PYTHON_BIN" scripts/plot_loss_curves.py --output_dir "$output_dir"
  else
    echo "Skipping Paso 7 loss plots; loss_curves.csv not found in $output_dir"
  fi
}

plot_lambda_scanpath() {
  local output_dir="$1"
  if [[ -f "$output_dir/lambda_scanpath_log.csv" ]]; then
    "$PYTHON_BIN" scripts/plot_lambda_scanpath.py --output_dir "$output_dir"
  else
    echo "Skipping lambda scanpath plots; lambda_scanpath_log.csv not found in $output_dir"
  fi
}

plot_downstream_losses() {
  local output_dir="$1"
  if [[ -f "$output_dir/trainer_state.json" ]]; then
    "$PYTHON_BIN" scripts/plot_trainer_loss_curves.py --output_dir "$output_dir"
  else
    echo "Skipping downstream loss plots; trainer_state.json not found in $output_dir"
  fi
}

prune_downstream_training_state() {
  local output_dir="$1"
  find "$output_dir" -type f \( -name optimizer.pt -o -name scheduler.pt \) -print -delete
}

echo "Corrida full-data 20ep sin checkpoint_epoch_*."
echo "- Output root: $OUTPUT_ROOT"
echo "- MLM train: $MLM_TRAIN_FILE"
echo "- MLM eval: $MLM_EVAL_FILE"
echo "- Downstream k: $DOWNSTREAM_TRAIN_SAMPLES"

# A. Pretraining
"$PYTHON_BIN" train_mlm_beto_baseline_step7_pretrain.py \
  --train_file "$MLM_TRAIN_FILE" \
  --eval_file "$MLM_EVAL_FILE" \
  --model_name_or_path "$BETO_BASE_MODEL" \
  --measured_text_field text \
  --measured_word_id_field word_id \
  --output_dir "$BASELINE_PRETRAIN_OUT" \
  --max_seq_length 128 \
  --max_train_samples -1 \
  --max_eval_samples -1 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --num_train_epochs 20 \
  --learning_rate 5e-5 \
  --max_masked_positions 3 \
  --split_strategy precomputed_files \
  --split_report_path "$BASELINE_PRETRAIN_OUT/split_report.json" \
  --save_every_epoch False \
  --seed "$SEED_PRETRAINING"
plot_pretraining_losses "$BASELINE_PRETRAIN_OUT"

"$PYTHON_BIN" train_mlm_combined_step7.py \
  --train_file "$MLM_TRAIN_FILE" \
  --eval_file "$MLM_EVAL_FILE" \
  --model_name_or_path "$BETO_BASE_MODEL" \
  --measured_text_field text \
  --measured_word_id_field word_id \
  --output_dir "$SCANPATH_PRETRAIN_OUT" \
  --max_seq_length 128 \
  --max_train_samples -1 \
  --max_eval_samples -1 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --num_train_epochs 20 \
  --learning_rate 5e-5 \
  --max_masked_positions 3 \
  --split_strategy precomputed_files \
  --split_report_path "$SCANPATH_PRETRAIN_OUT/split_report.json" \
  --aux_weight 0.1 \
  --adaptive_scanpath_weight True \
  --scanpath_weight_min 0.05 \
  --scanpath_weight_max 0.5 \
  --scanpath_weight_warmup_epochs 1 \
  --scanpath_weight_update_metric auto \
  --scanpath_weight_log_path "$SCANPATH_PRETRAIN_OUT/lambda_scanpath_log.csv" \
  --save_every_epoch False \
  --seed "$SEED_PRETRAINING"
plot_pretraining_losses "$SCANPATH_PRETRAIN_OUT"
plot_lambda_scanpath "$SCANPATH_PRETRAIN_OUT"

# B. Downstream desde BETO solo preentrenado
"$PYTHON_BIN" train_spanish_downstream_baseline.py \
  --model_name_or_path "$BASELINE_BEST_CKPT" \
  --task_name xnli_es \
  --output_dir "$BASELINE_XNLI_OUT" \
  --max_train_samples "$DOWNSTREAM_TRAIN_SAMPLES" \
  --max_eval_samples "$DOWNSTREAM_EVAL_SAMPLES" \
  --num_train_epochs 20 \
  --learning_rate 2e-5 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --max_seq_length 128 \
  --seed "$SEED_DOWNSTREAM" \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --load_best_model_at_end \
  --save_total_limit 1 \
  --expected_num_labels 3
plot_downstream_losses "$BASELINE_XNLI_OUT"
prune_downstream_training_state "$BASELINE_XNLI_OUT"

"$PYTHON_BIN" train_glue_LM_baseline.py \
  --model_name_or_path "$BASELINE_BEST_CKPT" \
  --task_name rioplatense_hate_binary \
  --output_dir "$BASELINE_RIO_OUT" \
  --num_train_epochs 20 \
  --learning_rate 2e-5 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --max_seq_length 128 \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --metric_for_best_model macro_f1 \
  --greater_is_better True \
  --train_as_val True \
  --max_train_samples "$DOWNSTREAM_TRAIN_SAMPLES" \
  --max_eval_samples "$DOWNSTREAM_EVAL_SAMPLES" \
  --low_resource_data_seed "$SEED_DOWNSTREAM" \
  --seed "$SEED_DOWNSTREAM" \
  --logging_steps 10 \
  --report_to none \
  --load_best_model_at_end \
  --overwrite_output_dir \
  --do_train \
  --do_eval \
  --do_predict
plot_downstream_losses "$BASELINE_RIO_OUT"
prune_downstream_training_state "$BASELINE_RIO_OUT"

# C. Downstream desde BETO + Scanpath preentrenado
"$PYTHON_BIN" train_spanish_downstream_baseline.py \
  --model_name_or_path "$SCANPATH_BEST_CKPT" \
  --task_name xnli_es \
  --output_dir "$SCANPATH_XNLI_OUT" \
  --max_train_samples "$DOWNSTREAM_TRAIN_SAMPLES" \
  --max_eval_samples "$DOWNSTREAM_EVAL_SAMPLES" \
  --num_train_epochs 20 \
  --learning_rate 2e-5 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --max_seq_length 128 \
  --seed "$SEED_DOWNSTREAM" \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --load_best_model_at_end \
  --save_total_limit 1 \
  --expected_num_labels 3
plot_downstream_losses "$SCANPATH_XNLI_OUT"
prune_downstream_training_state "$SCANPATH_XNLI_OUT"

"$PYTHON_BIN" train_glue_LM_baseline.py \
  --model_name_or_path "$SCANPATH_BEST_CKPT" \
  --task_name rioplatense_hate_binary \
  --output_dir "$SCANPATH_RIO_OUT" \
  --num_train_epochs 20 \
  --learning_rate 2e-5 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --max_seq_length 128 \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --metric_for_best_model macro_f1 \
  --greater_is_better True \
  --train_as_val True \
  --max_train_samples "$DOWNSTREAM_TRAIN_SAMPLES" \
  --max_eval_samples "$DOWNSTREAM_EVAL_SAMPLES" \
  --low_resource_data_seed "$SEED_DOWNSTREAM" \
  --seed "$SEED_DOWNSTREAM" \
  --logging_steps 10 \
  --report_to none \
  --load_best_model_at_end \
  --overwrite_output_dir \
  --do_train \
  --do_eval \
  --do_predict
plot_downstream_losses "$SCANPATH_RIO_OUT"
prune_downstream_training_state "$SCANPATH_RIO_OUT"

# D. BETO base directo
"$PYTHON_BIN" train_spanish_downstream_baseline.py \
  --model_name_or_path "$BETO_BASE_MODEL" \
  --task_name xnli_es \
  --output_dir "$BETO_BASE_XNLI_OUT" \
  --max_train_samples "$DOWNSTREAM_TRAIN_SAMPLES" \
  --max_eval_samples "$DOWNSTREAM_EVAL_SAMPLES" \
  --num_train_epochs 20 \
  --learning_rate 2e-5 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --max_seq_length 128 \
  --seed "$SEED_DOWNSTREAM" \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --load_best_model_at_end \
  --save_total_limit 1 \
  --expected_num_labels 3
plot_downstream_losses "$BETO_BASE_XNLI_OUT"
prune_downstream_training_state "$BETO_BASE_XNLI_OUT"

"$PYTHON_BIN" train_glue_LM_baseline.py \
  --model_name_or_path "$BETO_BASE_MODEL" \
  --task_name rioplatense_hate_binary \
  --output_dir "$BETO_BASE_RIO_OUT" \
  --num_train_epochs 20 \
  --learning_rate 2e-5 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --max_seq_length 128 \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --metric_for_best_model macro_f1 \
  --greater_is_better True \
  --train_as_val True \
  --max_train_samples "$DOWNSTREAM_TRAIN_SAMPLES" \
  --max_eval_samples "$DOWNSTREAM_EVAL_SAMPLES" \
  --low_resource_data_seed "$SEED_DOWNSTREAM" \
  --seed "$SEED_DOWNSTREAM" \
  --logging_steps 10 \
  --report_to none \
  --load_best_model_at_end \
  --overwrite_output_dir \
  --do_train \
  --do_eval \
  --do_predict
plot_downstream_losses "$BETO_BASE_RIO_OUT"
prune_downstream_training_state "$BETO_BASE_RIO_OUT"
