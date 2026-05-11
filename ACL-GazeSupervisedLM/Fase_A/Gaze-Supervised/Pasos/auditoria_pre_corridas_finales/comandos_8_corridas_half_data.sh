#!/usr/bin/env bash
set -euo pipefail

# Corridas finales en modo half-data.
# Este archivo no ejecuta nada salvo que se llame explicitamente con:
# RUN_EXPERIMENTS=1 bash Pasos/auditoria_pre_corridas_finales/comandos_8_corridas_half_data.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

if [[ "${RUN_EXPERIMENTS:-0}" != "1" ]]; then
  echo "Modo seguro: no se ejecutaron corridas half-data."
  echo "Para ejecutar realmente: RUN_EXPERIMENTS=1 bash Pasos/auditoria_pre_corridas_finales/comandos_8_corridas_half_data.sh"
  exit 0
fi

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
MLM_TRAIN_FILE="${MLM_TRAIN_FILE:-/home/tomi/tesis_LIAA/reading-et/mlm_dataset_limpio_train_test/train.jsonl}"
MLM_EVAL_FILE="${MLM_EVAL_FILE:-/home/tomi/tesis_LIAA/reading-et/mlm_dataset_limpio_train_test/test.jsonl}"
BETO_BASE_MODEL="${BETO_BASE_MODEL:-dccuchile/bert-base-spanish-wwm-cased}"

# Mitad exacta redondeada hacia abajo de los JSONL limpios:
# train.jsonl = 38518 -> 19259
# test.jsonl = 9630 -> 4815
MLM_HALF_TRAIN_SAMPLES="${MLM_HALF_TRAIN_SAMPLES:-19259}"
MLM_HALF_EVAL_SAMPLES="${MLM_HALF_EVAL_SAMPLES:-4815}"

# Mitad del setup downstream low-resource auditado: k1000 -> k500.
DOWNSTREAM_HALF_TRAIN_SAMPLES="${DOWNSTREAM_HALF_TRAIN_SAMPLES:-500}"
DOWNSTREAM_HALF_EVAL_SAMPLES="${DOWNSTREAM_HALF_EVAL_SAMPLES:-500}"

SEED_PRETRAINING="${SEED_PRETRAINING:-13}"
SEED_DOWNSTREAM="${SEED_DOWNSTREAM:-111}"

BASELINE_PRETRAIN_OUT="${BASELINE_PRETRAIN_OUT:-Pasos/corridas_finales_half_data/pretraining_beto_mlm_limpio_half}"
SCANPATH_PRETRAIN_OUT="${SCANPATH_PRETRAIN_OUT:-Pasos/corridas_finales_half_data/pretraining_beto_scanpath_lambda_limpio_half}"
BASELINE_BEST_CKPT="$BASELINE_PRETRAIN_OUT/best_checkpoint"
SCANPATH_BEST_CKPT="$SCANPATH_PRETRAIN_OUT/best_checkpoint"

BASELINE_XNLI_OUT="Pasos/corridas_finales_half_data/downstream/beto_pretrained/xnli_es/k${DOWNSTREAM_HALF_TRAIN_SAMPLES}/seed_${SEED_DOWNSTREAM}"
BASELINE_RIO_OUT="Pasos/corridas_finales_half_data/downstream/beto_pretrained/rioplatense_hate_binary/k${DOWNSTREAM_HALF_TRAIN_SAMPLES}/seed_${SEED_DOWNSTREAM}"
SCANPATH_XNLI_OUT="Pasos/corridas_finales_half_data/downstream/scanpath_pretrained/xnli_es/k${DOWNSTREAM_HALF_TRAIN_SAMPLES}/seed_${SEED_DOWNSTREAM}"
SCANPATH_RIO_OUT="Pasos/corridas_finales_half_data/downstream/scanpath_pretrained/rioplatense_hate_binary/k${DOWNSTREAM_HALF_TRAIN_SAMPLES}/seed_${SEED_DOWNSTREAM}"
BETO_BASE_XNLI_OUT="Pasos/corridas_finales_half_data/downstream/beto_base/xnli_es/k${DOWNSTREAM_HALF_TRAIN_SAMPLES}/seed_${SEED_DOWNSTREAM}"
BETO_BASE_RIO_OUT="Pasos/corridas_finales_half_data/downstream/beto_base/rioplatense_hate_binary/k${DOWNSTREAM_HALF_TRAIN_SAMPLES}/seed_${SEED_DOWNSTREAM}"

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

echo "Usando Train/Test MLM limpios por cuentos:"
echo "- Train: $MLM_TRAIN_FILE"
echo "- Eval/Test: $MLM_EVAL_FILE"
echo "- MLM half train samples: $MLM_HALF_TRAIN_SAMPLES"
echo "- MLM half eval samples: $MLM_HALF_EVAL_SAMPLES"
echo "- La submuestra es deterministica y shuffled por seed dentro de cada archivo."

# A. Pretraining
# 1) BETO solo
"$PYTHON_BIN" train_mlm_beto_baseline_step7_pretrain.py \
  --train_file "$MLM_TRAIN_FILE" \
  --eval_file "$MLM_EVAL_FILE" \
  --model_name_or_path "$BETO_BASE_MODEL" \
  --measured_text_field text \
  --measured_word_id_field word_id \
  --output_dir "$BASELINE_PRETRAIN_OUT" \
  --max_seq_length 128 \
  --max_train_samples "$MLM_HALF_TRAIN_SAMPLES" \
  --max_eval_samples "$MLM_HALF_EVAL_SAMPLES" \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --num_train_epochs 8 \
  --learning_rate 5e-5 \
  --max_masked_positions 3 \
  --split_strategy precomputed_files \
  --split_report_path "$BASELINE_PRETRAIN_OUT/split_report.json" \
  --save_every_epoch True \
  --seed "$SEED_PRETRAINING"
plot_pretraining_losses "$BASELINE_PRETRAIN_OUT"

# 2) BETO + Scanpath, sin fijaciones numericas, con word_id y lambda adaptativo
"$PYTHON_BIN" train_mlm_combined_step7.py \
  --train_file "$MLM_TRAIN_FILE" \
  --eval_file "$MLM_EVAL_FILE" \
  --model_name_or_path "$BETO_BASE_MODEL" \
  --measured_text_field text \
  --measured_word_id_field word_id \
  --output_dir "$SCANPATH_PRETRAIN_OUT" \
  --max_seq_length 128 \
  --max_train_samples "$MLM_HALF_TRAIN_SAMPLES" \
  --max_eval_samples "$MLM_HALF_EVAL_SAMPLES" \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --num_train_epochs 8 \
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
  --save_every_epoch True \
  --seed "$SEED_PRETRAINING"
plot_pretraining_losses "$SCANPATH_PRETRAIN_OUT"
plot_lambda_scanpath "$SCANPATH_PRETRAIN_OUT"

# B. Downstream desde BETO solo preentrenado
# 3) XNLI
"$PYTHON_BIN" train_spanish_downstream_baseline.py \
  --model_name_or_path "$BASELINE_BEST_CKPT" \
  --task_name xnli_es \
  --output_dir "$BASELINE_XNLI_OUT" \
  --max_train_samples "$DOWNSTREAM_HALF_TRAIN_SAMPLES" \
  --max_eval_samples "$DOWNSTREAM_HALF_EVAL_SAMPLES" \
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

# 4) Tarea rioplatense correcta definida para estas corridas: rioplatense_hate_binary
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
  --max_train_samples "$DOWNSTREAM_HALF_TRAIN_SAMPLES" \
  --max_eval_samples "$DOWNSTREAM_HALF_EVAL_SAMPLES" \
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

# C. Downstream desde BETO + Scanpath preentrenado
# 5) XNLI
"$PYTHON_BIN" train_spanish_downstream_baseline.py \
  --model_name_or_path "$SCANPATH_BEST_CKPT" \
  --task_name xnli_es \
  --output_dir "$SCANPATH_XNLI_OUT" \
  --max_train_samples "$DOWNSTREAM_HALF_TRAIN_SAMPLES" \
  --max_eval_samples "$DOWNSTREAM_HALF_EVAL_SAMPLES" \
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

# 6) Tarea rioplatense correcta definida para estas corridas: rioplatense_hate_binary
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
  --max_train_samples "$DOWNSTREAM_HALF_TRAIN_SAMPLES" \
  --max_eval_samples "$DOWNSTREAM_HALF_EVAL_SAMPLES" \
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

# D. BETO base directo
# 7) XNLI
"$PYTHON_BIN" train_spanish_downstream_baseline.py \
  --model_name_or_path "$BETO_BASE_MODEL" \
  --task_name xnli_es \
  --output_dir "$BETO_BASE_XNLI_OUT" \
  --max_train_samples "$DOWNSTREAM_HALF_TRAIN_SAMPLES" \
  --max_eval_samples "$DOWNSTREAM_HALF_EVAL_SAMPLES" \
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

# 8) Tarea rioplatense correcta definida para estas corridas: rioplatense_hate_binary
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
  --max_train_samples "$DOWNSTREAM_HALF_TRAIN_SAMPLES" \
  --max_eval_samples "$DOWNSTREAM_HALF_EVAL_SAMPLES" \
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
