#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

RUN_PRETRAINING="${RUN_PRETRAINING:-1}"
RUN_STEP9="${RUN_STEP9:-1}"

PRETRAINING_RUN_TAG="${PRETRAINING_RUN_TAG:-half_data_8ep_loss_curves}"
STEP9_RUN_TAG="${STEP9_RUN_TAG:-xnli_rioplatense_low_resource}"

# half = Paso 9 usa los checkpoints generados por el Paso 7 half-data de este flujo.
# full = Paso 9 usa los best_checkpoint full ya existentes.
STEP9_CHECKPOINT_SOURCE="${STEP9_CHECKPOINT_SOURCE:-half}"

BASELINE_REFERENCE_DIR="${BASELINE_REFERENCE_DIR:-Pasos/Paso 7 Preentrenamiento BETO Full}"
SCANPATH_REFERENCE_DIR="${SCANPATH_REFERENCE_DIR:-Pasos/paso_7_scanpath_full_8ep}"

BETO_BASE_MODEL="${BETO_BASE_MODEL:-dccuchile/bert-base-spanish-wwm-cased}"

STEP9_SEED="${STEP9_SEED:-111}"
STEP9_NUM_TRAIN_EPOCHS="${STEP9_NUM_TRAIN_EPOCHS:-20}"
STEP9_LEARNING_RATE="${STEP9_LEARNING_RATE:-2e-5}"
STEP9_MAX_SEQ_LENGTH="${STEP9_MAX_SEQ_LENGTH:-128}"
STEP9_PER_DEVICE_TRAIN_BATCH_SIZE="${STEP9_PER_DEVICE_TRAIN_BATCH_SIZE:-32}"
STEP9_PER_DEVICE_EVAL_BATCH_SIZE="${STEP9_PER_DEVICE_EVAL_BATCH_SIZE:-32}"
STEP9_LOGGING_STEPS="${STEP9_LOGGING_STEPS:-10}"
STEP9_SAVE_TOTAL_LIMIT="${STEP9_SAVE_TOTAL_LIMIT:-1}"

XNLI_MAX_TRAIN_SAMPLES="${XNLI_MAX_TRAIN_SAMPLES:-1000}"
XNLI_MAX_EVAL_SAMPLES="${XNLI_MAX_EVAL_SAMPLES:-1000}"
RIOPLATENSE_MAX_TRAIN_SAMPLES="${RIOPLATENSE_MAX_TRAIN_SAMPLES:-1000}"
RIOPLATENSE_DATASET_PATH="${RIOPLATENSE_DATASET_PATH:-}"

BASELINE_HALF_DIR="$BASELINE_REFERENCE_DIR/$PRETRAINING_RUN_TAG"
SCANPATH_HALF_DIR="$SCANPATH_REFERENCE_DIR/$PRETRAINING_RUN_TAG"

case "$STEP9_CHECKPOINT_SOURCE" in
  half)
    BETO_PRETRAINED_MODEL="$BASELINE_HALF_DIR/best_checkpoint"
    SCANPATH_PRETRAINED_MODEL="$SCANPATH_HALF_DIR/best_checkpoint"
    BETO_PRETRAINED_ARTIFACT_ROOT="$BASELINE_HALF_DIR/downstream/$STEP9_RUN_TAG"
    SCANPATH_ARTIFACT_ROOT="$SCANPATH_HALF_DIR/downstream/$STEP9_RUN_TAG"
    ;;
  full)
    BETO_PRETRAINED_MODEL="$BASELINE_REFERENCE_DIR/best_checkpoint"
    SCANPATH_PRETRAINED_MODEL="$SCANPATH_REFERENCE_DIR/best_checkpoint"
    BETO_PRETRAINED_ARTIFACT_ROOT="$BASELINE_REFERENCE_DIR/downstream/$STEP9_RUN_TAG"
    SCANPATH_ARTIFACT_ROOT="$SCANPATH_REFERENCE_DIR/downstream/$STEP9_RUN_TAG"
    ;;
  *)
    echo "Invalid STEP9_CHECKPOINT_SOURCE=$STEP9_CHECKPOINT_SOURCE. Use half or full."
    exit 1
    ;;
esac

BETO_BASE_ARTIFACT_ROOT="${BETO_BASE_ARTIFACT_ROOT:-results/paso_9/$STEP9_RUN_TAG/beto_base}"

TOTAL_STAGES=0
if [[ "$RUN_PRETRAINING" == "1" ]]; then
  TOTAL_STAGES=$((TOTAL_STAGES + 1))
fi
if [[ "$RUN_STEP9" == "1" ]]; then
  TOTAL_STAGES=$((TOTAL_STAGES + 6))
fi
CURRENT_STAGE=0

stage_start() {
  local label="$1"
  CURRENT_STAGE=$((CURRENT_STAGE + 1))
  local percent=$((CURRENT_STAGE * 100 / TOTAL_STAGES))
  echo
  echo "============================================================"
  echo "[$CURRENT_STAGE/$TOTAL_STAGES] ${percent}% - $label"
  echo "============================================================"
  date -Is
}

stage_done() {
  local label="$1"
  echo "Finished: $label"
  date -Is
}

require_local_checkpoint() {
  local path="$1"
  local label="$2"
  if [[ ! -d "$path" ]]; then
    echo "Missing checkpoint for $label:"
    echo "  $path"
    echo "If STEP9_CHECKPOINT_SOURCE=half, run with RUN_PRETRAINING=1 first."
    exit 1
  fi
}

plot_downstream_losses() {
  local output_dir="$1"
  if [[ -f "$output_dir/trainer_state.json" ]]; then
    "$PYTHON_BIN" scripts/plot_trainer_loss_curves.py --output_dir "$output_dir"
  else
    echo "Skipping loss plots; trainer_state.json not found in $output_dir"
  fi
}

write_downstream_description() {
  local output_dir="$1"
  local model_key="$2"
  local model_name_or_path="$3"
  local task_name="$4"
  local metric_name="$5"
  local max_train_samples="$6"
  local max_eval_samples="$7"
  local description_file="$output_dir/DESCRIPCION_CORRIDA.md"

  mkdir -p "$output_dir"
  {
    printf '# Paso 9 - %s - %s\n\n' "$task_name" "$model_key"
    printf 'Generated at: `%s`\n\n' "$(date -Is)"
    printf '## Descripcion\n\n'
    printf 'Fine-tuning downstream de Paso 9 usando un backbone BETO/Scanpath ya inicializado. Esta etapa si es supervisada; el Paso 7 previo fue preentrenamiento MLM.\n\n'
    printf '## Modelo\n\n'
    printf -- '- model_key: `%s`\n' "$model_key"
    printf -- '- model_name_or_path: `%s`\n' "$model_name_or_path"
    printf -- '- checkpoint_source: `%s`\n\n' "$STEP9_CHECKPOINT_SOURCE"
    printf '## Tarea\n\n'
    printf -- '- task_name: `%s`\n' "$task_name"
    printf -- '- metric_for_best_model: `%s`\n\n' "$metric_name"
    printf '## Hiperparametros\n\n'
    printf -- '- max_train_samples: `%s`\n' "$max_train_samples"
    printf -- '- max_eval_samples: `%s`\n' "$max_eval_samples"
    printf -- '- seed: `%s`\n' "$STEP9_SEED"
    printf -- '- num_train_epochs: `%s`\n' "$STEP9_NUM_TRAIN_EPOCHS"
    printf -- '- learning_rate: `%s`\n' "$STEP9_LEARNING_RATE"
    printf -- '- max_seq_length: `%s`\n' "$STEP9_MAX_SEQ_LENGTH"
    printf -- '- per_device_train_batch_size: `%s`\n' "$STEP9_PER_DEVICE_TRAIN_BATCH_SIZE"
    printf -- '- per_device_eval_batch_size: `%s`\n' "$STEP9_PER_DEVICE_EVAL_BATCH_SIZE"
    printf -- '- logging_steps: `%s`\n\n' "$STEP9_LOGGING_STEPS"
    printf '## Artefactos principales\n\n'
    printf -- '- output_dir: `%s`\n' "$output_dir"
    printf -- '- trainer_state: `%s/trainer_state.json`\n' "$output_dir"
    printf -- '- train_results: `%s/train_results.txt`\n' "$output_dir"
    printf -- '- eval_results: `%s/eval_results.json`\n' "$output_dir"
    printf -- '- test_results: `%s/test_results.json` si la tarea tiene test\n' "$output_dir"
    printf -- '- loss_curves_csv: `%s/trainer_loss_curves.csv`\n' "$output_dir"
    printf -- '- loss_plots_dir: `%s/loss_plots`\n\n' "$output_dir"
    if command -v jq >/dev/null 2>&1 && [[ -f "$output_dir/eval_results.json" ]]; then
      printf '## Resumen de resultados\n\n'
      jq -r 'to_entries[] | select(.key | startswith("eval_")) | "- \(.key): `\(.value)`"' "$output_dir/eval_results.json"
      if [[ -f "$output_dir/test_results.json" ]]; then
        jq -r 'to_entries[] | select(.key | startswith("test_")) | "- \(.key): `\(.value)`"' "$output_dir/test_results.json"
      fi
    fi
  } > "$description_file"
}

run_xnli_for_model() {
  local model_key="$1"
  local model_name_or_path="$2"
  local artifact_root="$3"
  local output_dir="$artifact_root/xnli_es/k${XNLI_MAX_TRAIN_SAMPLES}/seed_${STEP9_SEED}"

  "$PYTHON_BIN" train_spanish_downstream_baseline.py \
    --model_name_or_path "$model_name_or_path" \
    --task_name xnli_es \
    --output_dir "$output_dir" \
    --max_train_samples "$XNLI_MAX_TRAIN_SAMPLES" \
    --max_eval_samples "$XNLI_MAX_EVAL_SAMPLES" \
    --num_train_epochs "$STEP9_NUM_TRAIN_EPOCHS" \
    --learning_rate "$STEP9_LEARNING_RATE" \
    --per_device_train_batch_size "$STEP9_PER_DEVICE_TRAIN_BATCH_SIZE" \
    --per_device_eval_batch_size "$STEP9_PER_DEVICE_EVAL_BATCH_SIZE" \
    --max_seq_length "$STEP9_MAX_SEQ_LENGTH" \
    --seed "$STEP9_SEED" \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end \
    --save_total_limit "$STEP9_SAVE_TOTAL_LIMIT" \
    --expected_num_labels 3

  plot_downstream_losses "$output_dir"
  write_downstream_description "$output_dir" "$model_key" "$model_name_or_path" "xnli_es" "accuracy" "$XNLI_MAX_TRAIN_SAMPLES" "$XNLI_MAX_EVAL_SAMPLES"
}

run_rioplatense_for_model() {
  local model_key="$1"
  local model_name_or_path="$2"
  local artifact_root="$3"
  local output_dir="$artifact_root/rioplatense_hate_binary/k${RIOPLATENSE_MAX_TRAIN_SAMPLES}/seed_${STEP9_SEED}"
  local dataset_args=()

  if [[ -n "$RIOPLATENSE_DATASET_PATH" ]]; then
    dataset_args+=(--dataset_path "$RIOPLATENSE_DATASET_PATH")
  fi

  "$PYTHON_BIN" train_glue_LM_baseline.py \
    --model_name_or_path "$model_name_or_path" \
    --task_name rioplatense_hate_binary \
    "${dataset_args[@]}" \
    --output_dir "$output_dir" \
    --num_train_epochs "$STEP9_NUM_TRAIN_EPOCHS" \
    --learning_rate "$STEP9_LEARNING_RATE" \
    --per_device_train_batch_size "$STEP9_PER_DEVICE_TRAIN_BATCH_SIZE" \
    --per_device_eval_batch_size "$STEP9_PER_DEVICE_EVAL_BATCH_SIZE" \
    --max_seq_length "$STEP9_MAX_SEQ_LENGTH" \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --metric_for_best_model macro_f1 \
    --greater_is_better True \
    --train_as_val True \
    --max_train_samples "$RIOPLATENSE_MAX_TRAIN_SAMPLES" \
    --low_resource_data_seed "$STEP9_SEED" \
    --seed "$STEP9_SEED" \
    --logging_steps "$STEP9_LOGGING_STEPS" \
    --report_to none \
    --load_best_model_at_end \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --do_predict

  plot_downstream_losses "$output_dir"
  write_downstream_description "$output_dir" "$model_key" "$model_name_or_path" "rioplatense_hate_binary" "macro_f1" "$RIOPLATENSE_MAX_TRAIN_SAMPLES" "1000"
}

echo "Repository: $SCRIPT_DIR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "STEP9_CHECKPOINT_SOURCE=$STEP9_CHECKPOINT_SOURCE"

if [[ "$RUN_PRETRAINING" == "1" ]]; then
  stage_start "Paso 7 MLM pretraining: BETO baseline y BETO + Scanpath"
  RUN_TAG="$PRETRAINING_RUN_TAG" \
  BASELINE_REFERENCE_DIR="$BASELINE_REFERENCE_DIR" \
  SCANPATH_REFERENCE_DIR="$SCANPATH_REFERENCE_DIR" \
    ./run_beto_pretraining_8ep_half_loss_curves.sh
  stage_done "Paso 7 MLM pretraining"
fi

if [[ "$RUN_STEP9" == "1" ]]; then
  require_local_checkpoint "$BETO_PRETRAINED_MODEL" "BETO preentrenado"
  require_local_checkpoint "$SCANPATH_PRETRAINED_MODEL" "BETO + Scanpath preentrenado"

  stage_start "Paso 9 XNLI - BETO base"
  run_xnli_for_model "beto_base" "$BETO_BASE_MODEL" "$BETO_BASE_ARTIFACT_ROOT"
  stage_done "Paso 9 XNLI - BETO base"

  stage_start "Paso 9 Rioplatense - BETO base"
  run_rioplatense_for_model "beto_base" "$BETO_BASE_MODEL" "$BETO_BASE_ARTIFACT_ROOT"
  stage_done "Paso 9 Rioplatense - BETO base"

  stage_start "Paso 9 XNLI - BETO preentrenado"
  run_xnli_for_model "beto_pretrained" "$BETO_PRETRAINED_MODEL" "$BETO_PRETRAINED_ARTIFACT_ROOT"
  stage_done "Paso 9 XNLI - BETO preentrenado"

  stage_start "Paso 9 Rioplatense - BETO preentrenado"
  run_rioplatense_for_model "beto_pretrained" "$BETO_PRETRAINED_MODEL" "$BETO_PRETRAINED_ARTIFACT_ROOT"
  stage_done "Paso 9 Rioplatense - BETO preentrenado"

  stage_start "Paso 9 XNLI - BETO + Scanpath preentrenado"
  run_xnli_for_model "scanpath_pretrained" "$SCANPATH_PRETRAINED_MODEL" "$SCANPATH_ARTIFACT_ROOT"
  stage_done "Paso 9 XNLI - BETO + Scanpath preentrenado"

  stage_start "Paso 9 Rioplatense - BETO + Scanpath preentrenado"
  run_rioplatense_for_model "scanpath_pretrained" "$SCANPATH_PRETRAINED_MODEL" "$SCANPATH_ARTIFACT_ROOT"
  stage_done "Paso 9 Rioplatense - BETO + Scanpath preentrenado"
fi

echo
echo "All requested stages finished."
echo "Paso 7 BETO artifacts: $BASELINE_HALF_DIR"
echo "Paso 7 Scanpath artifacts: $SCANPATH_HALF_DIR"
echo "Paso 9 BETO base artifacts: $BETO_BASE_ARTIFACT_ROOT"
echo "Paso 9 BETO pretrained artifacts: $BETO_PRETRAINED_ARTIFACT_ROOT"
echo "Paso 9 Scanpath artifacts: $SCANPATH_ARTIFACT_ROOT"
