#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
DATASET_PATH="${DATASET_PATH:-}"
SEED="${SEED:-111}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-1000}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-20}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-32}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-32}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-128}"
OUTPUT_ROOT="${OUTPUT_ROOT:-}"
RUN_TAG="${RUN_TAG:-rioplatense_hate_binary_low_resource}"
RUN_DESCRIPTION_FILENAME="${RUN_DESCRIPTION_FILENAME:-DESCRIPCION_CORRIDA.md}"

BETO_BASE_MODEL="${BETO_BASE_MODEL:-dccuchile/bert-base-spanish-wwm-cased}"
BETO_FULL_MODEL="${BETO_FULL_MODEL:-Pasos/Paso 7 Preentrenamiento BETO Full/best_checkpoint}"
SCANPATH_MODEL="${SCANPATH_MODEL:-Pasos/paso_7_scanpath_full_8ep/best_checkpoint}"

BETO_BASE_ARTIFACT_ROOT="${BETO_BASE_ARTIFACT_ROOT:-results/paso_9/$RUN_TAG/beto_base}"
BETO_FULL_ARTIFACT_ROOT="${BETO_FULL_ARTIFACT_ROOT:-Pasos/Paso 7 Preentrenamiento BETO Full/downstream/$RUN_TAG}"
SCANPATH_ARTIFACT_ROOT="${SCANPATH_ARTIFACT_ROOT:-Pasos/paso_7_scanpath_full_8ep/downstream/$RUN_TAG}"

if [[ "$#" -gt 0 ]]; then
  EXTRA_ARGS_DESCRIPTION="$*"
else
  EXTRA_ARGS_DESCRIPTION="none"
fi

DATASET_ARGS=()
if [[ -n "$DATASET_PATH" ]]; then
  DATASET_ARGS+=(--dataset_path "$DATASET_PATH")
fi

write_run_description() {
  local output_dir="$1"
  local run_name="$2"
  local model_name_or_path="$3"
  local architecture_reference="$4"
  local description_file="$output_dir/$RUN_DESCRIPTION_FILENAME"

  mkdir -p "$output_dir"
  {
    printf '# Rioplatense hate binary - %s\n\n' "$run_name"
    printf 'Generated at: `%s`\n\n' "$(date -Is)"
    printf '## Descripcion\n\n'
    printf 'Corrida downstream de Paso 9 para clasificacion binaria de hate speech rioplatense. El output queda junto a la carpeta de la arquitectura usada como punto de partida, para que resultados, metricas, predicciones y configuracion queden trazables al checkpoint correspondiente.\n\n'
    printf '## Arquitectura de referencia\n\n'
    printf -- '- `%s`\n\n' "$architecture_reference"
    printf '## Hiperparametros\n\n'
    printf -- '- model_name_or_path: `%s`\n' "$model_name_or_path"
    printf -- '- task_name: `rioplatense_hate_binary`\n'
    printf -- '- dataset_path: `%s`\n' "${DATASET_PATH:-GitHub raw finiteautomata/rioplatense_hate_speech}"
    printf -- '- train_as_val: `True`\n'
    printf -- '- max_train_samples: `%s`\n' "$MAX_TRAIN_SAMPLES"
    printf -- '- low_resource_data_seed: `%s`\n' "$SEED"
    printf -- '- num_train_epochs: `%s`\n' "$NUM_TRAIN_EPOCHS"
    printf -- '- learning_rate: `%s`\n' "$LEARNING_RATE"
    printf -- '- max_seq_length: `%s`\n' "$MAX_SEQ_LENGTH"
    printf -- '- per_device_train_batch_size: `%s`\n' "$PER_DEVICE_TRAIN_BATCH_SIZE"
    printf -- '- per_device_eval_batch_size: `%s`\n' "$PER_DEVICE_EVAL_BATCH_SIZE"
    printf -- '- evaluation_strategy: `epoch`\n'
    printf -- '- save_strategy: `epoch`\n'
    printf -- '- metric_for_best_model: `macro_f1`\n'
    printf -- '- greater_is_better: `True`\n'
    printf -- '- load_best_model_at_end: `True`\n'
    printf -- '- do_train/do_eval/do_predict: `True/True/True`\n'
    printf -- '- extra_cli_args: `%s`\n\n' "$EXTRA_ARGS_DESCRIPTION"
    printf '## Artefactos principales\n\n'
    printf -- '- output_dir: `%s`\n' "$output_dir"
    printf -- '- model_init_check: `%s/model_init_check.json`\n' "$output_dir"
    printf -- '- class_distribution: `%s/class_distribution.json`\n' "$output_dir"
    printf -- '- train_results: `%s/train_results.txt`\n' "$output_dir"
    printf -- '- eval_results: `%s/eval_results.json`\n' "$output_dir"
    printf -- '- test_results: `%s/test_results.json`\n' "$output_dir"
    printf -- '- test_predictions: `%s/test_results_rioplatense_hate_binary.txt`\n\n' "$output_dir"
    if command -v jq >/dev/null 2>&1 && [[ -f "$output_dir/eval_results.json" ]]; then
      printf '## Resumen de resultados\n\n'
      jq -r '"- eval_loss: `\(.eval_loss)`\n- eval_accuracy: `\(.eval_accuracy)`\n- eval_macro_f1: `\(.eval_macro_f1)`"' "$output_dir/eval_results.json"
      if [[ -f "$output_dir/test_results.json" ]]; then
        jq -r '"- test_loss: `\(.test_loss)`\n- test_accuracy: `\(.test_accuracy)`\n- test_macro_f1: `\(.test_macro_f1)`"' "$output_dir/test_results.json"
      fi
    fi
  } > "$description_file"
}

run_one() {
  local run_name="$1"
  local model_name_or_path="$2"
  local artifact_root="$3"
  local architecture_reference="$4"
  local output_dir

  if [[ -n "$OUTPUT_ROOT" ]]; then
    output_dir="$OUTPUT_ROOT/$run_name/k${MAX_TRAIN_SAMPLES}/seed_${SEED}"
  else
    output_dir="$artifact_root/k${MAX_TRAIN_SAMPLES}/seed_${SEED}"
  fi

  echo "Running rioplatense_hate_binary: $run_name"
  echo "  model: $model_name_or_path"
  echo "  output: $output_dir"

  "$PYTHON_BIN" train_glue_LM_baseline.py \
    --model_name_or_path "$model_name_or_path" \
    --task_name rioplatense_hate_binary \
    "${DATASET_ARGS[@]}" \
    --output_dir "$output_dir" \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --metric_for_best_model macro_f1 \
    --greater_is_better True \
    --train_as_val True \
    --max_train_samples "$MAX_TRAIN_SAMPLES" \
    --low_resource_data_seed "$SEED" \
    --load_best_model_at_end \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --do_predict

  write_run_description "$output_dir" "$run_name" "$model_name_or_path" "$architecture_reference"
}

run_one "beto_base" "$BETO_BASE_MODEL" "$BETO_BASE_ARTIFACT_ROOT" "dccuchile/bert-base-spanish-wwm-cased"
run_one "beto_full" "$BETO_FULL_MODEL" "$BETO_FULL_ARTIFACT_ROOT" "Pasos/Paso 7 Preentrenamiento BETO Full"
run_one "scanpath_full_8ep" "$SCANPATH_MODEL" "$SCANPATH_ARTIFACT_ROOT" "Pasos/paso_7_scanpath_full_8ep"

if [[ -n "$OUTPUT_ROOT" ]]; then
  echo "Done. Outputs under $OUTPUT_ROOT"
else
  echo "Done. Outputs under:"
  echo "  beto_base: $BETO_BASE_ARTIFACT_ROOT"
  echo "  beto_full: $BETO_FULL_ARTIFACT_ROOT"
  echo "  scanpath_full_8ep: $SCANPATH_ARTIFACT_ROOT"
fi
