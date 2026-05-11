#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
MEASURED_SCANPATH_FILE="${MEASURED_SCANPATH_FILE:-/home/tomi/tesis_LIAA/reading-et/aligned_output/aligned_scanpaths.jsonl}"
MLM_TRAIN_FILE="${MLM_TRAIN_FILE:-/home/tomi/tesis_LIAA/reading-et/mlm_dataset_limpio_train_test/train.jsonl}"
MLM_EVAL_FILE="${MLM_EVAL_FILE:-/home/tomi/tesis_LIAA/reading-et/mlm_dataset_limpio_train_test/test.jsonl}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-dccuchile/bert-base-spanish-wwm-cased}"

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-8}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-128}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-4}"
LEARNING_RATE="${LEARNING_RATE:-5e-05}"
MAX_MASKED_POSITIONS="${MAX_MASKED_POSITIONS:-3}"
SEED="${SEED:-13}"
AUX_WEIGHT="${AUX_WEIGHT:-0.1}"
SPLIT_STRATEGY="${SPLIT_STRATEGY:-precomputed_files}"
EVAL_SENTENCE_POSITION_MOD="${EVAL_SENTENCE_POSITION_MOD:-10}"
EVAL_SENTENCE_POSITION_REMAINDER="${EVAL_SENTENCE_POSITION_REMAINDER:-5}"
DRY_RUN_SPLIT_ONLY="${DRY_RUN_SPLIT_ONLY:-False}"

# Mitad aproximada de las corridas full previas:
# - BETO baseline full: train=63000, eval=2000
# - BETO+scanpath full: train=63671, eval=2000
BASELINE_MAX_TRAIN_SAMPLES="${BASELINE_MAX_TRAIN_SAMPLES:-31500}"
SCANPATH_MAX_TRAIN_SAMPLES="${SCANPATH_MAX_TRAIN_SAMPLES:-31836}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-1000}"

BASELINE_REFERENCE_DIR="${BASELINE_REFERENCE_DIR:-Pasos/Paso 7 Preentrenamiento BETO Full}"
SCANPATH_REFERENCE_DIR="${SCANPATH_REFERENCE_DIR:-Pasos/paso_7_scanpath_full_8ep}"
RUN_TAG="${RUN_TAG:-half_data_8ep_loss_curves}"
RUN_DESCRIPTION_FILENAME="${RUN_DESCRIPTION_FILENAME:-DESCRIPCION_CORRIDA.md}"

BASELINE_OUTPUT_DIR="${BASELINE_OUTPUT_DIR:-$BASELINE_REFERENCE_DIR/$RUN_TAG}"
SCANPATH_OUTPUT_DIR="${SCANPATH_OUTPUT_DIR:-$SCANPATH_REFERENCE_DIR/$RUN_TAG}"

if [[ "$#" -gt 0 ]]; then
  EXTRA_ARGS_DESCRIPTION="$*"
else
  EXTRA_ARGS_DESCRIPTION="none"
fi

build_split_args() {
  if [[ "$SPLIT_STRATEGY" == "precomputed_files" ]]; then
    printf '%s\n' --train_file "$MLM_TRAIN_FILE" --eval_file "$MLM_EVAL_FILE"
  else
    printf '%s\n' --measured_scanpath_file "$MEASURED_SCANPATH_FILE"
  fi
}

write_run_description() {
  local output_dir="$1"
  local run_title="$2"
  local architecture_reference="$3"
  local max_train_samples="$4"
  local aux_weight_value="$5"
  local description_file="$output_dir/$RUN_DESCRIPTION_FILENAME"

  mkdir -p "$output_dir"
  {
    printf '# %s\n\n' "$run_title"
    printf 'Generated at: `%s`\n\n' "$(date -Is)"
    printf '## Descripcion\n\n'
    printf 'Corrida de preentrenamiento MLM de Paso 7 usando aproximadamente la mitad de los datos de las corridas full previas. Los artefactos quedan dentro de la carpeta de la arquitectura de referencia para mantener juntos checkpoints, curvas y resultados.\n\n'
    printf '## Arquitectura de referencia\n\n'
    printf -- '- `%s`\n\n' "$architecture_reference"
    printf '## Hiperparametros\n\n'
    printf -- '- model_name_or_path: `%s`\n' "$MODEL_NAME_OR_PATH"
    printf -- '- measured_scanpath_file: `%s`\n' "$MEASURED_SCANPATH_FILE"
    printf -- '- mlm_train_file: `%s`\n' "$MLM_TRAIN_FILE"
    printf -- '- mlm_eval_file: `%s`\n' "$MLM_EVAL_FILE"
    printf -- '- split: `train`\n'
    printf -- '- max_train_samples: `%s`\n' "$max_train_samples"
    printf -- '- max_eval_samples: `%s`\n' "$MAX_EVAL_SAMPLES"
    printf -- '- num_train_epochs: `%s`\n' "$NUM_TRAIN_EPOCHS"
    printf -- '- max_seq_length: `%s`\n' "$MAX_SEQ_LENGTH"
    printf -- '- per_device_train_batch_size: `%s`\n' "$PER_DEVICE_TRAIN_BATCH_SIZE"
    printf -- '- per_device_eval_batch_size: `%s`\n' "$PER_DEVICE_EVAL_BATCH_SIZE"
    printf -- '- learning_rate: `%s`\n' "$LEARNING_RATE"
    printf -- '- max_masked_positions: `%s`\n' "$MAX_MASKED_POSITIONS"
    printf -- '- split_strategy: `%s`\n' "$SPLIT_STRATEGY"
    printf -- '- eval_sentence_position_mod: `%s`\n' "$EVAL_SENTENCE_POSITION_MOD"
    printf -- '- eval_sentence_position_remainder: `%s`\n' "$EVAL_SENTENCE_POSITION_REMAINDER"
    printf -- '- dry_run_split_only: `%s`\n' "$DRY_RUN_SPLIT_ONLY"
    printf -- '- seed: `%s`\n' "$SEED"
    printf -- '- aux_weight: `%s`\n' "$aux_weight_value"
    printf -- '- save_every_epoch: `True` unless overridden by extra_cli_args\n'
    printf -- '- extra_cli_args: `%s`\n\n' "$EXTRA_ARGS_DESCRIPTION"
    printf '## Artefactos principales\n\n'
    printf -- '- output_dir: `%s`\n' "$output_dir"
    printf -- '- loss_curves_csv: `%s/loss_curves.csv`\n' "$output_dir"
    printf -- '- loss_plots_dir: `%s/loss_plots`\n' "$output_dir"
    printf -- '- checkpoint_final: `%s/checkpoint_final`\n' "$output_dir"
    printf -- '- best_checkpoint: `%s/best_checkpoint`\n' "$output_dir"
  } > "$description_file"
}

echo "Running BETO baseline: $BASELINE_OUTPUT_DIR"
mapfile -t SPLIT_ARGS < <(build_split_args)
"$PYTHON_BIN" train_mlm_beto_baseline_step7_pretrain.py \
  "${SPLIT_ARGS[@]}" \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --measured_text_field text \
  --measured_word_id_field word_id \
  --split train \
  --output_dir "$BASELINE_OUTPUT_DIR" \
  --max_seq_length "$MAX_SEQ_LENGTH" \
  --max_train_samples "$BASELINE_MAX_TRAIN_SAMPLES" \
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
  --seed "$SEED" \
  "$@"

if [[ "$DRY_RUN_SPLIT_ONLY" != "True" && "$DRY_RUN_SPLIT_ONLY" != "true" && "$DRY_RUN_SPLIT_ONLY" != "1" ]]; then
  "$PYTHON_BIN" scripts/plot_loss_curves.py --output_dir "$BASELINE_OUTPUT_DIR"
fi
write_run_description \
  "$BASELINE_OUTPUT_DIR" \
  "BETO baseline half-data pretraining" \
  "$BASELINE_REFERENCE_DIR" \
  "$BASELINE_MAX_TRAIN_SAMPLES" \
  "N/A"

echo "Running BETO + scanpath: $SCANPATH_OUTPUT_DIR"
mapfile -t SPLIT_ARGS < <(build_split_args)
"$PYTHON_BIN" train_mlm_combined_step7.py \
  "${SPLIT_ARGS[@]}" \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --measured_text_field text \
  --measured_word_id_field word_id \
  --split train \
  --output_dir "$SCANPATH_OUTPUT_DIR" \
  --max_seq_length "$MAX_SEQ_LENGTH" \
  --max_train_samples "$SCANPATH_MAX_TRAIN_SAMPLES" \
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
  --seed "$SEED" \
  "$@"

if [[ "$DRY_RUN_SPLIT_ONLY" != "True" && "$DRY_RUN_SPLIT_ONLY" != "true" && "$DRY_RUN_SPLIT_ONLY" != "1" ]]; then
  "$PYTHON_BIN" scripts/plot_loss_curves.py --output_dir "$SCANPATH_OUTPUT_DIR"
fi
write_run_description \
  "$SCANPATH_OUTPUT_DIR" \
  "BETO + scanpath half-data pretraining" \
  "$SCANPATH_REFERENCE_DIR" \
  "$SCANPATH_MAX_TRAIN_SAMPLES" \
  "$AUX_WEIGHT"

echo "Done."
echo "Baseline artifacts: $BASELINE_OUTPUT_DIR"
echo "Scanpath artifacts: $SCANPATH_OUTPUT_DIR"
