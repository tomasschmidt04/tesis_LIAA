#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

RUN_PRETRAINING="${RUN_PRETRAINING:-1}"
RUN_DOWNSTREAM="${RUN_DOWNSTREAM:-1}"

# By default Step 9 uses the previous full Paso 7 checkpoints.
# Set USE_HALF_PRETRAINED_FOR_STEP9=1 to fine-tune the half-data checkpoints
# produced by run_beto_pretraining_8ep_half_loss_curves.sh in the same run.
USE_HALF_PRETRAINED_FOR_STEP9="${USE_HALF_PRETRAINED_FOR_STEP9:-0}"

PRETRAINING_RUN_TAG="${PRETRAINING_RUN_TAG:-half_data_8ep_loss_curves}"
DOWNSTREAM_RUN_TAG="${DOWNSTREAM_RUN_TAG:-rioplatense_hate_binary_low_resource}"

BASELINE_REFERENCE_DIR="${BASELINE_REFERENCE_DIR:-Pasos/Paso 7 Preentrenamiento BETO Full}"
SCANPATH_REFERENCE_DIR="${SCANPATH_REFERENCE_DIR:-Pasos/paso_7_scanpath_full_8ep}"

BETO_HALF_CHECKPOINT="$BASELINE_REFERENCE_DIR/$PRETRAINING_RUN_TAG/best_checkpoint"
SCANPATH_HALF_CHECKPOINT="$SCANPATH_REFERENCE_DIR/$PRETRAINING_RUN_TAG/best_checkpoint"

echo "Repository: $SCRIPT_DIR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "PYTHON_BIN=$PYTHON_BIN"

if [[ "$RUN_PRETRAINING" == "1" ]]; then
  echo
  echo "== Paso 7 half-data pretraining =="
  RUN_TAG="$PRETRAINING_RUN_TAG" \
  BASELINE_REFERENCE_DIR="$BASELINE_REFERENCE_DIR" \
  SCANPATH_REFERENCE_DIR="$SCANPATH_REFERENCE_DIR" \
    ./run_beto_pretraining_8ep_half_loss_curves.sh
fi

if [[ "$RUN_DOWNSTREAM" == "1" ]]; then
  echo
  echo "== Paso 9 rioplatense hate binary low-resource =="

  if [[ "$USE_HALF_PRETRAINED_FOR_STEP9" == "1" ]]; then
    if [[ ! -d "$BETO_HALF_CHECKPOINT" || ! -d "$SCANPATH_HALF_CHECKPOINT" ]]; then
      echo "Missing half-data checkpoints."
      echo "Expected:"
      echo "  $BETO_HALF_CHECKPOINT"
      echo "  $SCANPATH_HALF_CHECKPOINT"
      echo "Run pretraining first or set USE_HALF_PRETRAINED_FOR_STEP9=0."
      exit 1
    fi

    BETO_FULL_MODEL="$BETO_HALF_CHECKPOINT" \
    SCANPATH_MODEL="$SCANPATH_HALF_CHECKPOINT" \
    RUN_TAG="$DOWNSTREAM_RUN_TAG" \
      ./run_step9_rioplatense_hate_binary_low_resource.sh
  else
    RUN_TAG="$DOWNSTREAM_RUN_TAG" \
      ./run_step9_rioplatense_hate_binary_low_resource.sh
  fi
fi

echo
echo "Done."
