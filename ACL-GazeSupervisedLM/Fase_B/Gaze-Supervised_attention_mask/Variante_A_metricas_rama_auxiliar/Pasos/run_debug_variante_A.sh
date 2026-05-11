#!/usr/bin/env bash
set -euo pipefail

MEASURED_SCANPATH_FILE="../reading-et/results_all_alligned_2/La noche de los feos/sub-007.json"
OUTPUT_DIR="/tmp/variante_A_metricas_debug_model_output"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

env -u LOCAL_RANK -u RANK -u WORLD_SIZE -u MASTER_ADDR -u MASTER_PORT CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" "$PYTHON_BIN" train_glue_gazesup_bert_low_resource.py \
  --model_name_or_path dccuchile/bert-base-spanish-wwm-cased \
  --task_name custom_aligned_scanpath \
  --scanpath_source measured \
  --measured_scanpath_file "$MEASURED_SCANPATH_FILE" \
  --measured_text_field text \
  --measured_word_id_field word_id \
  --label_name coverage \
  --use_gaze_features True \
  --gaze_feature_dim 3 \
  --gaze_feature_names FFD,TRT,nFix \
  --debug_gaze_features True \
  --max_seq_length 64 \
  --max_train_samples 2 \
  --max_eval_samples 2 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --learning_rate 2e-5 \
  --output_dir "$OUTPUT_DIR" \
  --overwrite_output_dir True \
  --overwrite_cache True \
  --report_to none \
  --use_cpu True \
  --do_train \
  --do_eval
