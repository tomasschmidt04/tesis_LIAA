# BETO Gaze-Supervised Transformers for Spanish

This repository currently focuses on a Spanish gaze-supervised pipeline built on top of **BETO** (`dccuchile/bert-base-spanish-wwm-cased`) and **measured scanpaths**. The main goal is to study whether scanpath-aware auxiliary supervision improves the internal representations learned by a Spanish Transformer and whether that improvement transfers to downstream tasks in Spanish.

The original English BERT/RoBERTa GLUE scripts from the ACL Gaze-Supervised LM project are still included for reference, but the active development in this repository is the **Spanish BETO pipeline**:

- measured scanpath ingestion,
- scanpath-aware auxiliary MLM training,
- reusable BETO backbone export,
- Spanish downstream transfer from the trained checkpoint,
- direct BETO baseline comparison.

## Repository Source

This project lives inside the following GitHub repository:

```text
https://github.com/tomasschmidt04/tesis_LIAA
```

The folder view

```text
https://github.com/tomasschmidt04/tesis_LIAA/tree/main/ACL-GazeSupervisedLM
```

is **not** a clonable Git remote by itself. To clone the project correctly, clone the repository root and then move into the subdirectory:

```bash
git clone https://github.com/tomasschmidt04/tesis_LIAA.git
cd tesis_LIAA/ACL-GazeSupervisedLM
```

## What This Repository Implements

The repository now supports two levels of experimentation:

- **Spanish BETO pipeline**: the main focus of this repository.
- **Original English GLUE experiments**: preserved mainly as reference launchers.

The Spanish pipeline is organized around a measured-scanpath auxiliary branch that operates during training only. Downstream inference remains a standard Transformer classification workflow.

## BETO Architecture

The current Spanish architecture combines a standard masked-language-model objective with an auxiliary scanpath-level objective.

### Core idea

1. Text is encoded by **BETO** in its original token order.
2. A measured scanpath provides a sequence of lexical fixation positions.
3. The lexical fixation sequence is expanded to token/subtoken positions.
4. BETO hidden states are selected in **scanpath order**.
5. The selected sequence is passed through a **GRU**.
6. An auxiliary MLM head predicts directly over the scanpath-level sequence.

### Current auxiliary branch design

The current implementation keeps the GRU output at scanpath length `S` and predicts directly over that sequence:

```text
Text -> BETO -> hidden states (B, T, d)
Measured scanpath -> token-level gaze positions (B, S)
Selected hidden states -> GRU -> scanpath hidden states (B, S, d)
Auxiliary MLM head -> scanpath logits (B, S, vocab_size)
```

This is the important design choice in the current codebase:

- the GRU output is **not** re-aggregated back to original token length `T`,
- the auxiliary MLM loss is computed directly over scanpath length `S`,
- repeated fixations contribute multiple times to the auxiliary loss.

### Combined training objective

For the combined BETO MLM model, the total loss is:

```text
total_loss = main_mlm_loss + aux_weight * scanpath_mlm_loss
```

This is implemented in the current BETO training pipeline and is the basis for the `l7b` experiments.

### Downstream usage

After pretraining, downstream evaluation uses **only the main Transformer backbone**:

- no scanpath branch,
- no GRU,
- no auxiliary MLM loss,
- only a standard downstream classification head.

That design keeps downstream transfer comparable to a standard Transformer baseline.

## Key Files for the Spanish BETO Pipeline

These are the most important files if your goal is to reproduce the Spanish experiments.

### Model and utilities

- [Gazesup_bert_model.py](./Gazesup_bert_model.py)
- [Gazesup_bert_mlm_model.py](./Gazesup_bert_mlm_model.py)
- [Gazesup_bert_combined_mlm_model.py](./Gazesup_bert_combined_mlm_model.py)
- [measured_scanpath_utils.py](./measured_scanpath_utils.py)

### BETO pretraining

- [train_mlm_scanpath_step5.py](./train_mlm_scanpath_step5.py)
- [train_mlm_combined_step6.py](./train_mlm_combined_step6.py)
- [train_mlm_combined_step7.py](./train_mlm_combined_step7.py)

### Backbone export and downstream transfer

- [export_and_verify_backbone_step8.py](./export_and_verify_backbone_step8.py)
- [train_spanish_downstream_baseline.py](./train_spanish_downstream_baseline.py)
- [run_spanish_downstream_from_l7b_step9.py](./run_spanish_downstream_from_l7b_step9.py)
- [run_spanish_baseline_beto_step9b.py](./run_spanish_baseline_beto_step9b.py)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/tomasschmidt04/tesis_LIAA.git
cd tesis_LIAA/ACL-GazeSupervisedLM
```

### 2. Create an environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

On Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Feature normalization files

The repository already contains the normalization files required by the original gaze-supervised code:

- `bert_feature_norm_celer.pickle`
- `roberta_feature_norm_celer.pickle`

They are small and are kept in the repository because the scripts use them directly.

## Data Requirements

### Measured scanpaths

The Spanish BETO pipeline expects an aligned measured-scanpath file in `json`, `jsonl`, or `csv` format. At minimum, the file should contain:

- `text`: the sentence to tokenize,
- `word_id`: the 1-based lexical fixation sequence.

The helper in [measured_scanpath_utils.py](./measured_scanpath_utils.py) converts that information into the token-level structures used by the model.

The current commands assume a file like:

```text
..\reading-et\aligned_output\aligned_scanpaths.jsonl
```

That aligned reading dataset is external to this repository.

### Spanish downstream tasks

The current downstream evaluation in Spanish uses exactly two tasks:

- `xnli_es`
- `intertass2020`

They are loaded automatically by the downstream scripts.

## Reproducing the Spanish BETO Pipeline

This is the main section of the repository.

### Optional validation of the auxiliary branch

Small-scale validation of the scanpath-only auxiliary MLM branch:

```powershell
.\.venv\Scripts\python.exe .\train_mlm_scanpath_step5.py --measured_scanpath_file "..\reading-et\results_all_alligned\Ahora debería reírme, si no estuviera muerto\sub-001.json" --model_name_or_path bert-base-cased --measured_text_field text --measured_word_id_field word_id --split train --output_dir Pasos/paso_5 --max_seq_length 128 --max_train_samples 8 --per_device_train_batch_size 2 --num_train_epochs 1 --learning_rate 5e-05 --max_masked_positions 3 --seed 13
```

Small-scale validation of the combined main MLM + auxiliary scanpath MLM model:

```powershell
.\.venv\Scripts\python.exe .\train_mlm_combined_step6.py --measured_scanpath_file "..\reading-et\results_all_alligned\Ahora debería reírme, si no estuviera muerto\sub-001.json" --model_name_or_path bert-base-cased --measured_text_field text --measured_word_id_field word_id --split train --output_dir Pasos/paso_6 --max_seq_length 128 --max_train_samples 8 --per_device_train_batch_size 2 --num_train_epochs 1 --learning_rate 5e-05 --max_masked_positions 3 --aux_weight 0.3 --seed 13
```

These are smoke-style checks. The main BETO experiment starts in `step7`.

### Main BETO pretraining experiment

Current canonical long-running BETO training command:

```powershell
.\.venv\Scripts\python.exe .\train_mlm_combined_step7.py --measured_scanpath_file "..\reading-et\aligned_output\aligned_scanpaths.jsonl" --model_name_or_path dccuchile/bert-base-spanish-wwm-cased --measured_text_field text --measured_word_id_field word_id --split train --output_dir Pasos/paso_7b --max_seq_length 128 --max_train_samples 2000 --max_eval_samples 256 --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --num_train_epochs 5 --learning_rate 5e-05 --max_masked_positions 3 --aux_weight 0.1 --save_every_epoch True --seed 13
```

This produces reusable checkpoints such as:

- `Pasos/paso_7b/best_checkpoint`
- `Pasos/paso_7b/checkpoint_final`

### Export the reusable BETO backbone

```powershell
.\.venv\Scripts\python.exe .\export_and_verify_backbone_step8.py --source_checkpoint "Pasos/paso_7b/best_checkpoint" --output_dir Pasos/paso_8b --export_dirname export_backbone_hf --downstream_num_labels 2
```

This step is useful to inspect and separate the reusable main Transformer backbone from the auxiliary scanpath branch.

### Spanish downstream transfer from the trained `l7b` experiment

```powershell
.\.venv\Scripts\python.exe .\run_spanish_downstream_from_l7b_step9.py --l7b_dir .\Pasos\paso_7b --output_dir .\Pasos\paso_9_es_l7b --max_train_samples 500 --max_eval_samples 200 --num_train_epochs 3 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --max_seq_length 128 --learning_rate 2e-5 --seed 13
```

This script:

- resolves the best usable checkpoint from `l7b`,
- loads it as downstream initialization,
- runs `xnli_es`,
- runs `intertass2020`,
- stores task-specific outputs under `Pasos/paso_9_es_l7b/`.

### Direct BETO baseline for comparison

```powershell
.\.venv\Scripts\python.exe .\run_spanish_baseline_beto_step9b.py --model_name_or_path dccuchile/bert-base-spanish-wwm-cased --output_dir .\Pasos\paso_9b_beto --step9_reference_dir .\Pasos\paso_9_es_l7b --max_train_samples 500 --max_eval_samples 200 --num_train_epochs 3 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --max_seq_length 128 --learning_rate 2e-5 --seed 13
```

This is the clean comparison baseline:

- same tasks,
- same training budget,
- same evaluation setup,
- same tokenizer interface per task,
- only the initialization changes.

## What Changed in This BETO Extension

Compared to the original ACL English setup, the current repository adds:

- support for **measured scanpaths**,
- BETO-compatible BERT-style training,
- a scanpath-level auxiliary MLM design over sequence length `S`,
- Spanish downstream transfer from the trained BETO checkpoint,
- a direct BETO baseline for fair comparison.

The most relevant current comparison is:

- **Paso 9**: downstream from the trained `l7b` BETO checkpoint
- **Paso 9b**: direct BETO baseline without that pretraining

## Original English Experiments

The English BERT/RoBERTa GLUE scripts are still available. Here they are as simple command references.

### BERT baseline

```bash
bash run_glue_LMbert_baseline_high_resource.sh
bash run_glue_LMbert_baseline_low_resource.sh
```

### Gaze-supervised BERT

```bash
bash run_glue_gazesup_bert_high_resource.sh
bash run_glue_gazesup_bert_low_resource.sh
```

### RoBERTa baseline

```bash
bash run_glue_LMroberta_baseline_high_resource.sh
bash run_glue_LMroberta_baseline_low_resource.sh
```

### Gaze-supervised RoBERTa

```bash
bash run_glue_gazesup_roberta_high_resource.sh
bash run_glue_gazesup_roberta_low_resource.sh
```

## Reproducibility Notes

For a fair Spanish comparison, keep the following fixed between `paso_9` and `paso_9b`:

- same tasks,
- same sample caps,
- same number of epochs,
- same batch sizes,
- same learning rate,
- same max sequence length,
- same seed.

Only the model initialization should differ.

## Citation

If you use this repository in academic work, please cite the ACL 2024 paper:

```bibtex
@inproceedings{deng-etal-2024-gazesuplm,
  title = {Fine-Tuning Pre-Trained Language Models with Gaze Supervision},
  author = {Deng, Shuwen and
            Prasse, Paul and
            Reich, David and
            Scheffer, Tobias and
            Jaeger, Lena},
  booktitle = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics},
  year = {2024}
}
```
