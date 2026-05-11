# Organizacion realizada

Fecha/hora: 2026-05-07 23:13:23 -0300

## Objetivo

Reorganizar el proyecto dentro de `ACL-GazeSupervisedLM` en:

- `Fase_A/Gaze-Supervised/`: copia del estado actual del codigo.
- `Fase_B/Gaze-Supervised_attention_mask/`: copia inicial para la Variante B con eye-tracking attention mask.

No se implementaron cambios de arquitectura masking.

## Comandos usados

```bash
rm -rf /home/tomi/tesis_LIAA/Fase_1 /home/tomi/tesis_LIAA/Fase2
mkdir -p Fase_A/Gaze-Supervised Fase_B/Gaze-Supervised_attention_mask \
  Fase_B/Pasos/paso_1_entender_paper \
  Fase_B/Pasos/paso_2_revisar_codigo_paper \
  Fase_B/Pasos/paso_3_prototipo_beto_attention \
  Fase_B/Pasos/paso_4_integracion_pipeline_actual

rsync -a ./ Fase_A/Gaze-Supervised/ \
  --exclude='Fase_A/' \
  --exclude='Fase_B/' \
  --exclude='.git/' \
  --exclude='.venv/' \
  --exclude='__pycache__/' \
  --exclude='.ipynb_checkpoints/' \
  --exclude='result/' \
  --exclude='results/' \
  --exclude='wandb/' \
  --exclude='checkpoints/' \
  --exclude='checkpoint*/' \
  --exclude='best_checkpoint/' \
  --exclude='cache/'

rsync -a ./ Fase_B/Gaze-Supervised_attention_mask/ \
  --exclude='Fase_A/' \
  --exclude='Fase_B/' \
  --exclude='.git/' \
  --exclude='.venv/' \
  --exclude='__pycache__/' \
  --exclude='.ipynb_checkpoints/' \
  --exclude='result/' \
  --exclude='results/' \
  --exclude='wandb/' \
  --exclude='checkpoints/' \
  --exclude='checkpoint*/' \
  --exclude='best_checkpoint/' \
  --exclude='cache/'
```

## Carpetas creadas

- `Fase_A/`
- `Fase_A/Gaze-Supervised/`
- `Fase_B/`
- `Fase_B/Gaze-Supervised_attention_mask/`
- `Fase_B/Pasos/paso_1_entender_paper/`
- `Fase_B/Pasos/paso_2_revisar_codigo_paper/`
- `Fase_B/Pasos/paso_3_prototipo_beto_attention/`
- `Fase_B/Pasos/paso_4_integracion_pipeline_actual/`

## Carpetas copiadas

- `./` hacia `Fase_A/Gaze-Supervised/`
- `./` hacia `Fase_B/Gaze-Supervised_attention_mask/`

## Exclusiones

Se excluyeron carpetas pesadas o regenerables:

- `.git/`
- `.venv/`
- `__pycache__/`
- `.ipynb_checkpoints/`
- `result/`
- `results/`
- `wandb/`
- `checkpoints/`
- `checkpoint*/`
- `best_checkpoint/`
- `cache/`
- `Fase_A/`
- `Fase_B/`

## Notas

No existia una carpeta llamada exactamente `Gaze-Supervised` en la raiz actual. La fuente usada fue el contenido actual de `ACL-GazeSupervisedLM`, y las copias se crearon con los nombres pedidos dentro de las fases.

Antes de rehacer la organizacion dentro del repo actual, se eliminaron las copias parciales creadas por error en `/home/tomi/tesis_LIAA/Fase_1` y `/home/tomi/tesis_LIAA/Fase2`.

## Tamanos

Resultado de `du -sh Fase_A Fase_B`:

```text
852M	Fase_A
852M	Fase_B
```

## Estado git final

Resultado de `git status --short`:

```text
 M Gazesup_bert_combined_mlm_model.py
 M Gazesup_bert_model.py
 M measured_scanpath_utils.py
 M requirements.txt
 M train_glue_LM_baseline.py
 M train_glue_gazesup_bert_low_resource.py
 M train_mlm_combined_step6.py
 M train_mlm_combined_step7.py
 M train_spanish_downstream_baseline.py
 M trainers.py
?? DATASETS_DOWNSTREAM.md
?? Fase_A/
?? Fase_B/
?? Variante_A_metricas_rama_auxiliar/
?? analysis/
?? comandos
?? create_mlm_train_test_from_clean_dataset.py
?? logs/
?? loss_curve_utils.py
?? procesar_por_largo_input.py
?? requirements_20260507_pre_cache_cleanup_backup.txt
?? results/
?? run_all_half_pretraining_and_rioplatense.sh
?? run_all_paso7_paso9_xnli_rioplatense.sh
?? run_beto_pretraining_8ep_half_loss_curves.sh
?? run_paso7_clean_sentence_split_full.sh
?? run_spanish_downstream_serio_step9.py
?? run_step9_rioplatense_hate_binary_low_resource.sh
?? scripts/
?? summarize_low_resource_results.py
?? train_mlm_beto_baseline_step7_pretrain.py
?? ../reading-et/aligned_output_synthetic_backup_20260423/
?? ../reading-et/mlm_dataset_limpio_train_test/
?? ../reading-et/procesamiento_por_largo_input/
?? ../reading-et/procesamiento_por_largo_input_con_fijaciones/
?? ../reading-et/results_all_alligned_2_limpio/
?? ../reading-et/results_all_alligned_2_limpio_alineado/
?? ../reading-et/results_all_alligned_synthetic_backup_20260423/
?? ../reading-et/scripts/data_processing/add_all_reading_measures_to_aligned.py
```
