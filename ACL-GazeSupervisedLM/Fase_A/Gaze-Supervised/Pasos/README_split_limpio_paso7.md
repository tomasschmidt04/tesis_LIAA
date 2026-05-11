# Paso 7 - Split limpio para MLM intrinseco

## Por que se cambio

El split anterior cargaba `aligned_scanpaths.jsonl` como un unico split `train` y despues tomaba filas contiguas:

- primeras `max_train_samples` filas para train
- siguientes `max_eval_samples` filas para eval

Eso producia leakage porque una misma oracion puede aparecer muchas veces: distintos participantes leen el mismo cuento y generan distintas lecturas/scanpaths para el mismo texto. Entonces la misma `text` exacta podia quedar en train y eval.

## Modo recomendado actual: split precomputado por cuentos

Despues de limpiar por largo de input se creo:

- `reading-et/mlm_dataset_limpio_train_test/train.jsonl`
- `reading-et/mlm_dataset_limpio_train_test/test.jsonl`

Ese split es por cuento completo. Ningun cuento aparece en ambos splits, por lo que evita leakage entre lecturas/scanpaths del mismo cuento.

Para usarlo en Paso 7, ahora los scripts aceptan:

```text
--split_strategy precomputed_files
--train_file /home/tomi/tesis_LIAA/reading-et/mlm_dataset_limpio_train_test/train.jsonl
--eval_file /home/tomi/tesis_LIAA/reading-et/mlm_dataset_limpio_train_test/test.jsonl
```

En este modo el entrenamiento no vuelve a dividir el dataset: carga `train_file` como Train y `eval_file` como Eval/Test. El `split_report.json` valida que no haya cuentos compartidos entre Train y Eval.

Este es el modo recomendado para:

- BETO MLM baseline
- BETO + rama scanpath
- BETO + lambda adaptativo
- futuras variantes arquitectonicas

## Modo anterior: split por posicion de oracion

Con `--split_strategy sentence_position`, la decision ya no se toma por fila individual. Se infiere un campo de cuento (`story_id`) y un campo de posicion de oracion (`sentence_position`, `sentence_id`, `trial_position`, `period_block_start`, `segment_index` o `global_word_start`, en ese orden).

Una fila va a eval si:

```text
sentence_position % eval_sentence_position_mod == eval_sentence_position_remainder
```

Por ejemplo, con:

```text
--eval_sentence_position_mod 10
--eval_sentence_position_remainder 5
```

las posiciones `5, 15, 25, ...` van a eval para todos los cuentos y todas sus lecturas/scanpaths. Las demas posiciones van a train.

El script escribe `split_report.json` y falla con `ValueError` si:

- train o eval queda vacio
- hay textos exactos compartidos entre train y eval
- hay pares `(story_id, sentence_position)` compartidos entre train y eval
- no encuentra campos razonables para cuento o posicion

El reporte tambien documenta si la posicion parece `0-based`, `1-based` o si arranca en otro valor.

Nota: en el archivo alineado actual puede ocurrir que el mismo `text` exacto aparezca en posiciones distintas dentro del mismo cuento. En ese caso el dry-run guarda el reporte y falla antes de entrenar, porque la condicion estricta `exact_text_overlap_count == 0` no se cumple. Esa falla es intencional: obliga a corregir el campo de posicion elegido o a revisar la deduplicacion antes de usar la loss de eval como metrica limpia.

## Dry-run sin entrenar

Baseline BETO:

```bash
.venv/bin/python train_mlm_beto_baseline_step7_pretrain.py \
  --train_file /home/tomi/tesis_LIAA/reading-et/mlm_dataset_limpio_train_test/train.jsonl \
  --eval_file /home/tomi/tesis_LIAA/reading-et/mlm_dataset_limpio_train_test/test.jsonl \
  --model_name_or_path dccuchile/bert-base-spanish-wwm-cased \
  --measured_text_field text \
  --measured_word_id_field word_id \
  --output_dir Pasos/paso_7_clean_sentence_split_beto_baseline_full \
  --max_train_samples -1 \
  --max_eval_samples -1 \
  --split_strategy precomputed_files \
  --split_report_path Pasos/paso_7_clean_sentence_split_beto_baseline_full/split_report.json \
  --dry_run_split_only True
```

BETO + scanpath:

```bash
.venv/bin/python train_mlm_combined_step7.py \
  --train_file /home/tomi/tesis_LIAA/reading-et/mlm_dataset_limpio_train_test/train.jsonl \
  --eval_file /home/tomi/tesis_LIAA/reading-et/mlm_dataset_limpio_train_test/test.jsonl \
  --model_name_or_path dccuchile/bert-base-spanish-wwm-cased \
  --measured_text_field text \
  --measured_word_id_field word_id \
  --output_dir Pasos/paso_7_clean_sentence_split_full_scanpath \
  --max_train_samples -1 \
  --max_eval_samples -1 \
  --split_strategy precomputed_files \
  --split_report_path Pasos/paso_7_clean_sentence_split_full_scanpath/split_report.json \
  --dry_run_split_only True
```

Tambien se puede correr ambos dry-runs con:

```bash
DRY_RUN_SPLIT_ONLY=True ./run_paso7_clean_sentence_split_full.sh
```

## Corrida full con split limpio

Para correr ambos preentrenamientos full:

```bash
./run_paso7_clean_sentence_split_full.sh
```

Carpetas esperadas:

- `Pasos/paso_7_clean_sentence_split_beto_baseline_full/`
- `Pasos/paso_7_clean_sentence_split_full_scanpath/`

Cada carpeta guarda:

- `split_report.json`
- `comandos_y_funciones.txt`
- `run_args.json`
- `trainer_state.json`
- salida/resumen de entrenamiento
- `loss_curves.csv`
- checkpoints por epoch, `best_checkpoint/` y `checkpoint_final/` cuando no es dry-run
