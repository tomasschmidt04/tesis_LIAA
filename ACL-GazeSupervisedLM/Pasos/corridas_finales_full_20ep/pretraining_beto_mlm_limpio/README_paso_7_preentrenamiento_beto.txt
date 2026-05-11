PASO 7 BASELINE - README
===========================

Que se hizo
- Se creo un script baseline llamado train_mlm_beto_baseline_step7_pretrain.py.
- Este script toma el mismo dataset alineado usado por el pipeline measured.
- Usa solo el campo de texto para preentrenar BETO con MLM estandar.
- No usa la rama scanpath.
- No usa GRU.
- No usa loss auxiliar.
- Usa un split limpio por posicion de oracion cuando split_strategy=sentence_position.
- Si split_strategy=precomputed_files, usa directamente train_file/eval_file ya separados por cuentos.

Que se mantiene comparable con el paso 7 scanpath
- Mismo archivo de entrada: None
- train_file si aplica: /home/tomi/tesis_LIAA/reading-et/mlm_dataset_limpio_train_test/train.jsonl
- eval_file si aplica: /home/tomi/tesis_LIAA/reading-et/mlm_dataset_limpio_train_test/test.jsonl
- Mismo modelo base: dccuchile/bert-base-spanish-wwm-cased
- Misma longitud maxima: 128
- Mismo esquema de mascara estatica usado en estos scripts del repo.

Que produce
- split_report.json con validaciones del split limpio
- Checkpoints por epoca si save_every_epoch=True
- best_checkpoint/
- checkpoint_final/
- loss_curves.csv con medias train/eval por epoch

Parametros principales
- split_strategy = precomputed_files
- eval_sentence_position_mod = 10
- eval_sentence_position_remainder = 5
- split_report_path = Pasos/corridas_finales_full_20ep/pretraining_beto_mlm_limpio/split_report.json
- max_train_samples = 38518
- max_eval_samples = 9630
- num_train_epochs = 20
- per_device_train_batch_size = 4
- per_device_eval_batch_size = 4
- learning_rate = 5e-05
- max_seq_length = 128
- max_masked_positions = 3
- seed = 13
- device = cuda

Diferencia conceptual respecto del paso 7b
- Paso 7b: BETO + MLM principal + rama scanpath + GRU + loss auxiliar.
- Este baseline: BETO + MLM principal solamente.

Split limpio por posicion de oracion
- El split anterior por filas contiguas podia poner la misma oracion en train y eval porque cada texto aparece varias veces con distintas lecturas/scanpaths.
- Con split_strategy=sentence_position se infiere story_id y sentence_position, y eval recibe las posiciones donde sentence_position % eval_sentence_position_mod == eval_sentence_position_remainder.
- La misma posicion queda en eval para todos los cuentos y todas sus lecturas; el resto queda en train.
- El reporte valida que no haya texto exacto compartido ni pares (story_id, sentence_position) compartidos.
- Para auditar sin entrenar: agregar --dry_run_split_only True.

Split precomputado por cuentos
- Con split_strategy=precomputed_files no se vuelve a dividir el dataset.
- El script carga train_file y eval_file directamente.
- El reporte valida que ningun cuento aparezca en ambos splits.
- Este modo es el recomendado para usar reading-et/mlm_dataset_limpio_train_test/train.jsonl y test.jsonl.

Siguiente uso esperado
- Comparar downstream del backbone exportado desde este baseline contra el backbone exportado desde paso_7b.
