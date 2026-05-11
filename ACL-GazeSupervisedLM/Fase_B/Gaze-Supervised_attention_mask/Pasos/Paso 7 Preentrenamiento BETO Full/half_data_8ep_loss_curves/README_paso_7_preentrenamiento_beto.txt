PASO 7 BASELINE - README
===========================

Que se hizo
- Se creo un script baseline llamado train_mlm_beto_baseline_step7_pretrain.py.
- Este script toma el mismo dataset alineado usado por el pipeline measured.
- Usa solo el campo de texto para preentrenar BETO con MLM estandar.
- No usa la rama scanpath.
- No usa GRU.
- No usa loss auxiliar.

Que se mantiene comparable con el paso 7 scanpath
- Mismo archivo de entrada: /home/tomi/tesis_LIAA/reading-et/aligned_output/aligned_scanpaths.jsonl
- Mismo modelo base: dccuchile/bert-base-spanish-wwm-cased
- Misma longitud maxima: 128
- Mismo esquema de mascara estatica usado en estos scripts del repo.

Que produce
- Checkpoints por epoca si save_every_epoch=True
- best_checkpoint/
- checkpoint_final/
- loss_curves.csv con medias train/eval por epoch

Parametros principales
- max_train_samples = 31500
- max_eval_samples = 1000
- num_train_epochs = 8
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

Siguiente uso esperado
- Comparar downstream del backbone exportado desde este baseline contra el backbone exportado desde paso_7b.
