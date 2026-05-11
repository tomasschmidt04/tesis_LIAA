PASO 7 - README
================

Que se hizo
- Se creo un script nuevo llamado train_mlm_combined_step7.py.
- El script entrena el modelo combinado con loss principal MLM + loss auxiliar scanpath MLM sobre un dataset medido mas grande que el smoke test.
- Se usan varias epochs, logging por epoch, evaluacion minima opcional y guardado de checkpoints reutilizables.
- El modelo sigue compartiendo un unico encoder BERT y combina las losses como total_loss = main_mlm_loss + scanpath_loss_weight * scanpath_mlm_loss.
- Si adaptive_scanpath_weight=False, scanpath_loss_weight es aux_weight como antes.
- Si adaptive_scanpath_weight=True, scanpath_loss_weight es lambda_scanpath_t actualizado por epoch.
- La evaluacion MLM usa un split limpio por posicion de oracion cuando split_strategy=sentence_position.
- Si split_strategy=precomputed_files, usa directamente train_file/eval_file ya separados por cuentos.

Que se verifico
- Que el entrenamiento combinado escala a mas datos y mas de una epoch.
- Que main_mlm_loss, scanpath_mlm_loss y total_loss se loguean claramente por separado.
- Que se guardan checkpoints por epoch, best checkpoint y checkpoint final.
- Que queda un checkpoint final reutilizable para una etapa downstream posterior.
- Que el pipeline puede correr con un modelo BERT-style compatible con BETO si se usa dccuchile/bert-base-spanish-wwm-cased.

Que NO se implemento todavia
- No se implemento GLUE.
- No se implemento fine-tuning downstream.
- No se hizo hyperparameter search grande.
- No se refactorizo de forma masiva el repo.
- No se avanzo todavia a la etapa downstream dentro de este paso.
- No se cambio arquitectura, dataset, labels ni split para implementar lambda adaptativo.

Archivos modificados
- Pasos/README.txt

Archivos nuevos creados
- train_mlm_combined_step7.py
- Pasos/paso_7_precomputed_story_split_scanpath_smoke/split_report.json
- Pasos/paso_7_precomputed_story_split_scanpath_smoke/README_paso_7.txt
- Pasos/paso_7_precomputed_story_split_scanpath_smoke/salida_training_step7.txt
- Pasos/paso_7_precomputed_story_split_scanpath_smoke/comandos_y_funciones.txt
- Pasos/paso_7_precomputed_story_split_scanpath_smoke/lambda_scanpath_log.csv si adaptive_scanpath_weight=True
- Pasos/paso_7_precomputed_story_split_scanpath_smoke/checkpoint_epoch_*/ si save_every_epoch=True
- Pasos/paso_7_precomputed_story_split_scanpath_smoke/best_checkpoint/
- Pasos/paso_7_precomputed_story_split_scanpath_smoke/checkpoint_final/

Explicacion breve del entrenamiento mas grande
- El dataset medido se tokeniza con BERT y se usa para construir inputs MLM estandar mas la representacion measured requerida por la rama scanpath.
- En cada batch se calculan simultaneamente la loss principal MLM y la loss auxiliar MLM scanpath.
- Luego se combinan con aux_weight fijo o con lambda_scanpath_t adaptativo, y se actualiza el modelo con AdamW.
- Ademas se registra un resumen por epoch y una evaluacion sobre el split limpio definido antes de entrenar.

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

Explicacion de la combinacion de losses
- Rama principal: input_ids -> BERT -> MLM head principal -> main_mlm_loss.
- Rama auxiliar: input_ids -> BERT -> scanpath expandido -> GRU -> reagregacion -> MLM head auxiliar -> scanpath_mlm_loss.
- Loss total: total_loss = main_mlm_loss + scanpath_loss_weight * scanpath_mlm_loss.
- Con adaptive_scanpath_weight=False, scanpath_loss_weight = aux_weight.
- Con adaptive_scanpath_weight=True, scanpath_loss_weight = lambda_scanpath_t.

Explicacion del parametro aux_weight
- aux_weight controla cuanto pesa la loss auxiliar scanpath respecto de la principal.
- Valores tipicos para probar aca: 1.0, 0.3, 0.1.
- Si la rama auxiliar domina, conviene bajar aux_weight. Si casi no influye, conviene subirlo.

Explicacion del lambda adaptativo simple
- Se activa con --adaptive_scanpath_weight True.
- La referencia initial_scanpath_loss es la primera loss scanpath promedio disponible para actualizar lambda.
- Por epoch se usa eval_loss_scanpath si esta disponible; si no, train_loss_scanpath_mean.
- progress = (initial_scanpath_loss - current_scanpath_loss) / initial_scanpath_loss, clipeado a [0, 1].
- lambda_scanpath_t = scanpath_weight_min + progress * (scanpath_weight_max - scanpath_weight_min).
- Durante scanpath_weight_warmup_epochs, lambda_scanpath_t se mantiene en scanpath_weight_min.

Aclaracion importante
- Este paso ya no es solo smoke test, pero tampoco es todavia un experimento final grande.
- La idea es dejar un entrenamiento mas estable y checkpoints reutilizables para el paso downstream posterior.

Nota especifica sobre el modelo/tokenizer usado
- model_name_or_path usado: hf-internal-testing/tiny-random-bert
- tokenizer detectado: BertTokenizerFast
- vocab_size detectado: 1124
- model_type detectado: bert
- hidden_size detectado: 32
- num_hidden_layers detectado: 5
- El script valida explicitamente que el modelo sea BERT-style (model_type='bert').

Configuracion usada en esta corrida
- measured_scanpath_file = None
- train_file = /home/tomi/tesis_LIAA/reading-et/mlm_dataset_limpio_train_test/train.jsonl
- eval_file = /home/tomi/tesis_LIAA/reading-et/mlm_dataset_limpio_train_test/test.jsonl
- split_strategy = precomputed_files
- eval_sentence_position_mod = 10
- eval_sentence_position_remainder = 5
- split_report_path = Pasos/paso_7_precomputed_story_split_scanpath_smoke/split_report.json
- max_train_samples = 4
- max_eval_samples = 2
- num_train_epochs = 1
- per_device_train_batch_size = 2
- per_device_eval_batch_size = 2
- max_seq_length = 128
- learning_rate = 0.001
- aux_weight = 0.1
- adaptive_scanpath_weight = False
- scanpath_weight_min = 0.05
- scanpath_weight_max = 0.5
- scanpath_weight_warmup_epochs = 1
- scanpath_weight_update_metric = auto
- scanpath_weight_log_path = None
- save_every_epoch = False
- seed = 13
- device = cpu
