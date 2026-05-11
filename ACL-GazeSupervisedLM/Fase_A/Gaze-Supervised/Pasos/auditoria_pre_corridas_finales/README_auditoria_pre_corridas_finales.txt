AUDITORIA PRE-CORRIDAS FINALES
==============================

Fecha de auditoria: 2026-05-06T19:19:56
Repositorio: /home/tomi/tesis_LIAA/ACL-GazeSupervisedLM

Estado ejecutivo
----------------
- No se lanzo ningun entrenamiento ni ninguna corrida larga.
- Se hizo una auditoria estatica del pipeline, de los reportes de dataset y de los scripts principales.
- Los scripts Python principales compilan con `.venv/bin/python -m py_compile`.
- El pipeline esta razonablemente listo para: pretraining BETO solo, pretraining BETO + scanpath, XNLI desde checkpoints y XNLI desde BETO base.
- El pipeline NO esta completamente listo si la tarea pedida es estrictamente "Sentiment rioplatense": lo implementado hoy es `rioplatense_hate_binary` / hate speech rioplatense, no sentimiento rioplatense. Tambien existe `intertass2020`, que es sentimiento en espanol, pero no especificamente rioplatense.
- Los comandos quedan generados en `comandos_8_corridas.sh`, pero el script esta protegido: si se ejecuta sin `RUN_EXPERIMENTS=1`, no corre los experimentos.

1. Dataset MLM limpio
---------------------
Dataset limpio por largo de input:
- Carpeta: `/home/tomi/tesis_LIAA/reading-et/procesamiento_por_largo_input/data_filtrada`
- Regla aplicada previamente: conservar solo ejemplos con `n_words >= 4`.
- Conteo de palabras documentado: `str(text).strip().split()` contando tokens con al menos un caracter alfanumerico.
- Ejemplos originales antes de limpiar: 65.671
- Ejemplos conservados: 48.148
- Ejemplos eliminados: 17.523
- Porcentaje eliminado: 26,683%
- Los JSON originales no fueron modificados.

Archivos finales para MLM:
- Train: `/home/tomi/tesis_LIAA/reading-et/mlm_dataset_limpio_train_test/train.jsonl`
- Test/Eval: `/home/tomi/tesis_LIAA/reading-et/mlm_dataset_limpio_train_test/test.jsonl`

Split Train/Test para MLM:
- Unidad de split: cuento completo, no fila/trial individual.
- Cuentos Train: 15
- Cuentos Test: 5
- Ejemplos Train: 38.518
- Ejemplos Test: 9.630
- Porcentaje por ejemplos: Train 79,9992% / Test 20,0008%
- `story_overlap_count`: 0
- `exact_text_overlap_count`: 0
- `min_words` Train/Test: 4
- Filas con `word_id`: 38.518 Train / 9.630 Test
- Filas con `scanpath_text`: 38.518 Train / 9.630 Test

Cuentos Train:
- Cómo funciona caminar en la nieve
- Cómo funcionan los bolsillos
- Educar para escalar y bucear
- El almohadón de plumas
- El espejo
- Embarrar la magia
- La canción que cantábamos todos los días
- La de la Obsesión por la Patineta
- La gallina degollada
- La lluvia de fuego
- La noche de los feos
- Las fotografías
- Rubí y el lago danzante
- Una rosa para Emilia
- Wakefield

Cuentos Test:
- Ahora debería reírme, si no estuviera muerto
- Buenos Aires
- El golpe de gracia
- La máscara de la Muerte Roja
- La salud de los enfermos

Conclusion dataset:
- OK: el split MLM principal evita leakage por cuento.
- OK: no hay texto exacto compartido entre Train y Test.
- OK: todos los ejemplos finales tienen 4 o mas palabras reales.

2. Dataset limpio con fijaciones / results_all_alligned_2
---------------------------------------------------------
Carpetas detectadas:
- `/home/tomi/tesis_LIAA/reading-et/results_all_alligned_2_limpio`
- `/home/tomi/tesis_LIAA/reading-et/results_all_alligned_2_limpio_alineado`

Reporte revisado:
- `/home/tomi/tesis_LIAA/reading-et/procesamiento_por_largo_input_con_fijaciones/alineacion_dataset_limpio_stats.json`

Resumen de alineacion:
- `examples_processed`: 48.148
- `examples_with_feature_match`: 48.148
- `examples_without_feature_match`: 0
- `files_written`: 1.015
- `stories_written`: 20
- Campos copiados desde `results_all_alligned_2`: `feature_names`, `reading_features_by_word`, `reading_features_by_fixation`, `reading_features_mask_by_word`, `reading_features_mask_by_fixation`, `num_text_tokens`, `num_fixations`.

Conclusion fijaciones:
- OK: existe una version filtrada/alineada tipo `results_all_alligned_2` para el dataset limpio.
- WARNING: el script actual de pretraining combinado del Paso 7 usa principalmente `word_id` como scanpath medido. No consume explicitamente los campos numericos de fijaciones de `/home/tomi/tesis_LIAA/reading-et/results_all_alligned_2_limpio_alineado` en el comando planificado. Si la corrida B debe usar medidas continuas de fijacion, hace falta confirmar o adaptar el script correspondiente.

3. Pretraining BETO solo
------------------------
Script identificado:
- `train_mlm_beto_baseline_step7_pretrain.py`

Dataset usado por el comando recomendado:
- `--train_file /home/tomi/tesis_LIAA/reading-et/mlm_dataset_limpio_train_test/train.jsonl`
- `--eval_file /home/tomi/tesis_LIAA/reading-et/mlm_dataset_limpio_train_test/test.jsonl`
- `--split_strategy precomputed_files`

Modelo inicial:
- `dccuchile/bert-base-spanish-wwm-cased`

Comportamiento verificado:
- Carga `AutoModelForMaskedLM.from_pretrained(...)`.
- Tokeniza el campo `text`.
- Usa labels MLM generados por `build_static_masked_inputs_and_labels`.
- No pasa `LM_word_ids`, `measured_word_ids` ni `measured_sp_len` al modelo.
- No instancia `Gazesup_BERTForCombinedMaskedLM`.
- No usa GRU, rama scanpath ni loss auxiliar.

Checkpoints y metricas:
- Guarda `checkpoint_epoch_*` si `--save_every_epoch True`.
- Guarda `best_checkpoint` cuando mejora la loss de referencia.
- Si hay eval, el best checkpoint se elige por menor `eval mean_mlm_loss`.
- Guarda `checkpoint_final`, `loss_curves.csv`, `trainer_state.json` y salida debug del propio script.

Estado:
- OK para correr con el dataset limpio por cuentos.

4. Pretraining BETO + Scanpath
------------------------------
Script identificado:
- `train_mlm_combined_step7.py`

Modelo usado:
- `Gazesup_BERTForCombinedMaskedLM` en `Gazesup_bert_combined_mlm_model.py`.

Dataset usado por el comando recomendado:
- `--train_file /home/tomi/tesis_LIAA/reading-et/mlm_dataset_limpio_train_test/train.jsonl`
- `--eval_file /home/tomi/tesis_LIAA/reading-et/mlm_dataset_limpio_train_test/test.jsonl`
- `--split_strategy precomputed_files`

Scanpaths usados:
- Campo `word_id` del JSONL limpio.
- En `preprocess_examples`, ese campo se transforma en `measured_word_ids` y `measured_sp_len`.
- Tambien se conserva `LM_word_ids`, obtenido desde la tokenizacion, para mapear palabras a subtokens.

Conversion scanpath palabra -> token:
- `Gazesup_BERTForCombinedMaskedLM.forward` recibe `LM_word_ids`, `measured_word_ids` y `measured_sp_len`.
- `_compute_gaze_token_pos` usa `SP_Encoder._prepare_measured_word_scanpath` y `convert_word_pos_seq_to_token_pos_seq`.
- Refixaciones/regresiones permanecen como posiciones de palabra repetidas o hacia atras y luego se mapean a posiciones token/subtoken.

Labels auxiliares:
- La rama MLM principal calcula loss contra `labels` sobre la secuencia tokenizada normal.
- `_expand_labels_to_scanpath` toma esos mismos labels MLM y los reordena sobre la secuencia de posiciones scanpath.
- Pasos invalidos del scanpath se enmascaran con `-100`.

Loss total:
- Antes/fijo: `total_loss = main_mlm_loss + aux_weight * scanpath_mlm_loss`.
- Ahora: `total_loss = main_mlm_loss + effective_scanpath_weight * scanpath_mlm_loss`.
- Si `current_scanpath_weight is None`, `effective_scanpath_weight = aux_weight`.
- Si `current_scanpath_weight` tiene valor, se usa ese lambda adaptativo.

Lambda adaptativo:
- Implementado en `train_mlm_combined_step7.py` y usado por `Gazesup_bert_combined_mlm_model.py`.
- Argumentos CLI: `--adaptive_scanpath_weight`, `--scanpath_weight_min`, `--scanpath_weight_max`, `--scanpath_weight_warmup_epochs`, `--scanpath_weight_update_metric`, `--scanpath_weight_log_path`.
- Formula:
  `progress = (initial_scanpath_loss - current_scanpath_loss) / initial_scanpath_loss`
  `progress = clip(progress, 0.0, 1.0)`
  `lambda_scanpath_t = lambda_min + progress * (lambda_max - lambda_min)`
- La metrica de actualizacion es por epoch, no por batch: primero `eval_loss_scanpath`, si no existe `train_loss_scanpath_mean`.
- Durante warmup, lambda queda en `scanpath_weight_min`.
- Para volver al comportamiento anterior: omitir `--adaptive_scanpath_weight` o usar `--adaptive_scanpath_weight False`; el modelo vuelve a `aux_weight` fijo.

Checkpoints y metricas:
- Guarda `checkpoint_epoch_*`, `best_checkpoint`, `checkpoint_final`, `loss_curves.csv`.
- Si `adaptive_scanpath_weight=True`, guarda `lambda_scanpath_log.csv`, README/debug de lambda dentro del output dir.
- Best checkpoint: menor `eval mean_total_loss` cuando hay eval; si no hay eval, menor train total loss.

Estado:
- OK para correr como BETO + scanpath por `word_id` medido.
- WARNING: los launchers existentes no siempre activan lambda adaptativo por defecto. El comando generado para la corrida final B si lo activa explicitamente.

5. Downstream / Fine-tuning
---------------------------
Scripts identificados:
- XNLI espanol: `train_spanish_downstream_baseline.py`
- Rioplatense disponible hoy: `train_glue_LM_baseline.py --task_name rioplatense_hate_binary`

Carga de checkpoints:
- Downstream carga con `AutoModelForSequenceClassification.from_pretrained(..., ignore_mismatched_sizes=True)`.
- Esto crea una cabeza de clasificacion nueva segun `num_labels` de cada tarea.
- Para checkpoints MLM con scanpath, las claves extra de la rama scanpath no son parte de `AutoModelForSequenceClassification`; se espera que se ignoren, mientras se cargan los pesos compatibles del backbone `bert.*`.
- Downstream no recibe `word_id`, no recibe scanpaths, no corre GRU y no calcula loss auxiliar.

Resultados downstream:
- `train_spanish_downstream_baseline.py` guarda `train_results.txt`, `trainer_state.json`, `eval_results.json`, `test_results.json` si aplica y `task_metadata.json`.
- `train_glue_LM_baseline.py` guarda `train_results.txt`, `trainer_state.json`, `eval_results.json`, `test_results.json`, `test_results_*.txt`, `class_distribution.json` y `model_init_check.json` para rioplatense.

6. Tareas downstream
--------------------
XNLIes:
- Fuente: `load_dataset("xnli", "es")`.
- Train: split oficial `train`, luego cap/shuffle por seed si `--max_train_samples` esta seteado.
- Eval: split oficial `validation`, luego cap/shuffle por seed si `--max_eval_samples` esta seteado.
- Columnas textuales: `premise`, `hypothesis`.
- `num_labels`: 3.
- Label mapping: se toma dinamicamente de `raw["train"].features["label"].names`; normalmente `0=entailment`, `1=neutral`, `2=contradiction`.
- Metrica principal: `accuracy`.

Sentiment rioplatense:
- PENDIENTE si se interpreta literalmente como analisis de sentimiento rioplatense.
- El repo tiene `intertass2020`, que es sentimiento en espanol, pero no rioplatense.
- El repo tiene `rioplatense_hate_binary`, que es clasificacion binaria de hate speech rioplatense, no sentimiento.
- Los comandos generados usan `rioplatense_hate_binary` porque es la tarea rioplatense disponible y ya integrada en los launchers.

Rioplatense hate binary actual:
- Fuente por defecto: CSV raw de `finiteautomata/rioplatense_hate_speech` desde GitHub, o `--dataset_path` local.
- Train: `test_01.csv`, `test_02.csv`, `test_03.csv` concatenados.
- Validation oficial: `test_04.csv`; pero con `--train_as_val True`, la eval usada es un holdout estratificado del train pool sin overlap con el subconjunto de entrenamiento.
- Test: `test_05.csv`, evaluado con `--do_predict`.
- Columna textual: `text`.
- `num_labels`: 2.
- Label mapping: `0=no_hate`, `1=hate`.
- Metrica principal para best model: `macro_f1`.
- Metricas adicionales: `accuracy`, `precision`, `recall`, `f1`, `f1_hate`, `f1_no_hate`.

7. BETO base directo
--------------------
Modelo inicial:
- `dccuchile/bert-base-spanish-wwm-cased`

Scripts:
- XNLI: `train_spanish_downstream_baseline.py`
- Rioplatense disponible: `train_glue_LM_baseline.py`

Comparabilidad:
- Los comandos base directo usan los mismos hiperparametros downstream que las corridas desde checkpoints preentrenados.
- La unica diferencia buscada es `--model_name_or_path`: checkpoint preentrenado versus BETO base.

8. Comparabilidad entre corridas
--------------------------------
Aspectos OK:
- Mismo tokenizer/modelo base compatible con BETO.
- Mismos splits downstream por script y seed.
- Mismos `max_train_samples` y `max_eval_samples` en cada tarea.
- Mismas epochs downstream: 20.
- Mismo learning rate downstream: `2e-5`.
- Mismo batch size downstream: 32 train / 32 eval.
- Misma seed downstream: 111.
- Misma metrica por tarea.
- Para MLM, ambos pretrainings usan el mismo Train/Test limpio por cuentos.

Aspectos a confirmar antes de corridas grandes:
- Si la tarea final debe ser sentiment rioplatense real, falta dataset/script especifico.
- Si BETO + Scanpath debe usar tambien medidas continuas de fijacion, el script de Paso 7 actual no consume los campos de `/home/tomi/tesis_LIAA/reading-et/results_all_alligned_2_limpio_alineado` en el comando generado.
- Los datasets downstream se descargan de Hugging Face/GitHub si no estan cacheados; para reproducibilidad fuerte conviene cachear o fijar paths locales.

9. Comandos
-----------
Los comandos exactos estan en:
- `Pasos/auditoria_pre_corridas_finales/comandos_8_corridas.sh`
- `Pasos/auditoria_pre_corridas_finales/tabla_experimentos_planificados.csv`

El archivo `.sh` no ejecuta nada salvo que se lo llame asi:
`RUN_EXPERIMENTS=1 bash Pasos/auditoria_pre_corridas_finales/comandos_8_corridas.sh`

10. Riesgos / pendientes
------------------------
Implementado y OK:
- Limpieza por largo de input: solo `n_words >= 4`.
- Split MLM por cuentos completos.
- Train/Test MLM sin leakage por cuento ni texto exacto.
- JSONL final con `text`, `word_id`, `scanpath_text`, `cuento`, `source_file`, `split`, `n_words`.
- Version alineada tipo `results_all_alligned_2` para dataset limpio.
- Pretraining BETO solo con MLM estandar.
- Pretraining BETO + scanpath por `word_id` medido.
- Lambda adaptativo simple implementado y con fallback a `aux_weight` fijo.
- Downstream XNLI listo.
- BETO base directo listo.

Pendiente / warning:
- `Sentiment rioplatense` no esta implementado como tal. Hay que decidir si se acepta `rioplatense_hate_binary` o si se agrega un dataset real de sentimiento rioplatense.
- Los launchers viejos tienen nombres como `sentence_split` o `half_data`; para la corrida final conviene usar los output dirs nuevos propuestos en `Pasos/corridas_finales/`.
- El uso de `results_all_alligned_2_limpio_alineado` no esta conectado al pretraining combinado planificado, salvo que otra variante lo consuma.
- Los checkpoints de `Pasos/corridas_finales/pretraining_*` todavia no existen porque no se corrieron las corridas grandes. Los downstream desde esos checkpoints dependen de correr primero los experimentos 1 y 2.
- El repositorio tiene cambios sin commitear y archivos nuevos; no se revirtio nada.
