PASO 5 - README
================

Que se hizo
- Se creo un script nuevo de entrenamiento corto llamado train_mlm_scanpath_step5.py.
- El script carga un dataset medido local con campos text y word_id.
- Reutiliza el preprocesamiento medido ya validado para construir input_ids, attention_mask, LM_word_ids, measured_word_ids y measured_sp_len.
- Construye labels MLM simples para smoke test usando masking estatico con -100 en posiciones ignoradas.
- Instancia la clase MLM-compatible del paso 4 y ejecuta un loop corto con forward, loss, backward y optimizer.step.
- Guarda un checkpoint local reutilizable dentro de Pasos/paso_5/checkpoint_smoke.

Que se verifico
- Que el forward del modelo MLM auxiliar corre en un entrenamiento real, no solo en inference/debug.
- Que scanpath_mlm_loss se calcula correctamente durante varias iteraciones.
- Que backward corre con la loss auxiliar definida sobre la secuencia scanpath-level.
- Que optimizer.step corre y actualiza el modelo.
- Que se puede guardar un checkpoint local al final de la corrida.

Que NO se implemento todavia
- No se implemento GLUE ni fine-tuning downstream.
- No se implemento combinacion con una rama MLM estandar adicional.
- No se implemento scheduler sofisticado ni hyperparameter search.
- No se implemento export especial para GLUE.
- No se avanzo al paso 6.
- No se convirtio esto en un experimento final serio.

Archivos modificados
- train_mlm_scanpath_step5.py

Archivos nuevos creados
- train_mlm_scanpath_step5.py
- Pasos/paso_5/README_paso_5.txt
- Pasos/paso_5/salida_training_debug.txt
- Pasos/paso_5/comandos_y_funciones.txt
- Pasos/paso_5/checkpoint_smoke/

Explicacion breve del entrenamiento corto realizado
- Cada ejemplo medido se tokeniza con BERT y se convierte a la representacion interna measured del repo.
- measured_word_ids se expande internamente a nivel token/subtoken dentro del modelo.
- La secuencia scanpath-level pasa por la GRU y esa misma salida se usa como representacion final de la rama auxiliar.
- Sobre gru_output se aplica una cabeza MLM lineal hidden_size -> vocab_size.
- Los labels originales de shape (B, T) se expanden a scanpath_labels_expanded con shape (B, S) usando gaze_token_pos.
- La loss es CrossEntropyLoss(ignore_index=-100) aplicada sobre la secuencia scanpath-level expandida.
- Si un token masked aparece multiples veces en el scanpath, entonces contribuye multiples veces a la loss auxiliar.

Configuracion usada
- measured_scanpath_file = ../reading-et/results_all_alligned/Ahora debería reírme, si no estuviera muerto/sub-001.json
- model_name_or_path = bert-base-cased
- split = train
- max_train_samples = 8
- num_train_epochs = 1
- per_device_train_batch_size = 2
- learning_rate = 5e-05
- max_seq_length = 128
- max_masked_positions = 3
- seed = 13
- device = cuda

Aclaracion importante
- Este paso es un smoke test funcional y documentado.
- Sirve para validar que el pipeline measured + scanpath + GRU + cabeza MLM scanpath-level corre de punta a punta.
- No debe interpretarse como un experimento final ni como una configuracion optimizada.
