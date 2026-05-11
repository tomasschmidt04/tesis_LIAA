PASO 6 - README
================

Que se hizo
- Se creo un script nuevo de entrenamiento corto llamado train_mlm_combined_step6.py.
- El script carga un dataset medido con campos text y word_id.
- Reutiliza el preprocesamiento medido ya validado para construir el input MLM y los tensores necesarios para la rama scanpath.
- Instancia un modelo combinado que comparte BERT y produce simultaneamente:
  * main_mlm_logits y main_mlm_loss
  * scanpath_mlm_logits y scanpath_mlm_loss
  * total_loss = main_mlm_loss + aux_weight * scanpath_mlm_loss
- Ejecuta un smoke training corto con forward, backward, optimizer.step y guardado de checkpoint.

Que se verifico
- Que la loss principal MLM se calcula correctamente.
- Que la loss auxiliar scanpath MLM se calcula correctamente.
- Que ambas losses pueden coexistir y combinarse sin romper el entrenamiento.
- Que total_loss se mantiene finita durante la corrida.
- Que backward y optimizer.step funcionan con la loss combinada.
- Que se puede guardar un checkpoint local al final.

Que NO se implemento todavia
- No se implemento GLUE ni fine-tuning downstream.
- No se hizo hyperparameter search serio.
- No se implemento export final para GLUE.
- No se avanzo al paso 7.
- No se convirtio esto en un experimento final grande.

Archivos modificados
- Pasos/README.txt
- Gazesup_bert_combined_mlm_model.py

Archivos nuevos creados
- Gazesup_bert_combined_mlm_model.py
- train_mlm_combined_step6.py
- Pasos/paso_6/README_paso_6.txt
- Pasos/paso_6/salida_training_combined_debug.txt
- Pasos/paso_6/comandos_y_funciones.txt
- Pasos/paso_6/checkpoint_smoke/

Explicacion breve de la combinacion de losses
- La rama principal usa el camino directo del Transformer y una cabeza MLM estandar sobre los hidden states originales de BERT.
- La rama auxiliar usa los mismos hidden states originales de BERT, pero los reordena segun el scanpath medido expandido, los pasa por GRU y aplica la cabeza MLM directamente sobre la secuencia scanpath-level.
- Para la loss auxiliar, los labels originales de shape (B, T) se expanden a scanpath_labels_expanded con shape (B, S) usando gaze_token_pos.
- Ambas ramas producen logits sobre el vocabulario y ambas losses usan CrossEntropyLoss(ignore_index=-100).
- La combinacion se hace como total_loss = main_mlm_loss + aux_weight * scanpath_mlm_loss.

Explicacion de la rama principal
- input_ids masked -> BERT -> hidden states originales -> cabeza MLM estandar -> main_mlm_logits -> main_mlm_loss.

Explicacion de la rama auxiliar
- input_ids masked -> BERT -> hidden states originales -> scanpath expandido a token/subtoken -> GRU -> cabeza MLM auxiliar sobre S -> scanpath_mlm_logits -> scanpath_mlm_loss.
- La loss auxiliar se calcula contra scanpath_labels_expanded, de modo que repeticiones del mismo token en el scanpath generan multiples contribuciones supervisionadas.

Explicacion de lambda / aux_weight
- aux_weight es el lambda que controla cuanto pesa la loss auxiliar en la loss total.
- Valores simples para probar en este paso: 1.0, 0.3 y 0.1.
- Si la loss auxiliar domina demasiado, conviene bajar aux_weight. Si casi no influye, conviene subirlo.

Aclaracion importante
- Este paso sigue siendo un smoke test funcional y documentado.
- Sirve para validar la convivencia de ambas losses y el entrenamiento corto conjunto.
- No debe interpretarse como un experimento final ni como una configuracion optimizada.

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
- aux_weight = 0.3
- seed = 13
- device = cuda
