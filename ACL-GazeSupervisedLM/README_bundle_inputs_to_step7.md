# Bundle: inputs crudos -> train_mlm_combined_step7.py

Este zip contiene los archivos de codigo y reportes utiles para revisar el flujo que lleva los datos alineados/limpios hasta el modelo de preentrenamiento `train_mlm_combined_step7.py`.

## Flujo resumido

1. `../reading-et/results_all_alligned_2/`
   - Carpeta de entrada con lecturas/scanpaths alineados por cuento y sujeto.
   - No se incluye completa en el zip porque pesa cientos de MB.

2. Limpieza por largo/input
   - `procesar_por_largo_input.py`
   - Produce versiones limpias/filtradas de los JSON alineados.
   - Reportes incluidos:
     - `reading-et/procesamiento_por_largo_input/informe_procesamiento_por_largo_input.txt`
     - `reading-et/procesamiento_por_largo_input_con_fijaciones/informe_procesamiento_por_largo_input.txt`

3. Armado train/test MLM limpio por cuentos
   - `create_mlm_train_test_from_clean_dataset.py`
   - Produce:
     - `../reading-et/mlm_dataset_limpio_train_test/train.jsonl`
     - `../reading-et/mlm_dataset_limpio_train_test/test.jsonl`
   - Reporte incluido:
     - `reading-et/mlm_dataset_limpio_train_test/informe_mlm_dataset_limpio_train_test.txt`

4. Carga y feature building dentro del repo ACL
   - `measured_scanpath_utils.py`
   - Funciones clave:
     - `load_measured_scanpath_train_eval_dataset`
     - `build_precomputed_file_train_eval_datasets`
     - `build_measured_single_sentence_features`
     - `build_measured_scanpath`

5. Masking/collate MLM
   - `train_mlm_scanpath_step5.py`
   - Funciones clave:
     - `_candidate_mask_positions`
     - `apply_dynamic_mlm_to_batch`
     - `collate_measured_mlm_batch`

6. Entrenamiento Paso 7
   - `train_mlm_combined_step7.py`
   - Funciones clave:
     - `preprocess_examples`
     - `train_step7`
     - `evaluate_model`

7. Modelo BETO + scanpath
   - `Gazesup_bert_combined_mlm_model.py`
   - `Gazesup_bert_model.py`
   - Funciones/clases clave:
     - `Gazesup_BERTForCombinedMaskedLM.forward`
     - `_expand_labels_to_scanpath`
     - `SP_Encoder`
     - `_prepare_measured_word_scanpath`
     - `convert_word_pos_seq_to_token_pos_seq`

## Nota

El zip no incluye checkpoints ni datasets completos. Incluye codigo, scripts de ejecucion y reportes/comandos para que otro modelo pueda seguir el flujo sin recibir cientos de MB de datos.
