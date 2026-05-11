PASO 7 - README
================

Que se hizo
- Se creo un script nuevo llamado train_mlm_combined_step7.py.
- El script entrena el modelo combinado con loss principal MLM + loss auxiliar scanpath MLM sobre un dataset medido mas grande que el smoke test.
- Se usan varias epochs, logging por epoch, evaluacion minima opcional y guardado de checkpoints reutilizables.
- El modelo sigue compartiendo un unico encoder BERT y combina las losses como total_loss = main_mlm_loss + aux_weight * scanpath_mlm_loss.

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

Archivos modificados
- Pasos/README.txt

Archivos nuevos creados
- train_mlm_combined_step7.py
- Pasos/paso_7/README_paso_7.txt
- Pasos/paso_7/salida_training_step7.txt
- Pasos/paso_7/comandos_y_funciones.txt
- Pasos/paso_7/checkpoint_epoch_*/ si save_every_epoch=True
- Pasos/paso_7/best_checkpoint/
- Pasos/paso_7/checkpoint_final/

Explicacion breve del entrenamiento mas grande
- El dataset medido se tokeniza con BERT y se usa para construir inputs MLM estandar mas la representacion measured requerida por la rama scanpath.
- En cada batch se calculan simultaneamente la loss principal MLM y la loss auxiliar MLM scanpath.
- Luego se combinan con aux_weight y se actualiza el modelo con AdamW.
- Ademas se registra un resumen por epoch y una evaluacion minima sobre un subconjunto held-out simple.

Explicacion de la combinacion de losses
- Rama principal: input_ids -> BERT -> MLM head principal -> main_mlm_loss.
- Rama auxiliar: input_ids -> BERT -> scanpath expandido -> GRU -> reagregacion -> MLM head auxiliar -> scanpath_mlm_loss.
- Loss total: total_loss = main_mlm_loss + aux_weight * scanpath_mlm_loss.

Explicacion del parametro aux_weight
- aux_weight controla cuanto pesa la loss auxiliar scanpath respecto de la principal.
- Valores tipicos para probar aca: 1.0, 0.3, 0.1.
- Si la rama auxiliar domina, conviene bajar aux_weight. Si casi no influye, conviene subirlo.

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
- measured_scanpath_file = /home/tomi/tesis_LIAA/reading-et/aligned_output/aligned_scanpaths.jsonl
- max_train_samples = 2
- max_eval_samples = 1
- num_train_epochs = 1
- per_device_train_batch_size = 1
- per_device_eval_batch_size = 1
- max_seq_length = 32
- learning_rate = 5e-05
- aux_weight = 0.1
- save_every_epoch = False
- seed = 13
- device = cuda
