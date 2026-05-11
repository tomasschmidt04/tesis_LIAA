PASO 9 SERIO - README
=====================

Objetivo
- Ejecutar fine-tuning downstream/extrinseco en espanol a partir del modelo entrenado en el pipeline principal BETO + rama auxiliar scanpath.
- En downstream se usa solo el encoder principal cargado desde el checkpoint seleccionado.
- No se usa scanpath, no se usa GRU y no se usa loss auxiliar durante downstream.
- Cada tarea crea una cabeza de clasificacion nueva mediante AutoModelForSequenceClassification.

Checkpoint seleccionado
- main_pipeline_dir: /home/tomi/tesis_LIAA/ACL-GazeSupervisedLM/Pasos/Paso 7 Preentrenamiento BETO Full
- checkpoint seleccionado: /home/tomi/tesis_LIAA/ACL-GazeSupervisedLM/Pasos/Paso 7 Preentrenamiento BETO Full/best_checkpoint
- criterio de seleccion: best_checkpoint_dir
- motivo: Existe el directorio best_checkpoint dentro del entrenamiento principal full 8ep.

Criterio de seleccion implementado
1. Usar best_checkpoint si existe.
2. Usar best checkpoint registrado en logs o trainer_state si existe.
3. Usar checkpoint con mejor metrica registrada si no hay best directo.
4. Usar ultimo checkpoint disponible si no hay best.
5. Usar el directorio raiz como fallback.

Tareas usadas
- XNLIes
  tipo: sentence pair classification
  columnas usadas: ['premise', 'hypothesis']
  num_labels: 3
  metrica principal: accuracy

- InterTass2020
  tipo: single sentence classification
  columnas usadas: ['text']
  num_labels: 3
  metrica principal: macro-F1

Configuracion
- num_train_epochs = 10
- max_train_samples = 1000
- max_eval_samples = 1000
- per_device_train_batch_size = 8
- per_device_eval_batch_size = 8
- max_seq_length = 128
- learning_rate = 2e-05
- seed = 13
- evaluation_strategy = epoch
- save_strategy = epoch
- load_best_model_at_end = True

Que se reutiliza del paso 9 anterior
- Se reutiliza el esquema de wrapper que ejecuta una tarea downstream por vez.
- Se reutiliza train_spanish_downstream_baseline.py para cargar datasets, tokenizar, entrenar y guardar metricas.
- Se reutiliza la logica de resumen por tarea con shapes de ejemplo, metricas y directorios de salida.

Que cambia respecto del smoke test
- La corrida por defecto pasa de una prueba liviana a 10 epochs, hasta 5000 ejemplos de train y hasta 1000 de evaluacion.
- Se elige automaticamente el checkpoint del pipeline principal full 8ep.
- Se documenta explicitamente el checkpoint usado, el criterio de seleccion y el mejor checkpoint downstream por tarea.
- Se corren solo las dos tareas solicitadas: XNLIes e InterTass2020.

Comparabilidad con literatura
- Esta corrida sigue el espiritu de la literatura: fine-tuning downstream estandar, evaluacion por tarea, uso del encoder preentrenado y sin gaze/scanpath en downstream.
- La configuracion es mas cercana a una corrida experimental real que a un smoke test.
- No replica grillas grandes de multiples seeds ni multiples learning rates; queda documentada como una version seria reducida.
