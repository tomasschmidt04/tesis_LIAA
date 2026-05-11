# Variante A - metricas en rama auxiliar

Implementacion incremental para agregar metricas de fijacion a la rama auxiliar scanpath medida.

La logica de entrenamiento se mantiene:

```text
loss_total = loss_standard + augweight * loss_scanpath
```

No se modifica `OurTrainer`, optimizer, scheduler, `save_strategy`, `evaluation_strategy`, `load_best_model_at_end` ni `metric_for_best_model`.

Pasos:

- `paso_A_dataset_metricas`: lectura, mapeo y normalizacion de `FFD`, `TRT`, `nFix`.
- `paso_B_batch_collator`: padding batchable de `measured_gaze_features`.
- `paso_C_model_fusion`: fusion suave antes de la GRU.
- `paso_D_alignment_subtokens`: expansion de features de fijacion a subtokens.

El comando minimo esta en `run_debug_variante_A.sh`. Los checkpoints del debug se escriben fuera de esta carpeta, en `/tmp/variante_A_metricas_debug_model_output`.
