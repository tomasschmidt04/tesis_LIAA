# Paso B - Batch / Collator

Este paso convierte ejemplos individuales con distinta cantidad de fijaciones en tensores batchables.

`measured_gaze_features` llega como listas de shape variable, por ejemplo `[20, 3]`, `[34, 3]`, `[12, 3]`. El collator las paddea con `0.0` hasta `max_scanpath_len` del batch y produce:

- `measured_word_ids`
- `measured_sp_len`
- `measured_gaze_features`
- `measured_gaze_feature_mask`

Este paso trabaja a nivel fijacion/`word_id`. La expansion a subtokens se hace y se inspecciona en el Paso D.

Con `--debug_gaze_features True`, el primer batch escribe `batch_shapes.txt` y `sample_batch_summary.json`.
