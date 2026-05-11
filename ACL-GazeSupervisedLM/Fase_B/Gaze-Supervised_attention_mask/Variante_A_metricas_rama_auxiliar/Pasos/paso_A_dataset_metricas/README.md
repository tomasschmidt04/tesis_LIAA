# Paso A - Dataset con metricas

Lee archivos JSONL de `../reading-et/results_all_alligned_2`, equivalentes a los `results_all_alligned` ya usados para scanpaths medidos, pero con columnas de metricas de lectura.

Features seleccionadas: `FFD`, `TRT`, `nFix`. Si `TRT` no existe usa `TFD`; si `nFix` no existe usa `FC`. El mapeo real se guarda en `gaze_feature_mapping.json`.

Se usa `reading_features_by_fixation`, no `reading_features_by_word`. Se valida que `len(word_id) == len(reading_features_by_fixation)` y los casos invalidos se registran en `mismatches.jsonl`.

La normalizacion se calcula sobre train con media/std por feature y se guarda en `gaze_feature_norm.json`. NaN/inf se reemplazan por `0.0`; si existe `reading_features_mask_by_fixation`, se usa para decidir valores validos.

Outputs principales: `sample_examples.jsonl`, `gaze_feature_mapping.json`, `gaze_feature_norm.json`, `dataset_metricas_summary.json`, `mismatches.jsonl`.
