# Paso D - Alignment con subtokens

La expansion de metricas ocurre en `SP_Encoder.convert_word_pos_seq_to_token_pos_seq`.

Cuando una fijacion apunta a una palabra que el tokenizer divide en varios subtokens, la fila `[FFD, TRT, nFix]` de esa fijacion se repite para cada subtoken. Si una palabra aparece varias veces en `word_id` por refijacion, sus features se repiten en cada aparicion.

El chequeo esperado dentro del modelo es:

```text
x_sp.shape[1] == measured_gaze_features_token_level.shape[1]
```

Si no coincide, se registra en `alignment_warnings.jsonl` y se aplica fallback de truncado/padding con ceros.

`inspect_subtoken_alignment.py` genera ejemplos chicos en `subtoken_alignment_examples.jsonl` y un resumen en `alignment_summary.json`.
