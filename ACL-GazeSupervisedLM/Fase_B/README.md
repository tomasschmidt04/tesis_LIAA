# Fase B - Variante con attention mask

Fase B implementa la Variante B: eye-tracking attention mask / gaze attention bias.

La idea es usar metricas de eye-tracking como FFD, TRT y nFix para construir un sesgo de atencion que pueda incorporarse durante la etapa de adaptacion con eye-tracking. En fine-tuning downstream sin scanpath, la mascara debe poder apagarse para reutilizar el backbone de forma comparable.

La copia base del codigo esta en `Gaze-Supervised_attention_mask/`.
