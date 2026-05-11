# Paso C - Modelo / Fusion suave

La fusion se inserta en `SP_Encoder` justo despues de construir `x_sp` y antes del dropout/GRU.

Formula:

```text
gaze_proj = Linear([FFD, TRT, nFix])
gate = sigmoid(Linear(concat(x_sp, gaze_proj)))
x_fused = LayerNorm(x_sp + gate * gaze_proj)
```

La salida conserva dimension `hidden_size`, por lo que la GRU recibe la misma forma que antes: `[B, T, hidden_size]`.

El bias de `gaze_gate` se inicializa en `-2.0` para que la fusion arranque suave. Si `--use_gaze_features False` o no llegan features, el camino baseline queda igual.

Con `--debug_gaze_features True`, el primer forward con fusion escribe `model_fusion_shapes.txt`.
