PASO 3 - Lambda adaptativo simple para loss scanpath
====================================================

Que se cambio
- Se agrego un modo opcional para reemplazar el peso fijo aux_weight de la rama scanpath por lambda_scanpath_t.
- No se cambio arquitectura, dataset, labels MLM, labels scanpath, conversion palabra-token, GRU, head MLM, downstream ni split.

Donde estaba antes la loss total
- Archivo: Gazesup_bert_combined_mlm_model.py
- Antes: total_loss = main_mlm_loss + float(aux_weight) * scanpath_mlm_loss

Donde se usa ahora
- Archivo: Gazesup_bert_combined_mlm_model.py
- Ahora el forward acepta current_scanpath_weight.
- Si current_scanpath_weight=None, usa aux_weight como antes.
- Si current_scanpath_weight tiene valor, usa ese peso efectivo:
  total_loss = main_mlm_loss + current_scanpath_weight * scanpath_mlm_loss

Donde se calcula lambda
- Archivo: train_mlm_combined_step7.py
- Funcion principal: train_step7
- Lambda se calcula al cierre de cada epoch usando una loss scanpath promedio.
- El valor calculado queda guardado como current_scanpath_weight y se pasa al forward durante la epoch siguiente.

Formula
- initial_scanpath_loss = primera loss scanpath promedio disponible.
- current_scanpath_loss = loss scanpath promedio usada para actualizar.
- progress = (initial_scanpath_loss - current_scanpath_loss) / initial_scanpath_loss
- progress se clipea a [0, 1].
- lambda_scanpath_t = scanpath_weight_min + progress * (scanpath_weight_max - scanpath_weight_min)

Metrica usada
- auto: usa eval_loss_scanpath si existe; si no existe, usa train_loss_scanpath_mean.
- eval_loss_scanpath: fuerza usar evaluacion.
- train_loss_scanpath_mean: fuerza usar entrenamiento.

Warmup
- Durante las primeras scanpath_weight_warmup_epochs, lambda se mantiene en scanpath_weight_min.
- Default: 1 epoch.

Como volver al comportamiento anterior
- Omitir --adaptive_scanpath_weight o usar --adaptive_scanpath_weight False.
- En ese modo se usa exactamente la forma fija:
  loss_total = loss_mlm + aux_weight * loss_scanpath

Argumentos nuevos
- --adaptive_scanpath_weight
- --scanpath_weight_min
- --scanpath_weight_max
- --scanpath_weight_warmup_epochs
- --scanpath_weight_update_metric
- --scanpath_weight_log_path

Limitaciones de esta estrategia simple
- Lambda depende de una unica loss scanpath promedio por epoch.
- No balancea gradientes como GradNorm.
- No mira si MLM empeora al mismo tiempo.
- No es una grilla de hiperparametros; es una regla simple y auditable.

Archivos de este paso
- lambda_scanpath_log.csv: evolucion numerica por epoch.
- salida_debug_lambda_adaptativo.txt: tabla legible de la corrida corta.
- comandos_y_funciones.txt: comandos y funciones usadas.
- README_lambda_adaptativo.txt: este documento.

Configuracion de esta corrida
- output_dir: Pasos/corridas_finales_full_20ep/pretraining_beto_scanpath_lambda_limpio
- adaptive_scanpath_weight: True
- scanpath_weight_min: 0.05
- scanpath_weight_max: 0.5
- scanpath_weight_warmup_epochs: 1
- scanpath_weight_update_metric: auto
- scanpath_weight_log_path: Pasos/corridas_finales_full_20ep/pretraining_beto_scanpath_lambda_limpio/lambda_scanpath_log.csv
