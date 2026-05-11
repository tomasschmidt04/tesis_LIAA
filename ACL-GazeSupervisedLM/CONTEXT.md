# Contexto del proyecto: BETO con supervisión de scanpath de lectura

Este documento describe la arquitectura, el pipeline de entrenamiento, los archivos clave y las corridas finales del experimento de preentrenamiento BETO supervisado con movimientos oculares en español.

---

## 1. ¿Qué hace este proyecto?

El proyecto implementa un modelo de lenguaje en español basado en BETO (`dccuchile/bert-base-spanish-wwm-cased`) que incorpora una **rama auxiliar de scanpath de lectura**. Durante el preentrenamiento MLM, además de la pérdida estándar de enmascaramiento, el modelo entrena simultáneamente una segunda cabeza que aprende a predecir tokens enmascarados siguiendo el orden en que los lectores reales fijaron los ojos sobre cada oración.

La hipótesis central es que supervisar las representaciones con scanpaths humanos medidos —que codifican qué palabras llaman la atención, cuántas veces se releen y en qué orden— enriquece los embeddings del modelo de forma complementaria al MLM estándar, lo que debería mejorar su desempeño en tareas downstream de español.

---

## 2. Arquitectura del modelo

### 2.1 Rama principal (MLM estándar)

```
input_ids → BERT (BETO) → MLM head lineal → main_mlm_loss
```

La rama principal es idéntica a un BERT estándar con enmascaramiento. La cabeza MLM es `BertOnlyMLMHead` de HuggingFace.

### 2.2 Rama auxiliar (MLM supervisado por scanpath)

```
input_ids → BERT (BETO) → selección de hidden states por posición de fijación
         → packing con longitudes reales → GRU unidireccional
         → scanpath_mlm_head lineal → scanpath_mlm_loss
```

El truco central: las posiciones de fijación ocular (`measured_word_ids`, a nivel de palabra léxica 1-based) se mapean a posiciones de token mediante `LM_word_ids`. Luego se extraen los hidden states de BERT en esas posiciones en el orden de lectura, se procesan con una GRU, y se aplica una cabeza lineal `hidden_size → vocab_size` sobre la secuencia scanpath-level.

Los labels de la rama auxiliar se construyen expandiendo los labels MLM originales (`shape: B × T`) a la longitud del scanpath (`shape: B × S`): si un token enmascarado aparece varias veces en la secuencia de fijaciones, aporta varias veces a la pérdida auxiliar.

### 2.3 Pérdida combinada

```
total_loss = main_mlm_loss + λ × scanpath_mlm_loss
```

`λ` puede ser fijo (`--aux_weight`) o adaptativo (`--adaptive_scanpath_weight True`). Con lambda adaptativo, se calcula al final de cada época basándose en el progreso de la pérdida de scanpath en el set de evaluación:

```
progress = max(0, (loss_scanpath_inicial - loss_scanpath_actual) / loss_scanpath_inicial)
λ_next   = λ_min + progress × (λ_max − λ_min)
```

En la corrida final, la pérdida de scanpath sobre eval **aumentó** progresivamente (la oración de lectura se volvía más difícil de predecir a medida que el modelo mejoraba en MLM), por lo que `progress` permaneció en 0 y `λ` quedó fijo en `λ_min = 0.05` durante las 20 épocas.

### 2.4 Clase principal del modelo

**Archivo:** [Gazesup_bert_combined_mlm_model.py](Gazesup_bert_combined_mlm_model.py)

```python
class Gazesup_BERTForCombinedMaskedLM(BertPreTrainedModel):
    def __init__(self, config):
        self.bert = BertModel(config)           # backbone compartido
        self.cls = BertOnlyMLMHead(config)      # cabeza MLM principal
        self.sp_encoder = SP_Encoder(config, scanpath_source="measured")  # GRU + utilidades
        self.scanpath_mlm_head = nn.Linear(config.hidden_size, config.vocab_size)
```

El output del forward es `GazesupCombinedMaskedLMOutput` que incluye:
- `loss` (total), `main_mlm_loss`, `scanpath_mlm_loss`
- `main_mlm_logits`, `scanpath_mlm_logits`
- `bert_last_hidden_state`, `gaze_token_pos`, `sp_len`
- `scanpath_selected_hidden_states`, `gru_output`
- `scanpath_labels_expanded`

---

## 3. El cambio en el cálculo del masking: estático → dinámico

### Antes (masking estático por cantidad fija)

El parámetro `--max_masked_positions N` fijaba un número absoluto de posiciones a enmascarar por ejemplo. El código seleccionaba exactamente `N` tokens candidatos, independientemente de la longitud de la oración.

### Ahora (masking dinámico por probabilidad)

El parámetro `--mlm_probability 0.15` (por defecto) define la **fracción** de tokens candidatos a enmascarar en cada ejemplo. La implementación actual en [`train_mlm_scanpath_step5.py`](train_mlm_scanpath_step5.py) es:

```python
DEFAULT_MLM_PROBABILITY = 0.15

def build_dynamic_masked_inputs_and_labels(input_ids, attention_mask, tokenizer, mlm_probability=0.15):
    candidate_positions = _candidate_mask_positions(input_ids, attention_mask, tokenizer)
    num_to_mask = max(1, int(round(len(candidate_positions) * mlm_probability)))
    selected_positions = sorted(random.sample(candidate_positions, num_to_mask))

    for position in selected_positions:
        labels[position] = input_ids[position]
        r = random.random()
        if r < 0.8:
            masked_input_ids[position] = tokenizer.mask_token_id   # 80% → [MASK]
        elif r < 0.9:
            masked_input_ids[position] = random.randrange(vocab_size)  # 10% → token aleatorio
        # 10% restante → sin cambio
```

El masking se aplica **en tiempo de colación** (dentro de `collate_measured_mlm_batch`), lo que significa que **cada vez que un ejemplo pasa por el dataloader obtiene un conjunto distinto de posiciones enmascaradas**. Esto implementa el mismo esquema dinámico que usa RoBERTa y es más robusto que un masking estático guardado en disco.

El parámetro `--max_masked_positions` fue deprecado y se mantiene solo por compatibilidad con comandos anteriores; no tiene efecto en el código actual (la función `build_static_masked_inputs_and_labels` redirige internamente a la versión dinámica).

---

## 4. Dataset de preentrenamiento

### 4.1 Archivos

| Archivo | Ejemplos | Descripción |
|---|---|---|
| `reading-et/mlm_dataset_limpio_train_test/train.jsonl` | 38 518 | Scanpaths de entrenamiento |
| `reading-et/mlm_dataset_limpio_train_test/test.jsonl` | 9 630 | Scanpaths de evaluación |

El split train/test es **por cuento** (sin solapamiento de historias entre conjuntos), con las oraciones del cuento asignadas al conjunto de test de forma estratificada.

### 4.2 Campos de cada ejemplo

| Campo | Tipo | Descripción |
|---|---|---|
| `text` | `str` | Oración original (texto limpio, input al tokenizador) |
| `word_id` | `List[int]` | Secuencia de posiciones de palabra (1-based) visitadas por el lector |
| `scanpath_text` | `str` | Reconstrucción textual del scanpath (palabras en orden de fijación) |
| `scanpath_tokens` | `List[str]` | Tokens visitados en orden de fijación |
| `text_tokens` | `List[str]` | Tokens del texto original |
| `trial_id` | `str` | Identificador único de la corrida: `cuento::sujeto::segmento` |
| `segment_index` | `int` | Posición de la oración dentro del cuento (0-based) |
| `cuento` | `str` | Título del cuento |
| `source_file` | `str` | Archivo de origen del participante |
| `split` | `str` | `"train"` o `"test"` |
| `n_words` | `int` | Número de palabras en la oración original |
| `match_quality` | `str` | Calidad del alineamiento oración↔scanpath |
| `coverage` | `float` | Fracción de palabras del texto cubiertas por el scanpath |

### 4.3 Interpretación de `word_id`

`word_id` es la secuencia central. Cada elemento es el índice 1-based de la palabra del texto que el lector fijó. Valores repetidos indican refijaciones. Valores salteados indican palabras skipeadas. La longitud de `word_id` puede ser mayor, menor o igual a `n_words`.

**Ejemplo:** si el texto tiene 4 palabras `["¡Abrigate!", "Mi", "mamá", "me"]` y `word_id = [1, 1, 2, 2, 4]`, el lector fijó dos veces la palabra 1, dos veces la palabra 2, y saltó directamente a la palabra 4 (ignoró "mamá").

Durante el preentrenamiento, la función `build_measured_scanpath()` agrega sentinelas CLS/SEP artificiales:

```python
measured_word_ids = [0] + lexical_positions + [sentence_word_count + 1]
```

---

## 5. Corridas finales: `corridas_finales_full_20ep`

Estas son las corridas de preentrenamiento definitivas del proyecto, con el dataset completo (sin truncamiento) y 20 épocas.

**Directorio base:** `Pasos/corridas_finales_full_20ep/`

```
corridas_finales_full_20ep/
├── pretraining_beto_mlm_limpio/          # BETO baseline (MLM puro, sin scanpath)
│   ├── best_checkpoint/                  # Mejor checkpoint por eval_loss
│   ├── checkpoint_final/                 # Checkpoint del último epoch
│   ├── loss_curves.csv                   # Pérdida por época (train + eval)
│   ├── loss_plots/                       # Gráficos PNG de curvas de pérdida
│   ├── run_args.json                     # Hiperparámetros exactos de la corrida
│   ├── split_report.json                 # Estadísticas del split train/eval
│   └── trainer_state.json               # Estado del Trainer de HuggingFace
│
├── pretraining_beto_scanpath_lambda_limpio/   # BETO + Scanpath con lambda adaptativo
│   ├── best_checkpoint/
│   ├── checkpoint_final/
│   ├── lambda_scanpath_log.csv           # Evolución de λ por época
│   ├── loss_curves.csv                   # Pérdida total + MLM + scanpath por época
│   ├── loss_plots/
│   ├── run_args.json
│   ├── split_report.json
│   └── trainer_state.json
│
└── downstream/
    └── beto_pretrained/                  # Fine-tuning del checkpoint BETO baseline
```

### 5.1 Hiperparámetros de la corrida BETO baseline

```json
{
  "model_name_or_path": "dccuchile/bert-base-spanish-wwm-cased",
  "train_file": ".../mlm_dataset_limpio_train_test/train.jsonl",
  "eval_file": ".../mlm_dataset_limpio_train_test/test.jsonl",
  "num_train_epochs": 20,
  "per_device_train_batch_size": 4,
  "learning_rate": 5e-05,
  "max_seq_length": 128,
  "split_strategy": "precomputed_files",
  "seed": 13,
  "save_every_epoch": false
}
```

Script de entrenamiento: [`train_mlm_beto_baseline_step7_pretrain.py`](train_mlm_beto_baseline_step7_pretrain.py)

### 5.2 Hiperparámetros de la corrida BETO + Scanpath (lambda adaptativo)

```json
{
  "model_name_or_path": "dccuchile/bert-base-spanish-wwm-cased",
  "train_file": ".../mlm_dataset_limpio_train_test/train.jsonl",
  "eval_file": ".../mlm_dataset_limpio_train_test/test.jsonl",
  "num_train_epochs": 20,
  "per_device_train_batch_size": 4,
  "learning_rate": 5e-05,
  "max_seq_length": 128,
  "split_strategy": "precomputed_files",
  "seed": 13,
  "aux_weight": 0.1,
  "adaptive_scanpath_weight": true,
  "scanpath_weight_min": 0.05,
  "scanpath_weight_max": 0.5,
  "scanpath_weight_warmup_epochs": 1,
  "scanpath_weight_update_metric": "auto"
}
```

Script de entrenamiento: [`train_mlm_combined_step7.py`](train_mlm_combined_step7.py)

### 5.3 Evolución de pérdidas (BETO + Scanpath, 20 épocas)

El dataset de entrenamiento tiene **9630 batches por época** (38518 ejemplos / batch_size 4), y el de evaluación **2408 batches** (9630 ejemplos / batch_size 4).

| Época | Train total | Train MLM | Train Scanpath | Eval total | Eval MLM | Eval Scanpath | λ |
|-------|------------|-----------|----------------|------------|----------|----------------|---|
| 1  | 0.777 | 0.592 | 3.686 | 6.077 | 5.683 | 7.890 | 0.05 |
| 2  | 0.310 | 0.250 | 1.199 | 7.928 | 7.505 | 8.455 | 0.05 |
| 5  | 0.130 | 0.109 | 0.407 | 9.838 | 9.406 | 8.626 | 0.05 |
| 10 | 0.086 | 0.076 | 0.197 | 10.772 | 10.326 | 8.911 | 0.05 |
| 15 | 0.066 | 0.059 | 0.124 | 10.801 | 10.349 | 9.034 | 0.05 |
| 20 | 0.057 | 0.053 | 0.093 | 10.963 | 10.490 | 9.473 | 0.05 |

**Observaciones sobre las curvas:**
- La pérdida de **train** baja consistentemente en las tres componentes, indicando que el modelo memoriza bien el dataset de entrenamiento.
- La pérdida de **eval total** sube de ~6 a ~11, señal de que el modelo sobre-ajusta al dataset de entrenamiento (esperable dado que el preentrenamiento usa el mismo corpus de scanpaths como train y eval, y no Wikipedia).
- La **eval scanpath loss** sube de 7.89 a ~9.5, lo que significa que `progress = 0` en todas las épocas y `λ` quedó fijado en `λ_min = 0.05` por el mecanismo adaptativo.
- La pérdida de **train scanpath** cae de 3.69 → 0.09, mostrando que la rama auxiliar aprende la estructura de los scanpaths de entrenamiento.

### 5.4 Archivos de artefactos por corrida

**`loss_curves.csv`** — columnas:
```
epoch, split, loss_total_mean, loss_standard_mean, loss_scanpath_mean, augweight, num_batches
```

**`lambda_scanpath_log.csv`** — columnas (solo corrida con scanpath):
```
epoch, lambda_scanpath_t, lambda_scanpath_next, metric_used_for_update,
current_scanpath_loss_for_update, initial_scanpath_loss, progress,
train_loss_total_mean, train_loss_mlm_mean, train_loss_scanpath_mean,
eval_loss_total_mean, eval_loss_mlm_mean, eval_loss_scanpath_mean, warning
```

**`run_args.json`** — snapshot exacto de todos los hiperparámetros de la corrida.

**`split_report.json`** — estadísticas del split: cantidad de ejemplos por partición, cuentos en cada split, solapamiento de textos (verificación de que no hay data leakage).

---

## 6. Pipeline de pasos (Pasos/)

El pipeline está organizado en pasos numerados de menor a mayor complejidad:

| Paso | Directorio | Script | Descripción |
|------|-----------|--------|-------------|
| 5 | `paso_5/` | `train_mlm_scanpath_step5.py` | Smoke test de la rama auxiliar MLM (solo scanpath) |
| 6 | `paso_6/` | `train_mlm_combined_step6.py` | Smoke test del modelo combinado (main + aux) |
| 7 | `paso_7*/` | `train_mlm_combined_step7.py` | Preentrenamiento a escala completa con BETO |
| 7-baseline | `paso_7*/` | `train_mlm_beto_baseline_step7_pretrain.py` | BETO puro (sin scanpath) |
| 9 | `results/paso_9/` | `train_spanish_downstream_baseline.py` | Fine-tuning downstream (XNLI, Rioplatense) |

Las corridas en `corridas_finales_full_20ep/` son la versión final definitiva del Paso 7, con el dataset completo y 20 épocas.

---

## 7. Estrategias de split para preentrenamiento

El script `train_mlm_combined_step7.py` soporta tres estrategias (`--split_strategy`):

### `precomputed_files` (usado en las corridas finales)
```bash
--train_file .../train.jsonl
--eval_file  .../test.jsonl
```
Carga archivos JSONL pre-separados. El script valida que no haya solapamiento de cuentos entre train y eval.

### `sentence_position`
```bash
--measured_scanpath_file .../aligned_scanpaths.jsonl
--eval_sentence_position_mod 10
--eval_sentence_position_remainder 5
```
Envía al eval todas las oraciones cuya posición en el cuento satisface `pos % 10 == 5`. Garantiza separación sin solapamiento de textos exactos.

### `contiguous`
Los primeros `max_train_samples` ejemplos van a train, el resto a eval. Solo para smoke tests.

---

## 8. Scripts de corrida

### `run_paso7_clean_sentence_split_full.sh`
Ejecuta BETO baseline + BETO+scanpath con la estrategia `precomputed_files`. Configurable via variables de entorno:
```bash
NUM_TRAIN_EPOCHS=20 AUX_WEIGHT=0.1 ./run_paso7_clean_sentence_split_full.sh
```

### `run_all_paso7_paso9_xnli_rioplatense.sh`
Pipeline completo: Paso 7 preentrenamiento → Paso 9 downstream (XNLI + Rioplatense).
Tres variantes de modelo: `beto_base`, `beto_pretrained`, `scanpath_pretrained`.

```bash
# Solo downstream con checkpoints full ya existentes
RUN_PRETRAINING=0 STEP9_CHECKPOINT_SOURCE=full ./run_all_paso7_paso9_xnli_rioplatense.sh
```

### `run_all_half_pretraining_and_rioplatense.sh`
Pipeline completo con datos parciales (half-data, 8 épocas) para exploración rápida.

---

## 9. Tareas downstream

| Tarea | Script | Métrica | Labels |
|-------|--------|---------|--------|
| XNLI (es) | `train_spanish_downstream_baseline.py` | accuracy | entailment / neutral / contradiction |
| Rioplatense hate binary | `train_glue_LM_baseline.py` | macro_f1 | hateful / non-hateful |

En el régimen **low-resource**, se usan solo 1000 ejemplos de entrenamiento (controlados por `--max_train_samples 1000` y una semilla de datos `--low_resource_data_seed`). Las corridas usan semillas 111, 222, 333, 444, 555 para estimar varianza.

Fine-tuning: 20 épocas, lr=2e-5, batch=32, evaluación por época, `load_best_model_at_end`.

---

## 10. Archivos clave

| Archivo | Rol |
|---------|-----|
| [`Gazesup_bert_combined_mlm_model.py`](Gazesup_bert_combined_mlm_model.py) | Modelo combinado (main + aux branch) |
| [`Gazesup_bert_model.py`](Gazesup_bert_model.py) | SP_Encoder, GRU, conversión word→token pos |
| [`train_mlm_combined_step7.py`](train_mlm_combined_step7.py) | Script de preentrenamiento principal |
| [`train_mlm_beto_baseline_step7_pretrain.py`](train_mlm_beto_baseline_step7_pretrain.py) | Preentrenamiento BETO puro (baseline) |
| [`train_mlm_scanpath_step5.py`](train_mlm_scanpath_step5.py) | Dynamic MLM masking + smoke test auxiliar |
| [`measured_scanpath_utils.py`](measured_scanpath_utils.py) | Carga de dataset, split, feature extraction |
| [`loss_curve_utils.py`](loss_curve_utils.py) | Logging de pérdidas a CSV |
| [`trainers.py`](trainers.py) | Trainer personalizado (HuggingFace) |
| [`train_spanish_downstream_baseline.py`](train_spanish_downstream_baseline.py) | Fine-tuning XNLI |
| [`train_glue_LM_baseline.py`](train_glue_LM_baseline.py) | Fine-tuning Rioplatense |
| [`scripts/plot_loss_curves.py`](scripts/plot_loss_curves.py) | Visualización de curvas de pérdida |
| [`scripts/plot_lambda_scanpath.py`](scripts/plot_lambda_scanpath.py) | Visualización de evolución de λ |
| [`run_paso7_clean_sentence_split_full.sh`](run_paso7_clean_sentence_split_full.sh) | Script de corrida Paso 7 |
| [`run_all_paso7_paso9_xnli_rioplatense.sh`](run_all_paso7_paso9_xnli_rioplatense.sh) | Pipeline completo Paso 7 + Paso 9 |

---

## 11. Instalación y entorno

```bash
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

Backbone descargado automáticamente por HuggingFace la primera vez:
```
dccuchile/bert-base-spanish-wwm-cased
```

Variables de entorno útiles:
```bash
export PYTHON_BIN=.venv/bin/python
export CUDA_VISIBLE_DEVICES=0
```

---

## 12. Reproducir las corridas finales

```bash
# Preentrenamiento BETO baseline (20 épocas, dataset completo)
BASELINE_OUTPUT_DIR=Pasos/corridas_finales_full_20ep/pretraining_beto_mlm_limpio \
NUM_TRAIN_EPOCHS=20 \
MAX_TRAIN_SAMPLES=-1 \
MAX_EVAL_SAMPLES=-1 \
MLM_TRAIN_FILE=/path/to/train.jsonl \
MLM_EVAL_FILE=/path/to/test.jsonl \
./run_paso7_clean_sentence_split_full.sh
```

```bash
# Preentrenamiento BETO + Scanpath (20 épocas, lambda adaptativo)
SCANPATH_OUTPUT_DIR=Pasos/corridas_finales_full_20ep/pretraining_beto_scanpath_lambda_limpio \
NUM_TRAIN_EPOCHS=20 \
MAX_TRAIN_SAMPLES=-1 \
MAX_EVAL_SAMPLES=-1 \
AUX_WEIGHT=0.1 \
MLM_TRAIN_FILE=/path/to/train.jsonl \
MLM_EVAL_FILE=/path/to/test.jsonl \
./run_paso7_clean_sentence_split_full.sh
```

Para habilitar lambda adaptativo, el script `train_mlm_combined_step7.py` debe llamarse con:
```bash
--adaptive_scanpath_weight True \
--scanpath_weight_min 0.05 \
--scanpath_weight_max 0.5 \
--scanpath_weight_warmup_epochs 1
```
