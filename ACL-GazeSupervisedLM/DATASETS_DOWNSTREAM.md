# Datasets downstream usados en el proyecto

Este documento resume los datasets usados para evaluar tareas downstream en el proyecto, cómo se cargan en los scripts, qué splits se usan y cuántos ejemplos entran en las corridas low-resource. Las referencias principales del repo son `train_glue_LM_baseline.py`, `run_glue_from_export_step9.py`, `train_spanish_downstream_baseline.py`, `run_spanish_downstream_serio_step9.py` y `Pasos/resumen_low_resource/resumen_metricas_low_resource.txt`.

## 1. Resumen general

| Tarea | Dataset exacto | Fuente | Configuración | Tipo de tarea | Splits disponibles | Split usado para train | Split usado para evaluación | Cantidad total aproximada | Cantidad usada en las corridas low-resource |
|---|---|---|---|---|---|---|---|---:|---|
| SST-2 | `load_dataset("glue", "sst2")` | Hugging Face Datasets / GLUE | `sst2` | Clasificación binaria de sentimiento en una oración | `train`: 67.349, `validation`: 872, `test`: 1.821 | `train` | En corridas estándar: `validation`. En scripts low-resource con `train_as_val=True`: 1.000 ejemplos tomados del `train` después del subconjunto de entrenamiento | 70.042 ejemplos | Scripts low-resource GLUE: `MAX_TRAIN_SAMPLES in {200, 500, 1000}`. El paso 9 de debug puede usar `--max_train_samples` menor para prueba rápida |
| RTE | `load_dataset("glue", "rte")` | Hugging Face Datasets / GLUE | `rte` | Clasificación binaria de inferencia textual / entailment | `train`: 2.490, `validation`: 277, `test`: 3.000 | `train` | En corridas estándar: `validation`. En scripts low-resource con `train_as_val=True`: 1.000 ejemplos tomados del `train` después del subconjunto de entrenamiento | 5.767 ejemplos | Scripts low-resource GLUE: `MAX_TRAIN_SAMPLES in {200, 500, 1000}` con seeds `111, 222, 333, 444, 555` |
| XNLI español | `load_dataset("xnli", "es")` | Hugging Face Datasets / XNLI | `es` | Clasificación multiclase de inferencia textual en español | `train`: 392.702, `validation`: 2.490, `test`: 5.010 | `train` | `validation` oficial de XNLI/es | 400.202 ejemplos | Corridas registradas: 1.000 ejemplos de train y 1.000 de eval; corrida intermedia: 20.000 de train y 2.490 de eval |
| InterTASS 2020 | Parquet local/cacheado desde `iberbench/iberbench_all` | Hugging Face Hub, archivo raw parquet `iberlef-tass-sentiment_analysis-2020-spanish/train-00000-of-00001.parquet` | Sin configuración de `datasets`; descarga directa del parquet y cache local como `intertass2020_train.parquet` | Clasificación multiclase de sentimiento en tweets en español | La fuente usada expone `train` solamente: 4.797 ejemplos | Split sintético `train` creado con `train_test_split(test_size=0.2, seed=13, stratify_by_column="label")` | Split sintético `test`, renombrado como `validation` en el `DatasetDict` del script | 4.797 ejemplos | Corridas registradas: 1.000 ejemplos de train y 960 de eval; corrida intermedia pide 20.000, pero usa 3.837 porque es todo el train sintético disponible, y 960 de eval |
| Rioplatense Hate Binary | `load_dataset("csv", data_files=...)` sobre CSV raw de GitHub, o `--dataset_path` local | `finiteautomata/rioplatense_hate_speech` | `data/test_01.csv`...`data/test_05.csv` | Clasificación binaria de hate speech rioplatense | CSVs `test_01` a `test_05`; en el script: `train` = `test_01`-`test_03`, `validation` = `test_04`, `test` = `test_05` | `train` sintético a partir de los tres primeros CSV | Con `train_as_val=True`: 1.000 ejemplos estratificados reservados del train original sin overlap; si no, `test_04` | Aproximadamente 6.5k ejemplos en los cinco CSV | Low-resource principal: 1.000 ejemplos de train, seed `111`, `num_train_epochs=20`, `metric_for_best_model=macro_f1` |

## 2. Datasets en inglés: GLUE

El paso 9 original usa Hugging Face Datasets para cargar tareas GLUE con:

```python
from datasets import load_dataset

ds = load_dataset("glue", task)
```

En el repo, `task` se instancia como `"sst2"` o `"rte"` desde `run_glue_from_export_step9.py`, mientras que `train_glue_LM_baseline.py` hace la carga efectiva con:

```python
raw_datasets = load_dataset("glue", data_args.task_name)
```

Las columnas de texto se resuelven con `task_to_keys`:

```python
task_to_keys = {
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
}
```

Esto significa que SST-2 se tokeniza como una sola secuencia, y RTE se tokeniza como par de secuencias.

### 2.1 GLUE SST-2

**Carga reproducible**

```python
from datasets import load_dataset

sst2 = load_dataset("glue", "sst2")
```

**Splits**

| Split | Ejemplos | Uso en el proyecto |
|---|---:|---|
| `train` | 67.349 | Entrenamiento |
| `validation` | 872 | Evaluación estándar cuando `train_as_val=False` |
| `test` | 1.821 | No se usa para métricas locales porque no trae labels públicas (`label = -1`) |

**Labels**

| ID | Label |
|---:|---|
| 0 | `negative` |
| 1 | `positive` |

**Low-resource**

Los scripts low-resource GLUE usan subconjuntos del split `train`:

```bash
for MAX_TRAIN_SAMPLES in 200 500 1000
do
  ...
  --max_train_samples $MAX_TRAIN_SAMPLES
  --low_resource_data_seed $DATA_SEED
  --train_as_val True
done
```

Cuando `train_as_val=True`, el script baraja el `train` con `low_resource_data_seed`, toma los primeros `MAX_TRAIN_SAMPLES` para entrenamiento y usa los siguientes 1.000 ejemplos del mismo `train` como validación interna:

```python
train_dataset_all = train_dataset.shuffle(seed=data_args.low_resource_data_seed)
train_dataset = train_dataset_all.select(range(data_args.max_train_samples))
eval_dataset = train_dataset_all.select(
    range(data_args.max_train_samples, data_args.max_train_samples + 1000)
)
```

**Ejemplos reales**

| Split | Campos relevantes |
|---|---|
| `train[0]` | `sentence`: `hide new secretions from the parental units `; `label`: `0` (`negative`); `idx`: `0` |
| `validation[0]` | `sentence`: `it 's a charming and often affecting journey . `; `label`: `1` (`positive`); `idx`: `0` |
| `test[0]` | `sentence`: `uneasy mishmash of styles and genres .`; `label`: `-1`; `idx`: `0` |

### 2.2 GLUE RTE

**Carga reproducible**

```python
from datasets import load_dataset

rte = load_dataset("glue", "rte")
```

**Splits**

| Split | Ejemplos | Uso en el proyecto |
|---|---:|---|
| `train` | 2.490 | Entrenamiento |
| `validation` | 277 | Evaluación estándar cuando `train_as_val=False` |
| `test` | 3.000 | No se usa para métricas locales porque no trae labels públicas (`label = -1`) |

**Labels**

| ID | Label |
|---:|---|
| 0 | `entailment` |
| 1 | `not_entailment` |

**Low-resource**

Las corridas low-resource GLUE registradas en los scripts usan:

- `MAX_TRAIN_SAMPLES`: `200`, `500`, `1000`.
- `DATA_SEED`: `111`, `222`, `333`, `444`, `555`.
- `train_as_val=True`, por lo que la evaluación interna usa los 1.000 ejemplos siguientes del `train` barajado.

En RTE hay que tener en cuenta que `train` tiene 2.490 ejemplos. Por eso el esquema `1000 train + 1000 validación interna` entra completo dentro del split de entrenamiento original.

**Ejemplos reales**

| Split | Campos relevantes |
|---|---|
| `train[0]` | `sentence1`: `No Weapons of Mass Destruction Found in Iraq Yet.`; `sentence2`: `Weapons of Mass Destruction Found in Iraq.`; `label`: `1` (`not_entailment`); `idx`: `0` |
| `validation[0]` | `sentence1`: `Dana Reeve, the widow of the actor Christopher Reeve, has died of lung cancer at age 44, according to the Christopher Reeve Foundation.`; `sentence2`: `Christopher Reeve had an accident.`; `label`: `1` (`not_entailment`); `idx`: `0` |
| `test[0]` | `sentence1`: `Mangla was summoned after Madhumita's sister Nidhi Shukla, who was the first witness in the case.`; `sentence2`: `Shukla is related to Mangla.`; `label`: `-1`; `idx`: `0` |

## 3. Datasets en español

Las tareas downstream en español están centralizadas en `train_spanish_downstream_baseline.py`. El wrapper `run_spanish_downstream_serio_step9.py` ejecuta siempre:

```python
TASK_ORDER = ["xnli_es", "intertass2020"]
```

Ambas tareas se entrenan con `AutoModelForSequenceClassification`, `AutoTokenizer`, `Trainer` y `DataCollatorWithPadding`. Por defecto, el script español usa:

```bash
--max_train_samples 5000
--max_eval_samples 1000
--num_train_epochs 10
--per_device_train_batch_size 8
--per_device_eval_batch_size 8
--max_seq_length 128
--learning_rate 2e-5
--seed 13
```

Las corridas low-resource resumidas en `Pasos/resumen_low_resource/resumen_metricas_low_resource.txt` incluyen configuraciones con 1.000 ejemplos de entrenamiento y una corrida intermedia con 20.000 ejemplos solicitados.

### 3.1 XNLI español

**Carga reproducible**

```python
from datasets import load_dataset

xnli_es = load_dataset("xnli", "es")
```

En el script:

```python
raw = load_dataset("xnli", "es")
train_dataset = cap_dataset(raw["train"], args.max_train_samples, args.seed)
eval_dataset = cap_dataset(raw["validation"], args.max_eval_samples, args.seed)
```

`cap_dataset` baraja con `seed=13` si el split tiene más ejemplos que el máximo pedido, y luego selecciona `range(max_samples)`.

**Splits**

| Split | Ejemplos | Uso en el proyecto |
|---|---:|---|
| `train` | 392.702 | Entrenamiento |
| `validation` | 2.490 | Evaluación |
| `test` | 5.010 | Disponible, pero no usado en las corridas documentadas |

**Labels**

| ID | Label |
|---:|---|
| 0 | `entailment` |
| 1 | `neutral` |
| 2 | `contradiction` |

**Low-resource**

Corridas encontradas en el repo:

| Corrida | Train usado | Eval usado | Seed | Fuente |
|---|---:|---:|---:|---|
| `Pasos/paso_9_serio_low_resource_k1000_seed13/xnli_es` | 1.000 | 1.000 | 13 | `task_metadata.json` |
| `Pasos/paso_9_serio_beto_full_low_resource_k1000_seed13/xnli_es` | 1.000 | 1.000 | 13 | `task_metadata.json` |
| `Pasos/paso_9_serio_intermedio_seed13/xnli_es` | 20.000 | 2.490 | 13 | `task_metadata.json` |

**Ejemplos reales**

| Split | Campos relevantes |
|---|---|
| `train[0]` | `premise`: `Los robando de crema conceptualmente tienen dos dimensiones básicas : producto y geografía .`; `hypothesis`: `El producto y la geografía son los que hacen que la crema funcione .`; `label`: `1` (`neutral`) |
| `validation[0]` | `premise`: `Y él dijo: Mamá, estoy en casa.`; `hypothesis`: `Llamó a su madre tan pronto como el autobús escolar lo dejó.`; `label`: `1` (`neutral`) |
| `test[0]` | `premise`: `Bien, ni estaba pensando en eso, pero estaba tan frustrada y empecé a hablar con él de nuevo.`; `hypothesis`: `No he vuelto a hablar con él.`; `label`: `2` (`contradiction`) |

### 3.2 InterTASS 2020

InterTASS 2020 no se carga con `load_dataset(...)` en el script del proyecto. Se descarga un parquet desde `iberbench/iberbench_all`, se cachea localmente y se convierte a `Dataset` de Hugging Face.

**Fuente exacta**

```python
INTERTASS2020_URL = (
    "https://huggingface.co/datasets/iberbench/iberbench_all/resolve/main/"
    "iberlef-tass-sentiment_analysis-2020-spanish/train-00000-of-00001.parquet?download=true"
)
```

**Carga reproducible según el repo**

```python
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict

parquet_path = Path(args.intertass_cache_dir) / "intertass2020_train.parquet"
dataframe = pd.read_parquet(parquet_path)
dataframe = dataframe[["text", "label", "language"]].dropna(
    subset=["text", "label"]
).reset_index(drop=True)
dataframe["label"] = dataframe["label"].astype(str)

dataset = Dataset.from_pandas(dataframe, preserve_index=False)
dataset = dataset.class_encode_column("label")
split = dataset.train_test_split(
    test_size=0.2,
    seed=args.seed,
    stratify_by_column="label",
)

raw_datasets = DatasetDict({
    "train": split["train"],
    "validation": split["test"],
})
```

**Splits**

La fuente usada por el repo trae un único parquet de entrenamiento con 4.797 filas y tres columnas: `text`, `label`, `language`.

| Split | Ejemplos | Uso en el proyecto |
|---|---:|---|
| Parquet original `train` | 4.797 | Fuente completa descargada/cacheada |
| Split sintético `train` | 3.837 | Entrenamiento antes de aplicar `max_train_samples` |
| Split sintético `validation` | 960 | Evaluación |

La partición sintética se crea con `test_size=0.2`, `seed=13` y estratificación por `label`.

**Labels**

El script codifica la columna `label` como clase categórica después de convertirla a string. En las corridas cacheadas aparecen:

| ID interno | Label original |
|---:|---|
| 0 | `"0"` |
| 1 | `"1"` |
| 2 | `"2"` |

Distribución en el parquet cacheado:

| Label | Ejemplos |
|---|---:|
| `"0"` | 1.882 |
| `"1"` | 1.392 |
| `"2"` | 1.523 |

**Low-resource**

Corridas encontradas en el repo:

| Corrida | Train usado | Eval usado | Seed | Fuente |
|---|---:|---:|---:|---|
| `Pasos/paso_9_serio_low_resource_k1000_seed13/intertass2020` | 1.000 | 960 | 13 | `task_metadata.json` |
| `Pasos/paso_9_serio_beto_full_low_resource_k1000_seed13/intertass2020` | 1.000 | 960 | 13 | `task_metadata.json` |
| `Pasos/paso_9_serio_intermedio_seed13/intertass2020` | 3.837 | 960 | 13 | `task_metadata.json` |

En la corrida intermedia se pidió `--max_train_samples 20000`, pero el split sintético de entrenamiento tiene solo 3.837 ejemplos; por eso se usan todos los ejemplos disponibles.

**Ejemplos reales del parquet cacheado**

| Índice | `text` | `label` | `language` |
|---:|---|---|---|
| 0 | `@leomall2018 Según yo era como aviso, pero ahora sí ya es oficial` | `"2"` | `spanish` |
| 1 | `@benshorts a juzgar por mis comportamientos autodestructivos en las relaciones, aún quiero serlo` | `"0"` | `spanish` |
| 2 | `#BuenosDias mundo Twittero ya desperté y estoy listo para vivir un dia mas #ExcelenteMartes` | `"1"` | `spanish` |

### 3.3 Rioplatense Hate Binary

Esta tarea usa el dataset del repo `finiteautomata/rioplatense_hate_speech`, orientado a hate speech en español rioplatense. Para evitar depender de un clon local, `train_glue_LM_baseline.py` carga por defecto los CSV publicados en GitHub raw mediante:

```python
from datasets import load_dataset

raw = load_dataset(
    "csv",
    data_files={
        "train": [
            "https://raw.githubusercontent.com/finiteautomata/rioplatense_hate_speech/refs/heads/main/data/test_01.csv",
            "https://raw.githubusercontent.com/finiteautomata/rioplatense_hate_speech/refs/heads/main/data/test_02.csv",
            "https://raw.githubusercontent.com/finiteautomata/rioplatense_hate_speech/refs/heads/main/data/test_03.csv",
        ],
        "validation": "https://raw.githubusercontent.com/finiteautomata/rioplatense_hate_speech/refs/heads/main/data/test_04.csv",
        "test": "https://raw.githubusercontent.com/finiteautomata/rioplatense_hate_speech/refs/heads/main/data/test_05.csv",
    },
)
```

Si se prefiere usar una copia local, el script acepta `--dataset_path`, apuntando al repo clonado o directamente a su carpeta `data`.

**Columnas relevantes**

| Columna | Uso |
|---|---|
| `text` | Tweet/comentario principal. Es el input textual por defecto. |
| `title` | Titulo/noticia asociada. Solo se usa si `--use_context True`. |
| `context_tweet` | Contexto o tweet/noticia fuente. Solo se usa si `--use_context True`. |
| `HATEFUL` | Label binario oficial usado por la tarea: `0=no_hate`, `1=hate`. |
| `CALLS`, `WOMEN`, `LGBTI`, `RACISM`, `CLASS`, `POLITICS`, `DISABLED`, `APPEARANCE`, `CRIMINAL` | Categorias auxiliares del dataset. No son la etiqueta principal en esta tarea; sirven para auditoria/analisis. |

**Labels**

| ID | Label |
|---:|---|
| 0 | `no_hate` |
| 1 | `hate` |

**Low-resource**

La tarea nueva se llama:

```bash
--task_name rioplatense_hate_binary
```

Para imitar el setup low-resource del paper principal:

1. Se toma el split sintético `train`, formado por `test_01.csv`, `test_02.csv` y `test_03.csv`.
2. Se seleccionan `--max_train_samples 1000` ejemplos de forma estratificada por `HATEFUL`, usando `--low_resource_data_seed`.
3. Si `--train_as_val True`, se reserva una validación interna de 1.000 ejemplos estratificados desde el resto del train original, sin overlap con los 1.000 ejemplos de entrenamiento.
4. El split `test` oficial del script queda como `test_05.csv` y puede evaluarse con `--do_predict`.

**Métricas**

La clase positiva es `hate`. El script guarda:

- `accuracy`
- `macro_f1`
- `precision` para `hate`
- `recall` para `hate`
- `f1` / `f1_hate`
- `f1_no_hate`

Se recomienda:

```bash
--metric_for_best_model macro_f1
--greater_is_better True
```

porque la clase `hate` puede estar desbalanceada.

**Archivos extra guardados**

Además de los archivos estándar del `Trainer`, la tarea guarda:

| Archivo | Contenido |
|---|---|
| `class_distribution.json` | Distribución `no_hate`/`hate` en train, validación y test. |
| `model_init_check.json` | Check inicial con `task_name`, checkpoint, labels, tamaños de splits y configuración low-resource. |
| `eval_results.json` | Métricas finales de validación. |
| `test_results.json` | Métricas de test si se usa `--do_predict`. |
| `DESCRIPCION_CORRIDA.md` | Resumen generado por `run_step9_rioplatense_hate_binary_low_resource.sh` con arquitectura de referencia, hiperparámetros, rutas de artefactos y métricas principales. |

**Organización de outputs del script**

Por defecto, `run_step9_rioplatense_hate_binary_low_resource.sh` guarda cada corrida junto a la carpeta de la arquitectura usada:

| Variante | Carpeta default |
|---|---|
| `beto_base` | `results/paso_9/rioplatense_hate_binary_low_resource/beto_base/k1000/seed_111` |
| `beto_full` | `Pasos/Paso 7 Preentrenamiento BETO Full/downstream/rioplatense_hate_binary_low_resource/k1000/seed_111` |
| `scanpath_full_8ep` | `Pasos/paso_7_scanpath_full_8ep/downstream/rioplatense_hate_binary_low_resource/k1000/seed_111` |

Si se define `OUTPUT_ROOT`, el script mantiene el layout alternativo `OUTPUT_ROOT/<variante>/k<MAX_TRAIN_SAMPLES>/seed_<SEED>`.

**Script final Paso 7 + Paso 9**

Para correr el flujo completo con preentrenamiento MLM de Paso 7 y fine-tuning downstream de Paso 9:

```bash
./run_all_paso7_paso9_xnli_rioplatense.sh
```

El script ejecuta:

1. Paso 7 MLM half-data BETO baseline.
2. Paso 7 MLM half-data BETO + Scanpath.
3. Paso 9 `xnli_es` para `beto_base`, `beto_pretrained` y `scanpath_pretrained`.
4. Paso 9 `rioplatense_hate_binary` para `beto_base`, `beto_pretrained` y `scanpath_pretrained`.

Por defecto, Paso 9 usa los checkpoints half-data recién generados. Para usar los checkpoints full existentes:

```bash
STEP9_CHECKPOINT_SOURCE=full ./run_all_paso7_paso9_xnli_rioplatense.sh
```

Cada fine-tuning genera `trainer_loss_curves.csv` y gráficos en `loss_plots/`, además de `DESCRIPCION_CORRIDA.md`.

## 4. Reglas comunes de selección low-resource

En GLUE, la selección low-resource está implementada en `train_glue_LM_baseline.py` y en las variantes gaze-supervised. La regla es:

1. Cargar el dataset completo con `load_dataset("glue", task_name)`.
2. Tokenizar el split completo.
3. Si `max_train_samples` está definido, barajar `train` con `low_resource_data_seed`.
4. Tomar los primeros `max_train_samples` para entrenamiento.
5. Si `train_as_val=True`, tomar los siguientes 1.000 ejemplos del `train` barajado como validación interna.

En español, la selección está implementada en `train_spanish_downstream_baseline.py`:

1. Cargar XNLI/es o construir InterTASS 2020 desde parquet.
2. Para XNLI/es, usar `train` oficial y `validation` oficial.
3. Para InterTASS 2020, crear `train`/`validation` con split estratificado 80/20.
4. Aplicar `cap_dataset(dataset, max_samples, seed)`: si el split supera el máximo, se baraja con `seed` y se seleccionan los primeros `max_samples`.

## 5. Comandos mínimos para inspeccionar los datasets

GLUE:

```bash
.venv/bin/python - <<'PY'
from datasets import load_dataset

for task in ["sst2", "rte"]:
    ds = load_dataset("glue", task)
    print(task, {split: len(ds[split]) for split in ds})
    print(ds["train"][0])
PY
```

XNLI español:

```bash
.venv/bin/python - <<'PY'
from datasets import load_dataset

ds = load_dataset("xnli", "es")
print({split: len(ds[split]) for split in ds})
print(ds["train"][0])
PY
```

InterTASS 2020 cacheado:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import pandas as pd

parquet_path = Path("Pasos/paso_9_serio_low_resource_k1000_seed13/cache/intertass2020_train.parquet")
df = pd.read_parquet(parquet_path)
print(df.shape)
print(df["label"].astype(str).value_counts().sort_index())
print(df.head(3))
PY
```
