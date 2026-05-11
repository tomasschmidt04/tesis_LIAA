# reading-et — Context para proyectos vinculados

Este documento describe la estructura y los archivos clave del repositorio `reading-et`, orientado a dar contexto rápido a proyectos que consumen sus salidas (e.g., `ACL-GazeSupervisedLM`).

Para documentación completa del experimento y metodología, ver [README.md](README.md).

---

## Qué es este repo

Corpus de eye-tracking durante lectura natural en español. Publicado como:

> **Cuentos: A Large-Scale Eye-Tracking Reading Corpus on Spanish Narrative Texts**
> Sci Data (2026) · doi: [10.1038/s41597-026-06798-z](https://doi.org/10.1038/s41597-026-06798-z)

- **20 textos narrativos** en español rioplatense (11 cortos ~795 palabras, 10 largos ~3300 palabras)
- **113 participantes** nativos del español (estudio primario)
- **1015 participantes** en total (dataset consolidado `aligned_scanpaths.jsonl`)
- **65,671 ejemplos** alineados, todos de calidad `"high"`
- **48,148 ejemplos** limpios (segmentos ≥ 4 palabras), usados para train/test del pipeline BETO

---

## Archivos y directorios importantes

```
reading-et/
├── README.md                          # Documentación completa del corpus y metodología
├── CONTEXT.md                         # Este archivo
│
├── stimuli/                           # 21 archivos .mat con estímulos MATLAB (una por historia + Test)
├── texts/                             # Textos planos de las historias
├── metadata/
│   ├── stimuli_config.mat             # Configuración de pantalla (resolución, fuente, márgenes)
│   ├── stimuli_order.mat              # Orden pseudo-aleatorio de estímulos por participante
│   ├── stimuli_questions.mat          # Preguntas de comprensión por historia
│   └── texts_properties/
│       ├── words_freq.csv             # Frecuencias léxicas (base EsPal, subtítulos latinoamericanos)
│       └── texts_properties.json      # Estadísticas por texto (largo, complejidad)
│
├── scripts/
│   ├── data_processing/
│   │   ├── parse.py                   # Extracción de datos crudos desde archivos EDF (EyeLink)
│   │   ├── assign_fix_to_words.py     # Asignación de fijaciones a palabras
│   │   └── extract_measures.py        # Cómputo de métricas de lectura (ver tabla abajo)
│   └── experiment/                    # Código MATLAB del experimento (referencia)
│
├── build_scanpaths_from_trials.py     # Reconstruye scanpaths por sujeto y historia desde trials procesados
├── build_scanpath_alignment.py        # Alinea scanpaths con texto completo, produce outputs para BETO
├── export_scanpaths_table.py          # Exporta JSONs de scanpaths a tablas CSV planas
├── em_analysis.py                     # Análisis principal: métricas ET + modelos mixtos (LMM con R/pymer4)
├── edit_trial.py                      # GUI de curación manual de trials
│
├── aligned_output/
│   ├── aligned_scanpaths.jsonl        # ★ Dataset consolidado: 65,671 ejemplos (formato JSONL)
│   ├── alignment_summary.json         # Resumen: 20 historias, 1015 participantes, 65671 ejemplos
│   └── alignment_issues.jsonl         # Problemas de alineación (vacío: 0 issues)
│
├── results_all_alligned_2_limpio_alineado/
│   └── <historia>/
│       └── sub-NNN.json               # ★ Un JSON por sujeto y historia, con features ET alineadas
│
├── procesamiento_por_largo_input/     # Filtrado: ≥4 palabras → 48,148 de 65,671 ejemplos
├── procesamiento_por_largo_input_con_fijaciones/  # Ídem con datos de fijaciones incluidos
│
├── mlm_dataset_limpio_train_test/
│   ├── train.jsonl                    # 38,518 ejemplos (15 historias)
│   └── test.jsonl                     # 9,630 ejemplos (5 historias)
│
└── artifacts/
    ├── aligned_output.zip             # Archivo comprimido de aligned_output/ (para reproducibilidad)
    └── results_all_alligned.zip       # Archivo comprimido de los JSONs por sujeto/historia
```

---

## Métricas de eye-tracking

Todas las métricas se computan **por palabra**. Las columnas de `feature_names` siguen este orden:

| Nombre | Descripción |
|--------|-------------|
| FFD    | First Fixation Duration — duración de la primera fijación en la palabra |
| SFD    | Single Fixation Duration — FFD cuando la palabra recibe exactamente una fijación en primer pase |
| FPRT   | First Pass Reading Time (Gaze Duration) — suma de fijaciones en primer pase |
| RPD    | Regression Path Duration — tiempo desde la primera fijación hasta salir de la región por derecha |
| TFD    | Total Fixation Duration — suma de todas las fijaciones sobre la palabra |
| RRT    | Re-Reading Time — TFD - FPRT |
| SPRT   | Second Pass Reading Time — suma de fijaciones en el segundo pase |
| FC     | Fixation Count — número total de fijaciones |
| RC     | Regression Count — número de regresiones hacia la palabra |
| LS     | Likelihood of Skipping — proporción de participantes que saltaron la palabra |
| RR     | Regression Rate — proporción de participantes que regresaron a la palabra |

Valores `0` en FFD/SFD/FPRT/RPD/TFD/RRT/SPRT/FC/RC indican que la palabra fue saltada o no recibió fijaciones en ese pase. `NaN` en LS/RR indica que el valor no está disponible (primera/última palabra de línea, o adyacente a puntuación).

---

## Formato del dataset

### 1. Dataset consolidado — `aligned_output/aligned_scanpaths.jsonl`

Un ejemplo por línea (JSONL). Cada ejemplo corresponde a **un segmento de historia leído por un participante**.

```json
{
  "story_id": "Ahora debería reírme, si no estuviera muerto",
  "participant_id": "sub-001",
  "trial_id": "Ahora debería reírme, si no estuviera muerto::sub-001::seg_0000",
  "segment_index": 0,

  "text": "hubo dos mujeres casadas que se trenzaron en una disputa para ver cuál de las dos tenía un marido más imbécil.",
  "text_tokens": ["hubo", "dos", "mujeres", "casadas", "que", "se", "trenzaron", "en", "una",
                  "disputa", "para", "ver", "cuál", "de", "las", "dos", "tenía", "un", "marido", "más", "imbécil."],

  "scanpath_tokens": ["hubo", "dos", "dos", "mujeres", "casadas", "trenzaron", "trenzaron",
                      "trenzaron", "trenzaron", "disputa", "disputa", "para", "cuál", "de",
                      "cuál", "las", "dos", "tenía", "marido", "imbécil."],

  "aligned_scanpath_tokens": ["hubo", "dos", "dos", ...],

  "word_id": [1, 2, 2, 3, 4, 7, 7, 7, 7, 10, 10, 11, 13, 14, 13, 15, 16, 17, 19, 21],

  "match_quality": "high",
  "coverage": 1.0,

  "n_scanpath_tokens": 20,
  "n_aligned_tokens": 20,

  "global_word_start": 2,
  "global_word_end": 22,

  "source_text": "hubo dos dos mujeres casadas trenzaron ...",
  "source_vs_full_quality": "high"
}
```

**Notas clave:**
- `text_tokens` — palabras del texto original (21 tokens en este ejemplo)
- `scanpath_tokens` — secuencia de fijaciones del lector (puede repetir palabras por re-lecturas, o saltarse palabras)
- `word_id` — índice (1-based) de la palabra en `text_tokens` a la que corresponde cada fijación
- `global_word_start/end` — posición de este segmento dentro del texto completo de la historia

---

### 2. JSONs por sujeto — `results_all_alligned_2_limpio_alineado/<historia>/sub-NNN.json`

Versión enriquecida con **features de eye-tracking** alineadas al texto. Un archivo por (sujeto × historia), con múltiples segmentos embebidos.

```json
{
  "scanpath_text": "hubo dos dos mujeres casadas trenzaron trenzaron trenzaron trenzaron disputa disputa para cuál de cuál las dos tenía marido imbécil.",
  "text": "hubo dos mujeres casadas que se trenzaron en una disputa para ver cuál de las dos tenía un marido más imbécil.",
  "trial_id": "Ahora debería reírme, si no estuviera muerto::sub-001::seg_0000",
  "segment_index": 0,
  "match_quality": "high",
  "coverage": 1.0,

  "word_id": [1, 2, 2, 3, 4, 7, 7, 7, 7, 10, 10, 11, 13, 14, 13, 15, 16, 17, 19, 21],
  "scanpath_tokens": ["hubo", "dos", "dos", "mujeres", "casadas", "trenzaron", ...],
  "text_tokens":     ["hubo", "dos", "mujeres", "casadas", "que", "se", "trenzaron", ...],

  "num_text_tokens": 21,
  "num_fixations": 20,

  "feature_names": ["FFD", "SFD", "FPRT", "RPD", "TFD", "RRT", "SPRT", "FC", "RC", "LS", "RR"],

  "reading_features_by_word": [
    [319, 319, 319, 319, 319, 0, 0, 1, 0, 0.472, 0.075],
    [280,   0, 364, 364, 364, 0, 0, 2, 0, 0.491, 0.208],
    ...
  ],

  "reading_features_by_fixation": [
    [319, 319, 319, 319, 319, 0, 0, 1, 0, 0.472, 0.075],
    [280,   0, 364, 364, 364, 0, 0, 2, 0, 0.491, 0.208],
    [280,   0, 364, 364, 364, 0, 0, 2, 0, 0.491, 0.208],
    ...
  ],

  "reading_features_mask_by_word":     [[1,1,1,1,1,1,1,1,1,1,1], ...],
  "reading_features_mask_by_fixation": [[1,1,1,1,1,1,1,1,1,1,1], ...]
}
```

**Diferencias clave respecto al JSONL consolidado:**
- `reading_features_by_word` — shape `[num_text_tokens, 11]`: una fila por palabra del texto
- `reading_features_by_fixation` — shape `[num_fixations, 20]`: una fila por fijación del scanpath (repite las features de la palabra fijada)
- `reading_features_mask_by_word/fixation` — máscara binaria: `0` donde el valor es `NaN` (LS/RR no disponibles para primera/última palabra de línea)

---

## Qué usa cada paso del pipeline BETO

| Paso BETO | Entrada desde reading-et |
|-----------|--------------------------|
| step 4, 5, 6 (debug / smoke training) | `results_all_alligned_2_limpio_alineado/<historia>/sub-NNN.json` |
| step 7b (pretraining a escala) | `aligned_output/aligned_scanpaths.jsonl` |
| step 9 (downstream / evaluación) | `mlm_dataset_limpio_train_test/train.jsonl` + `test.jsonl` |

---

## Cómo restaurar los archivos comprimidos

Los directorios procesados grandes están versionados como archivos comprimidos en `artifacts/`. Para extraerlos:

```bash
# Linux
unzip artifacts/aligned_output.zip -d .
unzip artifacts/results_all_alligned.zip -d .
```

```powershell
# Windows
Expand-Archive -LiteralPath .\artifacts\aligned_output.zip -DestinationPath . -Force
Expand-Archive -LiteralPath .\artifacts\results_all_alligned.zip -DestinationPath . -Force
```

---

## Cita

```bibtex
@article{travi_cuentos_2026,
  title   = {Cuentos: A Large-Scale Eye-Tracking Reading Corpus on Spanish Narrative Texts},
  author  = {Travi, Fermin and Bianchi, Bruno and Slezak, Diego Fernandez and Kamienkowski, Juan E.},
  journal = {Scientific Data},
  year    = {2026},
  doi     = {10.1038/s41597-026-06798-z}
}
```
