# Ejemplos del dataset de preentrenamiento (scanpaths medidos)

Este archivo contiene ejemplos reales del dataset de scanpaths usados para el preentrenamiento del modelo BETO supervisado con movimientos oculares. Los ejemplos provienen del archivo `reading-et/mlm_dataset_limpio_train_test/train.jsonl`.

---

## Formato de cada ejemplo

Cada línea del JSONL es un objeto con los siguientes campos:

```
text           → oración original (input al tokenizador BETO)
word_id        → secuencia de posiciones de palabra visitadas por el lector (1-based)
scanpath_text  → reconstrucción textual del scanpath
text_tokens    → palabras del texto en orden original
scanpath_tokens → palabras visitadas en orden de fijación
trial_id       → identificador: cuento::sujeto::segmento
segment_index  → posición de la oración en el cuento (0-based)
cuento         → título del cuento
source_file    → archivo del participante (sujeto)
split          → "train" o "test"
n_words        → cantidad de palabras en la oración original
match_quality  → calidad del alineamiento texto↔scanpath ("high", "medium", ...)
coverage       → fracción de palabras del texto cubiertas por el scanpath
```

---

## Ejemplo 1 — Oración corta con refijación

**Contexto:** Inicio de un cuento sobre caminar en la nieve. Oración muy breve.

```json
{
  "text": "¡Abrigate! Mi mamá me",
  "word_id": [1, 1, 2, 2, 4],
  "text_tokens": ["¡Abrigate!", "Mi", "mamá", "me"],
  "scanpath_tokens": ["Abrigate", "¡Abrigate!", "Mi", "Mi", "me."],
  "trial_id": "Cómo funciona caminar en la nieve::sub-003::seg_0000",
  "segment_index": 0,
  "cuento": "Cómo funciona caminar en la nieve",
  "source_file": "sub-003.json",
  "split": "train",
  "n_words": 4,
  "match_quality": "high",
  "coverage": 1.0
}
```

**Interpretación de `word_id = [1, 1, 2, 2, 4]`:**
- Fijación 1 → palabra 1 ("¡Abrigate!")
- Fijación 2 → palabra 1 ("¡Abrigate!") — **refijación**: el lector volvió a esta palabra
- Fijación 3 → palabra 2 ("Mi")
- Fijación 4 → palabra 2 ("Mi") — **refijación**
- Fijación 5 → palabra 4 ("me") — skip: la palabra 3 ("mamá") fue ignorada

Lo que el modelo ve en la rama auxiliar: una secuencia de 5+2 pasos (con sentinelas CLS/SEP agregados) en lugar de los 4 tokens del texto. El GRU procesa este orden de lectura real.

---

## Ejemplo 2 — Oración mediana con skips y refijaciones

**Contexto:** Segunda oración del mismo cuento. El lector relee "tanto molestia" varias veces.

```json
{
  "text": "El invierno puede ser tanto una molestia como una aventura cada vez que salimos de casa.",
  "word_id": [1, 3, 5, 7, 6, 5, 7, 3, 5, 7, 9, 10, 12, 13, 14, 16],
  "text_tokens": ["El", "invierno", "puede", "ser", "tanto", "una", "molestia", "como", "una", "aventura", "cada", "vez", "que", "salimos", "de", "casa."],
  "scanpath_tokens": ["El", "puede", "tanto", "molestia", "una", "tanto", "molestia", "puede", "tanto", "molestia", "una", "aventura", "vez", "que", "salimos", "casa."],
  "trial_id": "Cómo funciona caminar en la nieve::sub-003::seg_0001",
  "segment_index": 1,
  "n_words": 16,
  "match_quality": "high",
  "coverage": 1.0
}
```

**Interpretación de `word_id`:**
El lector fijó las palabras en este orden: El (1) → puede (3) → tanto (5) → molestia (7) → una (6) → tanto (5) → molestia (7) → puede (3) → tanto (5) → molestia (7) → una (9) → aventura (10) → vez (12) → que (13) → salimos (14) → casa (16).

Se observan:
- **Skips hacia adelante:** El lector salta "invierno" (2) y "ser" (4) la primera vez.
- **Regresiones:** Vuelve de la palabra 7 (molestia) hacia atrás a la 6 (una), y de 7 a 3 (puede).
- **Trigramas repetidos:** "puede → tanto → molestia" aparece 3 veces en el scanpath (patrón de relectura).
- Las palabras "como" (8), "cada" (11), "de" (15) nunca son fijadas.

---

## Ejemplo 3 — Oración larga con cobertura completa

**Contexto:** Descripción del paisaje invernal. 53 palabras, 36 fijaciones.

```json
{
  "text": "En muy poco tiempo el paisaje pierde la mayoría de sus colores y si me guiara solo por mis ojos tranquilamente podría concluir que todo es parte de una película antigua en escala de grises, con los copos de nieve haciendo las veces del ruido en la imagen de las viejas proyecciones cinematográficas.",
  "word_id": [2, 4, 1, 4, 6, 7, 9, 12, 12, 16, 17, 18, 20, 21, 22, 23, 25, 27, 29, 30, 31, 33, 35, 36, 38, 40, 41, 43, 43, 45, 46, 48, 49, 51, 52, 53],
  "trial_id": "Cómo funciona caminar en la nieve::sub-003::seg_0002",
  "segment_index": 2,
  "n_words": 53,
  "match_quality": "high",
  "coverage": 1.0
}
```

**Interpretación:**
- La oración tiene 53 palabras pero el scanpath tiene solo 36 fijaciones → el lector skipea ~17 palabras.
- Las dos primeras fijaciones van a palabras 2 ("muy") y 4 ("tiempo"), pero la tercera va a la palabra 1 ("En") — **regresión al inicio de la oración**.
- La palabra 12 ("colores") aparece **dos veces** en el scanpath (posiciones 8 y 9) → refijación inmediata.
- La palabra 43 ("veces") también aparece dos veces (posiciones 28 y 29).
- Las palabras funcionales cortas (artículos, preposiciones intermedias) tienden a ser skipeadas.

---

## Ejemplo 4 — Oración con estructura de relectura extensa

**Contexto:** Oración compleja con regresiones largas.

```json
{
  "text": "Envuelto en el silencio con el que empiezan todos los poemas invernales, el crujir de mis pasos insinúa que aunque el horizonte se disuelva ya no estoy en el mismo lugar desde el que partí.",
  "word_id": [1, 3, 2, 5, 8, 9, 11, 12, 12, 14, 17, 17, 18, 20, 22, 24, 25, 27, 29, 30, 28, 31, 33, 35, 33],
  "n_words": 35,
  "match_quality": "high",
  "coverage": 1.0
}
```

**Interpretación:**
- Fijación 1→3→2: el lector salta "en" (2) para ir a "el" (3), luego retrocede a "en" — regresión de un token.
- `word_id = 12` aparece dos veces (refijación de "invernales,").
- `word_id = 17` aparece dos veces (refijación de "pasos").
- `word_id = 28` ("desde") aparece entre 30 y 31 — regresión de dos palabras hacia atrás.
- `word_id = 33` ("que") aparece dos veces — refijación al final de la oración.

---

## Ejemplo 5 — Oración muy larga con patrón de lectura no lineal marcado

**Contexto:** Descripción de un árbol en el bosque invernal. 72 palabras, 59 fijaciones.

```json
{
  "text": "Cada tanto irrumpe el sonido de una rama que ya no pudo soportar el azote del clima y cedió para llegar finalmente al suelo, donde quizá se convierta en la casita de una familia recién formada de liebres y algún día en el alimento de un retoño como el que supieron ser esos árboles imponentes que ahora me protegen de la nieve, que cae con todas las intenciones de teñirme de blanco.",
  "word_id": [1, 2, 2, 1, 2, 3, 5, 7, 9, 11, 12, 13, 15, 17, 17, 20, 19, 20, 21, 22, 23, 25, 26, 28, 31, 30, 31, 34, 34, 36, 36, 37, 39, 41, 44, 47, 45, 47, 49, 51, 51, 53, 54, 55, 56, 55, 55, 56, 57, 55, 58, 59, 61, 63, 65, 66, 68, 70, 72],
  "n_words": 72,
  "match_quality": "high",
  "coverage": 1.0
}
```

**Interpretación:**
- Los primeros 5 pasos alternan entre "Cada" (1) y "tanto" (2): el lector relee el inicio dos veces antes de avanzar.
- La palabra 17 ("azote") aparece dos veces; las palabras 19 ("cedió") y 20 ("para") se leen en orden invertido (regresión).
- Las palabras 20, 21, 22 se leen dos veces cada una en algunos casos.
- Las palabras 55 y 56 forman un bucle de relectura de 5 fijaciones al final.
- La palabra 72 ("blanco.") es la última fijación — el lector sí termina de leer la oración.

---

## Cómo se usa `word_id` en el modelo

### 1. Parseo en `build_measured_scanpath()`

```python
def build_measured_scanpath(word_id_value, sentence_word_count):
    # Parsear la lista de posiciones
    lexical_positions = _parse_word_id_sequence(word_id_value)
    # Filtrar posiciones fuera del rango de la oración
    lexical_positions = [p for p in lexical_positions if 1 <= p <= sentence_word_count]
    # Agregar sentinelas: 0 = CLS sintético, sentence_word_count+1 = SEP sintético
    measured_word_ids = [0] + lexical_positions + [sentence_word_count + 1]
    measured_sp_len = len(measured_word_ids)
    return measured_word_ids, measured_sp_len
```

Para el Ejemplo 1 con `n_words=4` y `word_id=[1,1,2,2,4]`:
```
measured_word_ids = [0, 1, 1, 2, 2, 4, 5]   # longitud 7 (5 fijaciones + 2 sentinelas)
measured_sp_len   = 7
```

### 2. Conversión palabra → token en el modelo

El tokenizador BETO es de subpalabras (WordPiece). Una sola palabra como "cinematográficas" puede generar varios tokens. La función `convert_word_pos_seq_to_token_pos_seq()` del `SP_Encoder` convierte cada posición de palabra léxica a la posición del **primer subtoken** correspondiente en la secuencia tokenizada.

Para el Ejemplo 1:
```
text_tokens = ["¡Abrigate!", "Mi", "mamá", "me"]

Tokenización BETO:
  [CLS] ¡ ##Ab ##rigate ! Mi mamá me [SEP]
  pos:   0  1    2       3  4   5   6   7

word_ids (HuggingFace):  [None, 0, 0, 0, 1, 2, 3, None]
→ lm_word_ids (ajustado): [ 0,  1, 1, 1, 2, 3, 4,  5 ]

word_id secuencia medida: [0, 1, 1, 2, 2, 4, 5]
→ gaze_token_pos: mapeado al primer subtoken de cada palabra
   0 → 0 (CLS), 1 → 1 (primer subtoken de "¡Abrigate!"), 2 → 4 ("Mi"), 4 → 6 ("me"), 5 → 7 (SEP)
```

### 3. Construcción de la pérdida auxiliar

Si el label original (post-masking) es `labels = [-100, -100, 42, -100, -100, -100, -100, -100]` (solo el token "¡Ab" fue enmascarado, original_id=42):

```python
scanpath_labels_expanded = labels.gather(1, gaze_token_pos.long())
# → [-100, 42, 42, -100, -100, 42, -100]
# La posición del token 1 aparece 3 veces en gaze_token_pos → contribuye 3 veces a la pérdida
```

---

## Estadísticas del dataset completo

| Partición | Ejemplos | Cuentos | Batches (batch_size=4) |
|-----------|----------|---------|------------------------|
| train     | 38 518   | —       | 9 630                  |
| test      | 9 630    | —       | 2 408                  |
| **total** | **48 148** | — | **12 038**           |

El split fue calculado **por cuento** para evitar data leakage: todas las oraciones de un mismo cuento están en el mismo split. La validación del split se hace automáticamente al inicio de cada corrida y falla si detecta solapamiento de textos exactos entre train y eval.
