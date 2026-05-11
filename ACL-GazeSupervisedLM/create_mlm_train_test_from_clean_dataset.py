import argparse
import csv
import json
import random
import shlex
import sys
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_INPUT_DIR = "../reading-et/procesamiento_por_largo_input/data_filtrada"
DEFAULT_OUTPUT_DIR = "../reading-et/mlm_dataset_limpio_train_test"
DEFAULT_TRAIN_RATIO = 0.8
DEFAULT_SEED = 13
DEFAULT_TEXT_FIELD = "text"
DEFAULT_WORD_ID_FIELD = "word_id"
MIN_WORDS_ASSUMED = 4

METADATA_COLUMNS = [
    "split",
    "cuento",
    "archivo",
    "trial_id",
    "text",
    "n_words",
    "n_scanpath_positions",
]

SUMMARY_COLUMNS = [
    "cuento",
    "split",
    "n_json_files",
    "n_trials",
    "porcentaje_trials_total",
    "n_words_total",
    "n_words_mean",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Crea train/test MLM por cuentos desde el dataset ya limpio por largo de input."
    )
    parser.add_argument("--input_dir", default=DEFAULT_INPUT_DIR, help="Carpeta data_filtrada ya limpia.")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Carpeta donde se escriben train/test JSONL y reportes.")
    parser.add_argument("--train_ratio", type=float, default=DEFAULT_TRAIN_RATIO, help="Proporcion objetivo aproximada para Train.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Seed usada para desempates reproducibles.")
    parser.add_argument("--text_field", default=DEFAULT_TEXT_FIELD, help="Campo textual usado para validar y contar palabras.")
    parser.add_argument("--word_id_field", default=DEFAULT_WORD_ID_FIELD, help="Campo word_id que debe preservarse si existe.")
    return parser.parse_args()


def count_real_words(text: Any) -> int:
    if text is None:
        return 0
    tokens = str(text).strip().split()
    return sum(1 for token in tokens if any(char.isalnum() for char in token))


def count_scanpath_positions(example: Dict[str, Any], word_id_field: str) -> Optional[int]:
    word_id_value = example.get(word_id_field)
    if isinstance(word_id_value, list):
        return len(word_id_value)
    if isinstance(word_id_value, tuple):
        return len(word_id_value)
    if isinstance(word_id_value, str) and word_id_value.strip():
        try:
            parsed = json.loads(word_id_value)
            if isinstance(parsed, list):
                return len(parsed)
        except json.JSONDecodeError:
            return len([part for part in word_id_value.replace(",", " ").split() if part])

    scanpath_tokens = example.get("scanpath_tokens")
    if isinstance(scanpath_tokens, list):
        return len(scanpath_tokens)

    scanpath_text = example.get("scanpath_text")
    if scanpath_text is not None:
        return count_real_words(scanpath_text)

    return None


def read_examples(path: Path) -> List[Dict[str, Any]]:
    raw_text = path.read_text(encoding="utf-8")
    stripped = raw_text.strip()
    if not stripped:
        return []

    if stripped.startswith("["):
        parsed = json.loads(stripped)
        if not isinstance(parsed, list):
            raise ValueError(f"Expected a JSON list in {path}")
        if not all(isinstance(example, dict) for example in parsed):
            raise ValueError(f"Expected every JSON list item to be an object in {path}")
        return parsed

    if "\n" not in stripped:
        parsed = json.loads(stripped)
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected a JSON object in {path}")
        return [parsed]

    examples = []
    for line_number, line in enumerate(raw_text.splitlines(), start=1):
        if not line.strip():
            continue
        parsed = json.loads(line)
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected one JSON object per line in {path}:{line_number}")
        examples.append(parsed)
    return examples


def percent(part: int, total: int) -> float:
    return round((part / total) * 100, 4) if total else 0.0


def optional_csv_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def discover_clean_dataset(input_dir: Path, text_field: str, word_id_field: str) -> Dict[str, Any]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")

    story_dirs = sorted(path for path in input_dir.iterdir() if path.is_dir())
    if not story_dirs:
        raise ValueError(f"No story folders found under {input_dir}")

    stories: Dict[str, Dict[str, Any]] = {}
    validation_errors: List[str] = []
    total_trials = 0
    total_words = 0
    total_json_files = 0
    word_id_seen = False
    scanpath_text_seen = False

    for story_dir in story_dirs:
        story = story_dir.name
        json_files = sorted(story_dir.glob("*.json"))
        total_json_files += len(json_files)
        story_record = {
            "cuento": story,
            "json_files": json_files,
            "examples": [],
            "n_trials": 0,
            "n_words_total": 0,
            "word_counts": [],
        }

        for source_path in json_files:
            examples = read_examples(source_path)
            for example_index, example in enumerate(examples):
                if text_field not in example:
                    validation_errors.append(
                        f"Missing text field {text_field!r}: cuento={story}, archivo={source_path.name}, index={example_index}"
                    )
                    continue

                n_words = count_real_words(example[text_field])
                if n_words < MIN_WORDS_ASSUMED:
                    validation_errors.append(
                        f"Example has n_words={n_words} < {MIN_WORDS_ASSUMED}: cuento={story}, archivo={source_path.name}, "
                        f"trial_id={example.get('trial_id')!r}"
                    )

                if word_id_field in example:
                    word_id_seen = True
                if "scanpath_text" in example:
                    scanpath_text_seen = True

                story_record["examples"].append(
                    {
                        "example": example,
                        "cuento": story,
                        "source_file": source_path.name,
                        "n_words": n_words,
                        "n_scanpath_positions": count_scanpath_positions(example, word_id_field),
                    }
                )
                story_record["n_trials"] += 1
                story_record["n_words_total"] += n_words
                story_record["word_counts"].append(n_words)
                total_trials += 1
                total_words += n_words

        stories[story] = story_record

    if validation_errors:
        preview = "\n".join(f"- {error}" for error in validation_errors[:20])
        raise ValueError(
            "Dataset limpio invalido para crear Train/Test MLM. "
            f"Se encontraron {len(validation_errors)} problemas:\n{preview}"
        )
    if total_trials == 0:
        raise ValueError(f"No examples found under {input_dir}")
    return {
        "stories": stories,
        "total_trials": total_trials,
        "total_words": total_words,
        "total_json_files": total_json_files,
        "word_id_seen": word_id_seen,
        "scanpath_text_seen": scanpath_text_seen,
    }


def choose_test_stories(stories: Dict[str, Dict[str, Any]], train_ratio: float, seed: int) -> Tuple[List[str], List[str], Dict[str, Any]]:
    if not 0 < train_ratio < 1:
        raise ValueError("--train_ratio must be between 0 and 1.")

    positive_stories = [story for story, row in stories.items() if row["n_trials"] > 0]
    if len(positive_stories) < 2:
        raise ValueError("Need at least two non-empty stories to create a story-level Train/Test split.")

    total_trials = sum(stories[story]["n_trials"] for story in positive_stories)
    target_test_trials = total_trials * (1.0 - train_ratio)
    rng = random.Random(seed)
    tie_rank = {story: rng.random() for story in positive_stories}

    best_subset: Optional[Tuple[str, ...]] = None
    best_key: Optional[Tuple[float, int, float]] = None

    if len(positive_stories) <= 24:
        for subset_size in range(1, len(positive_stories)):
            for subset in combinations(positive_stories, subset_size):
                test_trials = sum(stories[story]["n_trials"] for story in subset)
                train_trials = total_trials - test_trials
                if train_trials <= 0 or test_trials <= 0:
                    continue
                tie_value = sum(tie_rank[story] for story in subset)
                key = (abs(test_trials - target_test_trials), test_trials, tie_value)
                if best_key is None or key < best_key:
                    best_key = key
                    best_subset = subset
    else:
        shuffled_stories = positive_stories[:]
        rng.shuffle(shuffled_stories)
        running_subset: List[str] = []
        running_count = 0
        for story in sorted(shuffled_stories, key=lambda item: stories[item]["n_trials"], reverse=True):
            candidate_count = running_count + stories[story]["n_trials"]
            if abs(candidate_count - target_test_trials) <= abs(running_count - target_test_trials) or not running_subset:
                running_subset.append(story)
                running_count = candidate_count
        if len(running_subset) == len(positive_stories):
            running_subset.pop()
        best_subset = tuple(running_subset)
        best_key = (abs(running_count - target_test_trials), running_count, 0.0)

    if not best_subset:
        raise ValueError("Could not create a non-empty Test split.")

    test_story_set = set(best_subset)
    train_story_set = {story for story in stories if story not in test_story_set}
    if not train_story_set:
        raise ValueError("Could not create a non-empty Train split.")

    train_stories = sorted(train_story_set)
    test_stories = sorted(test_story_set)
    train_trials = sum(stories[story]["n_trials"] for story in train_stories)
    test_trials = sum(stories[story]["n_trials"] for story in test_stories)

    split_info = {
        "target_train_ratio": train_ratio,
        "target_test_ratio": 1.0 - train_ratio,
        "target_test_trials": target_test_trials,
        "train_trials": train_trials,
        "test_trials": test_trials,
        "train_trial_percentage": percent(train_trials, train_trials + test_trials),
        "test_trial_percentage": percent(test_trials, train_trials + test_trials),
        "test_objective_abs_error_trials": best_key[0] if best_key else None,
        "tie_break_rule": "closest to target test trials; if tied, choose fewer test trials to leave more data in Train; final ties use seed",
    }
    return train_stories, test_stories, split_info


def build_augmented_example(source_item: Dict[str, Any], split: str) -> Dict[str, Any]:
    original = source_item["example"]
    augmented = dict(original)
    augmented["cuento"] = source_item["cuento"]
    augmented["source_file"] = source_item["source_file"]
    augmented["split"] = split
    augmented["n_words"] = source_item["n_words"]
    return augmented


def build_metadata_row(source_item: Dict[str, Any], split: str, text_field: str) -> Dict[str, Any]:
    example = source_item["example"]
    return {
        "split": split,
        "cuento": source_item["cuento"],
        "archivo": source_item["source_file"],
        "trial_id": optional_csv_value(example.get("trial_id")),
        "text": optional_csv_value(example.get(text_field)),
        "n_words": source_item["n_words"],
        "n_scanpath_positions": optional_csv_value(source_item.get("n_scanpath_positions")),
    }


def write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_split_outputs(
    dataset: Dict[str, Any],
    train_stories: List[str],
    test_stories: List[str],
    text_field: str,
) -> Dict[str, Any]:
    train_story_set = set(train_stories)
    test_story_set = set(test_stories)
    train_rows: List[Dict[str, Any]] = []
    test_rows: List[Dict[str, Any]] = []
    train_metadata: List[Dict[str, Any]] = []
    test_metadata: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for story, story_record in sorted(dataset["stories"].items()):
        if story in train_story_set:
            split = "train"
        elif story in test_story_set:
            split = "test"
        else:
            raise ValueError(f"Story {story!r} was not assigned to any split.")

        target_rows = train_rows if split == "train" else test_rows
        target_metadata = train_metadata if split == "train" else test_metadata

        for source_item in story_record["examples"]:
            target_rows.append(build_augmented_example(source_item, split))
            target_metadata.append(build_metadata_row(source_item, split, text_field))

        n_trials = story_record["n_trials"]
        n_words_mean = round(mean(story_record["word_counts"]), 4) if story_record["word_counts"] else ""
        summary_rows.append(
            {
                "cuento": story,
                "split": split,
                "n_json_files": len(story_record["json_files"]),
                "n_trials": n_trials,
                "porcentaje_trials_total": percent(n_trials, dataset["total_trials"]),
                "n_words_total": story_record["n_words_total"],
                "n_words_mean": n_words_mean,
            }
        )

    return {
        "train_rows": train_rows,
        "test_rows": test_rows,
        "train_metadata": train_metadata,
        "test_metadata": test_metadata,
        "summary_rows": summary_rows,
    }


def validate_outputs(outputs: Dict[str, Any], train_stories: List[str], test_stories: List[str], text_field: str, word_id_field: str):
    if not outputs["train_rows"]:
        raise ValueError("Validation failed: train.jsonl would be empty.")
    if not outputs["test_rows"]:
        raise ValueError("Validation failed: test.jsonl would be empty.")

    overlap = set(train_stories) & set(test_stories)
    if overlap:
        raise ValueError(f"Validation failed: stories appear in both splits: {sorted(overlap)}")

    for split_name in ("train", "test"):
        for index, row in enumerate(outputs[f"{split_name}_rows"]):
            if text_field not in row:
                raise ValueError(f"Validation failed: missing text in {split_name} row {index}")
            if count_real_words(row[text_field]) < MIN_WORDS_ASSUMED:
                raise ValueError(f"Validation failed: row with fewer than {MIN_WORDS_ASSUMED} words in {split_name} row {index}")
            if word_id_field in row and row[word_id_field] is None:
                raise ValueError(f"Validation failed: word_id exists but is None in {split_name} row {index}")
            if "scanpath_text" in row and row["scanpath_text"] is None:
                raise ValueError(f"Validation failed: scanpath_text exists but is None in {split_name} row {index}")


def build_report(
    input_dir: Path,
    output_dir: Path,
    train_stories: List[str],
    test_stories: List[str],
    outputs: Dict[str, Any],
    split_info: Dict[str, Any],
    seed: int,
    train_ratio: float,
) -> str:
    total_trials = len(outputs["train_rows"]) + len(outputs["test_rows"])
    train_story_percentage = percent(len(train_stories), len(train_stories) + len(test_stories))
    test_story_percentage = percent(len(test_stories), len(train_stories) + len(test_stories))
    train_trial_percentage = percent(len(outputs["train_rows"]), total_trials)
    test_trial_percentage = percent(len(outputs["test_rows"]), total_trials)
    target_test_percentage = round((1.0 - train_ratio) * 100, 4)
    test_deviation = round(test_trial_percentage - target_test_percentage, 4)

    lines = [
        "MLM DATASET LIMPIO - SPLIT TRAIN/TEST POR CUENTO",
        "==================================================",
        "",
        "Que se hizo",
        "- Se leyo el dataset ya limpio por largo de input.",
        "- Se creo una division Train/Test por cuentos completos.",
        "- Cada carpeta de cuento se trato como unidad indivisible.",
        "- Se escribieron train.jsonl y test.jsonl preservando todos los campos originales y agregando cuento, source_file, split y n_words.",
        "",
        "Dataset de entrada usado",
        f"- {input_dir}",
        "",
        "Regla de limpieza asumida",
        f"- El dataset de entrada ya conserva solo ejemplos con n_words >= {MIN_WORDS_ASSUMED}.",
        "- En este paso no se volvio a limpiar; solo se valido esa condicion.",
        "- Conteo usado para validar: str(text).strip().split() y tokens con al menos un caracter alfanumerico.",
        "",
        "Como se creo el split",
        f"- train_ratio objetivo: {train_ratio}",
        f"- seed: {seed}",
        "- Se conto la cantidad de trials por cuento.",
        "- Se busco un subconjunto de cuentos para Test cercano al 20% de trials.",
        "- En empates se eligio el subconjunto con menos trials en Test para dejar mas datos en Train.",
        f"- Regla de desempate exacta: {split_info['tie_break_rule']}.",
        "",
        "Totales finales",
        f"- cuentos Train: {len(train_stories)}",
        f"- cuentos Test: {len(test_stories)}",
        f"- ejemplos Train: {len(outputs['train_rows'])}",
        f"- ejemplos Test: {len(outputs['test_rows'])}",
        f"- porcentaje real Train/Test por cuentos: {train_story_percentage}% / {test_story_percentage}%",
        f"- porcentaje real Train/Test por examples/trials: {train_trial_percentage}% / {test_trial_percentage}%",
        f"- target Test por trials: {target_test_percentage}%",
        f"- desvio Test real - Test objetivo: {test_deviation} puntos porcentuales",
        "",
        "Cuentos Train",
    ]
    lines.extend(f"- {story}" for story in train_stories)
    lines.extend(["", "Cuentos Test"])
    lines.extend(f"- {story}" for story in test_stories)
    lines.extend(
        [
            "",
            "Validaciones",
            "- train.jsonl no esta vacio.",
            "- test.jsonl no esta vacio.",
            "- No hay cuentos compartidos entre Train y Test.",
            f"- Todos los ejemplos tienen campo text y n_words >= {MIN_WORDS_ASSUMED}.",
            "- Si word_id existia, fue preservado dentro de cada JSONL.",
            "- Si scanpath_text existia, fue preservado dentro de cada JSONL.",
            f"- total de ejemplos reportado: {total_trials}",
            "",
            "Leakage por cuento",
            "- Confirmado: no hay leakage por cuento porque cada cuento aparece en un unico split.",
            "",
            "Advertencias",
        ]
    )
    if test_deviation == 0:
        lines.append("- El split quedo exactamente en el porcentaje objetivo por trials.")
    else:
        lines.append("- El split no queda 80/20 exacto porque la unidad indivisible es el cuento completo.")
        lines.append("- Se priorizo aproximar el 20% de Test y, ante empates, dejar mas datos en Train.")

    lines.extend(
        [
            "",
            "Uso futuro",
            "- A partir de ahora los modelos MLM deben usar estos archivos:",
            f"- Train: {output_dir / 'train.jsonl'}",
            f"- Test/Eval: {output_dir / 'test.jsonl'}",
            "- Esto aplica a BETO MLM baseline, BETO + rama scanpath, BETO + lambda adaptativo y futuras variantes arquitectonicas.",
            "- No se debe volver a usar el dataset original sin filtrar para estos experimentos.",
            "",
            "Archivos originales",
            "- No se modifico data_filtrada.",
            "- No se modificaron, borraron ni movieron los archivos originales.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_commands_file(command: str, script_name: str, input_dir: Path, output_dir: Path) -> str:
    return f"""COMANDOS Y FUNCIONES - SPLIT MLM LIMPIO POR CUENTOS
===================================================

Comando ejecutado
- {command}

Script usado
- {script_name}

Funciones principales
- parse_args
- count_real_words
- read_examples
- discover_clean_dataset
- choose_test_stories
- build_split_outputs
- validate_outputs
- write_jsonl
- write_csv
- build_report
- build_commands_file

Carpetas leidas
- {input_dir}

Archivos generados
- {output_dir / 'train.jsonl'}
- {output_dir / 'test.jsonl'}
- {output_dir / 'train_metadata.csv'}
- {output_dir / 'test_metadata.csv'}
- {output_dir / 'resumen_split_mlm_limpio.csv'}
- {output_dir / 'informe_mlm_dataset_limpio_train_test.txt'}
- {output_dir / 'comandos_y_funciones.txt'}
"""


def main():
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    command = " ".join(shlex.quote(part) for part in [sys.executable, Path(__file__).name, *sys.argv[1:]])

    dataset = discover_clean_dataset(
        input_dir=input_dir,
        text_field=args.text_field,
        word_id_field=args.word_id_field,
    )
    train_stories, test_stories, split_info = choose_test_stories(
        stories=dataset["stories"],
        train_ratio=args.train_ratio,
        seed=args.seed,
    )
    outputs = build_split_outputs(
        dataset=dataset,
        train_stories=train_stories,
        test_stories=test_stories,
        text_field=args.text_field,
    )
    validate_outputs(
        outputs=outputs,
        train_stories=train_stories,
        test_stories=test_stories,
        text_field=args.text_field,
        word_id_field=args.word_id_field,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "train.jsonl", outputs["train_rows"])
    write_jsonl(output_dir / "test.jsonl", outputs["test_rows"])
    write_csv(output_dir / "train_metadata.csv", outputs["train_metadata"], METADATA_COLUMNS)
    write_csv(output_dir / "test_metadata.csv", outputs["test_metadata"], METADATA_COLUMNS)
    write_csv(output_dir / "resumen_split_mlm_limpio.csv", outputs["summary_rows"], SUMMARY_COLUMNS)
    (output_dir / "informe_mlm_dataset_limpio_train_test.txt").write_text(
        build_report(
            input_dir=input_dir,
            output_dir=output_dir,
            train_stories=train_stories,
            test_stories=test_stories,
            outputs=outputs,
            split_info=split_info,
            seed=args.seed,
            train_ratio=args.train_ratio,
        ),
        encoding="utf-8",
    )
    (output_dir / "comandos_y_funciones.txt").write_text(
        build_commands_file(
            command=command,
            script_name=Path(__file__).name,
            input_dir=input_dir,
            output_dir=output_dir,
        ),
        encoding="utf-8",
    )

    total_trials = len(outputs["train_rows"]) + len(outputs["test_rows"])
    print(f"Total examples: {total_trials}")
    print(f"Train stories: {len(train_stories)}")
    print(f"Test stories: {len(test_stories)}")
    print(f"Train examples: {len(outputs['train_rows'])} ({percent(len(outputs['train_rows']), total_trials)}%)")
    print(f"Test examples: {len(outputs['test_rows'])} ({percent(len(outputs['test_rows']), total_trials)}%)")
    print("Story overlap: 0")
    print(f"Wrote {output_dir / 'train.jsonl'}")
    print(f"Wrote {output_dir / 'test.jsonl'}")
    print(f"Wrote {output_dir / 'train_metadata.csv'}")
    print(f"Wrote {output_dir / 'test_metadata.csv'}")
    print(f"Wrote {output_dir / 'resumen_split_mlm_limpio.csv'}")
    print(f"Wrote {output_dir / 'informe_mlm_dataset_limpio_train_test.txt'}")
    print(f"Wrote {output_dir / 'comandos_y_funciones.txt'}")


if __name__ == "__main__":
    main()
