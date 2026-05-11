import argparse
import csv
import json
import shlex
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple


DEFAULT_INPUT_DIR = "../reading-et/results_all_alligned"
DEFAULT_OUTPUT_DIR = "../reading-et/procesamiento_por_largo_input"
DEFAULT_TEXT_FIELD = "text"
DEFAULT_MIN_WORDS = 4


def parse_args():
    parser = argparse.ArgumentParser(
        description="Procesa scanpaths medidos eliminando ejemplos con inputs textuales demasiado cortos."
    )
    parser.add_argument("--input_dir", default=DEFAULT_INPUT_DIR, help="Carpeta original results_all_alligned.")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Carpeta donde se escriben los datos filtrados y reportes.")
    parser.add_argument("--text_field", default=DEFAULT_TEXT_FIELD, help="Campo textual usado para contar palabras.")
    parser.add_argument("--min_words", type=int, default=DEFAULT_MIN_WORDS, help="Cantidad minima de palabras reales para conservar un ejemplo.")
    return parser.parse_args()


def count_real_words(text: Any) -> int:
    if text is None:
        return 0
    tokens = str(text).strip().split()
    return sum(1 for token in tokens if any(char.isalnum() for char in token))


def read_examples(path: Path) -> Tuple[List[Dict[str, Any]], str]:
    raw_text = path.read_text(encoding="utf-8")
    stripped = raw_text.strip()
    if not stripped:
        return [], "jsonl"

    if stripped.startswith("["):
        parsed = json.loads(stripped)
        if not isinstance(parsed, list):
            raise ValueError(f"Expected a JSON list in {path}")
        return parsed, "json"

    examples = []
    for line_number, line in enumerate(raw_text.splitlines(), start=1):
        if not line.strip():
            continue
        parsed = json.loads(line)
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected one JSON object per line in {path}:{line_number}")
        examples.append(parsed)
    return examples, "jsonl"


def write_examples(path: Path, examples: List[Dict[str, Any]], input_format: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    if input_format == "json":
        path.write_text(json.dumps(examples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return

    lines = [json.dumps(example, ensure_ascii=False) for example in examples]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def safe_stats(values: List[int]) -> Dict[str, Any]:
    if not values:
        return {"min": "", "max": "", "mean": ""}
    return {
        "min": min(values),
        "max": max(values),
        "mean": round(mean(values), 4),
    }


def percent(part: int, total: int) -> float:
    return round((part / total) * 100, 4) if total else 0.0


def serialize_optional(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def discover_split_members(output_dir: Path) -> Dict[str, set]:
    reading_et_dir = output_dir.parent
    split_dir = reading_et_dir / "splits_por_cuento"
    split_members = {}
    if not split_dir.exists():
        return split_members

    for split_name in ("train", "test"):
        path = split_dir / f"split_cuentos_{split_name}.txt"
        if path.exists():
            split_members[split_name] = {
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
    return split_members


def process_dataset(input_dir: Path, output_dir: Path, text_field: str, min_words: int) -> Dict[str, Any]:
    if min_words < 1:
        raise ValueError("--min_words must be >= 1")
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    data_output_dir = output_dir / "data_filtrada"
    data_output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    removed_rows = []
    story_totals = defaultdict(lambda: {"original": 0, "kept": 0, "removed": 0})
    split_members = discover_split_members(output_dir)
    split_filtered_counts = {split_name: 0 for split_name in split_members}

    json_files = sorted(input_dir.glob("*/*.json"))
    if not json_files:
        raise ValueError(f"No JSON files found under {input_dir}")

    total_original = 0
    total_kept = 0
    total_removed = 0

    for source_path in json_files:
        story = source_path.parent.name
        relative_path = source_path.relative_to(input_dir)
        target_path = data_output_dir / relative_path
        examples, input_format = read_examples(source_path)

        kept_examples = []
        original_word_counts = []
        kept_word_counts = []

        for example in examples:
            text = example.get(text_field, "")
            n_words = count_real_words(text)
            original_word_counts.append(n_words)
            if n_words >= min_words:
                kept_examples.append(example)
                kept_word_counts.append(n_words)
            else:
                removed_rows.append(
                    {
                        "cuento": story,
                        "archivo": source_path.name,
                        "trial_id": serialize_optional(example.get("trial_id")),
                        "text": serialize_optional(text),
                        "n_words": n_words,
                        "word_id": serialize_optional(example.get("word_id")),
                        "motivo_eliminacion": f"input_menos_de_{min_words}_palabras",
                    }
                )

        write_examples(target_path, kept_examples, input_format)

        n_original = len(examples)
        n_kept = len(kept_examples)
        n_removed = n_original - n_kept
        original_stats = safe_stats(original_word_counts)
        kept_stats = safe_stats(kept_word_counts)

        summary_rows.append(
            {
                "cuento": story,
                "archivo": source_path.name,
                "n_original": n_original,
                "n_conservados": n_kept,
                "n_eliminados": n_removed,
                "porcentaje_eliminado": percent(n_removed, n_original),
                "min_words_original": original_stats["min"],
                "max_words_original": original_stats["max"],
                "mean_words_original": original_stats["mean"],
                "min_words_filtrado": kept_stats["min"],
                "max_words_filtrado": kept_stats["max"],
                "mean_words_filtrado": kept_stats["mean"],
            }
        )

        story_totals[story]["original"] += n_original
        story_totals[story]["kept"] += n_kept
        story_totals[story]["removed"] += n_removed
        total_original += n_original
        total_kept += n_kept
        total_removed += n_removed

        for split_name, stories in split_members.items():
            if story in stories:
                split_filtered_counts[split_name] += n_kept

    return {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "data_output_dir": data_output_dir,
        "text_field": text_field,
        "min_words": min_words,
        "summary_rows": summary_rows,
        "removed_rows": removed_rows,
        "story_totals": story_totals,
        "split_members": split_members,
        "split_filtered_counts": split_filtered_counts,
        "total_original": total_original,
        "total_kept": total_kept,
        "total_removed": total_removed,
    }


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_report(result: Dict[str, Any], command: str) -> str:
    lines = [
        "PROCESAMIENTO POR LARGO DE INPUT",
        "=================================",
        "",
        "Que se hizo",
        "- Se recorrio la carpeta original de scanpaths medidos.",
        "- Para cada ejemplo se leyo el campo textual configurado.",
        "- Se conservaron solo ejemplos cuyo input textual tiene suficientes palabras reales.",
        "- Se escribio una copia filtrada preservando cuentos, nombres de archivos y campos originales.",
        "",
        "Regla aplicada",
        f"- Campo usado: {result['text_field']}",
        f"- min_words: {result['min_words']}",
        f"- Se conserva un ejemplo si n_words >= {result['min_words']}.",
        f"- Se elimina un ejemplo si n_words < {result['min_words']}.",
        "- Conteo de palabras: str(text).strip().split() y se cuentan solo tokens con al menos un caracter alfanumerico.",
        "",
        "Por que se eliminan inputs demasiado cortos",
        "- Inputs de muy pocas palabras no representan una oracion o contexto textual suficiente para una tarea MLM interpretable.",
        "- Tambien pueden inflar solapamientos triviales entre train/eval, por ejemplo palabras aisladas repetidas.",
        "- Este paso solo limpia por largo; no modifica modelo, loss, arquitectura ni split train/test.",
        "",
        "Totales",
        f"- ejemplos originales: {result['total_original']}",
        f"- ejemplos conservados: {result['total_kept']}",
        f"- ejemplos eliminados: {result['total_removed']}",
        f"- porcentaje eliminado: {percent(result['total_removed'], result['total_original'])}%",
        "",
        "Resumen por cuento",
    ]

    for story, totals in sorted(result["story_totals"].items()):
        lines.append(
            f"- {story}: original={totals['original']}, conservados={totals['kept']}, "
            f"eliminados={totals['removed']}, porcentaje_eliminado={percent(totals['removed'], totals['original'])}%"
        )

    lines.extend(["", "Ejemplos concretos eliminados"])
    for row in result["removed_rows"][:20]:
        lines.append(
            f"- cuento={row['cuento']} archivo={row['archivo']} trial_id={row['trial_id']} "
            f"n_words={row['n_words']} text={row['text']!r}"
        )
    if not result["removed_rows"]:
        lines.append("- No se eliminaron ejemplos.")

    lines.extend(["", "Relacion opcional con split por cuento"])
    if result["split_members"]:
        for split_name, count in sorted(result["split_filtered_counts"].items()):
            n_stories = len(result["split_members"].get(split_name, set()))
            lines.append(f"- {split_name}: cuentos={n_stories}, ejemplos_filtrados={count}")
        lines.append("- Esta seccion es solo informativa; no se creo ni modifico ningun split.")
    else:
        lines.append("- No se encontro reading-et/splits_por_cuento/. No se calculo resumen Train/Test.")

    lines.extend(
        [
            "",
            "Archivos originales",
            "- Los JSON originales no fueron modificados, borrados ni movidos.",
            "",
            "Comando ejecutado",
            f"- {command}",
            "",
            "Carpetas",
            f"- entrada: {result['input_dir']}",
            f"- salida: {result['output_dir']}",
            f"- datos filtrados: {result['data_output_dir']}",
        ]
    )
    return "\n".join(lines) + "\n"


def build_commands_file(result: Dict[str, Any], command: str, script_name: str) -> str:
    return f"""COMANDOS Y FUNCIONES - PROCESAMIENTO POR LARGO DE INPUT
========================================================

Comando ejecutado
- {command}

Script usado
- {script_name}

Funciones principales
- parse_args
- count_real_words
- read_examples
- write_examples
- process_dataset
- write_csv
- build_report
- build_commands_file

Carpeta de entrada
- {result['input_dir']}

Carpeta de salida
- {result['output_dir']}

Archivos generados
- {result['data_output_dir']}/
- {result['output_dir'] / 'resumen_procesamiento_largo.csv'}
- {result['output_dir'] / 'ejemplos_eliminados.csv'}
- {result['output_dir'] / 'informe_procesamiento_por_largo_input.txt'}
- {result['output_dir'] / 'comandos_y_funciones.txt'}

Regla de conteo
- Se usa el campo {result['text_field']!r}.
- Se divide con str(text).strip().split().
- Se ignoran tokens sin caracteres alfanumericos.
- Se conserva si n_words >= {result['min_words']}.
"""


def main():
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    command = " ".join(shlex.quote(part) for part in [sys.executable, Path(__file__).name, *sys.argv[1:]])

    result = process_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        text_field=args.text_field,
        min_words=args.min_words,
    )

    write_csv(
        output_dir / "resumen_procesamiento_largo.csv",
        result["summary_rows"],
        [
            "cuento",
            "archivo",
            "n_original",
            "n_conservados",
            "n_eliminados",
            "porcentaje_eliminado",
            "min_words_original",
            "max_words_original",
            "mean_words_original",
            "min_words_filtrado",
            "max_words_filtrado",
            "mean_words_filtrado",
        ],
    )
    write_csv(
        output_dir / "ejemplos_eliminados.csv",
        result["removed_rows"],
        [
            "cuento",
            "archivo",
            "trial_id",
            "text",
            "n_words",
            "word_id",
            "motivo_eliminacion",
        ],
    )
    (output_dir / "informe_procesamiento_por_largo_input.txt").write_text(
        build_report(result, command),
        encoding="utf-8",
    )
    (output_dir / "comandos_y_funciones.txt").write_text(
        build_commands_file(result, command, Path(__file__).name),
        encoding="utf-8",
    )

    print(f"Total original: {result['total_original']}")
    print(f"Total conservado: {result['total_kept']}")
    print(f"Total eliminado: {result['total_removed']}")
    print(f"Porcentaje eliminado: {percent(result['total_removed'], result['total_original'])}%")
    print(f"Wrote {result['data_output_dir']}")
    print(f"Wrote {output_dir / 'resumen_procesamiento_largo.csv'}")
    print(f"Wrote {output_dir / 'ejemplos_eliminados.csv'}")
    print(f"Wrote {output_dir / 'informe_procesamiento_por_largo_input.txt'}")
    print(f"Wrote {output_dir / 'comandos_y_funciones.txt'}")


if __name__ == "__main__":
    main()
