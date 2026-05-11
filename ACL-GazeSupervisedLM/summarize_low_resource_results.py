import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_SEARCH_DIRS = ["Pasos", "result", "."]
DEFAULT_OUTPUT_DIR = "Pasos/resumen_low_resource"
TASK_NAMES = {"xnli_es", "intertass2020"}


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize low-resource downstream results.")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--search_dirs", nargs="*", default=DEFAULT_SEARCH_DIRS)
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def parse_key_value_file(path: Path) -> Dict[str, object]:
    data: Dict[str, object] = {}
    if not path.exists():
        return data
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if " = " not in line:
            continue
        key, value = line.split(" = ", 1)
        data[key.strip()] = parse_scalar(value.strip())
    return data


def parse_scalar(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    try:
        if re.fullmatch(r"[-+]?\d+", text):
            return int(text)
        return float(text)
    except ValueError:
        return text


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def discover_relevant_files(search_dirs: Sequence[str]) -> List[Path]:
    patterns = {
        "task_metadata.json",
        "eval_results.json",
        "train_results.txt",
        "trainer_state.json",
        "all_results.json",
        "salida_spanish_downstream_serio.txt",
        "comandos_y_funciones.txt",
        "README_paso_9_serio.txt",
    }
    files: List[Path] = []
    seen = set()
    for root_text in search_dirs:
        root = Path(root_text)
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.name not in patterns:
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            files.append(path)
    return sorted(files)


def discover_task_dirs(search_dirs: Sequence[str]) -> List[Path]:
    candidates = set()
    for file_path in discover_relevant_files(search_dirs):
        if file_path.name == "task_metadata.json":
            candidates.add(file_path.parent)
        elif file_path.name == "trainer_state.json":
            if file_path.parent.name.startswith("checkpoint-"):
                task_dir = file_path.parent.parent
            else:
                task_dir = file_path.parent
            if task_dir.name in TASK_NAMES or any(name in str(task_dir) for name in TASK_NAMES):
                candidates.add(task_dir)
    return sorted(candidates)


def find_run_dir(task_dir: Path) -> Path:
    if task_dir.name in TASK_NAMES:
        return task_dir.parent
    return task_dir


def extract_command_text(run_dir: Path) -> str:
    return "\n".join(
        [
            read_text(run_dir / "comandos_y_funciones.txt"),
            read_text(run_dir / "salida_spanish_downstream_serio.txt"),
            read_text(run_dir / "README_paso_9_serio.txt"),
        ]
    )


def regex_first(text: str, patterns: Iterable[str]) -> Optional[str]:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.MULTILINE)
        if match:
            return match.group(1).strip()
    return None


def extract_arg_from_command(text: str, arg_name: str) -> Optional[str]:
    pattern = rf"--{re.escape(arg_name)}\s+(?:'([^']+)'|\"([^\"]+)\"|([^\s]+))"
    match = re.search(pattern, text)
    if not match:
        return None
    return next(group for group in match.groups() if group is not None)


def latest_eval_from_trainer_state(state: Dict[str, object]) -> Dict[str, object]:
    log_history = state.get("log_history", [])
    if not isinstance(log_history, list):
        return {}
    eval_entries = [
        entry for entry in log_history
        if isinstance(entry, dict) and any(str(key).startswith("eval_") for key in entry)
    ]
    if not eval_entries:
        return {}
    return {key: value for key, value in eval_entries[-1].items() if str(key).startswith("eval_") or key == "epoch"}


def latest_train_from_trainer_state(state: Dict[str, object]) -> Dict[str, object]:
    log_history = state.get("log_history", [])
    if not isinstance(log_history, list):
        return {}
    train_entries = [
        entry for entry in log_history
        if isinstance(entry, dict) and ("train_loss" in entry or "loss" in entry)
    ]
    return train_entries[-1] if train_entries else {}


def classify_run_type(model_path: Optional[str], run_dir: Path, command_text: str) -> str:
    combined = " ".join([model_path or "", str(run_dir), command_text]).lower()
    if "paso_7_scanpath" in combined or "l7b" in combined:
        return "l7b / scanpath"
    if "preentrenamiento beto full" in combined or "beto_full" in combined:
        return "BETO control"
    if "dccuchile/bert-base-spanish" in combined:
        return "BETO directo"
    if "beto" in combined:
        return "BETO"
    return "otro"


def number_or_none(value: object) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric


def row_from_task_dir(task_dir: Path) -> Tuple[Dict[str, object], List[str]]:
    run_dir = find_run_dir(task_dir)
    command_text = extract_command_text(run_dir)
    metadata = load_json(task_dir / "task_metadata.json")
    eval_metrics = load_json(task_dir / "eval_results.json")
    all_results = load_json(task_dir / "all_results.json")
    train_metrics = parse_key_value_file(task_dir / "train_results.txt")

    state_path = task_dir / "trainer_state.json"
    state = load_json(state_path)
    if not state and not (task_dir / "task_metadata.json").exists():
        checkpoint_states = sorted(task_dir.glob("checkpoint-*/trainer_state.json"))
        if checkpoint_states:
            state_path = checkpoint_states[-1]
            state = load_json(state_path)

    if not eval_metrics:
        eval_metrics = {**all_results, **latest_eval_from_trainer_state(state)}
    if not train_metrics:
        train_metrics = latest_train_from_trainer_state(state)

    task_name = str(metadata.get("task_name") or task_dir.name)
    model_path = str(
        metadata.get("model_name_or_path")
        or regex_first(command_text, [r"Checkpoint seleccionado\s*\n- ruta:\s*(.+)", r"checkpoint usado:\s*(.+)"])
        or extract_arg_from_command(command_text, "model_name_or_path")
        or ""
    )
    checkpoint_used = str(
        regex_first(command_text, [r"Checkpoint seleccionado\s*\n- ruta:\s*(.+)", r"checkpoint usado:\s*(.+)"])
        or model_path
        or ""
    )

    max_train_samples = metadata.get("max_train_samples") or extract_arg_from_command(command_text, "max_train_samples")
    max_eval_samples = metadata.get("max_eval_samples") or extract_arg_from_command(command_text, "max_eval_samples")
    epochs = metadata.get("num_train_epochs") or extract_arg_from_command(command_text, "num_train_epochs")
    learning_rate = metadata.get("learning_rate") or extract_arg_from_command(command_text, "learning_rate")
    batch_size = metadata.get("per_device_train_batch_size") or extract_arg_from_command(command_text, "per_device_train_batch_size")
    seed = metadata.get("seed") or extract_arg_from_command(command_text, "seed")

    output_dir = str(metadata.get("output_dir") or task_dir)
    row = {
        "run_dir": str(run_dir),
        "task_name": task_name,
        "model_used": model_path or "NA",
        "run_type": classify_run_type(model_path, run_dir, command_text),
        "max_train_samples": parse_scalar(max_train_samples),
        "max_eval_samples": parse_scalar(max_eval_samples),
        "epochs": parse_scalar(epochs),
        "learning_rate": parse_scalar(learning_rate),
        "batch_size": parse_scalar(batch_size),
        "seed": parse_scalar(seed),
        "eval_accuracy": eval_metrics.get("eval_accuracy"),
        "eval_macro_f1": eval_metrics.get("eval_macro_f1"),
        "eval_loss": eval_metrics.get("eval_loss"),
        "train_loss": train_metrics.get("train_loss") or train_metrics.get("loss"),
        "train_runtime": train_metrics.get("train_runtime"),
        "checkpoint_used": checkpoint_used or "NA",
        "best_downstream_checkpoint": state.get("best_model_checkpoint") or metadata.get("best_model_checkpoint") or "NA",
        "output_dir": output_dir,
        "status": "complete" if (task_dir / "eval_results.json").exists() else "partial_or_missing_final_metrics",
        "source_files": "; ".join(str(path) for path in source_files_for_task(task_dir, run_dir, state_path)),
    }
    return row, source_files_for_task(task_dir, run_dir, state_path)


def source_files_for_task(task_dir: Path, run_dir: Path, state_path: Path) -> List[Path]:
    paths = [
        task_dir / "task_metadata.json",
        task_dir / "eval_results.json",
        task_dir / "train_results.txt",
        task_dir / "all_results.json",
        task_dir / "trainer_state.json",
        state_path,
        run_dir / "comandos_y_funciones.txt",
        run_dir / "salida_spanish_downstream_serio.txt",
        run_dir / "README_paso_9_serio.txt",
    ]
    unique: List[Path] = []
    seen = set()
    for path in paths:
        if not path.exists() or path.resolve() in seen:
            continue
        seen.add(path.resolve())
        unique.append(path)
    return unique


def normalize_dataframe(rows: List[Dict[str, object]]) -> pd.DataFrame:
    columns = [
        "run_dir",
        "task_name",
        "model_used",
        "run_type",
        "max_train_samples",
        "max_eval_samples",
        "epochs",
        "learning_rate",
        "batch_size",
        "seed",
        "eval_accuracy",
        "eval_macro_f1",
        "eval_loss",
        "train_loss",
        "train_runtime",
        "checkpoint_used",
        "best_downstream_checkpoint",
        "output_dir",
        "status",
        "source_files",
    ]
    df = pd.DataFrame(rows, columns=columns)
    if df.empty:
        return df
    df = df.drop_duplicates(subset=["output_dir", "task_name", "model_used"], keep="last")
    for column in [
        "max_train_samples",
        "max_eval_samples",
        "epochs",
        "learning_rate",
        "batch_size",
        "seed",
        "eval_accuracy",
        "eval_macro_f1",
        "eval_loss",
        "train_loss",
        "train_runtime",
    ]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.sort_values(["task_name", "run_type", "run_dir"]).reset_index(drop=True)


def format_value(value: object, digits: int = 6) -> str:
    numeric = number_or_none(value)
    if numeric is None:
        return "NA"
    return f"{numeric:.{digits}f}"


def best_lines(df: pd.DataFrame, metric: str) -> List[str]:
    if df.empty or metric not in df.columns:
        return ["- NA"]
    subset = df.dropna(subset=[metric])
    if subset.empty:
        return ["- NA"]
    lines = []
    for task_name, group in subset.groupby("task_name"):
        best = group.sort_values(metric, ascending=False).iloc[0]
        lines.append(
            f"- {task_name}: {format_value(best[metric])} | {best['run_type']} | "
            f"samples={format_value(best['max_train_samples'], 0)} | output_dir={best['output_dir']}"
        )
    return lines


def metric_table_lines(df: pd.DataFrame, metric: str) -> List[str]:
    subset = df.dropna(subset=[metric]) if metric in df.columns else pd.DataFrame()
    if subset.empty:
        return ["- No hay datos disponibles."]
    lines = []
    for _, row in subset.sort_values(["task_name", "run_type", "max_train_samples"]).iterrows():
        lines.append(
            f"- {row['task_name']} | {row['run_type']} | {format_value(row[metric])} | "
            f"train={format_value(row['max_train_samples'], 0)} | seed={format_value(row['seed'], 0)} | {row['output_dir']}"
        )
    return lines


def comparison_lines(df: pd.DataFrame) -> List[str]:
    complete = df[df["status"] == "complete"] if "status" in df.columns else df
    if complete.empty:
        return ["- No hay corridas completas para comparar."]
    lines = []
    for task_name, group in complete.groupby("task_name"):
        metric = "eval_accuracy" if group["eval_accuracy"].notna().any() else "eval_macro_f1"
        metric_group = group.dropna(subset=[metric])
        if metric_group.empty:
            lines.append(f"- {task_name}: sin metrica principal disponible.")
            continue
        best = metric_group.sort_values(metric, ascending=False).iloc[0]
        values = [
            f"{row['run_type']}={format_value(row[metric])}"
            for _, row in metric_group.sort_values("run_type").iterrows()
        ]
        lines.append(
            f"- {task_name}: mejor {metric} = {format_value(best[metric])} con {best['run_type']}. "
            f"Valores: {', '.join(values)}."
        )
    return lines


def incomplete_lines(df: pd.DataFrame) -> List[str]:
    if df.empty or "status" not in df.columns:
        return ["- NA"]
    subset = df[df["status"] != "complete"]
    if subset.empty:
        return ["- No se detectaron corridas incompletas en las filas consolidadas."]
    return [f"- {row['task_name']} | {row['output_dir']} | {row['status']}" for _, row in subset.iterrows()]


def build_summary_text(df: pd.DataFrame, search_dirs: Sequence[str], read_files: Sequence[Path]) -> str:
    lines = [
        "RESUMEN METRICAS LOW-RESOURCE",
        "=============================",
        "",
        "Carpetas recorridas:",
    ]
    lines.extend(f"- {path}" for path in search_dirs)
    lines.extend(
        [
            "",
            f"Corridas/tareas encontradas: {len(df)}",
            f"Archivos fuente leidos/detectados: {len(read_files)}",
            "",
            "Mejores accuracy por tarea:",
        ]
    )
    lines.extend(best_lines(df, "eval_accuracy"))
    lines.extend(["", "Mejores macro-F1 por tarea:"])
    lines.extend(best_lines(df, "eval_macro_f1"))
    lines.extend(["", "Accuracy por tarea/modelo:"])
    lines.extend(metric_table_lines(df, "eval_accuracy"))
    lines.extend(["", "Macro-F1 por tarea/modelo:"])
    lines.extend(metric_table_lines(df, "eval_macro_f1"))
    lines.extend(["", "Comparacion breve entre modelos:"])
    lines.extend(comparison_lines(df))
    lines.extend(["", "Notas sobre corridas incompletas o metricas faltantes:"])
    lines.extend(incomplete_lines(df))
    return "\n".join(lines) + "\n"


def plot_metric(df: pd.DataFrame, metric: str, output_path: Path, note_path: Path):
    subset = df.dropna(subset=[metric]) if metric in df.columns else pd.DataFrame()
    if subset.empty:
        note_path.write_text(f"No habia datos suficientes para graficar {metric}.\n", encoding="utf-8")
        return

    labels = [
        f"{row['task_name']}\n{row['run_type']}\ntrain={int(row['max_train_samples']) if not pd.isna(row['max_train_samples']) else 'NA'}"
        for _, row in subset.sort_values(["task_name", "run_type", "max_train_samples"]).iterrows()
    ]
    values = subset.sort_values(["task_name", "run_type", "max_train_samples"])[metric].astype(float).tolist()
    width = max(8, len(values) * 1.7)
    fig, ax = plt.subplots(figsize=(width, 5))
    ax.bar(range(len(values)), values, color="#4c78a8")
    ax.set_title(metric)
    ax.set_ylabel(metric)
    ax.set_ylim(0, max(1.0, max(values) * 1.15))
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_commands_file(
    output_dir: Path,
    search_dirs: Sequence[str],
    read_files: Sequence[Path],
    generated_files: Sequence[Path],
):
    lines = [
        "COMANDOS Y FUNCIONES - RESUMEN LOW-RESOURCE",
        "===========================================",
        "",
        "Script usado",
        "- summarize_low_resource_results.py",
        "",
        "Comando ejecutado",
        f"- python summarize_low_resource_results.py --output_dir {output_dir}",
        "",
        "Carpetas recorridas",
    ]
    lines.extend(f"- {path}" for path in search_dirs)
    lines.extend(["", "Archivos leidos/detectados"])
    lines.extend(f"- {path}" for path in read_files)
    lines.extend(
        [
            "",
            "Funciones principales",
            "- discover_relevant_files",
            "- discover_task_dirs",
            "- row_from_task_dir",
            "- normalize_dataframe",
            "- build_summary_text",
            "- plot_metric",
            "",
            "Outputs generados",
        ]
    )
    lines.extend(f"- {path}" for path in generated_files)
    (output_dir / "comandos_y_funciones.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    read_files = discover_relevant_files(args.search_dirs)
    task_dirs = discover_task_dirs(args.search_dirs)
    rows = []
    row_source_files: List[Path] = []
    for task_dir in task_dirs:
        row, sources = row_from_task_dir(task_dir)
        rows.append(row)
        row_source_files.extend(sources)

    df = normalize_dataframe(rows)
    csv_path = output_dir / "resumen_metricas_low_resource.csv"
    txt_path = output_dir / "resumen_metricas_low_resource.txt"
    accuracy_path = output_dir / "grafico_accuracy_low_resource.png"
    macro_f1_path = output_dir / "grafico_macro_f1_low_resource.png"
    accuracy_note_path = output_dir / "grafico_accuracy_low_resource.txt"
    macro_f1_note_path = output_dir / "grafico_macro_f1_low_resource.txt"

    df.to_csv(csv_path, index=False, quoting=csv.QUOTE_MINIMAL, na_rep="NA")
    txt_path.write_text(build_summary_text(df, args.search_dirs, read_files), encoding="utf-8")
    plot_metric(df, "eval_accuracy", accuracy_path, accuracy_note_path)
    plot_metric(df, "eval_macro_f1", macro_f1_path, macro_f1_note_path)

    generated_files = [csv_path, txt_path, accuracy_path, macro_f1_path]
    if accuracy_note_path.exists():
        generated_files.append(accuracy_note_path)
    if macro_f1_note_path.exists():
        generated_files.append(macro_f1_note_path)
    generated_files.append(output_dir / "comandos_y_funciones.txt")
    write_commands_file(output_dir, args.search_dirs, sorted(set(read_files + row_source_files)), generated_files)

    print(f"Wrote {csv_path}")
    print(f"Wrote {txt_path}")
    print(f"Wrote {accuracy_path if accuracy_path.exists() else accuracy_note_path}")
    print(f"Wrote {macro_f1_path if macro_f1_path.exists() else macro_f1_note_path}")
    print(f"Wrote {output_dir / 'comandos_y_funciones.txt'}")
    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    main()
