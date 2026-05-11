import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

from transformers import AutoConfig, AutoTokenizer


DEFAULT_BETO_MODEL = "dccuchile/bert-base-spanish-wwm-cased"
DEFAULT_OUTPUT_DIR = "Pasos/paso_9b_beto"
DEFAULT_STEP9_REFERENCE_DIR = "Pasos/paso_9_es_l7b"
TASK_ORDER = ["xnli_es", "intertass2020"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a Spanish BETO baseline comparable to paso 9 downstream from l7b."
    )
    parser.add_argument(
        "--model_name_or_path",
        default=DEFAULT_BETO_MODEL,
        help="Baseline model used for downstream fine-tuning.",
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where paso_9b_beto artifacts will be written.",
    )
    parser.add_argument(
        "--step9_reference_dir",
        default=DEFAULT_STEP9_REFERENCE_DIR,
        help="Optional directory with paso 9 Spanish results used for a simple comparison table.",
    )
    parser.add_argument("--max_train_samples", type=int, default=500, help="Maximum number of training examples per task.")
    parser.add_argument("--max_eval_samples", type=int, default=200, help="Maximum number of evaluation examples per task.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of epochs per task.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Evaluation batch size.")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Maximum tokenized sequence length.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed.")
    return parser.parse_args()


def parse_key_value_file(path: Path) -> Dict[str, str]:
    metrics: Dict[str, str] = {}
    if not path.exists():
        return metrics
    for line in path.read_text(encoding="utf-8").splitlines():
        if " = " not in line:
            continue
        key, value = line.split(" = ", 1)
        metrics[key.strip()] = value.strip()
    return metrics


def extract_eval_metrics_from_trainer_state(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    state = json.loads(path.read_text(encoding="utf-8"))
    log_history = state.get("log_history", [])
    eval_entries = [entry for entry in log_history if any(key.startswith("eval_") for key in entry.keys())]
    if not eval_entries:
        return {}
    last_eval = eval_entries[-1]
    return {key: value for key, value in last_eval.items() if key.startswith("eval_")}


def extract_eval_metrics_from_task_dir(task_dir: Path) -> Dict[str, object]:
    eval_results_path = task_dir / "eval_results.json"
    if eval_results_path.exists():
        data = json.loads(eval_results_path.read_text(encoding="utf-8"))
        return {key: value for key, value in data.items() if key.startswith("eval_")}
    return extract_eval_metrics_from_trainer_state(task_dir / "trainer_state.json")


def build_baseline_command(args, task_name: str, task_output_dir: Path, intertass_cache_dir: Path) -> List[str]:
    return [
        sys.executable,
        "train_spanish_downstream_baseline.py",
        "--model_name_or_path",
        args.model_name_or_path,
        "--task_name",
        task_name,
        "--output_dir",
        str(task_output_dir),
        "--max_train_samples",
        str(args.max_train_samples),
        "--max_eval_samples",
        str(args.max_eval_samples),
        "--num_train_epochs",
        str(args.num_train_epochs),
        "--per_device_train_batch_size",
        str(args.per_device_train_batch_size),
        "--per_device_eval_batch_size",
        str(args.per_device_eval_batch_size),
        "--max_seq_length",
        str(args.max_seq_length),
        "--learning_rate",
        str(args.learning_rate),
        "--seed",
        str(args.seed),
        "--intertass_cache_dir",
        str(intertass_cache_dir),
    ]


def run_task(args, task_name: str, base_output_dir: Path, intertass_cache_dir: Path) -> Dict[str, object]:
    task_output_dir = base_output_dir / task_name
    task_output_dir.mkdir(parents=True, exist_ok=True)

    command = build_baseline_command(args, task_name, task_output_dir, intertass_cache_dir)
    result = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
        check=True,
    )

    train_metrics = parse_key_value_file(task_output_dir / "train_results.txt")
    eval_metrics = extract_eval_metrics_from_task_dir(task_output_dir)
    metadata_path = task_output_dir / "task_metadata.json"
    task_metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    checkpoint_dirs = sorted([str(path) for path in task_output_dir.glob("checkpoint-*") if path.is_dir()])
    saved_model_root = str(task_output_dir) if (task_output_dir / "pytorch_model.bin").exists() else None

    return {
        "task_name": task_name,
        "task_output_dir": str(task_output_dir),
        "command": command,
        "stdout_tail": "\n".join(result.stdout.strip().splitlines()[-10:]) if result.stdout else "",
        "stderr_tail": "\n".join(result.stderr.strip().splitlines()[-10:]) if result.stderr else "",
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "task_metadata": task_metadata,
        "checkpoint_dirs": checkpoint_dirs,
        "saved_model_root": saved_model_root,
    }


def extract_primary_metric(metric_name: Optional[str], eval_metrics: Dict[str, object]) -> Optional[float]:
    if not metric_name:
        return None
    candidates = [f"eval_{metric_name}", metric_name]
    for candidate in candidates:
        if candidate in eval_metrics:
            try:
                return float(eval_metrics[candidate])
            except (TypeError, ValueError):
                return None
    return None


def load_step9_reference_task(step9_reference_dir: Path, task_name: str) -> Dict[str, object]:
    task_dir = step9_reference_dir / task_name
    if not task_dir.exists():
        return {
            "task_name": task_name,
            "available": False,
            "reason": "no existe directorio de salida para esta tarea en paso_9.",
        }

    metadata_path = task_dir / "task_metadata.json"
    if not metadata_path.exists():
        return {
            "task_name": task_name,
            "available": False,
            "reason": "no existe task_metadata.json en la referencia de paso_9.",
        }

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    eval_metrics = extract_eval_metrics_from_task_dir(task_dir)
    metric_name = metadata.get("metric_name")
    primary_metric = extract_primary_metric(metric_name, eval_metrics)

    return {
        "task_name": task_name,
        "available": primary_metric is not None,
        "reason": None if primary_metric is not None else "no se encontro la metrica principal de evaluacion en la referencia de paso_9.",
        "metric_name": metric_name,
        "primary_metric": primary_metric,
        "task_metadata": metadata,
        "eval_metrics": eval_metrics,
        "task_output_dir": str(task_dir),
    }


def load_step9_reference_results(step9_reference_dir: Path) -> Dict[str, Dict[str, object]]:
    results: Dict[str, Dict[str, object]] = {}
    for task_name in TASK_ORDER:
        results[task_name] = load_step9_reference_task(step9_reference_dir, task_name)
    return results


def build_comparison_rows(
    summaries: List[Dict[str, object]], step9_reference_dir: Path
) -> List[Dict[str, object]]:
    reference_results = load_step9_reference_results(step9_reference_dir)
    comparison_rows: List[Dict[str, object]] = []

    for summary in summaries:
        task_name = summary["task_name"]
        metadata = summary["task_metadata"]
        metric_name = metadata.get("metric_name")
        paso_9b_metric = extract_primary_metric(metric_name, summary["eval_metrics"])
        paso_9_reference = reference_results.get(task_name, {})
        paso_9_metric = paso_9_reference.get("primary_metric")

        if paso_9_metric is not None and paso_9b_metric is not None:
            difference = paso_9b_metric - paso_9_metric
            note = "diferencia calculada como paso_9b_beto - paso_9_l7b"
        elif not paso_9_reference.get("available", False):
            difference = None
            note = paso_9_reference.get("reason", "referencia de paso_9 no disponible.")
        else:
            difference = None
            note = "no se pudo calcular la diferencia por metrica faltante en paso_9b."

        comparison_rows.append(
            {
                "task_name": task_name,
                "metric_name": metric_name,
                "paso_9_l7b": paso_9_metric,
                "paso_9b_beto": paso_9b_metric,
                "difference": difference,
                "note": note,
                "step9_reference_output_dir": paso_9_reference.get("task_output_dir"),
            }
        )

    return comparison_rows


def format_metric_value(value: Optional[float]) -> str:
    if value is None:
        return "N/D"
    return f"{value:.6f}"


def build_debug_output(args, summaries: List[Dict[str, object]], comparison_rows: List[Dict[str, object]]) -> str:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    config = AutoConfig.from_pretrained(args.model_name_or_path)

    lines: List[str] = [
        "PASO 9B BETO - Fine-tuning downstream en espanol con baseline directo",
        "",
        "----------------------------------------",
        "Modelo base",
        "----------------------------------------",
        f"model_name_or_path usado: {args.model_name_or_path}",
        "tipo de inicializacion: BETO directo sin pretraining scanpath previo de l7b",
        "",
        "----------------------------------------",
        "Configuracion general",
        "----------------------------------------",
        f"tokenizer: {tokenizer.__class__.__name__}",
        f"model_type: {config.model_type}",
        f"tasks ejecutadas: {[summary['task_name'] for summary in summaries]}",
        f"max_train_samples: {args.max_train_samples}",
        f"max_eval_samples: {args.max_eval_samples}",
        f"epochs: {args.num_train_epochs}",
        f"batch_size_train: {args.per_device_train_batch_size}",
        f"batch_size_eval: {args.per_device_eval_batch_size}",
        f"max_seq_length: {args.max_seq_length}",
        f"learning_rate: {args.learning_rate}",
        f"seed: {args.seed}",
        "",
    ]

    for summary in summaries:
        metadata = summary["task_metadata"]
        forward_debug = metadata.get("forward_debug", {})
        lines.extend(
            [
                "----------------------------------------",
                f"Task {summary['task_name']}",
                "----------------------------------------",
                f"nombre: {summary['task_name']}",
                f"tipo: {metadata.get('task_type')}",
                f"columnas textuales: {metadata.get('text_columns')}",
                f"dataset_source: {metadata.get('dataset_source')}",
                f"dataset_note: {metadata.get('dataset_note')}",
                f"eval_split_origin: {metadata.get('eval_split_origin')}",
                f"num_labels: {metadata.get('num_labels')}",
                f"labels: {metadata.get('label_list')}",
                f"metrica principal: {metadata.get('metric_name')}",
                f"input_ids.shape: {forward_debug.get('input_ids_shape')}",
                f"logits.shape: {forward_debug.get('logits_shape')}",
                f"output_dir: {summary['task_output_dir']}",
                "",
                "Train metrics:",
            ]
        )
        if summary["train_metrics"]:
            for key, value in summary["train_metrics"].items():
                lines.append(f"- {key}: {value}")
        else:
            lines.append("- no se encontraron train metrics en train_results.txt")

        lines.append("")
        lines.append("Evaluation:")
        if summary["eval_metrics"]:
            for key, value in summary["eval_metrics"].items():
                lines.append(f"- {key}: {value}")
        else:
            lines.append("- no se encontraron eval metrics en eval_results.json ni en trainer_state.json")

        lines.append("")
        lines.append("Checkpoints guardados:")
        if summary["checkpoint_dirs"]:
            for checkpoint_dir in summary["checkpoint_dirs"]:
                lines.append(f"- {checkpoint_dir}")
        else:
            lines.append("- no se encontraron checkpoint-* intermedios en la salida")
        if summary.get("saved_model_root") is not None:
            lines.append(f"- modelo final guardado en: {summary['saved_model_root']}")

        lines.extend(
            [
                "",
                "Nota final por tarea:",
                "- este paso usa BETO directo como baseline inicial.",
                "- no se uso scanpath, no se uso GRU y no se uso loss auxiliar en downstream.",
                "",
            ]
        )

    lines.extend(
        [
            "----------------------------------------",
            "Comparacion simple contra paso 9",
            "----------------------------------------",
            f"directorio de referencia paso_9: {args.step9_reference_dir}",
        ]
    )

    for row in comparison_rows:
        lines.extend(
            [
                f"- task: {row['task_name']}",
                f"  metric: {row['metric_name']}",
                f"  paso_9_l7b: {format_metric_value(row['paso_9_l7b'])}",
                f"  paso_9b_beto: {format_metric_value(row['paso_9b_beto'])}",
                f"  diferencia: {format_metric_value(row['difference'])}",
                f"  nota: {row['note']}",
            ]
        )

    return "\n".join(lines) + "\n"


def build_readme(args, summaries: List[Dict[str, object]], comparison_rows: List[Dict[str, object]], script_name: str) -> str:
    xnli_summary = next((summary for summary in summaries if summary["task_name"] == "xnli_es"), None)
    intertass_summary = next((summary for summary in summaries if summary["task_name"] == "intertass2020"), None)

    xnli_metadata = xnli_summary["task_metadata"] if xnli_summary else {}
    intertass_metadata = intertass_summary["task_metadata"] if intertass_summary else {}

    comparison_note = "La comparacion simple se arma leyendo las salidas disponibles de paso_9_es_l7b."
    if any(row["paso_9_l7b"] is None for row in comparison_rows):
        comparison_note += " Si alguna metrica de paso_9 no existe todavia, se informa N/D y no se rompe el script."

    return f"""PASO 9B BETO - README
=========================

Que cambia respecto del paso 9
- PASO 9 usa un checkpoint seleccionado automaticamente desde l7b.
- PASO 9b usa BETO directo como baseline: {args.model_name_or_path}
- Todo lo demas se mantiene alineado para hacer una comparacion lo mas justa posible.

Objetivo de este paso
- Medir si el modelo con pretraining previo de l7b aporta algo util frente a un baseline fuerte en espanol.
- Comparar el modelo downstream basado en l7b contra BETO directo sobre exactamente las mismas tareas.
- Mantener downstream sin scanpath, sin GRU y sin losses auxiliares.

Comparacion justa mantenida
- mismas tareas
- mismos benchmarks
- misma tokenizacion por tarea
- mismos max_train_samples y max_eval_samples
- mismas epochs
- mismo batch size
- mismo max_seq_length
- mismo learning rate
- misma seed
- mismas metricas
- mismo formato general de salida

Tareas elegidas
- XNLIes
  tipo: {xnli_metadata.get('task_type')}
  columnas usadas: {xnli_metadata.get('text_columns')}
  dataset source: {xnli_metadata.get('dataset_source')}
  labels: {xnli_metadata.get('label_list')}
  num_labels: {xnli_metadata.get('num_labels')}
  metrica principal: accuracy
  como se calcula: exactitud simple sobre la prediccion argmax contra el label gold.

- InterTass2020
  tipo: {intertass_metadata.get('task_type')}
  columnas usadas: {intertass_metadata.get('text_columns')}
  dataset source: {intertass_metadata.get('dataset_source')}
  labels: {intertass_metadata.get('label_list')}
  num_labels: {intertass_metadata.get('num_labels')}
  metrica principal: macro-F1
  como se calcula: promedio simple del F1 por clase sobre las 3 clases.
  nota de split: {intertass_metadata.get('eval_split_origin')}

Que parte reutiliza del paso 9
- Reutiliza el mismo baseline downstream en espanol: train_spanish_downstream_baseline.py
- Reutiliza el mismo conjunto de tareas: xnli_es e intertass2020
- Reutiliza la misma configuracion de entrenamiento y evaluacion
- Reutiliza la misma idea de wrapper orquestador y resumen por tarea

Que parte nueva agrega
- Un wrapper nuevo llamado {script_name}
- Seleccion fija del baseline BETO directo, en lugar de un checkpoint de l7b
- Una seccion comparativa simple contra paso_9_es_l7b cuando esa referencia existe

Aclaracion importante sobre downstream
- En este paso downstream sigue usando solo la rama principal del modelo.
- No se reintroduce scanpath.
- No se usa GRU.
- No se usa loss auxiliar.
- La cabeza downstream sigue siendo nueva y propia de cada tarea.

Configuracion usada
- num_train_epochs = {args.num_train_epochs}
- per_device_train_batch_size = {args.per_device_train_batch_size}
- per_device_eval_batch_size = {args.per_device_eval_batch_size}
- max_train_samples = {args.max_train_samples}
- max_eval_samples = {args.max_eval_samples}
- max_seq_length = {args.max_seq_length}
- learning_rate = {args.learning_rate}
- seed = {args.seed}

Relacion con el paper original
- Esta configuracion intenta acercarse mas al regimen del paper original que el paso 9 ingles ultraliviano.
- Aun asi sigue siendo una version reducida para no volver impracticable la corrida.
- Por eso no se replica la grilla completa de multiples seeds, multiples tamanos de dataset y largas barridas de epochs.

Comparacion con paso 9
- Directorio de referencia esperado: {args.step9_reference_dir}
- {comparison_note}
"""


def build_commands_file(args, summaries: List[Dict[str, object]], script_name: str) -> str:
    relative_output = Path(args.output_dir).as_posix()
    wrapper_command = (
        f".\\.venv\\Scripts\\python.exe {script_name} "
        f"--model_name_or_path {args.model_name_or_path} "
        f"--output_dir {relative_output} "
        f"--step9_reference_dir {args.step9_reference_dir} "
        f"--max_train_samples {args.max_train_samples} "
        f"--max_eval_samples {args.max_eval_samples} "
        f"--num_train_epochs {args.num_train_epochs} "
        f"--per_device_train_batch_size {args.per_device_train_batch_size} "
        f"--per_device_eval_batch_size {args.per_device_eval_batch_size} "
        f"--max_seq_length {args.max_seq_length} "
        f"--learning_rate {args.learning_rate} "
        f"--seed {args.seed}"
    )

    lines = [
        "COMANDOS Y FUNCIONES - PASO 9B BETO",
        "===================================",
        "",
        "Comando ejecutado",
        f"- {wrapper_command}",
        "",
        "Script principal usado",
        f"- {script_name}",
        "- baseline reutilizado/minimo: train_spanish_downstream_baseline.py",
        "",
        "Modelo base usado",
        f"- {args.model_name_or_path}",
        "- baseline directo en espanol sin checkpoint de l7b",
        "",
        "Funciones principales llamadas",
        "- AutoTokenizer.from_pretrained",
        "- AutoConfig.from_pretrained",
        "- subprocess.run(... train_spanish_downstream_baseline.py ...)",
        "- parse_key_value_file",
        "- extract_eval_metrics_from_task_dir",
        "- load_step9_reference_results",
        "- build_comparison_rows",
        "",
        "Outputs generados",
        f"- {Path(args.output_dir) / 'salida_spanish_baseline_debug.txt'}",
        f"- {Path(args.output_dir) / 'README_paso_9b_beto.txt'}",
        f"- {Path(args.output_dir) / 'comandos_y_funciones.txt'}",
    ]

    for summary in summaries:
        lines.append(f"- {summary['task_output_dir']}")
    lines.append("")

    lines.append("Comandos internos por tarea")
    for summary in summaries:
        lines.append(f"- {' '.join(summary['command'])}")

    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    intertass_cache_dir = output_dir / "cache"

    summaries = [run_task(args, task_name, output_dir, intertass_cache_dir) for task_name in TASK_ORDER]
    comparison_rows = build_comparison_rows(summaries, Path(args.step9_reference_dir))
    script_name = Path(__file__).name

    (output_dir / "salida_spanish_baseline_debug.txt").write_text(
        build_debug_output(args, summaries, comparison_rows),
        encoding="utf-8",
    )
    (output_dir / "README_paso_9b_beto.txt").write_text(
        build_readme(args, summaries, comparison_rows, script_name),
        encoding="utf-8",
    )
    (output_dir / "comandos_y_funciones.txt").write_text(
        build_commands_file(args, summaries, script_name),
        encoding="utf-8",
    )

    print(f"Wrote {output_dir / 'salida_spanish_baseline_debug.txt'}")
    print(f"Wrote {output_dir / 'README_paso_9b_beto.txt'}")
    print(f"Wrote {output_dir / 'comandos_y_funciones.txt'}")
    for summary in summaries:
        print(f"Finished downstream task {summary['task_name']} in {summary['task_output_dir']}")


if __name__ == "__main__":
    main()
