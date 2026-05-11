import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from transformers import AutoConfig, AutoTokenizer


DEFAULT_MAIN_PIPELINE_DIR = "Pasos/paso_7_scanpath_full_8ep"
DEFAULT_OUTPUT_DIR = "Pasos/paso_9_serio"
TASK_ORDER = ["xnli_es", "intertass2020"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run serious Spanish downstream fine-tuning from the main BETO+scanpath pipeline checkpoint."
    )
    parser.add_argument(
        "--main_pipeline_dir",
        default=DEFAULT_MAIN_PIPELINE_DIR,
        help="Directory that contains the main pipeline checkpoints.",
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where paso_9_serio artifacts will be written.",
    )
    parser.add_argument("--max_train_samples", type=int, default=5000, help="Maximum number of training examples per task.")
    parser.add_argument("--max_eval_samples", type=int, default=1000, help="Maximum number of evaluation examples per task.")
    parser.add_argument("--num_train_epochs", type=int, default=10, help="Number of epochs per task.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Evaluation batch size.")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Maximum tokenized sequence length.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed.")
    parser.add_argument("--evaluation_strategy", default="epoch", choices=["no", "steps", "epoch"])
    parser.add_argument("--save_strategy", default="epoch", choices=["no", "steps", "epoch"])
    parser.add_argument("--load_best_model_at_end", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save_total_limit", type=int, default=1)
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


def load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def extract_eval_metrics_from_task_dir(task_dir: Path) -> Dict[str, object]:
    eval_results = load_json(task_dir / "eval_results.json")
    if eval_results:
        return {key: value for key, value in eval_results.items() if key.startswith("eval_")}

    trainer_state = load_json(task_dir / "trainer_state.json")
    log_history = trainer_state.get("log_history", [])
    eval_entries = [entry for entry in log_history if any(key.startswith("eval_") for key in entry.keys())]
    if not eval_entries:
        return {}
    return {key: value for key, value in eval_entries[-1].items() if key.startswith("eval_")}


def is_valid_model_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    has_config = (path / "config.json").exists()
    has_weights = (path / "pytorch_model.bin").exists() or (path / "model.safetensors").exists()
    return has_config and has_weights


def resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value.strip())
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def parse_best_checkpoint_from_logs(main_dir: Path) -> Optional[Path]:
    log_candidates = sorted(main_dir.glob("salida_training*.txt")) + sorted(main_dir.glob("*training*.txt"))
    for log_path in log_candidates:
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.strip().startswith("best_checkpoint:"):
                candidate = line.split(":", 1)[1].strip()
                resolved = resolve_path(candidate, Path.cwd())
                if is_valid_model_dir(resolved):
                    return resolved
    return None


def parse_best_model_checkpoint_from_trainer_state(main_dir: Path) -> Optional[Path]:
    for state_path in sorted(main_dir.rglob("trainer_state.json")):
        state = load_json(state_path)
        best_model_checkpoint = state.get("best_model_checkpoint")
        if best_model_checkpoint:
            resolved = resolve_path(str(best_model_checkpoint), Path.cwd())
            if is_valid_model_dir(resolved):
                return resolved
    return None


def metric_from_step7_log(main_dir: Path, checkpoint_name: str) -> Optional[float]:
    log_candidates = sorted(main_dir.glob("salida_training*.txt")) + sorted(main_dir.glob("*training*.txt"))
    checkpoint_seen = False
    for log_path in log_candidates:
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if stripped.startswith("checkpoint_saved_to:") and checkpoint_name in stripped:
                checkpoint_seen = True
            elif checkpoint_seen and stripped.startswith("- eval_mean_total_loss:"):
                try:
                    return float(stripped.split(":", 1)[1].strip())
                except ValueError:
                    return None
            elif checkpoint_seen and stripped.startswith("----------------------------------------"):
                checkpoint_seen = False
    return None


def resolve_metric_best_checkpoint(main_dir: Path) -> Optional[Tuple[Path, float]]:
    candidates: List[Tuple[float, Path]] = []
    for child in main_dir.iterdir() if main_dir.exists() else []:
        if not re.fullmatch(r"checkpoint_epoch_\d+", child.name) or not is_valid_model_dir(child):
            continue
        metric = metric_from_step7_log(main_dir, child.name)
        if metric is not None:
            candidates.append((metric, child.resolve()))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    metric, path = candidates[0]
    return path, metric


def resolve_latest_checkpoint(main_dir: Path) -> Optional[Path]:
    final_dir = main_dir / "checkpoint_final"
    if is_valid_model_dir(final_dir):
        return final_dir.resolve()

    epoch_candidates: List[Tuple[int, Path]] = []
    for child in main_dir.iterdir() if main_dir.exists() else []:
        match = re.fullmatch(r"checkpoint_epoch_(\d+)", child.name)
        if match and is_valid_model_dir(child):
            epoch_candidates.append((int(match.group(1)), child.resolve()))
    if epoch_candidates:
        epoch_candidates.sort(key=lambda item: item[0], reverse=True)
        return epoch_candidates[0][1]

    generic_candidates = sorted(
        [path.resolve() for path in main_dir.glob("checkpoint*") if is_valid_model_dir(path)],
        key=lambda path: path.name,
        reverse=True,
    )
    return generic_candidates[0] if generic_candidates else None


def resolve_main_pipeline_checkpoint(main_pipeline_dir: Path) -> Dict[str, str]:
    main_dir = main_pipeline_dir.resolve()

    best_dir = main_dir / "best_checkpoint"
    if is_valid_model_dir(best_dir):
        return {
            "selected_checkpoint": str(best_dir.resolve()),
            "selection_kind": "best_checkpoint_dir",
            "selection_reason": "Existe el directorio best_checkpoint dentro del entrenamiento principal full 8ep.",
            "main_pipeline_dir": str(main_dir),
        }

    logged_best = parse_best_checkpoint_from_logs(main_dir)
    if logged_best is not None:
        return {
            "selected_checkpoint": str(logged_best),
            "selection_kind": "best_checkpoint_logged",
            "selection_reason": "Se encontro best_checkpoint registrado en los logs del entrenamiento principal.",
            "main_pipeline_dir": str(main_dir),
        }

    trainer_state_best = parse_best_model_checkpoint_from_trainer_state(main_dir)
    if trainer_state_best is not None:
        return {
            "selected_checkpoint": str(trainer_state_best),
            "selection_kind": "trainer_state_best_model_checkpoint",
            "selection_reason": "Se encontro best_model_checkpoint registrado en trainer_state.json.",
            "main_pipeline_dir": str(main_dir),
        }

    metric_best = resolve_metric_best_checkpoint(main_dir)
    if metric_best is not None:
        path, metric = metric_best
        return {
            "selected_checkpoint": str(path),
            "selection_kind": "best_metric_from_step7_log",
            "selection_reason": f"No habia best_checkpoint directo; se eligio el menor eval_mean_total_loss registrado ({metric}).",
            "main_pipeline_dir": str(main_dir),
        }

    latest = resolve_latest_checkpoint(main_dir)
    if latest is not None:
        return {
            "selected_checkpoint": str(latest),
            "selection_kind": "latest_checkpoint",
            "selection_reason": "No se encontro best ni metrica utilizable; se eligio el ultimo checkpoint disponible.",
            "main_pipeline_dir": str(main_dir),
        }

    return {
        "selected_checkpoint": str(main_dir),
        "selection_kind": "root_fallback",
        "selection_reason": "No se encontraron subcheckpoints validos; se usa el directorio raiz como fallback.",
        "main_pipeline_dir": str(main_dir),
    }


def bool_flag(name: str, enabled: bool) -> List[str]:
    return [f"--{name}" if enabled else f"--no-{name}"]


def build_task_command(args, selected_checkpoint: str, task_name: str, task_output_dir: Path, intertass_cache_dir: Path) -> List[str]:
    command = [
        sys.executable,
        "train_spanish_downstream_baseline.py",
        "--model_name_or_path",
        selected_checkpoint,
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
        "--evaluation_strategy",
        args.evaluation_strategy,
        "--save_strategy",
        args.save_strategy,
        "--save_total_limit",
        str(args.save_total_limit),
        "--intertass_cache_dir",
        str(intertass_cache_dir),
    ]
    command.extend(bool_flag("load_best_model_at_end", args.load_best_model_at_end))
    return command


def run_task(args, selected_checkpoint_info: Dict[str, str], task_name: str, base_output_dir: Path) -> Dict[str, object]:
    task_output_dir = base_output_dir / task_name
    task_output_dir.mkdir(parents=True, exist_ok=True)
    intertass_cache_dir = base_output_dir / "cache"

    command = build_task_command(
        args=args,
        selected_checkpoint=selected_checkpoint_info["selected_checkpoint"],
        task_name=task_name,
        task_output_dir=task_output_dir,
        intertass_cache_dir=intertass_cache_dir,
    )
    result = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
        check=True,
    )

    trainer_state = load_json(task_output_dir / "trainer_state.json")
    metadata_path = task_output_dir / "task_metadata.json"
    task_metadata = load_json(metadata_path)

    return {
        "task_name": task_name,
        "task_output_dir": str(task_output_dir),
        "command": command,
        "stdout_tail": "\n".join(result.stdout.strip().splitlines()[-20:]) if result.stdout else "",
        "stderr_tail": "\n".join(result.stderr.strip().splitlines()[-20:]) if result.stderr else "",
        "train_metrics": parse_key_value_file(task_output_dir / "train_results.txt"),
        "eval_metrics": extract_eval_metrics_from_task_dir(task_output_dir),
        "task_metadata": task_metadata,
        "trainer_state": trainer_state,
        "best_model_checkpoint": trainer_state.get("best_model_checkpoint") or task_metadata.get("best_model_checkpoint"),
        "checkpoint_dirs": sorted([str(path) for path in task_output_dir.glob("checkpoint-*") if path.is_dir()]),
        "saved_model_root": str(task_output_dir) if (task_output_dir / "config.json").exists() else None,
    }


def configuration_lines(args) -> List[str]:
    return [
        f"num_train_epochs: {args.num_train_epochs}",
        f"max_train_samples: {args.max_train_samples}",
        f"max_eval_samples: {args.max_eval_samples}",
        f"per_device_train_batch_size: {args.per_device_train_batch_size}",
        f"per_device_eval_batch_size: {args.per_device_eval_batch_size}",
        f"max_seq_length: {args.max_seq_length}",
        f"learning_rate: {args.learning_rate}",
        f"seed: {args.seed}",
        f"evaluation_strategy: {args.evaluation_strategy}",
        f"save_strategy: {args.save_strategy}",
        f"load_best_model_at_end: {args.load_best_model_at_end}",
        f"save_total_limit: {args.save_total_limit}",
    ]


def build_debug_output(args, selected_checkpoint_info: Dict[str, str], summaries: List[Dict[str, object]]) -> str:
    tokenizer = AutoTokenizer.from_pretrained(selected_checkpoint_info["selected_checkpoint"], use_fast=True)
    config = AutoConfig.from_pretrained(selected_checkpoint_info["selected_checkpoint"])
    lines: List[str] = [
        "PASO 9 SERIO - Fine-tuning downstream/extrinseco en espanol",
        "",
        "----------------------------------------",
        "Checkpoint usado",
        "----------------------------------------",
        f"main_pipeline_dir: {selected_checkpoint_info['main_pipeline_dir']}",
        f"checkpoint usado: {selected_checkpoint_info['selected_checkpoint']}",
        f"criterio de seleccion: {selected_checkpoint_info['selection_kind']}",
        f"motivo: {selected_checkpoint_info['selection_reason']}",
        "",
        "----------------------------------------",
        "Tokenizer y modelo base",
        "----------------------------------------",
        f"tokenizer: {tokenizer.__class__.__name__}",
        f"model_type: {config.model_type}",
        f"architectures_originales: {getattr(config, 'architectures', None)}",
        "downstream: AutoModelForSequenceClassification con cabeza nueva por tarea",
        "scanpath_downstream: no",
        "GRU_downstream: no",
        "loss_auxiliar_downstream: no",
        "",
        "----------------------------------------",
        "Task list",
        "----------------------------------------",
        f"tasks: {[summary['task_name'] for summary in summaries]}",
        "",
        "----------------------------------------",
        "Configuracion completa",
        "----------------------------------------",
    ]
    lines.extend(configuration_lines(args))
    lines.append("")

    for summary in summaries:
        metadata = summary["task_metadata"]
        forward_debug = metadata.get("forward_debug", {})
        lines.extend(
            [
                "----------------------------------------",
                f"Tarea: {summary['task_name']}",
                "----------------------------------------",
                f"tipo de tarea: {metadata.get('task_type')}",
                f"columnas usadas: {metadata.get('text_columns')}",
                f"dataset_source: {metadata.get('dataset_source')}",
                f"dataset_note: {metadata.get('dataset_note')}",
                f"eval_split_origin: {metadata.get('eval_split_origin')}",
                f"num_labels: {metadata.get('num_labels')}",
                f"labels: {metadata.get('label_list')}",
                f"metrica principal: {metadata.get('metric_name')}",
                f"input_ids.shape de ejemplo: {forward_debug.get('input_ids_shape')}",
                f"logits.shape de ejemplo: {forward_debug.get('logits_shape')}",
                f"best checkpoint downstream: {summary.get('best_model_checkpoint')}",
                f"output_dir: {summary['task_output_dir']}",
                "",
                "Train metrics:",
            ]
        )
        if summary["train_metrics"]:
            for key, value in summary["train_metrics"].items():
                lines.append(f"- {key}: {value}")
        else:
            lines.append("- no se encontraron train metrics")

        lines.append("")
        lines.append("Eval metrics:")
        if summary["eval_metrics"]:
            for key, value in summary["eval_metrics"].items():
                lines.append(f"- {key}: {value}")
        else:
            lines.append("- no se encontraron eval metrics")

        lines.append("")
        lines.append("Checkpoints downstream guardados:")
        if summary["checkpoint_dirs"]:
            for checkpoint_dir in summary["checkpoint_dirs"]:
                lines.append(f"- {checkpoint_dir}")
        else:
            lines.append("- no se encontraron checkpoint-*")
        if summary.get("saved_model_root"):
            lines.append(f"- modelo final guardado en: {summary['saved_model_root']}")
        lines.append("")

    return "\n".join(lines) + "\n"


def build_readme(args, selected_checkpoint_info: Dict[str, str], summaries: List[Dict[str, object]], script_name: str) -> str:
    by_task = {summary["task_name"]: summary for summary in summaries}
    xnli_metadata = by_task.get("xnli_es", {}).get("task_metadata", {})
    intertass_metadata = by_task.get("intertass2020", {}).get("task_metadata", {})

    return f"""PASO 9 SERIO - README
=====================

Objetivo
- Ejecutar fine-tuning downstream/extrinseco en espanol a partir del modelo entrenado en el pipeline principal BETO + rama auxiliar scanpath.
- En downstream se usa solo el encoder principal cargado desde el checkpoint seleccionado.
- No se usa scanpath, no se usa GRU y no se usa loss auxiliar durante downstream.
- Cada tarea crea una cabeza de clasificacion nueva mediante AutoModelForSequenceClassification.

Checkpoint seleccionado
- main_pipeline_dir: {selected_checkpoint_info['main_pipeline_dir']}
- checkpoint seleccionado: {selected_checkpoint_info['selected_checkpoint']}
- criterio de seleccion: {selected_checkpoint_info['selection_kind']}
- motivo: {selected_checkpoint_info['selection_reason']}

Criterio de seleccion implementado
1. Usar best_checkpoint si existe.
2. Usar best checkpoint registrado en logs o trainer_state si existe.
3. Usar checkpoint con mejor metrica registrada si no hay best directo.
4. Usar ultimo checkpoint disponible si no hay best.
5. Usar el directorio raiz como fallback.

Tareas usadas
- XNLIes
  tipo: {xnli_metadata.get('task_type')}
  columnas usadas: {xnli_metadata.get('text_columns')}
  num_labels: {xnli_metadata.get('num_labels')}
  metrica principal: accuracy

- InterTass2020
  tipo: {intertass_metadata.get('task_type')}
  columnas usadas: {intertass_metadata.get('text_columns')}
  num_labels: {intertass_metadata.get('num_labels')}
  metrica principal: macro-F1

Configuracion
- num_train_epochs = {args.num_train_epochs}
- max_train_samples = {args.max_train_samples}
- max_eval_samples = {args.max_eval_samples}
- per_device_train_batch_size = {args.per_device_train_batch_size}
- per_device_eval_batch_size = {args.per_device_eval_batch_size}
- max_seq_length = {args.max_seq_length}
- learning_rate = {args.learning_rate}
- seed = {args.seed}
- evaluation_strategy = {args.evaluation_strategy}
- save_strategy = {args.save_strategy}
- load_best_model_at_end = {args.load_best_model_at_end}

Que se reutiliza del paso 9 anterior
- Se reutiliza el esquema de wrapper que ejecuta una tarea downstream por vez.
- Se reutiliza train_spanish_downstream_baseline.py para cargar datasets, tokenizar, entrenar y guardar metricas.
- Se reutiliza la logica de resumen por tarea con shapes de ejemplo, metricas y directorios de salida.

Que cambia respecto del smoke test
- La corrida por defecto pasa de una prueba liviana a 10 epochs, hasta 5000 ejemplos de train y hasta 1000 de evaluacion.
- Se elige automaticamente el checkpoint del pipeline principal full 8ep.
- Se documenta explicitamente el checkpoint usado, el criterio de seleccion y el mejor checkpoint downstream por tarea.
- Se corren solo las dos tareas solicitadas: XNLIes e InterTass2020.

Comparabilidad con literatura
- Esta corrida sigue el espiritu de la literatura: fine-tuning downstream estandar, evaluacion por tarea, uso del encoder preentrenado y sin gaze/scanpath en downstream.
- La configuracion es mas cercana a una corrida experimental real que a un smoke test.
- No replica grillas grandes de multiples seeds ni multiples learning rates; queda documentada como una version seria reducida.
"""


def shell_join(command: List[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def build_commands_file(args, selected_checkpoint_info: Dict[str, str], summaries: List[Dict[str, object]], script_name: str) -> str:
    wrapper_command = [
        sys.executable,
        script_name,
        "--main_pipeline_dir",
        args.main_pipeline_dir,
        "--output_dir",
        args.output_dir,
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
        "--evaluation_strategy",
        args.evaluation_strategy,
        "--save_strategy",
        args.save_strategy,
        "--save_total_limit",
        str(args.save_total_limit),
    ]
    wrapper_command.extend(bool_flag("load_best_model_at_end", args.load_best_model_at_end))

    lines = [
        "COMANDOS Y FUNCIONES - PASO 9 SERIO",
        "====================================",
        "",
        "Comando ejecutado",
        f"- {shell_join(wrapper_command)}",
        "",
        "Script principal",
        f"- {script_name}",
        "",
        "Script downstream reutilizado",
        "- train_spanish_downstream_baseline.py",
        "",
        "Funciones principales llamadas",
        "- resolve_main_pipeline_checkpoint",
        "- parse_best_checkpoint_from_logs",
        "- resolve_metric_best_checkpoint",
        "- resolve_latest_checkpoint",
        "- build_task_command",
        "- run_task",
        "- train_spanish_downstream_baseline.load_task_dataset",
        "- train_spanish_downstream_baseline.tokenize_dataset",
        "- train_spanish_downstream_baseline.build_forward_debug",
        "- Trainer.train",
        "- Trainer.evaluate",
        "",
        "Checkpoint seleccionado",
        f"- ruta: {selected_checkpoint_info['selected_checkpoint']}",
        f"- criterio: {selected_checkpoint_info['selection_kind']}",
        f"- motivo: {selected_checkpoint_info['selection_reason']}",
        "",
        "Archivos generados",
        f"- {Path(args.output_dir) / 'README_paso_9_serio.txt'}",
        f"- {Path(args.output_dir) / 'salida_spanish_downstream_serio.txt'}",
        f"- {Path(args.output_dir) / 'comandos_y_funciones.txt'}",
    ]
    for summary in summaries:
        lines.append(f"- {summary['task_output_dir']}")

    lines.extend(["", "Comandos internos por tarea"])
    for summary in summaries:
        lines.append(f"- {shell_join(summary['command'])}")

    return "\n".join(lines) + "\n"


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_checkpoint_info = resolve_main_pipeline_checkpoint(Path(args.main_pipeline_dir))
    selected_checkpoint_path = Path(selected_checkpoint_info["selected_checkpoint"])
    if not is_valid_model_dir(selected_checkpoint_path):
        raise FileNotFoundError(
            "El checkpoint seleccionado no es un directorio de modelo cargable: "
            f"{selected_checkpoint_info['selected_checkpoint']}"
        )

    summaries = [run_task(args, selected_checkpoint_info, task_name, output_dir) for task_name in TASK_ORDER]
    script_name = Path(__file__).name

    (output_dir / "salida_spanish_downstream_serio.txt").write_text(
        build_debug_output(args, selected_checkpoint_info, summaries),
        encoding="utf-8",
    )
    (output_dir / "README_paso_9_serio.txt").write_text(
        build_readme(args, selected_checkpoint_info, summaries, script_name),
        encoding="utf-8",
    )
    (output_dir / "comandos_y_funciones.txt").write_text(
        build_commands_file(args, selected_checkpoint_info, summaries, script_name),
        encoding="utf-8",
    )

    print(f"Wrote {output_dir / 'README_paso_9_serio.txt'}")
    print(f"Wrote {output_dir / 'salida_spanish_downstream_serio.txt'}")
    print(f"Wrote {output_dir / 'comandos_y_funciones.txt'}")
    for summary in summaries:
        print(f"Finished downstream task {summary['task_name']} in {summary['task_output_dir']}")


if __name__ == "__main__":
    main()
