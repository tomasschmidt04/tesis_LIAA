import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from transformers import AutoConfig, AutoTokenizer


DEFAULT_L7B_DIR = "Pasos/paso_7b"
DEFAULT_OUTPUT_DIR = "Pasos/paso_9_es_l7b"
TASK_ORDER = ["xnli_es", "intertass2020"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Spanish downstream fine-tuning from the best available checkpoint inside l7b."
    )
    parser.add_argument("--l7b_dir", default=DEFAULT_L7B_DIR, help="Directory that contains the l7b experiment checkpoints.")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Directory where paso_9_es_l7b artifacts will be written.")
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


def is_valid_model_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    has_config = (path / "config.json").exists()
    has_weights = (path / "pytorch_model.bin").exists() or (path / "model.safetensors").exists()
    return has_config and has_weights


def parse_best_checkpoint_from_summary(l7b_dir: Path) -> Optional[Path]:
    summary_path = l7b_dir / "salida_training_step7.txt"
    if not summary_path.exists():
        return None

    best_checkpoint: Optional[Path] = None
    for line in summary_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("best_checkpoint:"):
            candidate = line.split(":", 1)[1].strip()
            best_checkpoint = Path(candidate)
            if not best_checkpoint.is_absolute():
                best_checkpoint = Path.cwd() / best_checkpoint
    return best_checkpoint


def parse_best_model_checkpoint_from_trainer_state(l7b_dir: Path) -> Optional[Path]:
    trainer_state_paths = list(l7b_dir.rglob("trainer_state.json"))
    for state_path in trainer_state_paths:
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        best_model_checkpoint = state.get("best_model_checkpoint")
        if best_model_checkpoint:
            resolved = Path(best_model_checkpoint)
            if not resolved.is_absolute():
                resolved = Path.cwd() / resolved
            return resolved
    return None


def resolve_latest_checkpoint(l7b_dir: Path) -> Optional[Tuple[Path, str]]:
    checkpoint_final = l7b_dir / "checkpoint_final"
    if is_valid_model_dir(checkpoint_final):
        return checkpoint_final, "latest_checkpoint_final"

    epoch_candidates: List[Tuple[int, Path]] = []
    for child in l7b_dir.iterdir():
        match = re.fullmatch(r"checkpoint_epoch_(\d+)", child.name)
        if match and is_valid_model_dir(child):
            epoch_candidates.append((int(match.group(1)), child))

    if epoch_candidates:
        epoch_candidates.sort(key=lambda item: item[0], reverse=True)
        latest_epoch, latest_path = epoch_candidates[0]
        return latest_path, f"latest_checkpoint_epoch_{latest_epoch}"

    generic_candidates = sorted(
        [path for path in l7b_dir.glob("checkpoint*") if path.is_dir() and is_valid_model_dir(path)],
        key=lambda path: path.name,
        reverse=True,
    )
    if generic_candidates:
        return generic_candidates[0], "latest_generic_checkpoint"

    return None


def resolve_l7b_checkpoint(l7b_dir: Path) -> Dict[str, str]:
    summary_best = parse_best_checkpoint_from_summary(l7b_dir)
    if summary_best is not None and is_valid_model_dir(summary_best):
        return {
            "selected_checkpoint": str(summary_best),
            "selection_kind": "best",
            "selection_reason": "best_checkpoint registrado en salida_training_step7.txt y directorio valido encontrado.",
        }

    trainer_state_best = parse_best_model_checkpoint_from_trainer_state(l7b_dir)
    if trainer_state_best is not None and is_valid_model_dir(trainer_state_best):
        return {
            "selected_checkpoint": str(trainer_state_best),
            "selection_kind": "best",
            "selection_reason": "best_model_checkpoint registrado en trainer_state.json y directorio valido encontrado.",
        }

    best_dir = l7b_dir / "best_checkpoint"
    if is_valid_model_dir(best_dir):
        return {
            "selected_checkpoint": str(best_dir),
            "selection_kind": "best",
            "selection_reason": "directorio best_checkpoint presente dentro de l7b.",
        }

    latest = resolve_latest_checkpoint(l7b_dir)
    if latest is not None:
        latest_path, latest_kind = latest
        return {
            "selected_checkpoint": str(latest_path),
            "selection_kind": "latest",
            "selection_reason": f"no se encontro best explicito utilizable; se uso {latest_kind}.",
        }

    return {
        "selected_checkpoint": str(l7b_dir),
        "selection_kind": "root",
        "selection_reason": "no se encontraron subcheckpoints validos; se usa l7b_dir como fallback.",
    }


def build_baseline_command(args, selected_checkpoint: str, task_name: str, task_output_dir: Path) -> List[str]:
    return [
        sys.executable,
        "train_spanish_downstream_baseline.py",
        "--model_name_or_path", selected_checkpoint,
        "--task_name", task_name,
        "--output_dir", str(task_output_dir),
        "--max_train_samples", str(args.max_train_samples),
        "--max_eval_samples", str(args.max_eval_samples),
        "--num_train_epochs", str(args.num_train_epochs),
        "--per_device_train_batch_size", str(args.per_device_train_batch_size),
        "--per_device_eval_batch_size", str(args.per_device_eval_batch_size),
        "--max_seq_length", str(args.max_seq_length),
        "--learning_rate", str(args.learning_rate),
        "--seed", str(args.seed),
    ]


def run_task(args, selected_checkpoint_info: Dict[str, str], task_name: str, base_output_dir: Path) -> Dict[str, object]:
    task_output_dir = base_output_dir / task_name
    task_output_dir.mkdir(parents=True, exist_ok=True)

    command = build_baseline_command(args, selected_checkpoint_info["selected_checkpoint"], task_name, task_output_dir)
    result = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
        check=True,
    )

    train_metrics = parse_key_value_file(task_output_dir / "train_results.txt")
    eval_metrics = extract_eval_metrics_from_trainer_state(task_output_dir / "trainer_state.json")
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


def build_debug_output(args, selected_checkpoint_info: Dict[str, str], summaries: List[Dict[str, object]]) -> str:
    tokenizer = AutoTokenizer.from_pretrained(selected_checkpoint_info["selected_checkpoint"], use_fast=True)
    config = AutoConfig.from_pretrained(selected_checkpoint_info["selected_checkpoint"])

    lines: List[str] = [
        "PASO 9 ES L7B - Fine-tuning downstream en espanol desde l7b",
        "",
        "----------------------------------------",
        "Seleccion del checkpoint",
        "----------------------------------------",
        f"l7b_dir: {args.l7b_dir}",
        f"checkpoint seleccionado: {selected_checkpoint_info['selected_checkpoint']}",
        f"criterio de seleccion: {selected_checkpoint_info['selection_kind']}",
        f"motivo: {selected_checkpoint_info['selection_reason']}",
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
            lines.append("- no se encontraron eval metrics en trainer_state.json")

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
                "- downstream uso solo la rama principal del modelo seleccionado desde l7b.",
                "- no se uso scanpath, no se uso GRU y no se uso loss auxiliar en downstream.",
                "",
            ]
        )

    return "\n".join(lines) + "\n"


def build_readme(args, selected_checkpoint_info: Dict[str, str], summaries: List[Dict[str, object]], script_name: str) -> str:
    xnli_summary = next((summary for summary in summaries if summary["task_name"] == "xnli_es"), None)
    intertass_summary = next((summary for summary in summaries if summary["task_name"] == "intertass2020"), None)

    xnli_metadata = xnli_summary["task_metadata"] if xnli_summary else {}
    intertass_metadata = intertass_summary["task_metadata"] if intertass_summary else {}

    return f"""PASO 9 ES L7B - README
===========================

Que cambia respecto del paso 9 anterior
- Ya no se usa por defecto el export del paso 8 como model_name_or_path.
- Ahora se selecciona automaticamente el mejor checkpoint posible dentro de l7b.
- Ya no se corren tareas GLUE en ingles.
- Ahora se corren solo dos tareas downstream en espanol:
  * xnli_es
  * intertass2020

Objetivo de este paso
- Reutilizar el modelo entrenado durante bastante tiempo en l7b como punto de partida downstream.
- Evaluar transferencia extrinseca en el mismo idioma del pretraining, en lugar de usar tareas en ingles.
- Mantener downstream sin scanpath, sin GRU y sin losses auxiliares.

Como se eligio el checkpoint de l7b
- l7b_dir usado: {args.l7b_dir}
- checkpoint seleccionado: {selected_checkpoint_info['selected_checkpoint']}
- criterio aplicado: {selected_checkpoint_info['selection_kind']}
- motivo: {selected_checkpoint_info['selection_reason']}

Logica de seleccion implementada
1. Si existe un best checkpoint explicito o registrado, se usa ese.
2. Si no existe best utilizable, se usa el checkpoint mas avanzado disponible.
3. Si no hay subcheckpoints validos, se usa l7b_dir como fallback.
4. Todo eso queda reflejado explicitamente en la salida debug.

Por que ahora las tareas son en espanol
- El checkpoint l7b fue entrenado con BETO y supervision auxiliar sobre datos en espanol.
- Para evaluar transferencia extrinseca tiene mas sentido medirlo sobre tareas downstream en el mismo idioma.
- Por eso se eligieron una tarea de inference sobre pares de oraciones y una tarea de clasificacion de texto unico, ambas en espanol.

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

Que parte del pipeline anterior se reutiliza
- Se mantiene la idea del wrapper del paso 9: un script orquestador llama un baseline downstream por tarea y luego resume resultados.
- Se sigue usando AutoTokenizer, AutoConfig y AutoModelForSequenceClassification para downstream estandar.
- La cabeza downstream sigue siendo nueva y propia de la tarea.

Que parte nueva se agrego
- Un wrapper nuevo llamado {script_name}.
- Un baseline minimo para tareas downstream en espanol.
- Seleccion automatica del checkpoint de l7b.
- Soporte explicito para xnli_es e intertass2020.
- Salidas nuevas en {args.output_dir}.

Aclaracion importante sobre downstream
- En este paso downstream sigue usando solo la rama principal del modelo.
- No se reintroduce scanpath.
- No se usa GRU.
- No se usa loss auxiliar.
- Los pesos reutilizados son los del backbone principal cargados desde el checkpoint seleccionado.

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
- Esta configuracion intenta acercarse mas al regimen del paper original que el paso 9 anterior ultra liviano.
- Aun asi sigue siendo una version reducida para no volver impracticable la corrida.
- Por eso no se replica la grilla completa de multiples seeds, multiples tamanos de dataset y 20 epochs u otros barridos mas costosos.
"""


def build_commands_file(args, selected_checkpoint_info: Dict[str, str], summaries: List[Dict[str, object]], script_name: str) -> str:
    relative_output = Path(args.output_dir).as_posix()
    wrapper_command = (
        f".\\.venv\\Scripts\\python.exe {script_name} "
        f"--l7b_dir {args.l7b_dir} "
        f"--output_dir {relative_output} "
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
        "COMANDOS Y FUNCIONES - PASO 9 ES L7B",
        "====================================",
        "",
        "Comando ejecutado",
        f"- {wrapper_command}",
        "",
        "Script principal usado",
        f"- {script_name}",
        "- baseline reutilizado/minimo: train_spanish_downstream_baseline.py",
        "",
        "Checkpoint seleccionado",
        f"- ruta: {selected_checkpoint_info['selected_checkpoint']}",
        f"- criterio: {selected_checkpoint_info['selection_kind']}",
        f"- motivo: {selected_checkpoint_info['selection_reason']}",
        "",
        "Funciones principales llamadas",
        "- resolve_l7b_checkpoint",
        "- AutoTokenizer.from_pretrained",
        "- AutoConfig.from_pretrained",
        "- subprocess.run(... train_spanish_downstream_baseline.py ...)",
        "- parse_key_value_file",
        "- extract_eval_metrics_from_trainer_state",
        "",
        "Outputs generados",
        f"- {Path(args.output_dir) / 'salida_spanish_downstream_debug.txt'}",
        f"- {Path(args.output_dir) / 'README_paso_9_es_l7b.txt'}",
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

    selected_checkpoint_info = resolve_l7b_checkpoint(Path(args.l7b_dir))
    selected_checkpoint_path = Path(selected_checkpoint_info["selected_checkpoint"])
    if not is_valid_model_dir(selected_checkpoint_path):
        raise FileNotFoundError(
            "The selected l7b checkpoint is not a loadable model directory: "
            f"{selected_checkpoint_info['selected_checkpoint']}"
        )

    summaries = [run_task(args, selected_checkpoint_info, task_name, output_dir) for task_name in TASK_ORDER]
    script_name = Path(__file__).name

    (output_dir / "salida_spanish_downstream_debug.txt").write_text(
        build_debug_output(args, selected_checkpoint_info, summaries),
        encoding="utf-8",
    )
    (output_dir / "README_paso_9_es_l7b.txt").write_text(
        build_readme(args, selected_checkpoint_info, summaries, script_name),
        encoding="utf-8",
    )
    (output_dir / "comandos_y_funciones.txt").write_text(
        build_commands_file(args, selected_checkpoint_info, summaries, script_name),
        encoding="utf-8",
    )

    print(f"Wrote {output_dir / 'salida_spanish_downstream_debug.txt'}")
    print(f"Wrote {output_dir / 'README_paso_9_es_l7b.txt'}")
    print(f"Wrote {output_dir / 'comandos_y_funciones.txt'}")
    for summary in summaries:
        print(f"Finished downstream task {summary['task_name']} in {summary['task_output_dir']}")


if __name__ == "__main__":
    main()
