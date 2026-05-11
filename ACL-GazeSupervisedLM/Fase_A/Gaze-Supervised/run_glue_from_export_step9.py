import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_MODEL = "Pasos/paso_8/export_backbone_hf"
DEFAULT_OUTPUT_DIR = "Pasos/paso_9"
TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run standard downstream GLUE fine-tuning from the exported backbone, without scanpath."
    )
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL, help="Exported backbone directory from PASO 8.")
    parser.add_argument("--task_name", default="sst2", help="Primary GLUE task to run.")
    parser.add_argument("--secondary_task_name", default="rte", help="Optional second GLUE task used to verify sentence-pair input handling.")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Directory where paso_9 artifacts will be written.")
    parser.add_argument("--max_train_samples", type=int, default=200, help="Maximum number of training examples per task.")
    parser.add_argument("--max_eval_samples", type=int, default=100, help="Maximum number of evaluation examples per task.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of epochs for the downstream smoke run.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Evaluation batch size.")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Maximum tokenized sequence length.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate used for downstream fine-tuning.")
    parser.add_argument("--seed", type=int, default=13, help="Seed passed to the downstream baseline.")
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


def get_num_labels(task_name: str) -> int:
    raw = load_dataset("glue", task_name)
    if task_name == "stsb":
        return 1
    return len(raw["train"].features["label"].names)


def run_forward_debug(model_name_or_path: str, task_name: str, max_seq_length: int) -> Dict[str, object]:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    num_labels = get_num_labels(task_name)
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        finetuning_task=task_name,
    )
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=config)
    model.eval()

    raw = load_dataset("glue", task_name)
    sample = raw["train"][0]
    sentence1_key, sentence2_key = TASK_TO_KEYS[task_name]
    texts = (sample[sentence1_key],) if sentence2_key is None else (sample[sentence1_key], sample[sentence2_key])
    batch = tokenizer(*texts, truncation=True, max_length=max_seq_length, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**batch)
    return {
        "tokenizer": tokenizer.__class__.__name__,
        "model_type": config.model_type,
        "num_labels": int(config.num_labels),
        "input_ids_shape": tuple(batch["input_ids"].shape),
        "logits_shape": tuple(outputs.logits.shape),
    }


def build_baseline_command(args, task_name: str, task_output_dir: Path) -> List[str]:
    return [
        sys.executable,
        "train_glue_LM_baseline.py",
        "--model_name_or_path", args.model_name_or_path,
        "--task_name", task_name,
        "--output_dir", str(task_output_dir),
        "--do_train",
        "--do_eval",
        "--train_as_val", "false",
        "--max_train_samples", str(args.max_train_samples),
        "--max_eval_samples", str(args.max_eval_samples),
        "--num_train_epochs", str(args.num_train_epochs),
        "--per_device_train_batch_size", str(args.per_device_train_batch_size),
        "--per_device_eval_batch_size", str(args.per_device_eval_batch_size),
        "--learning_rate", str(args.learning_rate),
        "--max_seq_length", str(args.max_seq_length),
        "--seed", str(args.seed),
        "--evaluation_strategy", "epoch",
        "--save_strategy", "epoch",
        "--save_total_limit", "1",
        "--load_best_model_at_end", "true",
        "--metric_for_best_model", "eval_loss",
        "--greater_is_better", "false",
        "--logging_strategy", "steps",
        "--logging_steps", "10",
        "--report_to", "none",
        "--overwrite_output_dir", "True",
    ]


def run_task(args, task_name: str, base_output_dir: Path) -> Dict[str, object]:
    task_output_dir = base_output_dir / f"glue_{task_name}"
    task_output_dir.mkdir(parents=True, exist_ok=True)

    forward_debug = run_forward_debug(args.model_name_or_path, task_name, args.max_seq_length)
    command = build_baseline_command(args, task_name, task_output_dir)
    result = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
        check=True,
    )

    train_metrics = parse_key_value_file(task_output_dir / "train_results.txt")
    eval_metrics = extract_eval_metrics_from_trainer_state(task_output_dir / "trainer_state.json")
    checkpoint_dirs = sorted([str(path) for path in task_output_dir.glob("checkpoint-*") if path.is_dir()])
    saved_model_root = str(task_output_dir) if (task_output_dir / "pytorch_model.bin").exists() else None

    return {
        "task_name": task_name,
        "task_output_dir": str(task_output_dir),
        "command": command,
        "stdout_tail": "\n".join(result.stdout.strip().splitlines()[-10:]) if result.stdout else "",
        "stderr_tail": "\n".join(result.stderr.strip().splitlines()[-10:]) if result.stderr else "",
        "forward_debug": forward_debug,
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "checkpoint_dirs": checkpoint_dirs,
        "saved_model_root": saved_model_root,
    }


def build_debug_output(args, summaries: List[Dict[str, object]]) -> str:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    lines: List[str] = [
        "PASO 9 - Fine-tuning downstream estandar desde el backbone exportado",
        "",
        "----------------------------------------",
        "Configuracion",
        "----------------------------------------",
        f"model_name_or_path: {args.model_name_or_path}",
        f"tokenizer: {tokenizer.__class__.__name__}",
        f"model_type: {config.model_type}",
        f"tasks ejecutadas: {[summary['task_name'] for summary in summaries]}",
        f"max_train_samples: {args.max_train_samples}",
        f"max_eval_samples: {args.max_eval_samples}",
        f"epochs: {args.num_train_epochs}",
        f"batch_size: {args.per_device_train_batch_size}",
        f"lr: {args.learning_rate}",
        f"max_seq_length: {args.max_seq_length}",
        f"seed: {args.seed}",
        "",
    ]

    for summary in summaries:
        forward_debug = summary["forward_debug"]
        lines.extend(
            [
                "----------------------------------------",
                f"Task {summary['task_name']}",
                "----------------------------------------",
                f"task_name: {summary['task_name']}",
                f"num_labels: {forward_debug['num_labels']}",
                f"input_ids.shape: {forward_debug['input_ids_shape']}",
                f"logits.shape: {forward_debug['logits_shape']}",
                f"task_output_dir: {summary['task_output_dir']}",
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
            lines.append("- no se encontraron checkpoints checkpoint-* en la salida")
        if summary.get("saved_model_root") is not None:
            lines.append(f"- best/final model saved in root output dir: {summary['saved_model_root']}")

        lines.extend(
            [
                "",
                "Nota final por tarea:",
                "- downstream corrio usando solo el backbone exportado, sin rama scanpath.",
                "- no se uso GRU ni loss auxiliar en este paso.",
                "",
            ]
        )

    return "\n".join(lines) + "\n"


def build_readme(args, summaries: List[Dict[str, object]], script_name: str) -> str:
    return f"""PASO 9 - README
================

Que se hizo
- Se creo un wrapper minimo llamado {script_name}.
- El wrapper reutiliza el baseline downstream existente train_glue_LM_baseline.py.
- Carga como model_name_or_path el export del paso 8: {args.model_name_or_path}
- Corre fine-tuning downstream estandar sin rama scanpath.
- En esta implementacion se ejecutaron dos tareas chicas para verificar dos formatos de input:
  * {summaries[0]['task_name'] if summaries else args.task_name}
  * {summaries[1]['task_name'] if len(summaries) > 1 else args.secondary_task_name}

Que se verifico
- Que el export del paso 8 puede usarse como model_name_or_path downstream.
- Que tokenizer/config/modelo downstream cargan correctamente desde ese directorio.
- Que el fine-tuning downstream corre.
- Que la evaluacion downstream corre.
- Que la rama scanpath ya no participa.
- Que el pipeline completo queda cerrado: pretraining con scanpath -> export backbone -> downstream estandar.

Que NO se implemento todavia
- No se reintrodujo scanpath en downstream.
- No se uso la GRU.
- No se usaron losses auxiliares.
- No se hizo hyperparameter search grande.
- No se refactorizo masivamente el repo.
- No se avanzo a experimentos finales grandes dentro de este paso.

Archivos modificados
- Pasos/README.txt

Archivos nuevos creados
- {script_name}
- Pasos/paso_9/README_paso_9.txt
- Pasos/paso_9/salida_glue_debug.txt
- Pasos/paso_9/comandos_y_funciones.txt
- Pasos/paso_9/glue_*/ con las corridas downstream de cada tarea

Explicacion breve de que este paso usa solo la rama principal
- El modelo downstream se carga con AutoModelForSequenceClassification desde el backbone exportado.
- Eso significa que se reutiliza solo el encoder principal BERT/BETO.
- La cabeza downstream de clasificacion es nueva y propia de la tarea.
- No se cargan ni usan sp_encoder.*, scanpath_mlm_head.* ni la logica de scanpath en el forward downstream.

Por que este paso cierra el pipeline
- El pipeline completo ahora queda asi:
  1. pretraining con supervision auxiliar scanpath
  2. export del backbone principal
  3. fine-tuning downstream estandar solo con el backbone
- Esto materializa exactamente la separacion buscada entre entrenamiento con gaze y uso downstream sin gaze.

Aclaracion explicita
- En este paso NO se uso la rama scanpath.
- No se uso GRU.
- No se uso loss auxiliar.

Parametros usados en esta corrida
- model_name_or_path = {args.model_name_or_path}
- tasks = {[summary['task_name'] for summary in summaries]}
- max_train_samples = {args.max_train_samples}
- max_eval_samples = {args.max_eval_samples}
- num_train_epochs = {args.num_train_epochs}
- per_device_train_batch_size = {args.per_device_train_batch_size}
- per_device_eval_batch_size = {args.per_device_eval_batch_size}
- max_seq_length = {args.max_seq_length}
- learning_rate = {args.learning_rate}
- seed = {args.seed}
"""


def build_commands_file(args, summaries: List[Dict[str, object]], script_name: str) -> str:
    relative_output = Path(args.output_dir).as_posix()
    command = (
        f".\\.venv\\Scripts\\python.exe {script_name} "
        f"--model_name_or_path {args.model_name_or_path} "
        f"--task_name {args.task_name} "
        f"--secondary_task_name {args.secondary_task_name} "
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
    return f"""COMANDOS Y FUNCIONES - PASO 9
=============================

Comando ejecutado
- {command}

Script principal o wrapper usado
- {script_name}
- baseline reutilizado: train_glue_LM_baseline.py

Funciones principales llamadas
- AutoTokenizer.from_pretrained
- AutoConfig.from_pretrained
- AutoModelForSequenceClassification.from_pretrained
- subprocess.run(... train_glue_LM_baseline.py ...)
- parse_key_value_file
- extract_eval_metrics_from_trainer_state

Output generado
- Pasos/paso_9/salida_glue_debug.txt con el resumen downstream.
- Pasos/paso_9/README_paso_9.txt con la explicacion del cierre del pipeline.
- Pasos/paso_9/comandos_y_funciones.txt con trazabilidad de la corrida.
- Pasos/paso_9/glue_{summaries[0]['task_name'] if summaries else args.task_name}/
- Pasos/paso_9/glue_{summaries[1]['task_name'] if len(summaries) > 1 else args.secondary_task_name}/
"""


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task_names = [args.task_name]
    secondary = (args.secondary_task_name or "").strip().lower()
    if secondary and secondary not in task_names:
        task_names.append(secondary)

    summaries = [run_task(args, task_name, output_dir) for task_name in task_names]
    script_name = Path(__file__).name

    (output_dir / "salida_glue_debug.txt").write_text(build_debug_output(args, summaries), encoding="utf-8")
    (output_dir / "README_paso_9.txt").write_text(build_readme(args, summaries, script_name), encoding="utf-8")
    (output_dir / "comandos_y_funciones.txt").write_text(build_commands_file(args, summaries, script_name), encoding="utf-8")

    print(f"Wrote {output_dir / 'salida_glue_debug.txt'}")
    print(f"Wrote {output_dir / 'README_paso_9.txt'}")
    print(f"Wrote {output_dir / 'comandos_y_funciones.txt'}")
    for summary in summaries:
        print(f"Finished downstream task {summary['task_name']} in {summary['task_output_dir']}")


if __name__ == "__main__":
    main()
