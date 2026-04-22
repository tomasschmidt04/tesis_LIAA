import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer

from Gazesup_bert_combined_mlm_model import Gazesup_BERTForCombinedMaskedLM
from measured_scanpath_utils import build_measured_single_sentence_features, load_measured_scanpath_dataset
from train_mlm_scanpath_step5 import (
    build_static_masked_inputs_and_labels,
    collate_measured_mlm_batch,
    move_tensor_batch_to_device,
    set_seed,
)


DEFAULT_OUTPUT_DIR = "Pasos/paso_7"
DEFAULT_CHECKPOINT_DIRNAME = "checkpoint_final"
NUM_DEBUG_BATCHES = 2


def str2bool(value):
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a larger combined MLM training with main + scanpath losses and reusable checkpoints."
    )
    parser.add_argument("--measured_scanpath_file", required=True, help="Path to a JSON/JSONL/CSV file with at least text and word_id fields.")
    parser.add_argument("--model_name_or_path", default="dccuchile/bert-base-spanish-wwm-cased", help="BERT-style model/tokenizer name or path. Examples: bert-base-cased, dccuchile/bert-base-spanish-wwm-cased.")
    parser.add_argument("--measured_text_field", default="text", help="Column that contains the plain text consumed by the tokenizer.")
    parser.add_argument("--measured_word_id_field", default="word_id", help="Column that contains the lexical 1-based scanpath positions.")
    parser.add_argument("--split", default="train", help="Dataset split to use after loading the measured file.")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Directory where paso_7 artifacts will be written.")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Maximum length passed to the BERT tokenizer.")
    parser.add_argument("--max_train_samples", type=int, default=128, help="Maximum number of measured examples used for training in this step.")
    parser.add_argument("--max_eval_samples", type=int, default=32, help="Maximum number of held-out measured examples used for lightweight evaluation.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Mini-batch size used by the training loop.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4, help="Mini-batch size used by the lightweight evaluation loop.")
    parser.add_argument("--num_train_epochs", type=int, default=2, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate used by AdamW.")
    parser.add_argument("--max_masked_positions", type=int, default=3, help="Maximum number of non-special tokens masked per example.")
    parser.add_argument("--aux_weight", type=float, default=0.3, help="Weight used in total_loss = main_mlm_loss + aux_weight * scanpath_mlm_loss.")
    parser.add_argument("--save_every_epoch", type=str2bool, default=True, help="Whether to save a checkpoint at the end of every epoch.")
    parser.add_argument("--remove_punctuation_space", action="store_true", help="Mirror the optional punctuation-space normalization used by the training scripts.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed used by the training loop.")
    parser.add_argument("--final_checkpoint_dirname", default=DEFAULT_CHECKPOINT_DIRNAME, help="Directory name used inside output_dir to save the final checkpoint.")
    return parser.parse_args()


def verify_bert_style_config(model_name_or_path: str):
    config = AutoConfig.from_pretrained(model_name_or_path)
    if getattr(config, "model_type", None) != "bert":
        raise ValueError(
            f"PASO 7 only supports BERT-style models for now. Received model_type={getattr(config, 'model_type', None)!r}"
        )
    return config


def preprocess_examples(dataset, tokenizer, args):
    processed_examples = []
    for example_index, example in enumerate(dataset):
        text = example[args.measured_text_field]
        feature = build_measured_single_sentence_features(
            text=text,
            word_id_value=example[args.measured_word_id_field],
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            remove_punctuation_space=args.remove_punctuation_space,
        )
        masked_input_ids, labels, masked_positions = build_static_masked_inputs_and_labels(
            input_ids=feature["input_ids"],
            attention_mask=feature["attention_mask"],
            tokenizer=tokenizer,
            max_masked_positions=args.max_masked_positions,
        )
        processed_examples.append(
            {
                "example_index": example_index,
                "text": text,
                "input_ids": masked_input_ids,
                "attention_mask": feature["attention_mask"],
                "token_type_ids": feature["token_type_ids"],
                "LM_word_ids": feature["word_ids"],
                "measured_word_ids": feature["measured_word_ids"][0],
                "measured_sp_len": feature["measured_sp_len"][0],
                "labels": labels,
                "masked_positions": masked_positions,
            }
        )
    return processed_examples


def build_train_eval_datasets(raw_dataset, max_train_samples: int, max_eval_samples: int):
    total_available = len(raw_dataset)
    train_count = min(max_train_samples, total_available)
    remaining = max(0, total_available - train_count)
    eval_count = min(max_eval_samples, remaining)

    train_dataset = raw_dataset.select(range(train_count))
    if eval_count > 0:
        eval_dataset = raw_dataset.select(range(train_count, train_count + eval_count))
    else:
        eval_dataset = None

    return train_dataset, eval_dataset


def compute_mean(values: List[float]):
    return sum(values) / len(values) if values else None


def evaluate_model(model, dataloader, device, aux_weight: float):
    model.eval()
    main_losses: List[float] = []
    scanpath_losses: List[float] = []
    total_losses: List[float] = []
    with torch.no_grad():
        for batch in dataloader:
            batch_on_device = move_tensor_batch_to_device(batch, device)
            outputs = model(
                input_ids=batch_on_device["input_ids"],
                attention_mask=batch_on_device["attention_mask"],
                token_type_ids=batch_on_device["token_type_ids"],
                LM_word_ids=batch_on_device["LM_word_ids"],
                measured_word_ids=batch_on_device["measured_word_ids"],
                measured_sp_len=batch_on_device["measured_sp_len"],
                labels=batch_on_device["labels"],
                aux_weight=aux_weight,
                return_dict=True,
            )
            main_losses.append(float(outputs.main_mlm_loss.detach().cpu().item()))
            scanpath_losses.append(float(outputs.scanpath_mlm_loss.detach().cpu().item()))
            total_losses.append(float(outputs.loss.detach().cpu().item()))
    model.train()
    return {
        "mean_main_mlm_loss": compute_mean(main_losses),
        "mean_scanpath_mlm_loss": compute_mean(scanpath_losses),
        "mean_total_loss": compute_mean(total_losses),
    }


def save_checkpoint(target_dir: Path, model, tokenizer):
    target_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target_dir)
    tokenizer.save_pretrained(target_dir)


def summarize_aux_weight(epoch_mean_main: float, epoch_mean_scan: float, aux_weight: float) -> List[str]:
    if epoch_mean_main is None or epoch_mean_scan is None or epoch_mean_main == 0:
        return ["- no hay suficientes datos para interpretar aux_weight."]
    ratio = epoch_mean_scan / epoch_mean_main
    weighted_ratio = (aux_weight * epoch_mean_scan) / epoch_mean_main
    lines = [
        f"- ratio scanpath/main: {ratio}",
        f"- ratio ponderado (aux_weight * scanpath)/main: {weighted_ratio}",
    ]
    if weighted_ratio > 1.5:
        lines.append("- sugerencia: la rama auxiliar pesa bastante; podria convenir bajar aux_weight a 0.1 o 0.2.")
    elif weighted_ratio < 0.5:
        lines.append("- sugerencia: la rama auxiliar pesa poco; si queres mas influencia scanpath, podria convenir subir aux_weight.")
    else:
        lines.append("- sugerencia: el aux_weight actual parece razonable para esta corrida.")
    return lines


def train_step7(args):
    set_seed(args.seed)
    config = verify_bert_style_config(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    raw_datasets = load_measured_scanpath_dataset(args.measured_scanpath_file)
    if args.split not in raw_datasets:
        raise ValueError(f"Split {args.split!r} not found. Available splits: {list(raw_datasets.keys())}")

    full_dataset = raw_datasets[args.split]
    train_dataset, eval_dataset = build_train_eval_datasets(full_dataset, args.max_train_samples, args.max_eval_samples)
    train_examples = preprocess_examples(train_dataset, tokenizer, args)
    eval_examples = preprocess_examples(eval_dataset, tokenizer, args) if eval_dataset is not None else []

    train_loader = DataLoader(
        train_examples,
        batch_size=args.per_device_train_batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_measured_mlm_batch(batch, tokenizer),
    )
    eval_loader = None
    if eval_examples:
        eval_loader = DataLoader(
            eval_examples,
            batch_size=args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_measured_mlm_batch(batch, tokenizer),
        )

    model = Gazesup_BERTForCombinedMaskedLM.from_pretrained(args.model_name_or_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    batch_debug: List[Dict[str, Any]] = []
    epoch_summaries: List[Dict[str, Any]] = []
    main_losses_all: List[float] = []
    scanpath_losses_all: List[float] = []
    total_losses_all: List[float] = []
    checkpoint_paths: List[str] = []
    best_checkpoint_path = None
    best_reference_total = None
    total_steps = 0
    training_ok = False
    checkpointing_ok = False

    for epoch_index in range(args.num_train_epochs):
        epoch_main_losses: List[float] = []
        epoch_scan_losses: List[float] = []
        epoch_total_losses: List[float] = []
        epoch_start = time.perf_counter()

        for batch_index, batch in enumerate(train_loader):
            optimizer.zero_grad()
            batch_on_device = move_tensor_batch_to_device(batch, device)
            outputs = model(
                input_ids=batch_on_device["input_ids"],
                attention_mask=batch_on_device["attention_mask"],
                token_type_ids=batch_on_device["token_type_ids"],
                LM_word_ids=batch_on_device["LM_word_ids"],
                measured_word_ids=batch_on_device["measured_word_ids"],
                measured_sp_len=batch_on_device["measured_sp_len"],
                labels=batch_on_device["labels"],
                aux_weight=args.aux_weight,
                return_dict=True,
            )
            total_loss = outputs.loss
            if total_loss is None or torch.isnan(total_loss).any():
                raise RuntimeError("PASO 7 produced an invalid total_loss.")

            total_loss.backward()
            optimizer.step()

            main_loss_value = float(outputs.main_mlm_loss.detach().cpu().item())
            scanpath_loss_value = float(outputs.scanpath_mlm_loss.detach().cpu().item())
            total_loss_value = float(total_loss.detach().cpu().item())

            epoch_main_losses.append(main_loss_value)
            epoch_scan_losses.append(scanpath_loss_value)
            epoch_total_losses.append(total_loss_value)
            main_losses_all.append(main_loss_value)
            scanpath_losses_all.append(scanpath_loss_value)
            total_losses_all.append(total_loss_value)
            total_steps += 1

            if len(batch_debug) < NUM_DEBUG_BATCHES:
                batch_debug.append(
                    {
                        "epoch": epoch_index + 1,
                        "batch_index": batch_index,
                        "input_ids_shape": tuple(batch["input_ids"].shape),
                        "labels_shape": tuple(batch["labels"].shape),
                        "main_mlm_logits_shape": tuple(outputs.main_mlm_logits.shape),
                        "scanpath_mlm_logits_shape": tuple(outputs.scanpath_mlm_logits.shape),
                        "main_mlm_loss": main_loss_value,
                        "scanpath_mlm_loss": scanpath_loss_value,
                        "total_loss": total_loss_value,
                    }
                )

        epoch_duration_sec = time.perf_counter() - epoch_start
        epoch_summary = {
            "epoch": epoch_index + 1,
            "mean_main_mlm_loss": compute_mean(epoch_main_losses),
            "mean_scanpath_mlm_loss": compute_mean(epoch_scan_losses),
            "mean_total_loss": compute_mean(epoch_total_losses),
            "num_steps": len(epoch_total_losses),
            "duration_sec": epoch_duration_sec,
        }

        if eval_loader is not None:
            epoch_summary["eval"] = evaluate_model(model, eval_loader, device, args.aux_weight)
            reference_total = epoch_summary["eval"]["mean_total_loss"]
        else:
            reference_total = epoch_summary["mean_total_loss"]

        if args.save_every_epoch:
            epoch_ckpt = Path(args.output_dir) / f"checkpoint_epoch_{epoch_index + 1}"
            save_checkpoint(epoch_ckpt, model, tokenizer)
            checkpoint_paths.append(str(epoch_ckpt))
            epoch_summary["checkpoint_saved_to"] = str(epoch_ckpt)
            checkpointing_ok = True

        if best_reference_total is None or (reference_total is not None and reference_total < best_reference_total):
            best_reference_total = reference_total
            best_ckpt = Path(args.output_dir) / "best_checkpoint"
            save_checkpoint(best_ckpt, model, tokenizer)
            best_checkpoint_path = str(best_ckpt)
            checkpointing_ok = True
            epoch_summary["best_checkpoint_updated"] = str(best_ckpt)

        epoch_summaries.append(epoch_summary)

    final_checkpoint_dir = Path(args.output_dir) / args.final_checkpoint_dirname
    save_checkpoint(final_checkpoint_dir, model, tokenizer)
    checkpoint_paths.append(str(final_checkpoint_dir))
    checkpointing_ok = True
    training_ok = True

    return {
        "config": config,
        "tokenizer_class": tokenizer.__class__.__name__,
        "device": str(device),
        "train_size": len(train_examples),
        "eval_size": len(eval_examples),
        "batch_debug": batch_debug,
        "epoch_summaries": epoch_summaries,
        "main_losses_all": main_losses_all,
        "scanpath_losses_all": scanpath_losses_all,
        "total_losses_all": total_losses_all,
        "total_steps": total_steps,
        "checkpoint_paths": checkpoint_paths,
        "best_checkpoint_path": best_checkpoint_path,
        "final_checkpoint_path": str(final_checkpoint_dir),
        "training_ok": training_ok,
        "checkpointing_ok": checkpointing_ok,
    }


def build_debug_output(args, summary: Dict[str, Any]) -> str:
    config = summary["config"]
    lines: List[str] = [
        "PASO 7 - Entrenamiento mas grande del modelo con losses combinadas + Beto",
        "",
        "----------------------------------------",
        "Configuracion",
        "----------------------------------------",
        f"dataset: {args.measured_scanpath_file}",
        f"model_name_or_path: {args.model_name_or_path}",
        f"tokenizer: {summary['tokenizer_class']}",
        f"vocab_size: {config.vocab_size}",
        f"model_type: {config.model_type}",
        f"hidden_size: {config.hidden_size}",
        f"num_hidden_layers: {config.num_hidden_layers}",
        f"max_train_samples: {summary['train_size']}",
        f"max_eval_samples: {summary['eval_size']}",
        f"epochs: {args.num_train_epochs}",
        f"batch_size: {args.per_device_train_batch_size}",
        f"lr: {args.learning_rate}",
        f"aux_weight: {args.aux_weight}",
        f"max_seq_length: {args.max_seq_length}",
        f"seed: {args.seed}",
        f"device: {summary['device']}",
        "",
    ]

    for debug_batch in summary["batch_debug"]:
        lines.extend(
            [
                "----------------------------------------",
                f"Batch debug {debug_batch['batch_index']} (epoch {debug_batch['epoch']})",
                "----------------------------------------",
                f"input_ids.shape: {debug_batch['input_ids_shape']}",
                f"labels.shape: {debug_batch['labels_shape']}",
                f"main_mlm_logits.shape: {debug_batch['main_mlm_logits_shape']}",
                f"scanpath_mlm_logits.shape: {debug_batch['scanpath_mlm_logits_shape']}",
                f"main_mlm_loss: {debug_batch['main_mlm_loss']}",
                f"scanpath_mlm_loss: {debug_batch['scanpath_mlm_loss']}",
                f"total_loss: {debug_batch['total_loss']}",
                "",
            ]
        )

    for epoch_summary in summary["epoch_summaries"]:
        lines.extend(
            [
                "----------------------------------------",
                f"Epoch {epoch_summary['epoch']} summary",
                "----------------------------------------",
                f"mean_main_mlm_loss: {epoch_summary['mean_main_mlm_loss']}",
                f"mean_scanpath_mlm_loss: {epoch_summary['mean_scanpath_mlm_loss']}",
                f"mean_total_loss: {epoch_summary['mean_total_loss']}",
                f"num_steps: {epoch_summary['num_steps']}",
                f"duration_sec: {epoch_summary['duration_sec']}",
            ]
        )
        if "checkpoint_saved_to" in epoch_summary:
            lines.append(f"checkpoint_saved_to: {epoch_summary['checkpoint_saved_to']}")
        if "best_checkpoint_updated" in epoch_summary:
            lines.append(f"best_checkpoint_updated: {epoch_summary['best_checkpoint_updated']}")
        if "eval" in epoch_summary:
            eval_summary = epoch_summary["eval"]
            lines.append("Evaluacion minima:")
            lines.append(f"- eval_mean_main_mlm_loss: {eval_summary['mean_main_mlm_loss']}")
            lines.append(f"- eval_mean_scanpath_mlm_loss: {eval_summary['mean_scanpath_mlm_loss']}")
            lines.append(f"- eval_mean_total_loss: {eval_summary['mean_total_loss']}")
        lines.append("")

    lines.extend(
        [
            "----------------------------------------",
            "Training final summary",
            "----------------------------------------",
            f"num_steps: {summary['total_steps']}",
            f"initial_main_loss: {summary['main_losses_all'][0] if summary['main_losses_all'] else None}",
            f"final_main_loss: {summary['main_losses_all'][-1] if summary['main_losses_all'] else None}",
            f"initial_scanpath_loss: {summary['scanpath_losses_all'][0] if summary['scanpath_losses_all'] else None}",
            f"final_scanpath_loss: {summary['scanpath_losses_all'][-1] if summary['scanpath_losses_all'] else None}",
            f"initial_total_loss: {summary['total_losses_all'][0] if summary['total_losses_all'] else None}",
            f"final_total_loss: {summary['total_losses_all'][-1] if summary['total_losses_all'] else None}",
            "checkpoints_guardados:",
        ]
    )
    for checkpoint_path in summary["checkpoint_paths"]:
        lines.append(f"- {checkpoint_path}")
    lines.extend(
        [
            f"best_checkpoint: {summary['best_checkpoint_path']}",
            f"final_checkpoint: {summary['final_checkpoint_path']}",
            "status:",
            f"- training {'OK' if summary['training_ok'] else 'FAILED'}",
            f"- checkpointing {'OK' if summary['checkpointing_ok'] else 'FAILED'}",
            "",
            "Interpretacion breve:",
        ]
    )

    final_epoch = summary["epoch_summaries"][-1] if summary["epoch_summaries"] else None
    if final_epoch is not None:
        lines.extend(summarize_aux_weight(final_epoch["mean_main_mlm_loss"], final_epoch["mean_scanpath_mlm_loss"], args.aux_weight))
        lines.append("- si las losses bajan o se mantienen estables entre epochs, el entrenamiento parece sano para esta escala intermedia.")
        if final_epoch.get("eval") is not None:
            lines.append("- la evaluacion minima ayuda a no depender solo de la loss de entrenamiento, aunque todavia no es un validation setup sofisticado.")
    else:
        lines.append("- no hubo epochs registradas para interpretar la corrida.")

    return "\n".join(lines) + "\n"


def build_readme(args, summary: Dict[str, Any], script_name: str) -> str:
    config = summary["config"]
    beto_note = ""
    if args.model_name_or_path == "dccuchile/bert-base-spanish-wwm-cased":
        beto_note = (
            "\nNota especifica sobre BETO\n"
            "- En esta corrida se uso dccuchile/bert-base-spanish-wwm-cased.\n"
            "- El tokenizer cargado fue compatible con el pipeline BERT-style y el config detectado mantiene model_type='bert'.\n"
            "- Esto lo hace compatible con la estructura actual del repo basada en BERT, sin cambiar a RoBERTa ni DeBERTa.\n"
        )

    return f"""PASO 7 - README
================

Que se hizo
- Se creo un script nuevo llamado {script_name}.
- El script entrena el modelo combinado con loss principal MLM + loss auxiliar scanpath MLM sobre un dataset medido mas grande que el smoke test.
- Se usan varias epochs, logging por epoch, evaluacion minima opcional y guardado de checkpoints reutilizables.
- El modelo sigue compartiendo un unico encoder BERT y combina las losses como total_loss = main_mlm_loss + aux_weight * scanpath_mlm_loss.

Que se verifico
- Que el entrenamiento combinado escala a mas datos y mas de una epoch.
- Que main_mlm_loss, scanpath_mlm_loss y total_loss se loguean claramente por separado.
- Que se guardan checkpoints por epoch, best checkpoint y checkpoint final.
- Que queda un checkpoint final reutilizable para una etapa downstream posterior.
- Que el pipeline puede correr con un modelo BERT-style compatible con BETO si se usa dccuchile/bert-base-spanish-wwm-cased.

Que NO se implemento todavia
- No se implemento GLUE.
- No se implemento fine-tuning downstream.
- No se hizo hyperparameter search grande.
- No se refactorizo de forma masiva el repo.
- No se avanzo todavia a la etapa downstream dentro de este paso.

Archivos modificados
- Pasos/README.txt

Archivos nuevos creados
- {script_name}
- Pasos/paso_7/README_paso_7.txt
- Pasos/paso_7/salida_training_step7.txt
- Pasos/paso_7/comandos_y_funciones.txt
- Pasos/paso_7/checkpoint_epoch_*/ si save_every_epoch=True
- Pasos/paso_7/best_checkpoint/
- Pasos/paso_7/{args.final_checkpoint_dirname}/

Explicacion breve del entrenamiento mas grande
- El dataset medido se tokeniza con BERT y se usa para construir inputs MLM estandar mas la representacion measured requerida por la rama scanpath.
- En cada batch se calculan simultaneamente la loss principal MLM y la loss auxiliar MLM scanpath.
- Luego se combinan con aux_weight y se actualiza el modelo con AdamW.
- Ademas se registra un resumen por epoch y una evaluacion minima sobre un subconjunto held-out simple.

Explicacion de la combinacion de losses
- Rama principal: input_ids -> BERT -> MLM head principal -> main_mlm_loss.
- Rama auxiliar: input_ids -> BERT -> scanpath expandido -> GRU -> reagregacion -> MLM head auxiliar -> scanpath_mlm_loss.
- Loss total: total_loss = main_mlm_loss + aux_weight * scanpath_mlm_loss.

Explicacion del parametro aux_weight
- aux_weight controla cuanto pesa la loss auxiliar scanpath respecto de la principal.
- Valores tipicos para probar aca: 1.0, 0.3, 0.1.
- Si la rama auxiliar domina, conviene bajar aux_weight. Si casi no influye, conviene subirlo.

Aclaracion importante
- Este paso ya no es solo smoke test, pero tampoco es todavia un experimento final grande.
- La idea es dejar un entrenamiento mas estable y checkpoints reutilizables para el paso downstream posterior.

Nota especifica sobre el modelo/tokenizer usado
- model_name_or_path usado: {args.model_name_or_path}
- tokenizer detectado: {summary['tokenizer_class']}
- vocab_size detectado: {config.vocab_size}
- model_type detectado: {config.model_type}
- hidden_size detectado: {config.hidden_size}
- num_hidden_layers detectado: {config.num_hidden_layers}
- El script valida explicitamente que el modelo sea BERT-style (model_type='bert').
{beto_note}
Configuracion usada en esta corrida
- measured_scanpath_file = {args.measured_scanpath_file}
- max_train_samples = {summary['train_size']}
- max_eval_samples = {summary['eval_size']}
- num_train_epochs = {args.num_train_epochs}
- per_device_train_batch_size = {args.per_device_train_batch_size}
- per_device_eval_batch_size = {args.per_device_eval_batch_size}
- max_seq_length = {args.max_seq_length}
- learning_rate = {args.learning_rate}
- aux_weight = {args.aux_weight}
- save_every_epoch = {args.save_every_epoch}
- seed = {args.seed}
- device = {summary['device']}
"""


def build_commands_file(args, script_name: str) -> str:
    relative_output = Path(args.output_dir).as_posix()
    command = (
        f".\\.venv\\Scripts\\python.exe {script_name} "
        f"--measured_scanpath_file \"{args.measured_scanpath_file}\" "
        f"--model_name_or_path {args.model_name_or_path} "
        f"--measured_text_field {args.measured_text_field} "
        f"--measured_word_id_field {args.measured_word_id_field} "
        f"--split {args.split} "
        f"--output_dir {relative_output} "
        f"--max_seq_length {args.max_seq_length} "
        f"--max_train_samples {args.max_train_samples} "
        f"--max_eval_samples {args.max_eval_samples} "
        f"--per_device_train_batch_size {args.per_device_train_batch_size} "
        f"--per_device_eval_batch_size {args.per_device_eval_batch_size} "
        f"--num_train_epochs {args.num_train_epochs} "
        f"--learning_rate {args.learning_rate} "
        f"--max_masked_positions {args.max_masked_positions} "
        f"--aux_weight {args.aux_weight} "
        f"--save_every_epoch {args.save_every_epoch} "
        f"--seed {args.seed}"
    )
    if args.remove_punctuation_space:
        command += " --remove_punctuation_space"

    return f"""COMANDOS Y FUNCIONES - PASO 7
=============================

Comando ejecutado
- {command}

Script principal usado
- {script_name}

Funciones principales llamadas
- measured_scanpath_utils.load_measured_scanpath_dataset
- measured_scanpath_utils.build_measured_single_sentence_features
- build_static_masked_inputs_and_labels
- collate_measured_mlm_batch
- Gazesup_BERTForCombinedMaskedLM.from_pretrained
- Gazesup_BERTForCombinedMaskedLM.forward
- evaluate_model
- save_checkpoint
- torch.optim.AdamW

Output generado
- Pasos/paso_7/salida_training_step7.txt con resumen por epoch, losses y checkpoints.
- Pasos/paso_7/README_paso_7.txt con el alcance de esta corrida mas grande.
- Pasos/paso_7/comandos_y_funciones.txt con trazabilidad de la corrida.
- Pasos/paso_7/checkpoint_epoch_*/, Pasos/paso_7/best_checkpoint/ y Pasos/paso_7/{args.final_checkpoint_dirname}/ con checkpoints reutilizables.
"""


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = train_step7(args)
    script_name = Path(__file__).name

    (output_dir / "salida_training_step7.txt").write_text(build_debug_output(args, summary), encoding="utf-8")
    (output_dir / "README_paso_7.txt").write_text(build_readme(args, summary, script_name), encoding="utf-8")
    (output_dir / "comandos_y_funciones.txt").write_text(build_commands_file(args, script_name), encoding="utf-8")

    print(f"Wrote {output_dir / 'salida_training_step7.txt'}")
    print(f"Wrote {output_dir / 'README_paso_7.txt'}")
    print(f"Wrote {output_dir / 'comandos_y_funciones.txt'}")
    print(f"Saved final checkpoint to {summary['final_checkpoint_path']}")
    if summary['best_checkpoint_path']:
        print(f"Saved best checkpoint to {summary['best_checkpoint_path']}")


if __name__ == "__main__":
    main()
