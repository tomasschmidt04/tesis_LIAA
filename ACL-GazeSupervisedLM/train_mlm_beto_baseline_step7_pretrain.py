import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

from loss_curve_utils import build_loss_curve_row, write_loss_curve_rows
from measured_scanpath_utils import (
    build_measured_single_sentence_features,
    build_mlm_train_eval_datasets,
    build_precomputed_file_train_eval_datasets,
    load_measured_scanpath_dataset,
)
from train_mlm_scanpath_step5 import DEFAULT_MLM_PROBABILITY, apply_dynamic_mlm_to_batch, set_seed


DEFAULT_OUTPUT_DIR = "Pasos/paso_7_clean_sentence_split_beto_baseline_full"
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
        description="Run baseline BETO MLM pretraining over the aligned texts only, without the scanpath branch."
    )
    parser.add_argument("--measured_scanpath_file", default=None, help="Path to a JSON/JSONL/CSV file with at least a text field.")
    parser.add_argument("--train_file", default=None, help="Precomputed MLM train JSONL/CSV file. Use with --split_strategy precomputed_files.")
    parser.add_argument("--eval_file", default=None, help="Precomputed MLM eval/test JSONL/CSV file. Use with --split_strategy precomputed_files.")
    parser.add_argument("--model_name_or_path", default="dccuchile/bert-base-spanish-wwm-cased", help="BERT-style model/tokenizer name or path.")
    parser.add_argument("--measured_text_field", default="text", help="Column that contains the plain text consumed by the tokenizer.")
    parser.add_argument("--measured_word_id_field", default="word_id", help="Accepted for dataset compatibility; ignored by the baseline model except for preprocessing parity.")
    parser.add_argument("--split", default="train", help="Dataset split to use after loading the measured file.")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Directory where baseline paso_7 artifacts will be written.")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Maximum length passed to the BETO tokenizer.")
    parser.add_argument("--max_train_samples", type=int, default=2000, help="Maximum number of examples used for training.")
    parser.add_argument("--max_eval_samples", type=int, default=256, help="Maximum number of held-out examples used for lightweight evaluation.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Mini-batch size used by the training loop.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4, help="Mini-batch size used by the evaluation loop.")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate used by AdamW.")
    parser.add_argument("--max_masked_positions", type=int, default=3, help="Deprecated; kept for command compatibility. Dynamic MLM now uses --mlm_probability.")
    parser.add_argument("--mlm_probability", type=float, default=DEFAULT_MLM_PROBABILITY, help="Probability of masking each non-special token for dynamic MLM.")
    parser.add_argument("--save_every_epoch", type=str2bool, default=True, help="Whether to save a checkpoint at the end of every epoch.")
    parser.add_argument("--remove_punctuation_space", action="store_true", help="Mirror the optional punctuation-space normalization used by the training scripts.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed used by the training loop.")
    parser.add_argument("--final_checkpoint_dirname", default=DEFAULT_CHECKPOINT_DIRNAME, help="Directory name used inside output_dir to save the final checkpoint.")
    parser.add_argument("--split_strategy", default="sentence_position", choices=["sentence_position", "contiguous", "precomputed_files"], help="How to build train/eval for intrinsic MLM.")
    parser.add_argument("--eval_sentence_position_mod", type=int, default=10, help="Modulo used by split_strategy=sentence_position.")
    parser.add_argument("--eval_sentence_position_remainder", type=int, default=5, help="Remainder sent to eval when split_strategy=sentence_position.")
    parser.add_argument("--split_report_path", default=None, help="Optional JSON path for the split report. Defaults to output_dir/split_report.json.")
    parser.add_argument("--story_id_field", default="auto", help="Story id field used by split_strategy=sentence_position. Use auto to infer.")
    parser.add_argument("--sentence_position_field", default="auto", help="Sentence position field used by split_strategy=sentence_position. Use auto to infer.")
    parser.add_argument("--dry_run_split_only", type=str2bool, default=False, help="Only build/validate/write the split report, then exit without training.")
    return parser.parse_args()


def verify_bert_style_config(model_name_or_path: str):
    config = AutoConfig.from_pretrained(model_name_or_path)
    if getattr(config, "model_type", None) != "bert":
        raise ValueError(
            f"Baseline paso_7 only supports BERT-style models for now. Received model_type={getattr(config, 'model_type', None)!r}"
        )
    return config


def uses_precomputed_split_files(args) -> bool:
    return args.split_strategy == "precomputed_files" or bool(args.train_file or args.eval_file)


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
        processed_examples.append(
            {
                "example_index": example_index,
                "text": text,
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
                "token_type_ids": feature["token_type_ids"],
            }
        )
    return processed_examples


def collate_baseline_mlm_batch(examples: List[Dict[str, Any]], tokenizer, mlm_probability: float = DEFAULT_MLM_PROBABILITY):
    batch_size = len(examples)
    max_seq_len = max(len(example["input_ids"]) for example in examples)

    input_ids = torch.full((batch_size, max_seq_len), tokenizer.pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    token_type_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    texts = []
    example_indices = []

    for batch_index, example in enumerate(examples):
        seq_len = len(example["input_ids"])
        input_ids[batch_index, :seq_len] = torch.tensor(example["input_ids"], dtype=torch.long)
        attention_mask[batch_index, :seq_len] = torch.tensor(example["attention_mask"], dtype=torch.long)
        token_type_ids[batch_index, :seq_len] = torch.tensor(example["token_type_ids"], dtype=torch.long)
        texts.append(example["text"])
        example_indices.append(int(example["example_index"]))

    input_ids, labels, masked_positions = apply_dynamic_mlm_to_batch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        tokenizer=tokenizer,
        mlm_probability=mlm_probability,
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "labels": labels,
        "texts": texts,
        "example_indices": example_indices,
        "masked_positions": masked_positions,
    }


def move_tensor_batch_to_device(batch: Dict[str, Any], device: torch.device):
    tensor_keys = ["input_ids", "attention_mask", "token_type_ids", "labels"]
    moved_batch = dict(batch)
    for key in tensor_keys:
        moved_batch[key] = batch[key].to(device)
    return moved_batch


def compute_mean(values: List[float]):
    return sum(values) / len(values) if values else None


def evaluate_model(model, dataloader, device):
    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for batch in dataloader:
            batch_on_device = move_tensor_batch_to_device(batch, device)
            outputs = model(
                input_ids=batch_on_device["input_ids"],
                attention_mask=batch_on_device["attention_mask"],
                token_type_ids=batch_on_device["token_type_ids"],
                labels=batch_on_device["labels"],
                return_dict=True,
            )
            losses.append(float(outputs.loss.detach().cpu().item()))
    model.train()
    return {"mean_mlm_loss": compute_mean(losses)}


def save_checkpoint(target_dir: Path, model, tokenizer):
    target_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target_dir)
    tokenizer.save_pretrained(target_dir)


def train_step7_baseline(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_report_path = args.split_report_path or str(output_dir / "split_report.json")

    set_seed(args.seed)

    if uses_precomputed_split_files(args):
        args.split_strategy = "precomputed_files"
        train_dataset, eval_dataset, split_report = build_precomputed_file_train_eval_datasets(
            train_file=args.train_file,
            eval_file=args.eval_file,
            max_train_samples=args.max_train_samples,
            max_eval_samples=args.max_eval_samples,
            seed=args.seed,
            text_field=args.measured_text_field,
            split_report_path=split_report_path,
            story_id_field=args.story_id_field,
        )
    else:
        if not args.measured_scanpath_file:
            raise ValueError("Either --measured_scanpath_file or both --train_file/--eval_file are required.")
        raw_datasets = load_measured_scanpath_dataset(args.measured_scanpath_file)
        if args.split not in raw_datasets:
            raise ValueError(f"Split {args.split!r} not found. Available splits: {list(raw_datasets.keys())}")

        full_dataset = raw_datasets[args.split]
        train_dataset, eval_dataset, split_report = build_mlm_train_eval_datasets(
            raw_dataset=full_dataset,
            split_strategy=args.split_strategy,
            max_train_samples=args.max_train_samples,
            max_eval_samples=args.max_eval_samples,
            seed=args.seed,
            text_field=args.measured_text_field,
            eval_sentence_position_mod=args.eval_sentence_position_mod,
            eval_sentence_position_remainder=args.eval_sentence_position_remainder,
            split_report_path=split_report_path,
            story_id_field=args.story_id_field,
            sentence_position_field=args.sentence_position_field,
        )
    if args.dry_run_split_only:
        return {
            "dry_run_split_only": True,
            "train_size": len(train_dataset),
            "eval_size": len(eval_dataset) if eval_dataset is not None else 0,
            "split_report": split_report,
            "split_report_path": split_report_path,
        }

    config = verify_bert_style_config(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    train_examples = preprocess_examples(train_dataset, tokenizer, args)
    eval_examples = preprocess_examples(eval_dataset, tokenizer, args) if eval_dataset is not None else []

    train_loader = DataLoader(
        train_examples,
        batch_size=args.per_device_train_batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_baseline_mlm_batch(batch, tokenizer, mlm_probability=args.mlm_probability),
    )
    eval_loader = None
    if eval_examples:
        eval_loader = DataLoader(
            eval_examples,
            batch_size=args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_baseline_mlm_batch(batch, tokenizer, mlm_probability=args.mlm_probability),
        )

    model = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    batch_debug: List[Dict[str, Any]] = []
    epoch_summaries: List[Dict[str, Any]] = []
    losses_all: List[float] = []
    checkpoint_paths: List[str] = []
    best_checkpoint_path = None
    best_reference_loss = None
    total_steps = 0
    loss_curve_rows: List[Dict[str, Any]] = []

    for epoch_index in range(args.num_train_epochs):
        epoch_losses: List[float] = []
        epoch_start = time.perf_counter()

        for batch_index, batch in enumerate(train_loader):
            optimizer.zero_grad()
            batch_on_device = move_tensor_batch_to_device(batch, device)
            outputs = model(
                input_ids=batch_on_device["input_ids"],
                attention_mask=batch_on_device["attention_mask"],
                token_type_ids=batch_on_device["token_type_ids"],
                labels=batch_on_device["labels"],
                return_dict=True,
            )
            loss = outputs.loss
            if loss is None or torch.isnan(loss).any():
                raise RuntimeError("Baseline paso_7 produced an invalid MLM loss.")

            loss.backward()
            optimizer.step()

            loss_value = float(loss.detach().cpu().item())
            epoch_losses.append(loss_value)
            losses_all.append(loss_value)
            total_steps += 1

            if len(batch_debug) < NUM_DEBUG_BATCHES:
                batch_debug.append(
                    {
                        "epoch": epoch_index + 1,
                        "batch_index": batch_index,
                        "input_ids_shape": tuple(batch["input_ids"].shape),
                        "labels_shape": tuple(batch["labels"].shape),
                        "logits_shape": tuple(outputs.logits.shape),
                        "mlm_loss": loss_value,
                    }
                )

        epoch_duration_sec = time.perf_counter() - epoch_start
        epoch_summary = {
            "epoch": epoch_index + 1,
            "mean_mlm_loss": compute_mean(epoch_losses),
            "num_steps": len(epoch_losses),
            "duration_sec": epoch_duration_sec,
        }

        if eval_loader is not None:
            epoch_summary["eval"] = evaluate_model(model, eval_loader, device)
            reference_loss = epoch_summary["eval"]["mean_mlm_loss"]
        else:
            reference_loss = epoch_summary["mean_mlm_loss"]

        if args.save_every_epoch:
            epoch_ckpt = Path(args.output_dir) / f"checkpoint_epoch_{epoch_index + 1}"
            save_checkpoint(epoch_ckpt, model, tokenizer)
            checkpoint_paths.append(str(epoch_ckpt))
            epoch_summary["checkpoint_saved_to"] = str(epoch_ckpt)

        if best_reference_loss is None or (reference_loss is not None and reference_loss < best_reference_loss):
            best_reference_loss = reference_loss
            best_ckpt = Path(args.output_dir) / "best_checkpoint"
            save_checkpoint(best_ckpt, model, tokenizer)
            best_checkpoint_path = str(best_ckpt)
            epoch_summary["best_checkpoint_updated"] = str(best_ckpt)

        epoch_summaries.append(epoch_summary)
        loss_curve_rows.append(
            build_loss_curve_row(
                epoch=epoch_index + 1,
                split="train",
                loss_total_mean=epoch_summary["mean_mlm_loss"],
                loss_standard_mean=epoch_summary["mean_mlm_loss"],
                loss_scanpath_mean=None,
                augweight=None,
                num_batches=epoch_summary["num_steps"],
            )
        )
        if "eval" in epoch_summary:
            loss_curve_rows.append(
                build_loss_curve_row(
                    epoch=epoch_index + 1,
                    split="eval",
                    loss_total_mean=epoch_summary["eval"]["mean_mlm_loss"],
                    loss_standard_mean=epoch_summary["eval"]["mean_mlm_loss"],
                    loss_scanpath_mean=None,
                    augweight=None,
                    num_batches=len(eval_loader) if eval_loader is not None else 0,
                )
            )
        write_loss_curve_rows(output_dir, loss_curve_rows)
        eval_loss_text = "NA"
        if "eval" in epoch_summary and epoch_summary["eval"]["mean_mlm_loss"] is not None:
            eval_loss_text = f"{epoch_summary['eval']['mean_mlm_loss']:.6f}"
        print(
            "[Paso 7 BETO] "
            f"epoch {epoch_index + 1}/{args.num_train_epochs} "
            f"({((epoch_index + 1) / args.num_train_epochs) * 100:.1f}%) "
            f"train_mlm_loss={epoch_summary['mean_mlm_loss']:.6f} "
            f"eval_mlm_loss={eval_loss_text} "
            f"steps={epoch_summary['num_steps']}",
            flush=True,
        )

    final_checkpoint_dir = Path(args.output_dir) / args.final_checkpoint_dirname
    save_checkpoint(final_checkpoint_dir, model, tokenizer)
    checkpoint_paths.append(str(final_checkpoint_dir))

    return {
        "config": config,
        "tokenizer_class": tokenizer.__class__.__name__,
        "device": str(device),
        "train_size": len(train_examples),
        "eval_size": len(eval_examples),
        "batch_debug": batch_debug,
        "epoch_summaries": epoch_summaries,
        "losses_all": losses_all,
        "total_steps": total_steps,
        "checkpoint_paths": checkpoint_paths,
        "best_checkpoint_path": best_checkpoint_path,
        "final_checkpoint_path": str(final_checkpoint_dir),
        "loss_curves_path": str(output_dir / "loss_curves.csv"),
        "split_report_path": split_report_path,
        "split_report": split_report,
    }


def build_debug_output(args, summary: Dict[str, Any]) -> str:
    config = summary["config"]
    lines: List[str] = [
        "PASO 7 BASELINE - Preentrenamiento BETO sin rama scanpath",
        "",
        "----------------------------------------",
        "Configuracion",
        "----------------------------------------",
        f"dataset: {args.measured_scanpath_file}",
        f"train_file: {args.train_file}",
        f"eval_file: {args.eval_file}",
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
        f"max_seq_length: {args.max_seq_length}",
        f"seed: {args.seed}",
        f"device: {summary['device']}",
        f"split_strategy: {args.split_strategy}",
        f"split_report: {summary.get('split_report_path')}",
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
                f"logits.shape: {debug_batch['logits_shape']}",
                f"mlm_loss: {debug_batch['mlm_loss']}",
                "",
            ]
        )

    for epoch_summary in summary["epoch_summaries"]:
        lines.extend(
            [
                "----------------------------------------",
                f"Epoch {epoch_summary['epoch']} summary",
                "----------------------------------------",
                f"mean_mlm_loss: {epoch_summary['mean_mlm_loss']}",
                f"num_steps: {epoch_summary['num_steps']}",
                f"duration_sec: {epoch_summary['duration_sec']}",
            ]
        )
        if "checkpoint_saved_to" in epoch_summary:
            lines.append(f"checkpoint_saved_to: {epoch_summary['checkpoint_saved_to']}")
        if "best_checkpoint_updated" in epoch_summary:
            lines.append(f"best_checkpoint_updated: {epoch_summary['best_checkpoint_updated']}")
        if "eval" in epoch_summary:
            lines.append("Evaluacion minima:")
            lines.append(f"- eval_mean_mlm_loss: {epoch_summary['eval']['mean_mlm_loss']}")
        lines.append("")

    lines.extend(
        [
            "----------------------------------------",
            "Training final summary",
            "----------------------------------------",
            f"num_steps: {summary['total_steps']}",
            f"initial_mlm_loss: {summary['losses_all'][0] if summary['losses_all'] else None}",
            f"final_mlm_loss: {summary['losses_all'][-1] if summary['losses_all'] else None}",
            "checkpoints_guardados:",
        ]
    )
    for checkpoint_path in summary["checkpoint_paths"]:
        lines.append(f"- {checkpoint_path}")
    lines.extend(
        [
            f"best_checkpoint: {summary['best_checkpoint_path']}",
            f"final_checkpoint: {summary['final_checkpoint_path']}",
            f"loss_curves_csv: {summary['loss_curves_path']}",
            "status:",
            "- training OK",
            "- checkpointing OK",
            "",
            "Nota:",
            "- Esta corrida usa los mismos textos alineados del pipeline measured, pero no instancia ni usa la rama scanpath.",
            "- La loss es solo MLM estandar del modelo BETO base.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_readme(args, summary: Dict[str, Any], script_name: str) -> str:
    return f"""PASO 7 BASELINE - README
===========================

Que se hizo
- Se creo un script baseline llamado {script_name}.
- Este script toma el mismo dataset alineado usado por el pipeline measured.
- Usa solo el campo de texto para preentrenar BETO con MLM estandar.
- No usa la rama scanpath.
- No usa GRU.
- No usa loss auxiliar.
- Usa un split limpio por posicion de oracion cuando split_strategy=sentence_position.
- Si split_strategy=precomputed_files, usa directamente train_file/eval_file ya separados por cuentos.

Que se mantiene comparable con el paso 7 scanpath
- Mismo archivo de entrada: {args.measured_scanpath_file}
- train_file si aplica: {args.train_file}
- eval_file si aplica: {args.eval_file}
- Mismo modelo base: {args.model_name_or_path}
- Misma longitud maxima: {args.max_seq_length}
- Mismo esquema de mascara estatica usado en estos scripts del repo.

Que produce
- split_report.json con validaciones del split limpio
- Checkpoints por epoca si save_every_epoch=True
- best_checkpoint/
- {args.final_checkpoint_dirname}/
- loss_curves.csv con medias train/eval por epoch

Parametros principales
- split_strategy = {args.split_strategy}
- eval_sentence_position_mod = {args.eval_sentence_position_mod}
- eval_sentence_position_remainder = {args.eval_sentence_position_remainder}
- split_report_path = {summary.get('split_report_path')}
- max_train_samples = {summary['train_size']}
- max_eval_samples = {summary['eval_size']}
- num_train_epochs = {args.num_train_epochs}
- per_device_train_batch_size = {args.per_device_train_batch_size}
- per_device_eval_batch_size = {args.per_device_eval_batch_size}
- learning_rate = {args.learning_rate}
- max_seq_length = {args.max_seq_length}
- max_masked_positions = {args.max_masked_positions}
- seed = {args.seed}
- device = {summary['device']}

Diferencia conceptual respecto del paso 7b
- Paso 7b: BETO + MLM principal + rama scanpath + GRU + loss auxiliar.
- Este baseline: BETO + MLM principal solamente.

Split limpio por posicion de oracion
- El split anterior por filas contiguas podia poner la misma oracion en train y eval porque cada texto aparece varias veces con distintas lecturas/scanpaths.
- Con split_strategy=sentence_position se infiere story_id y sentence_position, y eval recibe las posiciones donde sentence_position % eval_sentence_position_mod == eval_sentence_position_remainder.
- La misma posicion queda en eval para todos los cuentos y todas sus lecturas; el resto queda en train.
- El reporte valida que no haya texto exacto compartido ni pares (story_id, sentence_position) compartidos.
- Para auditar sin entrenar: agregar --dry_run_split_only True.

Split precomputado por cuentos
- Con split_strategy=precomputed_files no se vuelve a dividir el dataset.
- El script carga train_file y eval_file directamente.
- El reporte valida que ningun cuento aparezca en ambos splits.
- Este modo es el recomendado para usar reading-et/mlm_dataset_limpio_train_test/train.jsonl y test.jsonl.

Siguiente uso esperado
- Comparar downstream del backbone exportado desde este baseline contra el backbone exportado desde paso_7b.
"""


def build_commands_reference(args, script_name: str) -> str:
    command = (
        f"python {script_name} "
        f"--model_name_or_path {args.model_name_or_path} "
        f"--measured_text_field {args.measured_text_field} "
        f"--measured_word_id_field {args.measured_word_id_field} "
        f"--split {args.split} "
        f"--output_dir \"{args.output_dir}\" "
        f"--max_seq_length {args.max_seq_length} "
        f"--max_train_samples {args.max_train_samples} "
        f"--max_eval_samples {args.max_eval_samples} "
        f"--per_device_train_batch_size {args.per_device_train_batch_size} "
        f"--per_device_eval_batch_size {args.per_device_eval_batch_size} "
        f"--num_train_epochs {args.num_train_epochs} "
        f"--learning_rate {args.learning_rate} "
        f"--max_masked_positions {args.max_masked_positions} "
        f"--save_every_epoch {args.save_every_epoch} "
        f"--seed {args.seed} "
        f"--split_strategy {args.split_strategy} "
        f"--eval_sentence_position_mod {args.eval_sentence_position_mod} "
        f"--eval_sentence_position_remainder {args.eval_sentence_position_remainder} "
        f"--dry_run_split_only {args.dry_run_split_only}"
    )
    if args.measured_scanpath_file:
        command += f" --measured_scanpath_file \"{args.measured_scanpath_file}\""
    if args.train_file:
        command += f" --train_file \"{args.train_file}\""
    if args.eval_file:
        command += f" --eval_file \"{args.eval_file}\""
    if args.split_report_path:
        command += f" --split_report_path \"{args.split_report_path}\""
    if args.story_id_field != "auto":
        command += f" --story_id_field {args.story_id_field}"
    if args.sentence_position_field != "auto":
        command += f" --sentence_position_field {args.sentence_position_field}"
    if args.remove_punctuation_space:
        command += " --remove_punctuation_space"

    return f"""COMANDOS Y FUNCIONES - PASO 7 BASELINE
=====================================

Comando ejecutado
{command}

Funciones / modulos usados
- measured_scanpath_utils.load_measured_scanpath_dataset
- measured_scanpath_utils.build_mlm_train_eval_datasets
- measured_scanpath_utils.build_precomputed_file_train_eval_datasets
- measured_scanpath_utils.build_measured_single_sentence_features
- train_mlm_scanpath_step5.build_static_masked_inputs_and_labels
- transformers.AutoTokenizer.from_pretrained
- transformers.AutoModelForMaskedLM.from_pretrained
- torch.optim.AdamW
- model.save_pretrained
- tokenizer.save_pretrained
- loss_curve_utils.write_loss_curve_rows
"""


def build_dry_run_output(args, summary: Dict[str, Any]) -> str:
    return f"""PASO 7 BASELINE - Dry run de split limpio
==========================================

No se entreno el modelo porque dry_run_split_only=True.

Configuracion
- dataset: {args.measured_scanpath_file}
- train_file: {args.train_file}
- eval_file: {args.eval_file}
- split_strategy: {args.split_strategy}
- eval_sentence_position_mod: {args.eval_sentence_position_mod}
- eval_sentence_position_remainder: {args.eval_sentence_position_remainder}
- max_train_samples: {args.max_train_samples}
- max_eval_samples: {args.max_eval_samples}
- seed: {args.seed}

Resultado
- train_rows: {summary['train_size']}
- eval_rows: {summary['eval_size']}
- split_report: {summary['split_report_path']}
"""


def write_run_args(output_dir: Path, args):
    (output_dir / "run_args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_trainer_state(output_dir: Path, summary: Dict[str, Any]):
    state = {
        "total_steps": summary.get("total_steps", 0),
        "train_size": summary.get("train_size"),
        "eval_size": summary.get("eval_size"),
        "epoch_summaries": summary.get("epoch_summaries", []),
        "checkpoint_paths": summary.get("checkpoint_paths", []),
        "best_checkpoint_path": summary.get("best_checkpoint_path"),
        "final_checkpoint_path": summary.get("final_checkpoint_path"),
        "loss_curves_path": summary.get("loss_curves_path"),
        "split_report_path": summary.get("split_report_path"),
        "dry_run_split_only": summary.get("dry_run_split_only", False),
    }
    (output_dir / "trainer_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_failed_trainer_state(output_dir: Path, args, exc: Exception):
    state = {
        "total_steps": 0,
        "train_size": None,
        "eval_size": None,
        "epoch_summaries": [],
        "checkpoint_paths": [],
        "best_checkpoint_path": None,
        "final_checkpoint_path": None,
        "loss_curves_path": None,
        "split_report_path": args.split_report_path or str(output_dir / "split_report.json"),
        "dry_run_split_only": args.dry_run_split_only,
        "failed": True,
        "error_type": exc.__class__.__name__,
        "error_message": str(exc),
    }
    (output_dir / "trainer_state.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        summary = train_step7_baseline(args)
    except ValueError as exc:
        write_run_args(output_dir, args)
        write_failed_trainer_state(output_dir, args, exc)
        print(f"Wrote {output_dir / 'run_args.json'}")
        print(f"Wrote {output_dir / 'trainer_state.json'}")
        raise

    script_name = Path(__file__).name
    debug_path = output_dir / "salida_training_step7_beto_baseline.txt"
    readme_path = output_dir / "README_paso_7_preentrenamiento_beto.txt"
    commands_path = output_dir / "comandos_y_funciones.txt"

    if summary.get("dry_run_split_only"):
        debug_path.write_text(build_dry_run_output(args, summary), encoding="utf-8")
        commands_path.write_text(build_commands_reference(args, script_name), encoding="utf-8")
        write_run_args(output_dir, args)
        write_trainer_state(output_dir, summary)
        print(f"Wrote {debug_path}")
        print(f"Wrote {commands_path}")
        print(f"Wrote {output_dir / 'run_args.json'}")
        print(f"Wrote {output_dir / 'trainer_state.json'}")
        print(f"Wrote {summary['split_report_path']}")
        return

    debug_path.write_text(build_debug_output(args, summary), encoding="utf-8")
    readme_path.write_text(build_readme(args, summary, script_name), encoding="utf-8")
    commands_path.write_text(build_commands_reference(args, script_name), encoding="utf-8")
    write_run_args(output_dir, args)
    write_trainer_state(output_dir, summary)

    print(f"Wrote {debug_path}")
    print(f"Wrote {readme_path}")
    print(f"Wrote {commands_path}")
    print(f"Wrote {output_dir / 'run_args.json'}")
    print(f"Wrote {output_dir / 'trainer_state.json'}")
    print(f"Wrote {summary['split_report_path']}")
    print(f"Wrote {summary['loss_curves_path']}")
    print(f"Saved final checkpoint to {summary['final_checkpoint_path']}")
    print(f"Saved best checkpoint to {summary['best_checkpoint_path']}")


if __name__ == "__main__":
    main()
