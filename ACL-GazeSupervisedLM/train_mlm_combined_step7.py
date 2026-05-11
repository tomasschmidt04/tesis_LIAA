import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer

from Gazesup_bert_combined_mlm_model import Gazesup_BERTForCombinedMaskedLM
from loss_curve_utils import build_loss_curve_row, write_loss_curve_rows
from measured_scanpath_utils import (
    build_measured_single_sentence_features,
    build_mlm_train_eval_datasets,
    build_precomputed_file_train_eval_datasets,
    load_measured_scanpath_dataset,
)
from train_mlm_scanpath_step5 import (
    DEFAULT_MLM_PROBABILITY,
    collate_measured_mlm_batch,
    move_tensor_batch_to_device,
    set_seed,
)


DEFAULT_OUTPUT_DIR = "Pasos/paso_7_clean_sentence_split_full_scanpath"
DEFAULT_CHECKPOINT_DIRNAME = "checkpoint_final"
NUM_DEBUG_BATCHES = 2
SCANPATH_WEIGHT_LOG_COLUMNS = [
    "epoch",
    "lambda_scanpath_t",
    "lambda_scanpath_next",
    "metric_used_for_update",
    "current_scanpath_loss_for_update",
    "initial_scanpath_loss",
    "progress",
    "train_loss_total_mean",
    "train_loss_mlm_mean",
    "train_loss_scanpath_mean",
    "eval_loss_total_mean",
    "eval_loss_mlm_mean",
    "eval_loss_scanpath_mean",
    "warning",
]


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
    parser.add_argument("--measured_scanpath_file", default=None, help="Path to a JSON/JSONL/CSV file with at least text and word_id fields.")
    parser.add_argument("--train_file", default=None, help="Precomputed MLM train JSONL/CSV file. Use with --split_strategy precomputed_files.")
    parser.add_argument("--eval_file", default=None, help="Precomputed MLM eval/test JSONL/CSV file. Use with --split_strategy precomputed_files.")
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
    parser.add_argument("--max_masked_positions", type=int, default=3, help="Deprecated; kept for command compatibility. Dynamic MLM now uses --mlm_probability.")
    parser.add_argument("--mlm_probability", type=float, default=DEFAULT_MLM_PROBABILITY, help="Probability of masking each non-special token for dynamic MLM.")
    parser.add_argument("--aux_weight", type=float, default=0.3, help="Weight used in total_loss = main_mlm_loss + aux_weight * scanpath_mlm_loss.")
    parser.add_argument("--adaptive_scanpath_weight", type=str2bool, default=False, help="Use an epoch-level adaptive lambda for the scanpath loss instead of fixed aux_weight.")
    parser.add_argument("--scanpath_weight_min", type=float, default=0.05, help="Minimum adaptive scanpath loss weight.")
    parser.add_argument("--scanpath_weight_max", type=float, default=0.5, help="Maximum adaptive scanpath loss weight.")
    parser.add_argument("--scanpath_weight_warmup_epochs", type=int, default=1, help="Number of first epochs kept at scanpath_weight_min.")
    parser.add_argument("--scanpath_weight_update_metric", default="auto", choices=["auto", "eval_loss_scanpath", "train_loss_scanpath_mean"], help="Epoch-level scanpath loss used to update lambda.")
    parser.add_argument("--scanpath_weight_log_path", default=None, help="Optional CSV path for the adaptive scanpath weight log. Defaults to output_dir/lambda_scanpath_log.csv.")
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
            f"PASO 7 only supports BERT-style models for now. Received model_type={getattr(config, 'model_type', None)!r}"
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
                "LM_word_ids": feature["word_ids"],
                "measured_word_ids": feature["measured_word_ids"][0],
                "measured_sp_len": feature["measured_sp_len"][0],
            }
        )
    return processed_examples


def compute_mean(values: List[float]):
    return sum(values) / len(values) if values else None


def evaluate_model(model, dataloader, device, aux_weight: float, current_scanpath_weight: Optional[float] = None):
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
                current_scanpath_weight=current_scanpath_weight,
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
    model.save_pretrained(target_dir, safe_serialization=False)
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


def validate_scanpath_weight_args(args):
    if args.scanpath_weight_min < 0:
        raise ValueError("--scanpath_weight_min must be >= 0.")
    if args.scanpath_weight_max < args.scanpath_weight_min:
        raise ValueError("--scanpath_weight_max must be >= --scanpath_weight_min.")
    if args.scanpath_weight_warmup_epochs < 0:
        raise ValueError("--scanpath_weight_warmup_epochs must be >= 0.")


def compute_adaptive_scanpath_weight(
    initial_scanpath_loss: Optional[float],
    current_scanpath_loss: Optional[float],
    lambda_min: float,
    lambda_max: float,
) -> Tuple[float, float]:
    if initial_scanpath_loss is None or current_scanpath_loss is None or initial_scanpath_loss <= 0:
        return float(lambda_min), 0.0
    if not math.isfinite(float(initial_scanpath_loss)) or not math.isfinite(float(current_scanpath_loss)):
        return float(lambda_min), 0.0
    progress = (float(initial_scanpath_loss) - float(current_scanpath_loss)) / float(initial_scanpath_loss)
    progress = max(0.0, min(1.0, progress))
    lambda_scanpath_t = float(lambda_min) + progress * (float(lambda_max) - float(lambda_min))
    return lambda_scanpath_t, progress


def select_scanpath_weight_update_metric(epoch_summary: Dict[str, Any], metric_choice: str) -> Tuple[str, Optional[float], Optional[str]]:
    eval_summary = epoch_summary.get("eval") or {}
    eval_loss_scanpath = eval_summary.get("mean_scanpath_mlm_loss")
    train_loss_scanpath = epoch_summary.get("mean_scanpath_mlm_loss")

    if metric_choice == "eval_loss_scanpath":
        if eval_loss_scanpath is None:
            return "eval_loss_scanpath", None, "WARNING: eval_loss_scanpath no esta disponible; lambda queda en scanpath_weight_min."
        return "eval_loss_scanpath", eval_loss_scanpath, None

    if metric_choice == "train_loss_scanpath_mean":
        if train_loss_scanpath is None:
            return "train_loss_scanpath_mean", None, "WARNING: train_loss_scanpath_mean no esta disponible; lambda queda en scanpath_weight_min."
        return "train_loss_scanpath_mean", train_loss_scanpath, None

    if eval_loss_scanpath is not None:
        return "eval_loss_scanpath", eval_loss_scanpath, None
    if train_loss_scanpath is not None:
        return (
            "train_loss_scanpath_mean",
            train_loss_scanpath,
            "WARNING: eval_loss_scanpath no esta disponible; se uso train_loss_scanpath_mean para actualizar lambda.",
        )
    return "none", None, "WARNING: no hay loss scanpath promedio disponible; lambda queda en scanpath_weight_min."


def _csv_value(value):
    return "" if value is None else value


def build_scanpath_weight_log_row(
    epoch_summary: Dict[str, Any],
    lambda_scanpath_t: float,
    lambda_scanpath_next: float,
    metric_used_for_update: str,
    current_scanpath_loss_for_update: Optional[float],
    initial_scanpath_loss: Optional[float],
    progress: float,
    warning: Optional[str],
) -> Dict[str, Any]:
    eval_summary = epoch_summary.get("eval") or {}
    return {
        "epoch": epoch_summary["epoch"],
        "lambda_scanpath_t": lambda_scanpath_t,
        "lambda_scanpath_next": lambda_scanpath_next,
        "metric_used_for_update": metric_used_for_update,
        "current_scanpath_loss_for_update": current_scanpath_loss_for_update,
        "initial_scanpath_loss": initial_scanpath_loss,
        "progress": progress,
        "train_loss_total_mean": epoch_summary.get("mean_total_loss"),
        "train_loss_mlm_mean": epoch_summary.get("mean_main_mlm_loss"),
        "train_loss_scanpath_mean": epoch_summary.get("mean_scanpath_mlm_loss"),
        "eval_loss_total_mean": eval_summary.get("mean_total_loss"),
        "eval_loss_mlm_mean": eval_summary.get("mean_main_mlm_loss"),
        "eval_loss_scanpath_mean": eval_summary.get("mean_scanpath_mlm_loss"),
        "warning": warning,
    }


def write_scanpath_weight_log(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCANPATH_WEIGHT_LOG_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(row.get(column)) for column in SCANPATH_WEIGHT_LOG_COLUMNS})
    return path


def train_step7(args):
    validate_scanpath_weight_args(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_report_path = args.split_report_path or str(output_dir / "split_report.json")
    scanpath_weight_log_path = Path(args.scanpath_weight_log_path) if args.scanpath_weight_log_path else output_dir / "lambda_scanpath_log.csv"

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
        collate_fn=lambda batch: collate_measured_mlm_batch(batch, tokenizer, mlm_probability=args.mlm_probability),
    )
    eval_loader = None
    if eval_examples:
        eval_loader = DataLoader(
            eval_examples,
            batch_size=args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_measured_mlm_batch(batch, tokenizer, mlm_probability=args.mlm_probability),
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
    loss_curve_rows: List[Dict[str, Any]] = []
    scanpath_weight_log_rows: List[Dict[str, Any]] = []
    initial_scanpath_loss_for_lambda: Optional[float] = None
    current_scanpath_weight = float(args.scanpath_weight_min) if args.adaptive_scanpath_weight else float(args.aux_weight)

    for epoch_index in range(args.num_train_epochs):
        epoch_scanpath_weight = current_scanpath_weight if args.adaptive_scanpath_weight else float(args.aux_weight)
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
                current_scanpath_weight=epoch_scanpath_weight if args.adaptive_scanpath_weight else None,
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
                        "scanpath_loss_weight": epoch_scanpath_weight,
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
            "scanpath_loss_weight": epoch_scanpath_weight,
        }

        if eval_loader is not None:
            epoch_summary["eval"] = evaluate_model(
                model,
                eval_loader,
                device,
                args.aux_weight,
                current_scanpath_weight=epoch_scanpath_weight if args.adaptive_scanpath_weight else None,
            )
            reference_total = epoch_summary["eval"]["mean_total_loss"]
        else:
            reference_total = epoch_summary["mean_total_loss"]

        metric_used_for_update = "fixed_aux_weight"
        current_scanpath_loss_for_update = None
        scanpath_weight_progress = 0.0
        next_scanpath_weight = epoch_scanpath_weight
        scanpath_weight_warning = None
        if args.adaptive_scanpath_weight:
            metric_used_for_update, current_scanpath_loss_for_update, scanpath_weight_warning = select_scanpath_weight_update_metric(
                epoch_summary,
                args.scanpath_weight_update_metric,
            )
            if initial_scanpath_loss_for_lambda is None and current_scanpath_loss_for_update is not None:
                initial_scanpath_loss_for_lambda = float(current_scanpath_loss_for_update)

            proposed_next_weight, scanpath_weight_progress = compute_adaptive_scanpath_weight(
                initial_scanpath_loss=initial_scanpath_loss_for_lambda,
                current_scanpath_loss=current_scanpath_loss_for_update,
                lambda_min=args.scanpath_weight_min,
                lambda_max=args.scanpath_weight_max,
            )
            if epoch_index + 1 <= args.scanpath_weight_warmup_epochs:
                next_scanpath_weight = float(args.scanpath_weight_min)
                warmup_warning = (
                    f"warmup activo: epoch {epoch_index + 1} <= "
                    f"scanpath_weight_warmup_epochs={args.scanpath_weight_warmup_epochs}; "
                    "lambda_next queda en scanpath_weight_min."
                )
                scanpath_weight_warning = (
                    warmup_warning
                    if scanpath_weight_warning is None
                    else f"{scanpath_weight_warning} {warmup_warning}"
                )
            else:
                next_scanpath_weight = proposed_next_weight
            current_scanpath_weight = next_scanpath_weight
            epoch_summary["scanpath_weight_update_metric"] = metric_used_for_update
            epoch_summary["scanpath_loss_for_weight_update"] = current_scanpath_loss_for_update
            epoch_summary["initial_scanpath_loss_for_lambda"] = initial_scanpath_loss_for_lambda
            epoch_summary["scanpath_weight_progress"] = scanpath_weight_progress
            epoch_summary["next_scanpath_loss_weight"] = next_scanpath_weight
            epoch_summary["scanpath_weight_warning"] = scanpath_weight_warning
            scanpath_weight_log_rows.append(
                build_scanpath_weight_log_row(
                    epoch_summary=epoch_summary,
                    lambda_scanpath_t=epoch_scanpath_weight,
                    lambda_scanpath_next=next_scanpath_weight,
                    metric_used_for_update=metric_used_for_update,
                    current_scanpath_loss_for_update=current_scanpath_loss_for_update,
                    initial_scanpath_loss=initial_scanpath_loss_for_lambda,
                    progress=scanpath_weight_progress,
                    warning=scanpath_weight_warning,
                )
            )
            write_scanpath_weight_log(scanpath_weight_log_path, scanpath_weight_log_rows)

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
        loss_curve_rows.append(
            build_loss_curve_row(
                epoch=epoch_index + 1,
                split="train",
                loss_total_mean=epoch_summary["mean_total_loss"],
                loss_standard_mean=epoch_summary["mean_main_mlm_loss"],
                loss_scanpath_mean=epoch_summary["mean_scanpath_mlm_loss"],
                augweight=epoch_scanpath_weight,
                num_batches=epoch_summary["num_steps"],
            )
        )
        if "eval" in epoch_summary:
            eval_summary = epoch_summary["eval"]
            loss_curve_rows.append(
                build_loss_curve_row(
                    epoch=epoch_index + 1,
                    split="eval",
                    loss_total_mean=eval_summary["mean_total_loss"],
                    loss_standard_mean=eval_summary["mean_main_mlm_loss"],
                    loss_scanpath_mean=eval_summary["mean_scanpath_mlm_loss"],
                    augweight=epoch_scanpath_weight,
                    num_batches=len(eval_loader) if eval_loader is not None else 0,
                )
            )
        write_loss_curve_rows(output_dir, loss_curve_rows)
        eval_total_text = "NA"
        if "eval" in epoch_summary and epoch_summary["eval"]["mean_total_loss"] is not None:
            eval_total_text = f"{epoch_summary['eval']['mean_total_loss']:.6f}"
        print(
            "[Paso 7 Scanpath] "
            f"epoch {epoch_index + 1}/{args.num_train_epochs} "
            f"({((epoch_index + 1) / args.num_train_epochs) * 100:.1f}%) "
            f"train_total_loss={epoch_summary['mean_total_loss']:.6f} "
            f"train_main_loss={epoch_summary['mean_main_mlm_loss']:.6f} "
            f"train_scanpath_loss={epoch_summary['mean_scanpath_mlm_loss']:.6f} "
            f"eval_total_loss={eval_total_text} "
            f"scanpath_weight={epoch_scanpath_weight:.6f} "
            f"steps={epoch_summary['num_steps']}",
            flush=True,
        )

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
        "loss_curves_path": str(output_dir / "loss_curves.csv"),
        "scanpath_weight_log_path": str(scanpath_weight_log_path) if scanpath_weight_log_rows else None,
        "scanpath_weight_log_rows": scanpath_weight_log_rows,
        "initial_scanpath_loss_for_lambda": initial_scanpath_loss_for_lambda,
        "adaptive_scanpath_weight": args.adaptive_scanpath_weight,
        "split_report_path": split_report_path,
        "split_report": split_report,
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
        f"aux_weight: {args.aux_weight}",
        f"adaptive_scanpath_weight: {args.adaptive_scanpath_weight}",
        f"scanpath_weight_min: {args.scanpath_weight_min}",
        f"scanpath_weight_max: {args.scanpath_weight_max}",
        f"scanpath_weight_warmup_epochs: {args.scanpath_weight_warmup_epochs}",
        f"scanpath_weight_update_metric: {args.scanpath_weight_update_metric}",
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
                f"main_mlm_logits.shape: {debug_batch['main_mlm_logits_shape']}",
                f"scanpath_mlm_logits.shape: {debug_batch['scanpath_mlm_logits_shape']}",
                f"main_mlm_loss: {debug_batch['main_mlm_loss']}",
                f"scanpath_mlm_loss: {debug_batch['scanpath_mlm_loss']}",
                f"scanpath_loss_weight: {debug_batch.get('scanpath_loss_weight')}",
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
                f"scanpath_loss_weight_usado: {epoch_summary.get('scanpath_loss_weight')}",
                f"scanpath_weight_update_metric: {epoch_summary.get('scanpath_weight_update_metric')}",
                f"scanpath_loss_for_weight_update: {epoch_summary.get('scanpath_loss_for_weight_update')}",
                f"initial_scanpath_loss_for_lambda: {epoch_summary.get('initial_scanpath_loss_for_lambda')}",
                f"scanpath_weight_progress: {epoch_summary.get('scanpath_weight_progress')}",
                f"next_scanpath_loss_weight: {epoch_summary.get('next_scanpath_loss_weight')}",
                f"scanpath_weight_warning: {epoch_summary.get('scanpath_weight_warning')}",
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
            f"loss_curves_csv: {summary['loss_curves_path']}",
            f"lambda_scanpath_log_csv: {summary.get('scanpath_weight_log_path')}",
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
- El modelo sigue compartiendo un unico encoder BERT y combina las losses como total_loss = main_mlm_loss + scanpath_loss_weight * scanpath_mlm_loss.
- Si adaptive_scanpath_weight=False, scanpath_loss_weight es aux_weight como antes.
- Si adaptive_scanpath_weight=True, scanpath_loss_weight es lambda_scanpath_t actualizado por epoch.
- La evaluacion MLM usa un split limpio por posicion de oracion cuando split_strategy=sentence_position.
- Si split_strategy=precomputed_files, usa directamente train_file/eval_file ya separados por cuentos.

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
- No se cambio arquitectura, dataset, labels ni split para implementar lambda adaptativo.

Archivos modificados
- Pasos/README.txt

Archivos nuevos creados
- {script_name}
- {args.output_dir}/split_report.json
- {args.output_dir}/README_paso_7.txt
- {args.output_dir}/salida_training_step7.txt
- {args.output_dir}/comandos_y_funciones.txt
- {args.output_dir}/lambda_scanpath_log.csv si adaptive_scanpath_weight=True
- {args.output_dir}/checkpoint_epoch_*/ si save_every_epoch=True
- {args.output_dir}/best_checkpoint/
- {args.output_dir}/{args.final_checkpoint_dirname}/

Explicacion breve del entrenamiento mas grande
- El dataset medido se tokeniza con BERT y se usa para construir inputs MLM estandar mas la representacion measured requerida por la rama scanpath.
- En cada batch se calculan simultaneamente la loss principal MLM y la loss auxiliar MLM scanpath.
- Luego se combinan con aux_weight fijo o con lambda_scanpath_t adaptativo, y se actualiza el modelo con AdamW.
- Ademas se registra un resumen por epoch y una evaluacion sobre el split limpio definido antes de entrenar.

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

Explicacion de la combinacion de losses
- Rama principal: input_ids -> BERT -> MLM head principal -> main_mlm_loss.
- Rama auxiliar: input_ids -> BERT -> scanpath expandido -> GRU -> reagregacion -> MLM head auxiliar -> scanpath_mlm_loss.
- Loss total: total_loss = main_mlm_loss + scanpath_loss_weight * scanpath_mlm_loss.
- Con adaptive_scanpath_weight=False, scanpath_loss_weight = aux_weight.
- Con adaptive_scanpath_weight=True, scanpath_loss_weight = lambda_scanpath_t.

Explicacion del parametro aux_weight
- aux_weight controla cuanto pesa la loss auxiliar scanpath respecto de la principal.
- Valores tipicos para probar aca: 1.0, 0.3, 0.1.
- Si la rama auxiliar domina, conviene bajar aux_weight. Si casi no influye, conviene subirlo.

Explicacion del lambda adaptativo simple
- Se activa con --adaptive_scanpath_weight True.
- La referencia initial_scanpath_loss es la primera loss scanpath promedio disponible para actualizar lambda.
- Por epoch se usa eval_loss_scanpath si esta disponible; si no, train_loss_scanpath_mean.
- progress = (initial_scanpath_loss - current_scanpath_loss) / initial_scanpath_loss, clipeado a [0, 1].
- lambda_scanpath_t = scanpath_weight_min + progress * (scanpath_weight_max - scanpath_weight_min).
- Durante scanpath_weight_warmup_epochs, lambda_scanpath_t se mantiene en scanpath_weight_min.

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
- train_file = {args.train_file}
- eval_file = {args.eval_file}
- split_strategy = {args.split_strategy}
- eval_sentence_position_mod = {args.eval_sentence_position_mod}
- eval_sentence_position_remainder = {args.eval_sentence_position_remainder}
- split_report_path = {summary.get('split_report_path')}
- max_train_samples = {summary['train_size']}
- max_eval_samples = {summary['eval_size']}
- num_train_epochs = {args.num_train_epochs}
- per_device_train_batch_size = {args.per_device_train_batch_size}
- per_device_eval_batch_size = {args.per_device_eval_batch_size}
- max_seq_length = {args.max_seq_length}
- learning_rate = {args.learning_rate}
- aux_weight = {args.aux_weight}
- adaptive_scanpath_weight = {args.adaptive_scanpath_weight}
- scanpath_weight_min = {args.scanpath_weight_min}
- scanpath_weight_max = {args.scanpath_weight_max}
- scanpath_weight_warmup_epochs = {args.scanpath_weight_warmup_epochs}
- scanpath_weight_update_metric = {args.scanpath_weight_update_metric}
- scanpath_weight_log_path = {summary.get('scanpath_weight_log_path')}
- save_every_epoch = {args.save_every_epoch}
- seed = {args.seed}
- device = {summary['device']}
"""


def _fmt_debug_value(value) -> str:
    if value is None or value == "":
        return "NA"
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def build_lambda_debug_output(args, summary: Dict[str, Any]) -> str:
    epoch_by_number = {row["epoch"]: row for row in summary.get("epoch_summaries", [])}
    lines = [
        "PASO 3 - Debug lambda adaptativo scanpath",
        "=========================================",
        "",
        "Tabla de evolucion por epoch",
        "epoch | loss_scanpath_usada | progress | lambda_scanpath_t | lambda_scanpath_next | loss_mlm | loss_scanpath | loss_total | metric",
    ]

    for row in summary.get("scanpath_weight_log_rows", []):
        epoch = row["epoch"]
        epoch_summary = epoch_by_number.get(epoch, {})
        lines.append(
            " | ".join(
                [
                    str(epoch),
                    _fmt_debug_value(row.get("current_scanpath_loss_for_update")),
                    _fmt_debug_value(row.get("progress")),
                    _fmt_debug_value(row.get("lambda_scanpath_t")),
                    _fmt_debug_value(row.get("lambda_scanpath_next")),
                    _fmt_debug_value(epoch_summary.get("mean_main_mlm_loss")),
                    _fmt_debug_value(epoch_summary.get("mean_scanpath_mlm_loss")),
                    _fmt_debug_value(epoch_summary.get("mean_total_loss")),
                    str(row.get("metric_used_for_update")),
                ]
            )
        )
        if row.get("warning"):
            lines.append(f"warning epoch {epoch}: {row['warning']}")

    if not summary.get("scanpath_weight_log_rows"):
        lines.extend(
            [
                "",
                "No se genero log adaptativo porque adaptive_scanpath_weight=False.",
                "En ese modo el comportamiento queda fijo: loss_total = loss_mlm + aux_weight * loss_scanpath.",
            ]
        )

    lines.extend(
        [
            "",
            "Validaciones esperadas",
            f"- adaptive_scanpath_weight: {args.adaptive_scanpath_weight}",
            f"- lambda_min: {args.scanpath_weight_min}",
            f"- lambda_max: {args.scanpath_weight_max}",
            f"- warmup_epochs: {args.scanpath_weight_warmup_epochs}",
            f"- metric_choice: {args.scanpath_weight_update_metric}",
            f"- lambda_log_csv: {summary.get('scanpath_weight_log_path')}",
            "- lambda se actualiza solo con promedios por epoch, no por batch.",
            "- si current_scanpath_loss empeora respecto de initial_scanpath_loss, progress queda clipeado en 0.",
            "- si no hay eval_loss_scanpath en modo auto, se usa train_loss_scanpath_mean y se deja warning.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_lambda_readme(args, summary: Dict[str, Any], script_name: str) -> str:
    return f"""PASO 3 - Lambda adaptativo simple para loss scanpath
====================================================

Que se cambio
- Se agrego un modo opcional para reemplazar el peso fijo aux_weight de la rama scanpath por lambda_scanpath_t.
- No se cambio arquitectura, dataset, labels MLM, labels scanpath, conversion palabra-token, GRU, head MLM, downstream ni split.

Donde estaba antes la loss total
- Archivo: Gazesup_bert_combined_mlm_model.py
- Antes: total_loss = main_mlm_loss + float(aux_weight) * scanpath_mlm_loss

Donde se usa ahora
- Archivo: Gazesup_bert_combined_mlm_model.py
- Ahora el forward acepta current_scanpath_weight.
- Si current_scanpath_weight=None, usa aux_weight como antes.
- Si current_scanpath_weight tiene valor, usa ese peso efectivo:
  total_loss = main_mlm_loss + current_scanpath_weight * scanpath_mlm_loss

Donde se calcula lambda
- Archivo: {script_name}
- Funcion principal: train_step7
- Lambda se calcula al cierre de cada epoch usando una loss scanpath promedio.
- El valor calculado queda guardado como current_scanpath_weight y se pasa al forward durante la epoch siguiente.

Formula
- initial_scanpath_loss = primera loss scanpath promedio disponible.
- current_scanpath_loss = loss scanpath promedio usada para actualizar.
- progress = (initial_scanpath_loss - current_scanpath_loss) / initial_scanpath_loss
- progress se clipea a [0, 1].
- lambda_scanpath_t = scanpath_weight_min + progress * (scanpath_weight_max - scanpath_weight_min)

Metrica usada
- auto: usa eval_loss_scanpath si existe; si no existe, usa train_loss_scanpath_mean.
- eval_loss_scanpath: fuerza usar evaluacion.
- train_loss_scanpath_mean: fuerza usar entrenamiento.

Warmup
- Durante las primeras scanpath_weight_warmup_epochs, lambda se mantiene en scanpath_weight_min.
- Default: 1 epoch.

Como volver al comportamiento anterior
- Omitir --adaptive_scanpath_weight o usar --adaptive_scanpath_weight False.
- En ese modo se usa exactamente la forma fija:
  loss_total = loss_mlm + aux_weight * loss_scanpath

Argumentos nuevos
- --adaptive_scanpath_weight
- --scanpath_weight_min
- --scanpath_weight_max
- --scanpath_weight_warmup_epochs
- --scanpath_weight_update_metric
- --scanpath_weight_log_path

Limitaciones de esta estrategia simple
- Lambda depende de una unica loss scanpath promedio por epoch.
- No balancea gradientes como GradNorm.
- No mira si MLM empeora al mismo tiempo.
- No es una grilla de hiperparametros; es una regla simple y auditable.

Archivos de este paso
- lambda_scanpath_log.csv: evolucion numerica por epoch.
- salida_debug_lambda_adaptativo.txt: tabla legible de la corrida corta.
- comandos_y_funciones.txt: comandos y funciones usadas.
- README_lambda_adaptativo.txt: este documento.

Configuracion de esta corrida
- output_dir: {args.output_dir}
- adaptive_scanpath_weight: {args.adaptive_scanpath_weight}
- scanpath_weight_min: {args.scanpath_weight_min}
- scanpath_weight_max: {args.scanpath_weight_max}
- scanpath_weight_warmup_epochs: {args.scanpath_weight_warmup_epochs}
- scanpath_weight_update_metric: {args.scanpath_weight_update_metric}
- scanpath_weight_log_path: {summary.get('scanpath_weight_log_path')}
"""


def build_commands_file(args, script_name: str) -> str:
    relative_output = Path(args.output_dir).as_posix()
    command = (
        f"python {script_name} "
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
        f"--adaptive_scanpath_weight {args.adaptive_scanpath_weight} "
        f"--scanpath_weight_min {args.scanpath_weight_min} "
        f"--scanpath_weight_max {args.scanpath_weight_max} "
        f"--scanpath_weight_warmup_epochs {args.scanpath_weight_warmup_epochs} "
        f"--scanpath_weight_update_metric {args.scanpath_weight_update_metric} "
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
    if args.scanpath_weight_log_path:
        command += f" --scanpath_weight_log_path \"{args.scanpath_weight_log_path}\""
    if args.remove_punctuation_space:
        command += " --remove_punctuation_space"

    adaptive_command = command.replace("--adaptive_scanpath_weight False", "--adaptive_scanpath_weight True", 1)
    fixed_command = command.replace("--adaptive_scanpath_weight True", "--adaptive_scanpath_weight False", 1)

    return f"""COMANDOS Y FUNCIONES - PASO 7
=============================

Comando ejecutado
- {command}

Comando para correr con lambda adaptativo
- {adaptive_command}

Comando para correr con augweight fijo como antes
- {fixed_command}

Script principal usado
- {script_name}

Funciones principales llamadas
- measured_scanpath_utils.load_measured_scanpath_dataset
- measured_scanpath_utils.build_mlm_train_eval_datasets
- measured_scanpath_utils.build_precomputed_file_train_eval_datasets
- measured_scanpath_utils.build_measured_single_sentence_features
- build_static_masked_inputs_and_labels
- collate_measured_mlm_batch
- Gazesup_BERTForCombinedMaskedLM.from_pretrained
- Gazesup_BERTForCombinedMaskedLM.forward
- evaluate_model
- compute_adaptive_scanpath_weight
- select_scanpath_weight_update_metric
- write_scanpath_weight_log
- save_checkpoint
- torch.optim.AdamW

Output generado
- {args.output_dir}/split_report.json con validaciones del split limpio.
- {args.output_dir}/salida_training_step7.txt con resumen por epoch, losses y checkpoints.
- {args.output_dir}/loss_curves.csv con medias train/eval por epoch.
- {args.output_dir}/lambda_scanpath_log.csv con evolucion de lambda si adaptive_scanpath_weight=True.
- {args.output_dir}/README_lambda_adaptativo.txt si adaptive_scanpath_weight=True.
- {args.output_dir}/salida_debug_lambda_adaptativo.txt si adaptive_scanpath_weight=True.
- {args.output_dir}/README_paso_7.txt con el alcance de esta corrida mas grande.
- {args.output_dir}/comandos_y_funciones.txt con trazabilidad de la corrida.
- {args.output_dir}/checkpoint_epoch_*/, {args.output_dir}/best_checkpoint/ y {args.output_dir}/{args.final_checkpoint_dirname}/ con checkpoints reutilizables.
"""


def build_dry_run_output(args, summary: Dict[str, Any]) -> str:
    return f"""PASO 7 - Dry run de split limpio
=================================

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
        "scanpath_weight_log_path": summary.get("scanpath_weight_log_path"),
        "scanpath_weight_log_rows": summary.get("scanpath_weight_log_rows", []),
        "initial_scanpath_loss_for_lambda": summary.get("initial_scanpath_loss_for_lambda"),
        "adaptive_scanpath_weight": summary.get("adaptive_scanpath_weight", False),
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
        "scanpath_weight_log_path": args.scanpath_weight_log_path or str(output_dir / "lambda_scanpath_log.csv"),
        "scanpath_weight_log_rows": [],
        "adaptive_scanpath_weight": args.adaptive_scanpath_weight,
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
        summary = train_step7(args)
    except ValueError as exc:
        write_run_args(output_dir, args)
        write_failed_trainer_state(output_dir, args, exc)
        print(f"Wrote {output_dir / 'run_args.json'}")
        print(f"Wrote {output_dir / 'trainer_state.json'}")
        raise
    script_name = Path(__file__).name

    if summary.get("dry_run_split_only"):
        (output_dir / "salida_training_step7.txt").write_text(build_dry_run_output(args, summary), encoding="utf-8")
        (output_dir / "comandos_y_funciones.txt").write_text(build_commands_file(args, script_name), encoding="utf-8")
        write_run_args(output_dir, args)
        write_trainer_state(output_dir, summary)
        print(f"Wrote {output_dir / 'salida_training_step7.txt'}")
        print(f"Wrote {output_dir / 'comandos_y_funciones.txt'}")
        print(f"Wrote {output_dir / 'run_args.json'}")
        print(f"Wrote {output_dir / 'trainer_state.json'}")
        print(f"Wrote {summary['split_report_path']}")
        return

    (output_dir / "salida_training_step7.txt").write_text(build_debug_output(args, summary), encoding="utf-8")
    (output_dir / "README_paso_7.txt").write_text(build_readme(args, summary, script_name), encoding="utf-8")
    if args.adaptive_scanpath_weight:
        (output_dir / "README_lambda_adaptativo.txt").write_text(build_lambda_readme(args, summary, script_name), encoding="utf-8")
        (output_dir / "salida_debug_lambda_adaptativo.txt").write_text(build_lambda_debug_output(args, summary), encoding="utf-8")
    (output_dir / "comandos_y_funciones.txt").write_text(build_commands_file(args, script_name), encoding="utf-8")
    write_run_args(output_dir, args)
    write_trainer_state(output_dir, summary)

    print(f"Wrote {output_dir / 'salida_training_step7.txt'}")
    print(f"Wrote {output_dir / 'README_paso_7.txt'}")
    if args.adaptive_scanpath_weight:
        print(f"Wrote {output_dir / 'README_lambda_adaptativo.txt'}")
        print(f"Wrote {output_dir / 'salida_debug_lambda_adaptativo.txt'}")
        if summary.get("scanpath_weight_log_path"):
            print(f"Wrote {summary['scanpath_weight_log_path']}")
    print(f"Wrote {output_dir / 'comandos_y_funciones.txt'}")
    print(f"Wrote {output_dir / 'run_args.json'}")
    print(f"Wrote {output_dir / 'trainer_state.json'}")
    print(f"Wrote {summary['split_report_path']}")
    print(f"Wrote {summary['loss_curves_path']}")
    print(f"Saved final checkpoint to {summary['final_checkpoint_path']}")
    if summary['best_checkpoint_path']:
        print(f"Saved best checkpoint to {summary['best_checkpoint_path']}")


if __name__ == "__main__":
    main()
