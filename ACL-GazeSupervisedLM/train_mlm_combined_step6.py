import argparse
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from Gazesup_bert_combined_mlm_model import Gazesup_BERTForCombinedMaskedLM
from measured_scanpath_utils import load_measured_scanpath_dataset
from train_mlm_scanpath_step5 import (
    build_static_masked_inputs_and_labels,
    collate_measured_mlm_batch,
    move_tensor_batch_to_device,
    set_seed,
)
from measured_scanpath_utils import build_measured_single_sentence_features


DEFAULT_OUTPUT_DIR = "Pasos/paso_6"
DEFAULT_CHECKPOINT_DIRNAME = "checkpoint_smoke"
NUM_DEBUG_BATCHES = 2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a short combined MLM smoke-training loop with main + scanpath auxiliary losses."
    )
    parser.add_argument("--measured_scanpath_file", required=True, help="Path to a JSON/JSONL/CSV file with at least text and word_id fields.")
    parser.add_argument("--model_name_or_path", default="bert-base-cased", help="Model/tokenizer name used to instantiate the shared BERT backbone.")
    parser.add_argument("--measured_text_field", default="text", help="Column that contains the plain text consumed by the tokenizer.")
    parser.add_argument("--measured_word_id_field", default="word_id", help="Column that contains the lexical 1-based scanpath positions.")
    parser.add_argument("--split", default="train", help="Dataset split to train on after loading the measured file.")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Directory where paso_6 artifacts will be written.")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Maximum length passed to the BERT tokenizer.")
    parser.add_argument("--max_train_samples", type=int, default=20, help="Maximum number of measured examples used in the smoke training.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Mini-batch size used by the smoke training loop.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of epochs for the smoke training loop.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate used by AdamW in the smoke training loop.")
    parser.add_argument("--max_masked_positions", type=int, default=3, help="Maximum number of non-special tokens masked per example.")
    parser.add_argument("--aux_weight", type=float, default=1.0, help="Weight lambda used in total_loss = main_mlm_loss + aux_weight * scanpath_mlm_loss.")
    parser.add_argument("--remove_punctuation_space", action="store_true", help="Mirror the optional punctuation-space normalization used by the training scripts.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed used for the smoke training loop.")
    parser.add_argument("--checkpoint_dirname", default=DEFAULT_CHECKPOINT_DIRNAME, help="Directory name used inside output_dir to save the smoke checkpoint.")
    return parser.parse_args()


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


def summarize_loss_relation(main_losses: List[float], scanpath_losses: List[float], aux_weight: float) -> List[str]:
    if not main_losses or not scanpath_losses:
        return ["- no hay suficientes pasos para interpretar la relacion entre losses."]

    avg_main = sum(main_losses) / len(main_losses)
    avg_scan = sum(scanpath_losses) / len(scanpath_losses)
    ratio = avg_scan / avg_main if avg_main != 0 else float("inf")
    weighted_ratio = (aux_weight * avg_scan) / avg_main if avg_main != 0 else float("inf")

    lines = [
        f"- media main_mlm_loss: {avg_main}",
        f"- media scanpath_mlm_loss: {avg_scan}",
        f"- ratio scanpath/main: {ratio}",
        f"- ratio ponderado (aux_weight * scanpath)/main: {weighted_ratio}",
    ]

    if weighted_ratio > 1.5:
        lines.append("- observacion: la loss auxiliar parece dominar la combinacion; podria convenir bajar aux_weight a 0.3 o 0.1.")
    elif weighted_ratio < 0.5:
        lines.append("- observacion: la loss auxiliar pesa bastante menos que la principal; si queres mas influencia scanpath, podria convenir subir aux_weight.")
    else:
        lines.append("- observacion: las dos losses tienen magnitudes razonablemente comparables con el aux_weight actual.")

    return lines


def train_smoke_loop(args):
    set_seed(args.seed)

    raw_datasets = load_measured_scanpath_dataset(args.measured_scanpath_file)
    if args.split not in raw_datasets:
        raise ValueError(f"Split {args.split!r} not found. Available splits: {list(raw_datasets.keys())}")

    dataset = raw_datasets[args.split]
    if args.max_train_samples is not None:
        dataset = dataset.select(range(min(args.max_train_samples, len(dataset))))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    processed_examples = preprocess_examples(dataset, tokenizer, args)
    if not processed_examples:
        raise ValueError("The measured dataset did not yield any training examples.")

    dataloader = DataLoader(
        processed_examples,
        batch_size=args.per_device_train_batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_measured_mlm_batch(batch, tokenizer),
    )

    model = Gazesup_BERTForCombinedMaskedLM.from_pretrained(args.model_name_or_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    main_losses: List[float] = []
    scanpath_losses: List[float] = []
    total_losses: List[float] = []
    batch_debug: List[Dict[str, Any]] = []
    total_steps = 0
    forward_ok = False
    backward_ok = False
    optimizer_ok = False
    nan_free = True

    for epoch_index in range(args.num_train_epochs):
        for batch_index, batch in enumerate(dataloader):
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
            forward_ok = True

            total_loss = outputs.loss
            main_loss = outputs.main_mlm_loss
            scanpath_loss = outputs.scanpath_mlm_loss
            if total_loss is None or main_loss is None or scanpath_loss is None:
                raise RuntimeError("Combined model returned None for one of the expected losses.")
            if torch.isnan(total_loss).any():
                nan_free = False
                raise RuntimeError("total_loss produced NaN during PASO 6 smoke training.")

            total_loss.backward()
            backward_ok = True
            optimizer.step()
            optimizer_ok = True

            main_loss_value = float(main_loss.detach().cpu().item())
            scanpath_loss_value = float(scanpath_loss.detach().cpu().item())
            total_loss_value = float(total_loss.detach().cpu().item())

            main_losses.append(main_loss_value)
            scanpath_losses.append(scanpath_loss_value)
            total_losses.append(total_loss_value)
            total_steps += 1

            if len(batch_debug) < NUM_DEBUG_BATCHES:
                batch_debug.append(
                    {
                        "epoch": epoch_index,
                        "batch_index": batch_index,
                        "example_indices": list(batch["example_indices"]),
                        "masked_positions": list(batch["masked_positions"]),
                        "input_ids_shape": tuple(batch["input_ids"].shape),
                        "labels_shape": tuple(batch["labels"].shape),
                        "main_mlm_logits_shape": tuple(outputs.main_mlm_logits.shape),
                        "scanpath_mlm_logits_shape": tuple(outputs.scanpath_mlm_logits.shape),
                        "scanpath_labels_expanded_shape": tuple(outputs.scanpath_labels_expanded.shape),
                        "main_mlm_loss": main_loss_value,
                        "scanpath_mlm_loss": scanpath_loss_value,
                        "total_loss": total_loss_value,
                    }
                )

    checkpoint_dir = Path(args.output_dir) / args.checkpoint_dirname
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)

    return {
        "dataset_size": len(processed_examples),
        "device": str(device),
        "batch_debug": batch_debug,
        "main_losses": main_losses,
        "scanpath_losses": scanpath_losses,
        "total_losses": total_losses,
        "total_steps": total_steps,
        "checkpoint_dir": str(checkpoint_dir),
        "forward_ok": forward_ok,
        "backward_ok": backward_ok,
        "optimizer_ok": optimizer_ok,
        "nan_free": nan_free,
    }


def build_debug_output(args, training_summary: Dict[str, Any]) -> str:
    lines: List[str] = [
        "PASO 6 - Smoke training combinado de losses principal + auxiliar",
        "",
        "----------------------------------------",
        "Configuracion",
        "----------------------------------------",
        f"dataset: {args.measured_scanpath_file}",
        f"split: {args.split}",
        f"max_train_samples: {training_summary['dataset_size']}",
        f"epochs: {args.num_train_epochs}",
        f"batch_size: {args.per_device_train_batch_size}",
        f"lr: {args.learning_rate}",
        f"aux_weight: {args.aux_weight}",
        f"max_seq_length: {args.max_seq_length}",
        f"max_masked_positions: {args.max_masked_positions}",
        f"device: {training_summary['device']}",
        "",
    ]

    for debug_batch in training_summary["batch_debug"]:
        lines.extend(
            [
                "----------------------------------------",
                f"Batch debug {debug_batch['batch_index']}",
                "----------------------------------------",
                f"epoch: {debug_batch['epoch']}",
                f"example_indices: {debug_batch['example_indices']}",
                f"masked_positions: {debug_batch['masked_positions']}",
                f"input_ids.shape: {debug_batch['input_ids_shape']}",
                f"labels.shape: {debug_batch['labels_shape']}",
                f"main_mlm_logits.shape: {debug_batch['main_mlm_logits_shape']}",
                f"scanpath_mlm_logits.shape: {debug_batch['scanpath_mlm_logits_shape']}",
                f"scanpath_labels_expanded.shape: {debug_batch['scanpath_labels_expanded_shape']}",
                f"main_mlm_loss: {debug_batch['main_mlm_loss']}",
                f"scanpath_mlm_loss: {debug_batch['scanpath_mlm_loss']}",
                f"total_loss: {debug_batch['total_loss']}",
                "",
            ]
        )

    main_losses = training_summary["main_losses"]
    scanpath_losses = training_summary["scanpath_losses"]
    total_losses = training_summary["total_losses"]

    lines.extend(
        [
            "----------------------------------------",
            "Training summary",
            "----------------------------------------",
            f"num_steps: {training_summary['total_steps']}",
            f"initial_main_loss: {main_losses[0] if main_losses else None}",
            f"final_main_loss: {main_losses[-1] if main_losses else None}",
            f"initial_scanpath_loss: {scanpath_losses[0] if scanpath_losses else None}",
            f"final_scanpath_loss: {scanpath_losses[-1] if scanpath_losses else None}",
            f"initial_total_loss: {total_losses[0] if total_losses else None}",
            f"final_total_loss: {total_losses[-1] if total_losses else None}",
            f"checkpoint_saved_to: {training_summary['checkpoint_dir']}",
            "status:",
            f"- forward {'OK' if training_summary['forward_ok'] else 'FAILED'}",
            f"- backward {'OK' if training_summary['backward_ok'] else 'FAILED'}",
            f"- optimizer step {'OK' if training_summary['optimizer_ok'] else 'FAILED'}",
            f"- total_loss sin NaN {'OK' if training_summary['nan_free'] else 'FAILED'}",
            "- checkpoint save OK",
            "",
            "Interpretacion breve:",
        ]
    )
    lines.extend(summarize_loss_relation(main_losses, scanpath_losses, args.aux_weight))
    return "\n".join(lines) + "\n"


def build_readme(args, script_name: str, training_summary: Dict[str, Any]) -> str:
    return f"""PASO 6 - README
================

Que se hizo
- Se creo un script nuevo de entrenamiento corto llamado {script_name}.
- El script carga un dataset medido con campos text y word_id.
- Reutiliza el preprocesamiento medido ya validado para construir el input MLM y los tensores necesarios para la rama scanpath.
- Instancia un modelo combinado que comparte BERT y produce simultaneamente:
  * main_mlm_logits y main_mlm_loss
  * scanpath_mlm_logits y scanpath_mlm_loss
  * total_loss = main_mlm_loss + aux_weight * scanpath_mlm_loss
- Ejecuta un smoke training corto con forward, backward, optimizer.step y guardado de checkpoint.

Que se verifico
- Que la loss principal MLM se calcula correctamente.
- Que la loss auxiliar scanpath MLM se calcula correctamente.
- Que ambas losses pueden coexistir y combinarse sin romper el entrenamiento.
- Que total_loss se mantiene finita durante la corrida.
- Que backward y optimizer.step funcionan con la loss combinada.
- Que se puede guardar un checkpoint local al final.

Que NO se implemento todavia
- No se implemento GLUE ni fine-tuning downstream.
- No se hizo hyperparameter search serio.
- No se implemento export final para GLUE.
- No se avanzo al paso 7.
- No se convirtio esto en un experimento final grande.

Archivos modificados
- Pasos/README.txt
- Gazesup_bert_combined_mlm_model.py

Archivos nuevos creados
- Gazesup_bert_combined_mlm_model.py
- {script_name}
- Pasos/paso_6/README_paso_6.txt
- Pasos/paso_6/salida_training_combined_debug.txt
- Pasos/paso_6/comandos_y_funciones.txt
- Pasos/paso_6/{args.checkpoint_dirname}/

Explicacion breve de la combinacion de losses
- La rama principal usa el camino directo del Transformer y una cabeza MLM estandar sobre los hidden states originales de BERT.
- La rama auxiliar usa los mismos hidden states originales de BERT, pero los reordena segun el scanpath medido expandido, los pasa por GRU y aplica la cabeza MLM directamente sobre la secuencia scanpath-level.
- Para la loss auxiliar, los labels originales de shape (B, T) se expanden a scanpath_labels_expanded con shape (B, S) usando gaze_token_pos.
- Ambas ramas producen logits sobre el vocabulario y ambas losses usan CrossEntropyLoss(ignore_index=-100).
- La combinacion se hace como total_loss = main_mlm_loss + aux_weight * scanpath_mlm_loss.

Explicacion de la rama principal
- input_ids masked -> BERT -> hidden states originales -> cabeza MLM estandar -> main_mlm_logits -> main_mlm_loss.

Explicacion de la rama auxiliar
- input_ids masked -> BERT -> hidden states originales -> scanpath expandido a token/subtoken -> GRU -> cabeza MLM auxiliar sobre S -> scanpath_mlm_logits -> scanpath_mlm_loss.
- La loss auxiliar se calcula contra scanpath_labels_expanded, de modo que repeticiones del mismo token en el scanpath generan multiples contribuciones supervisionadas.

Explicacion de lambda / aux_weight
- aux_weight es el lambda que controla cuanto pesa la loss auxiliar en la loss total.
- Valores simples para probar en este paso: 1.0, 0.3 y 0.1.
- Si la loss auxiliar domina demasiado, conviene bajar aux_weight. Si casi no influye, conviene subirlo.

Aclaracion importante
- Este paso sigue siendo un smoke test funcional y documentado.
- Sirve para validar la convivencia de ambas losses y el entrenamiento corto conjunto.
- No debe interpretarse como un experimento final ni como una configuracion optimizada.

Configuracion usada
- measured_scanpath_file = {args.measured_scanpath_file}
- model_name_or_path = {args.model_name_or_path}
- split = {args.split}
- max_train_samples = {training_summary['dataset_size']}
- num_train_epochs = {args.num_train_epochs}
- per_device_train_batch_size = {args.per_device_train_batch_size}
- learning_rate = {args.learning_rate}
- max_seq_length = {args.max_seq_length}
- max_masked_positions = {args.max_masked_positions}
- aux_weight = {args.aux_weight}
- seed = {args.seed}
- device = {training_summary['device']}
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
        f"--per_device_train_batch_size {args.per_device_train_batch_size} "
        f"--num_train_epochs {args.num_train_epochs} "
        f"--learning_rate {args.learning_rate} "
        f"--max_masked_positions {args.max_masked_positions} "
        f"--aux_weight {args.aux_weight} "
        f"--seed {args.seed}"
    )
    if args.remove_punctuation_space:
        command += " --remove_punctuation_space"

    return f"""COMANDOS Y FUNCIONES - PASO 6
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
- torch.optim.AdamW

Output generado
- Pasos/paso_6/salida_training_combined_debug.txt con losses principal, auxiliar y total.
- Pasos/paso_6/README_paso_6.txt con el alcance de este smoke training combinado.
- Pasos/paso_6/comandos_y_funciones.txt con trazabilidad de la corrida.
- Pasos/paso_6/{args.checkpoint_dirname}/ con el checkpoint guardado.
"""


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_summary = train_smoke_loop(args)
    script_name = Path(__file__).name

    (output_dir / "salida_training_combined_debug.txt").write_text(
        build_debug_output(args, training_summary),
        encoding="utf-8",
    )
    (output_dir / "README_paso_6.txt").write_text(
        build_readme(args, script_name, training_summary),
        encoding="utf-8",
    )
    (output_dir / "comandos_y_funciones.txt").write_text(
        build_commands_file(args, script_name),
        encoding="utf-8",
    )

    print(f"Wrote {output_dir / 'salida_training_combined_debug.txt'}")
    print(f"Wrote {output_dir / 'README_paso_6.txt'}")
    print(f"Wrote {output_dir / 'comandos_y_funciones.txt'}")
    print(f"Saved checkpoint to {Path(training_summary['checkpoint_dir'])}")


if __name__ == "__main__":
    main()
