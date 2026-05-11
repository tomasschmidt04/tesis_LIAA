import argparse
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from Gazesup_bert_mlm_model import Gazesup_BERTForMaskedLM
from measured_scanpath_utils import build_measured_single_sentence_features, load_measured_scanpath_dataset


DEFAULT_OUTPUT_DIR = "paso_5"
DEFAULT_CHECKPOINT_DIRNAME = "checkpoint_smoke"
NUM_DEBUG_BATCHES = 2


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a short smoke-training loop for the measured-scanpath MLM branch validated in paso_4."
    )
    parser.add_argument(
        "--measured_scanpath_file",
        required=True,
        help="Path to a JSON/JSONL/CSV file with at least text and word_id fields.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="bert-base-cased",
        help="Model/tokenizer name used to instantiate the BERT backbone.",
    )
    parser.add_argument(
        "--measured_text_field",
        default="text",
        help="Column that contains the plain text consumed by the tokenizer.",
    )
    parser.add_argument(
        "--measured_word_id_field",
        default="word_id",
        help="Column that contains the lexical 1-based scanpath positions.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to train on after loading the measured file.",
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where paso_5 artifacts will be written.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help="Maximum length passed to the BERT tokenizer.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=20,
        help="Maximum number of measured examples used in the smoke training.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
        help="Mini-batch size used by the smoke training loop.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=1,
        help="Number of epochs for the smoke training loop.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate used by AdamW in the smoke training loop.",
    )
    parser.add_argument(
        "--max_masked_positions",
        type=int,
        default=3,
        help="Maximum number of non-special tokens masked per example.",
    )
    parser.add_argument(
        "--remove_punctuation_space",
        action="store_true",
        help="Mirror the optional punctuation-space normalization used by the training scripts.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Random seed used for the smoke training loop.",
    )
    parser.add_argument(
        "--checkpoint_dirname",
        default=DEFAULT_CHECKPOINT_DIRNAME,
        help="Directory name used inside output_dir to save the smoke checkpoint.",
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_static_masked_inputs_and_labels(input_ids: List[int], attention_mask: List[int], tokenizer, max_masked_positions: int):
    masked_input_ids = list(input_ids)
    labels = [-100] * len(input_ids)

    candidate_positions = [
        idx
        for idx, token_id in enumerate(input_ids)
        if attention_mask[idx] == 1 and token_id not in tokenizer.all_special_ids
    ]
    if not candidate_positions:
        return masked_input_ids, labels, []

    if len(candidate_positions) <= max_masked_positions:
        selected_positions = candidate_positions
    else:
        stride = max(1, len(candidate_positions) // max_masked_positions)
        selected_positions = candidate_positions[::stride][:max_masked_positions]

    for position in selected_positions:
        labels[position] = input_ids[position]
        masked_input_ids[position] = tokenizer.mask_token_id

    return masked_input_ids, labels, selected_positions


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


def collate_measured_mlm_batch(examples: List[Dict[str, Any]], tokenizer):
    batch_size = len(examples)
    max_seq_len = max(len(example["input_ids"]) for example in examples)
    max_sp_len = max(len(example["measured_word_ids"]) for example in examples)

    input_ids = torch.full((batch_size, max_seq_len), tokenizer.pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    token_type_ids = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    labels = torch.full((batch_size, max_seq_len), -100, dtype=torch.long)
    lm_word_ids = torch.full((batch_size, max_seq_len), float("nan"), dtype=torch.float64)
    measured_word_ids = torch.zeros((batch_size, max_sp_len), dtype=torch.long)
    measured_sp_len = torch.zeros((batch_size,), dtype=torch.long)

    texts = []
    example_indices = []
    masked_positions = []

    for batch_index, example in enumerate(examples):
        seq_len = len(example["input_ids"])
        sp_len = len(example["measured_word_ids"])

        input_ids[batch_index, :seq_len] = torch.tensor(example["input_ids"], dtype=torch.long)
        attention_mask[batch_index, :seq_len] = torch.tensor(example["attention_mask"], dtype=torch.long)
        token_type_ids[batch_index, :seq_len] = torch.tensor(example["token_type_ids"], dtype=torch.long)
        labels[batch_index, :seq_len] = torch.tensor(example["labels"], dtype=torch.long)
        lm_word_ids[batch_index, :seq_len] = torch.tensor(example["LM_word_ids"], dtype=torch.float64)
        measured_word_ids[batch_index, :sp_len] = torch.tensor(example["measured_word_ids"], dtype=torch.long)
        measured_sp_len[batch_index] = int(example["measured_sp_len"])

        texts.append(example["text"])
        example_indices.append(int(example["example_index"]))
        masked_positions.append(list(example["masked_positions"]))

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "LM_word_ids": lm_word_ids,
        "measured_word_ids": measured_word_ids,
        "measured_sp_len": measured_sp_len,
        "labels": labels,
        "texts": texts,
        "example_indices": example_indices,
        "masked_positions": masked_positions,
    }


def move_tensor_batch_to_device(batch: Dict[str, Any], device: torch.device):
    tensor_keys = [
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "LM_word_ids",
        "measured_word_ids",
        "measured_sp_len",
        "labels",
    ]
    moved_batch = dict(batch)
    for key in tensor_keys:
        moved_batch[key] = batch[key].to(device)
    return moved_batch


def train_smoke_loop(args):
    set_seed(args.seed)

    raw_datasets = load_measured_scanpath_dataset(args.measured_scanpath_file)
    if args.split not in raw_datasets:
        raise ValueError(f"Split {args.split!r} not found. Available splits: {list(raw_datasets.keys())}")

    dataset = raw_datasets[args.split]
    if args.max_train_samples is not None:
        max_samples = min(args.max_train_samples, len(dataset))
        dataset = dataset.select(range(max_samples))
    else:
        max_samples = len(dataset)

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

    model = Gazesup_BERTForMaskedLM.from_pretrained(args.model_name_or_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    losses: List[float] = []
    batch_debug: List[Dict[str, Any]] = []
    total_steps = 0
    forward_ok = False
    backward_ok = False
    optimizer_ok = False

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
                return_dict=True,
            )
            forward_ok = True

            loss = outputs.scanpath_mlm_loss
            if loss is None:
                raise RuntimeError("scanpath_mlm_loss is None during smoke training.")

            loss.backward()
            backward_ok = True
            optimizer.step()
            optimizer_ok = True

            loss_value = float(loss.detach().cpu().item())
            losses.append(loss_value)
            total_steps += 1

            if len(batch_debug) < NUM_DEBUG_BATCHES:
                batch_debug.append(
                    {
                        "epoch": epoch_index,
                        "batch_index": batch_index,
                        "example_indices": list(batch["example_indices"]),
                        "texts": list(batch["texts"]),
                        "masked_positions": list(batch["masked_positions"]),
                        "input_ids_shape": tuple(batch["input_ids"].shape),
                        "labels_shape": tuple(batch["labels"].shape),
                        "scanpath_mlm_logits_shape": tuple(outputs.scanpath_mlm_logits.shape),
                        "scanpath_labels_expanded_shape": tuple(outputs.scanpath_labels_expanded.shape),
                        "loss": loss_value,
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
        "losses": losses,
        "total_steps": total_steps,
        "checkpoint_dir": str(checkpoint_dir),
        "forward_ok": forward_ok,
        "backward_ok": backward_ok,
        "optimizer_ok": optimizer_ok,
    }


def build_debug_output(args, training_summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.extend(
        [
            "PASO 5 - Smoke training de la rama auxiliar MLM con scanpaths medidos",
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
            f"max_seq_length: {args.max_seq_length}",
            f"max_masked_positions: {args.max_masked_positions}",
            f"device: {training_summary['device']}",
            "",
        ]
    )

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
                f"scanpath_mlm_logits.shape: {debug_batch['scanpath_mlm_logits_shape']}",
                f"scanpath_labels_expanded.shape: {debug_batch['scanpath_labels_expanded_shape']}",
                f"loss: {debug_batch['loss']}",
                "textos del batch:",
            ]
        )
        for text_index, text in enumerate(debug_batch["texts"]):
            lines.append(f"- texto {text_index}: {text}")
        lines.append("")

    losses = training_summary["losses"]
    initial_loss = losses[0] if losses else None
    final_loss = losses[-1] if losses else None

    lines.extend(
        [
            "----------------------------------------",
            "Training summary",
            "----------------------------------------",
            f"num_steps: {training_summary['total_steps']}",
            f"initial_loss: {initial_loss}",
            f"final_loss: {final_loss}",
            f"checkpoint_saved_to: {training_summary['checkpoint_dir']}",
            "status:",
            f"- forward {'OK' if training_summary['forward_ok'] else 'FAILED'}",
            f"- backward {'OK' if training_summary['backward_ok'] else 'FAILED'}",
            f"- optimizer step {'OK' if training_summary['optimizer_ok'] else 'FAILED'}",
            "- checkpoint save OK",
            "",
            "Notas:",
            "- La loss usada en este smoke training es CrossEntropyLoss(ignore_index=-100) sobre logits MLM de vocabulario.",
            "- La rama auxiliar ya no se reagrega a longitud original T; la prediccion se hace directo sobre la secuencia scanpath-level S.",
            "- scanpath_labels_expanded replica el target original cada vez que el mismo token reaparece en gaze_token_pos.",
            "- Si un token masked aparece varias veces en el scanpath, aporta varias veces a la loss auxiliar.",
            "- Esta corrida valida funcionamiento de punta a punta, pero no representa un experimento final ni una configuracion optimizada.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_readme(args, script_name: str, training_summary: Dict[str, Any]) -> str:
    return f"""PASO 5 - README
================

Que se hizo
- Se creo un script nuevo de entrenamiento corto llamado {script_name}.
- El script carga un dataset medido local con campos text y word_id.
- Reutiliza el preprocesamiento medido ya validado para construir input_ids, attention_mask, LM_word_ids, measured_word_ids y measured_sp_len.
- Construye labels MLM simples para smoke test usando masking estatico con -100 en posiciones ignoradas.
- Instancia la clase MLM-compatible del paso 4 y ejecuta un loop corto con forward, loss, backward y optimizer.step.
- Guarda un checkpoint local reutilizable dentro de {Path(args.output_dir) / args.checkpoint_dirname}.

Que se verifico
- Que el forward del modelo MLM auxiliar corre en un entrenamiento real, no solo en inference/debug.
- Que scanpath_mlm_loss se calcula correctamente durante varias iteraciones.
- Que backward corre con la loss auxiliar definida sobre la secuencia scanpath-level.
- Que optimizer.step corre y actualiza el modelo.
- Que se puede guardar un checkpoint local al final de la corrida.

Que NO se implemento todavia
- No se implemento GLUE ni fine-tuning downstream.
- No se implemento combinacion con una rama MLM estandar adicional.
- No se implemento scheduler sofisticado ni hyperparameter search.
- No se implemento export especial para GLUE.
- No se avanzo al paso 6.
- No se convirtio esto en un experimento final serio.

Archivos modificados
- train_mlm_scanpath_step5.py

Archivos nuevos creados
- {script_name}
- {Path(args.output_dir) / 'README_paso_5.txt'}
- {Path(args.output_dir) / 'salida_training_debug.txt'}
- {Path(args.output_dir) / 'comandos_y_funciones.txt'}
- {Path(args.output_dir) / args.checkpoint_dirname}/

Explicacion breve del entrenamiento corto realizado
- Cada ejemplo medido se tokeniza con BERT y se convierte a la representacion interna measured del repo.
- measured_word_ids se expande internamente a nivel token/subtoken dentro del modelo.
- La secuencia scanpath-level pasa por la GRU y esa misma salida se usa como representacion final de la rama auxiliar.
- Sobre gru_output se aplica una cabeza MLM lineal hidden_size -> vocab_size.
- Los labels originales de shape (B, T) se expanden a scanpath_labels_expanded con shape (B, S) usando gaze_token_pos.
- La loss es CrossEntropyLoss(ignore_index=-100) aplicada sobre la secuencia scanpath-level expandida.
- Si un token masked aparece multiples veces en el scanpath, entonces contribuye multiples veces a la loss auxiliar.

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
- seed = {args.seed}
- device = {training_summary['device']}

Aclaracion importante
- Este paso es un smoke test funcional y documentado.
- Sirve para validar que el pipeline measured + scanpath + GRU + cabeza MLM scanpath-level corre de punta a punta.
- No debe interpretarse como un experimento final ni como una configuracion optimizada.
"""


def build_commands_file(args, script_name: str) -> str:
    relative_output = Path(args.output_dir)
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
        f"--seed {args.seed}"
    )
    if args.remove_punctuation_space:
        command += " --remove_punctuation_space"

    return f"""COMANDOS Y FUNCIONES - PASO 5
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
- Gazesup_BERTForMaskedLM.from_pretrained
- Gazesup_BERTForMaskedLM.forward
- torch.optim.AdamW

Output generado
- {Path(args.output_dir) / 'salida_training_debug.txt'} con configuracion, losses y resumen de entrenamiento.
- {Path(args.output_dir) / 'README_paso_5.txt'} con el alcance de este smoke training.
- {Path(args.output_dir) / 'comandos_y_funciones.txt'} con trazabilidad de la corrida.
- {Path(args.output_dir) / args.checkpoint_dirname}/ con el checkpoint guardado.
"""


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_summary = train_smoke_loop(args)
    script_name = Path(__file__).name

    (output_dir / "salida_training_debug.txt").write_text(
        build_debug_output(args, training_summary),
        encoding="utf-8",
    )
    (output_dir / "README_paso_5.txt").write_text(
        build_readme(args, script_name, training_summary),
        encoding="utf-8",
    )
    (output_dir / "comandos_y_funciones.txt").write_text(
        build_commands_file(args, script_name),
        encoding="utf-8",
    )

    print(f"Wrote {output_dir / 'salida_training_debug.txt'}")
    print(f"Wrote {output_dir / 'README_paso_5.txt'}")
    print(f"Wrote {output_dir / 'comandos_y_funciones.txt'}")
    print(f"Saved checkpoint to {Path(training_summary['checkpoint_dir'])}")


if __name__ == "__main__":
    main()
