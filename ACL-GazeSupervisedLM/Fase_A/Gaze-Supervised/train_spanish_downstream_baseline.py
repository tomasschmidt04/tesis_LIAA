import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


INTERTASS2020_URL = (
    "https://huggingface.co/datasets/iberbench/iberbench_all/resolve/main/"
    "iberlef-tass-sentiment_analysis-2020-spanish/train-00000-of-00001.parquet?download=true"
)

TASK_SPECS = {
    "xnli_es": {
        "task_type": "sentence pair classification",
        "text_columns": ["premise", "hypothesis"],
        "metric_name": "accuracy",
        "num_labels": 3,
        "dataset_source": 'datasets.load_dataset("xnli", "es")',
        "dataset_note": "Se usa train para entrenamiento y validation para evaluacion.",
    },
    "intertass2020": {
        "task_type": "single sentence classification",
        "text_columns": ["text"],
        "metric_name": "macro_f1",
        "num_labels": 3,
        "dataset_source": (
            "iberbench/iberbench_all raw parquet: "
            "iberlef-tass-sentiment_analysis-2020-spanish"
        ),
        "dataset_note": (
            "La fuente expone solo train; se construye validation con train_test_split "
            "estratificado por label usando seed fija."
        ),
    },
    "rioplatense_hate_speech": {
        "task_type": "single sentence binary classification",
        "text_columns": ["text"],
        "metric_name": "f1",
        "num_labels": 2,
        "dataset_source": (
            'datasets.load_dataset("piuba-bigdata/contextualized_hate_speech") '
            "referenciado por finiteautomata/rioplatense_hate_speech"
        ),
        "dataset_note": (
            "Se usa HATEFUL como etiqueta binaria. Por defecto: split train oficial "
            "para entrenamiento, dev oficial para validation y test oficial para test final."
        ),
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a minimal Spanish downstream fine-tuning baseline for xnli_es, "
            "intertass2020 or rioplatense_hate_speech."
        )
    )
    parser.add_argument("--model_name_or_path", required=True, help="Checkpoint or model directory used as downstream backbone.")
    parser.add_argument("--task_name", required=True, choices=sorted(TASK_SPECS.keys()), help="Spanish downstream task.")
    parser.add_argument("--output_dir", required=True, help="Directory where task outputs will be written.")
    parser.add_argument("--max_train_samples", type=int, default=500, help="Maximum number of training examples.")
    parser.add_argument("--max_eval_samples", type=int, default=200, help="Maximum number of evaluation examples.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Evaluation batch size.")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Maximum tokenized sequence length.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=13, help="Random seed.")
    parser.add_argument(
        "--expected_num_labels",
        type=int,
        default=None,
        help="Optional safety check: fail if the task does not use this number of labels.",
    )
    parser.add_argument("--evaluation_strategy", default="epoch", choices=["no", "steps", "epoch"], help="Trainer evaluation strategy.")
    parser.add_argument("--save_strategy", default="epoch", choices=["no", "steps", "epoch"], help="Trainer checkpoint save strategy.")
    parser.add_argument(
        "--load_best_model_at_end",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether Trainer should reload the best checkpoint at the end.",
    )
    parser.add_argument("--save_total_limit", type=int, default=1, help="Maximum number of downstream checkpoints to keep.")
    parser.add_argument(
        "--intertass_cache_dir",
        default="Pasos/paso_9_es_l7b/cache",
        help="Cache directory used to store the downloaded InterTass2020 parquet.",
    )
    parser.add_argument(
        "--rioplatense_dataset_name",
        default="piuba-bigdata/contextualized_hate_speech",
        help="Hugging Face dataset used for rioplatense_hate_speech.",
    )
    parser.add_argument(
        "--rioplatense_use_local_csvs",
        action="store_true",
        help="Use local CSVs from --rioplatense_hs_dir instead of the Hugging Face dataset.",
    )
    parser.add_argument(
        "--rioplatense_hs_dir",
        default="rioplatense_hate_speech",
        help="Path to the finiteautomata/rioplatense_hate_speech repository clone when using local CSVs.",
    )
    parser.add_argument(
        "--rioplatense_train_files",
        default="data/test_01.csv,data/test_02.csv,data/test_03.csv",
        help="Comma-separated CSV files used as the train pool for rioplatense_hate_speech.",
    )
    parser.add_argument(
        "--rioplatense_validation_file",
        default="data/test_04.csv",
        help="CSV file used as validation split for rioplatense_hate_speech.",
    )
    parser.add_argument(
        "--rioplatense_test_file",
        default="data/test_05.csv",
        help="CSV file used as final test split for rioplatense_hate_speech.",
    )
    return parser.parse_args()


def ensure_parent_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def download_with_cache(url: str, target_path: Path):
    if target_path.exists():
        return target_path

    ensure_parent_dir(target_path)
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    target_path.write_bytes(response.content)
    return target_path


def cap_dataset(dataset: Dataset, max_samples: int, seed: int) -> Dataset:
    if max_samples is None or max_samples <= 0 or len(dataset) <= max_samples:
        return dataset
    shuffled = dataset.shuffle(seed=seed)
    return shuffled.select(range(max_samples))


def split_csv_arg(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_dataset_file(dataset_root: Path, file_name: str) -> Path:
    path = Path(file_name)
    if path.is_absolute():
        if path.exists():
            return path
        raise FileNotFoundError(f"Dataset file not found: {path}")

    candidates = [dataset_root / path]
    if not str(path).startswith("data/"):
        candidates.append(dataset_root / "data" / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Dataset file not found: {file_name}. Tried: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def class_counts(dataframe: pd.DataFrame, label_column: str = "label") -> Dict[str, int]:
    counts = dataframe[label_column].value_counts().sort_index()
    return {str(int(label)): int(count) for label, count in counts.items()}


def compute_stratified_sample_counts(class_count_series: pd.Series, sample_size: int) -> Dict[int, int]:
    total_count = int(class_count_series.sum())
    raw_counts = class_count_series * (sample_size / total_count)
    sample_counts = np.floor(raw_counts).astype(int)

    remaining = sample_size - int(sample_counts.sum())
    remainders = (raw_counts - sample_counts).sort_values(ascending=False)
    for label in remainders.index:
        if remaining <= 0:
            break
        if sample_counts[label] < class_count_series[label]:
            sample_counts[label] += 1
            remaining -= 1

    labels = list(sample_counts.index)
    while remaining > 0:
        added_any = False
        for label in labels:
            if remaining <= 0:
                break
            if sample_counts[label] < class_count_series[label]:
                sample_counts[label] += 1
                remaining -= 1
                added_any = True
        if not added_any:
            break

    return {int(label): int(count) for label, count in sample_counts.items()}


def stratified_sample_dataframe(dataframe: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    if sample_size is None or sample_size <= 0 or len(dataframe) <= sample_size:
        return dataframe.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    counts = dataframe["label"].value_counts().sort_index()
    sample_counts = compute_stratified_sample_counts(counts, sample_size)
    sampled_parts = []
    for label, label_sample_size in sample_counts.items():
        if label_sample_size <= 0:
            continue
        label_rows = dataframe[dataframe["label"] == label]
        sampled_parts.append(label_rows.sample(n=label_sample_size, random_state=seed))

    sampled = pd.concat(sampled_parts, ignore_index=True)
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def prepare_rioplatense_dataframe(dataframe: pd.DataFrame, source_name: str) -> pd.DataFrame:
    required_columns = {"text", "HATEFUL"}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"{source_name} is missing required columns: {sorted(missing_columns)}")

    keep_columns = ["text", "HATEFUL"]
    if "id" in dataframe.columns:
        keep_columns.insert(0, "id")

    dataframe = dataframe[keep_columns].dropna(subset=["text", "HATEFUL"]).copy()
    dataframe["text"] = dataframe["text"].astype(str)
    dataframe["label"] = dataframe["HATEFUL"].astype(int)
    invalid_labels = sorted(set(dataframe["label"]) - {0, 1})
    if invalid_labels:
        raise ValueError(f"{source_name} has non-binary HATEFUL labels: {invalid_labels}")
    dataframe["source_file"] = source_name
    return dataframe.drop(columns=["HATEFUL"])


def read_rioplatense_csvs(paths: List[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        dataframe = pd.read_csv(path)
        frames.append(prepare_rioplatense_dataframe(dataframe, path.name))

    if not frames:
        raise ValueError("No rioplatense_hate_speech CSV files were provided.")

    return pd.concat(frames, ignore_index=True)


def load_xnli_es_dataset(args) -> Tuple[DatasetDict, Dict[str, object]]:
    raw = load_dataset("xnli", "es")
    train_dataset = cap_dataset(raw["train"], args.max_train_samples, args.seed)
    eval_dataset = cap_dataset(raw["validation"], args.max_eval_samples, args.seed)
    label_names = list(raw["train"].features["label"].names)

    metadata = {
        "task_name": "xnli_es",
        "task_type": TASK_SPECS["xnli_es"]["task_type"],
        "text_columns": TASK_SPECS["xnli_es"]["text_columns"],
        "metric_name": TASK_SPECS["xnli_es"]["metric_name"],
        "dataset_source": TASK_SPECS["xnli_es"]["dataset_source"],
        "dataset_note": TASK_SPECS["xnli_es"]["dataset_note"],
        "num_labels": len(label_names),
        "label_list": label_names,
        "train_size": len(train_dataset),
        "eval_size": len(eval_dataset),
        "eval_split_origin": "validation oficial de xnli/es",
    }
    return DatasetDict({"train": train_dataset, "validation": eval_dataset}), metadata


def load_intertass2020_dataset(args) -> Tuple[DatasetDict, Dict[str, object]]:
    cache_dir = Path(args.intertass_cache_dir)
    parquet_path = cache_dir / "intertass2020_train.parquet"
    download_with_cache(INTERTASS2020_URL, parquet_path)

    dataframe = pd.read_parquet(parquet_path)
    dataframe = dataframe[["text", "label", "language"]].dropna(subset=["text", "label"]).reset_index(drop=True)
    dataframe["label"] = dataframe["label"].astype(str)

    dataset = Dataset.from_pandas(dataframe, preserve_index=False)
    dataset = dataset.class_encode_column("label")
    split = dataset.train_test_split(test_size=0.2, seed=args.seed, stratify_by_column="label")

    train_dataset = cap_dataset(split["train"], args.max_train_samples, args.seed)
    eval_dataset = cap_dataset(split["test"], args.max_eval_samples, args.seed)
    label_names = list(train_dataset.features["label"].names)

    metadata = {
        "task_name": "intertass2020",
        "task_type": TASK_SPECS["intertass2020"]["task_type"],
        "text_columns": TASK_SPECS["intertass2020"]["text_columns"],
        "metric_name": TASK_SPECS["intertass2020"]["metric_name"],
        "dataset_source": TASK_SPECS["intertass2020"]["dataset_source"],
        "dataset_note": TASK_SPECS["intertass2020"]["dataset_note"],
        "num_labels": len(label_names),
        "label_list": label_names,
        "train_size": len(train_dataset),
        "eval_size": len(eval_dataset),
        "eval_split_origin": "train_test_split(test_size=0.2, stratify_by_column='label', seed=13)",
        "intertass_parquet_path": str(parquet_path),
    }
    return DatasetDict({"train": train_dataset, "validation": eval_dataset}), metadata


def build_rioplatense_dataset_dict(
    train_dataframe: pd.DataFrame,
    validation_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
    args,
) -> Tuple[DatasetDict, Dict[str, object]]:
    train_pool_size = len(train_dataframe)
    train_dataframe = stratified_sample_dataframe(
        train_dataframe,
        sample_size=args.max_train_samples,
        seed=args.seed,
    )

    train_dataset = Dataset.from_pandas(train_dataframe, preserve_index=False)
    eval_dataset = cap_dataset(
        Dataset.from_pandas(validation_dataframe, preserve_index=False),
        args.max_eval_samples,
        args.seed,
    )
    test_dataset = cap_dataset(
        Dataset.from_pandas(test_dataframe, preserve_index=False),
        args.max_eval_samples,
        args.seed,
    )
    label_names = ["no_hate", "hate"]

    dataset_dict = DatasetDict({"train": train_dataset, "validation": eval_dataset, "test": test_dataset})
    metadata = {
        "task_name": "rioplatense_hate_speech",
        "task_type": TASK_SPECS["rioplatense_hate_speech"]["task_type"],
        "text_columns": TASK_SPECS["rioplatense_hate_speech"]["text_columns"],
        "metric_name": TASK_SPECS["rioplatense_hate_speech"]["metric_name"],
        "dataset_source": TASK_SPECS["rioplatense_hate_speech"]["dataset_source"],
        "dataset_note": TASK_SPECS["rioplatense_hate_speech"]["dataset_note"],
        "num_labels": len(label_names),
        "label_list": label_names,
        "label_column": "HATEFUL",
        "label_definition": "0=no_hate, 1=hate",
        "train_size": len(train_dataset),
        "eval_size": len(eval_dataset),
        "test_size": len(test_dataset),
        "train_pool_size": train_pool_size,
        "train_class_counts": class_counts(train_dataframe),
        "eval_class_counts": class_counts(validation_dataframe),
        "test_class_counts": class_counts(test_dataframe),
        "sampling_strategy": "stratified by HATEFUL using seed",
    }
    return dataset_dict, metadata


def load_rioplatense_hate_speech_local_csv_dataset(args) -> Tuple[DatasetDict, Dict[str, object]]:
    dataset_root = Path(args.rioplatense_hs_dir)
    train_paths = [
        resolve_dataset_file(dataset_root, file_name)
        for file_name in split_csv_arg(args.rioplatense_train_files)
    ]
    validation_path = resolve_dataset_file(dataset_root, args.rioplatense_validation_file)
    test_path = resolve_dataset_file(dataset_root, args.rioplatense_test_file)

    train_pool_dataframe = read_rioplatense_csvs(train_paths)
    validation_dataframe = read_rioplatense_csvs([validation_path])
    test_dataframe = read_rioplatense_csvs([test_path])

    dataset_dict, metadata = build_rioplatense_dataset_dict(
        train_pool_dataframe,
        validation_dataframe,
        test_dataframe,
        args,
    )
    metadata.update(
        {
            "dataset_source": (
                "finiteautomata/rioplatense_hate_speech local CSVs: "
                "data/test_01.csv..data/test_05.csv"
            ),
            "dataset_note": (
                "Local CSV fallback. Por defecto: test_01-test_03 como pool de train, "
                "test_04 como validation y test_05 como test final."
            ),
            "train_pool_size": len(train_pool_dataframe),
            "train_pool_class_counts": class_counts(train_pool_dataframe),
            "eval_split_origin": str(validation_path),
            "test_split_origin": str(test_path),
            "train_files": [str(path) for path in train_paths],
            "rioplatense_hs_dir": str(dataset_root),
        }
    )
    return dataset_dict, metadata


def load_rioplatense_hate_speech_dataset(args) -> Tuple[DatasetDict, Dict[str, object]]:
    if args.rioplatense_use_local_csvs:
        return load_rioplatense_hate_speech_local_csv_dataset(args)

    raw = load_dataset(args.rioplatense_dataset_name)
    train_dataframe = prepare_rioplatense_dataframe(raw["train"].to_pandas(), "hf_train")
    validation_dataframe = prepare_rioplatense_dataframe(raw["dev"].to_pandas(), "hf_dev")
    test_dataframe = prepare_rioplatense_dataframe(raw["test"].to_pandas(), "hf_test")

    dataset_dict, metadata = build_rioplatense_dataset_dict(
        train_dataframe,
        validation_dataframe,
        test_dataframe,
        args,
    )
    metadata.update(
        {
            "dataset_name": args.rioplatense_dataset_name,
            "train_pool_size": len(train_dataframe),
            "train_pool_class_counts": class_counts(train_dataframe),
            "eval_split_origin": "official dev split",
            "test_split_origin": "official test split",
        }
    )
    return dataset_dict, metadata


def load_task_dataset(task_name: str, args) -> Tuple[DatasetDict, Dict[str, object]]:
    if task_name == "xnli_es":
        return load_xnli_es_dataset(args)
    if task_name == "intertass2020":
        return load_intertass2020_dataset(args)
    if task_name == "rioplatense_hate_speech":
        return load_rioplatense_hate_speech_dataset(args)
    raise ValueError(f"Unsupported task_name: {task_name}")


def tokenize_dataset(task_name: str, raw_datasets: DatasetDict, tokenizer, max_seq_length: int):
    text_columns = TASK_SPECS[task_name]["text_columns"]

    def preprocess_function(example):
        if len(text_columns) == 2:
            return tokenizer(
                example[text_columns[0]],
                example[text_columns[1]],
                truncation=True,
                max_length=max_seq_length,
            )
        return tokenizer(
            example[text_columns[0]],
            truncation=True,
            max_length=max_seq_length,
        )

    return raw_datasets.map(preprocess_function, batched=True, remove_columns=[])


def compute_accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    if labels.size == 0:
        return 0.0
    return float((predictions == labels).sum() / labels.size)


def compute_macro_f1(predictions: np.ndarray, labels: np.ndarray, num_labels: int) -> float:
    f1_values: List[float] = []
    for label_id in range(num_labels):
        true_positive = int(((predictions == label_id) & (labels == label_id)).sum())
        false_positive = int(((predictions == label_id) & (labels != label_id)).sum())
        false_negative = int(((predictions != label_id) & (labels == label_id)).sum())

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        f1_values.append(f1)

    return float(sum(f1_values) / len(f1_values)) if f1_values else 0.0


def compute_binary_f1(predictions: np.ndarray, labels: np.ndarray, positive_label: int = 1) -> float:
    true_positive = int(((predictions == positive_label) & (labels == positive_label)).sum())
    false_positive = int(((predictions == positive_label) & (labels != positive_label)).sum())
    false_negative = int(((predictions != positive_label) & (labels == positive_label)).sum())

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
    return float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0


def build_compute_metrics(task_name: str, num_labels: int):
    metric_name = TASK_SPECS[task_name]["metric_name"]

    def compute_metrics(eval_prediction):
        logits, labels = eval_prediction
        predictions = np.argmax(logits, axis=-1)
        metrics = {}
        if task_name == "rioplatense_hate_speech":
            metrics["f1"] = compute_binary_f1(predictions, labels, positive_label=1)
            metrics["accuracy"] = compute_accuracy(predictions, labels)
            metrics["macro_f1"] = compute_macro_f1(predictions, labels, num_labels)
        elif metric_name == "accuracy":
            metrics["accuracy"] = compute_accuracy(predictions, labels)
        elif metric_name == "macro_f1":
            metrics["macro_f1"] = compute_macro_f1(predictions, labels, num_labels)
        elif metric_name == "f1":
            metrics["f1"] = compute_binary_f1(predictions, labels, positive_label=1)
        else:
            raise ValueError(f"Unsupported metric_name: {metric_name}")
        return metrics

    return compute_metrics

def build_forward_debug(task_name: str, model_name_or_path: str, tokenizer, raw_datasets: DatasetDict, num_labels: int, max_seq_length: int):
    config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        finetuning_task=task_name,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        config=config,
        ignore_mismatched_sizes=True,
    )
    model.eval()

    sample = raw_datasets["train"][0]
    text_columns = TASK_SPECS[task_name]["text_columns"]
    if len(text_columns) == 2:
        batch = tokenizer(
            sample[text_columns[0]],
            sample[text_columns[1]],
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )
    else:
        batch = tokenizer(
            sample[text_columns[0]],
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )

    with np.errstate(all="ignore"):
        with torch.no_grad():
            outputs = model(**batch)

    return {
        "tokenizer": tokenizer.__class__.__name__,
        "model_class": model.__class__.__name__,
        "model_type": config.model_type,
        "num_labels": int(config.num_labels),
        "input_ids_shape": tuple(batch["input_ids"].shape),
        "logits_shape": tuple(outputs.logits.shape),
    }


def write_key_value_metrics(path: Path, metrics: Dict[str, object]):
    lines = [f"{key} = {value}" for key, value in metrics.items()]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    raw_datasets, task_metadata = load_task_dataset(args.task_name, args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    tokenized_datasets = tokenize_dataset(args.task_name, raw_datasets, tokenizer, args.max_seq_length)
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]
    test_dataset = tokenized_datasets["test"] if "test" in tokenized_datasets else None

    label_list = list(task_metadata["label_list"])
    num_labels = int(task_metadata["num_labels"])
    if args.expected_num_labels is not None and num_labels != args.expected_num_labels:
        raise ValueError(
            f"Task {args.task_name} resolved num_labels={num_labels}, "
            f"but --expected_num_labels={args.expected_num_labels}."
        )
    label2id = {label_name: index for index, label_name in enumerate(label_list)}
    id2label = {index: label_name for index, label_name in enumerate(label_list)}

    forward_debug = build_forward_debug(
        task_name=args.task_name,
        model_name_or_path=args.model_name_or_path,
        tokenizer=tokenizer,
        raw_datasets=raw_datasets,
        num_labels=num_labels,
        max_seq_length=args.max_seq_length,
    )

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        label2id=label2id,
        id2label=id2label,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        config=config,
        ignore_mismatched_sizes=True,
    )
    if args.expected_num_labels is not None and int(model.config.num_labels) != args.expected_num_labels:
        raise ValueError(
            f"Model config has num_labels={model.config.num_labels}, "
            f"but --expected_num_labels={args.expected_num_labels}."
        )

    metric_name = TASK_SPECS[args.task_name]["metric_name"]
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        do_train=True,
        do_eval=True,
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=metric_name,
        greater_is_better=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        max_steps=-1,
        logging_strategy="steps",
        logging_steps=10,
        report_to="none",
        overwrite_output_dir=True,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=build_compute_metrics(args.task_name, num_labels),
    )

    train_result = trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    train_metrics = dict(train_result.metrics)
    train_metrics["train_samples"] = len(train_dataset)
    eval_metrics = trainer.evaluate()
    eval_metrics["eval_samples"] = len(eval_dataset)
    test_metrics = None
    if test_dataset is not None:
        test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
        test_metrics["test_samples"] = len(test_dataset)

    write_key_value_metrics(output_dir / "train_results.txt", train_metrics)
    trainer.state.save_to_json(str(output_dir / "trainer_state.json"))
    (output_dir / "eval_results.json").write_text(json.dumps(eval_metrics, indent=2), encoding="utf-8")
    if test_metrics is not None:
        (output_dir / "test_results.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")

    task_metadata.update(
        {
            "label2id": label2id,
            "id2label": {str(key): value for key, value in id2label.items()},
            "forward_debug": forward_debug,
            "model_name_or_path": args.model_name_or_path,
            "tokenizer_class": tokenizer.__class__.__name__,
            "output_dir": str(output_dir),
            "max_train_samples": args.max_train_samples,
            "max_eval_samples": args.max_eval_samples,
            "num_train_epochs": args.num_train_epochs,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "max_seq_length": args.max_seq_length,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "expected_num_labels": args.expected_num_labels,
            "evaluation_strategy": args.evaluation_strategy,
            "save_strategy": args.save_strategy,
            "load_best_model_at_end": args.load_best_model_at_end,
            "save_total_limit": args.save_total_limit,
            "metric_name": metric_name,
            "best_model_checkpoint": trainer.state.best_model_checkpoint,
        }
    )
    (output_dir / "task_metadata.json").write_text(json.dumps(task_metadata, indent=2), encoding="utf-8")

    print(f"Wrote {output_dir / 'train_results.txt'}")
    print(f"Wrote {output_dir / 'trainer_state.json'}")
    print(f"Wrote {output_dir / 'eval_results.json'}")
    if test_metrics is not None:
        print(f"Wrote {output_dir / 'test_results.json'}")
    print(f"Wrote {output_dir / 'task_metadata.json'}")
    print(f"Saved model to {output_dir}")


if __name__ == "__main__":
    main()
