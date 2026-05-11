import ast
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from datasets import load_dataset

from utils import remove_punctuation_split

CUSTOM_ALIGNED_SCANPATH_TASK = "custom_aligned_scanpath"
DEFAULT_GAZE_FEATURE_NAMES = ["FFD", "TRT", "nFix"]
GAZE_FEATURE_ALIASES = {
    "FFD": ["FFD"],
    "TRT": ["TRT", "TFD"],
    "nFix": ["nFix", "FC"],
}

IGNORE_TOKEN_BY_TOKENIZER = [
    "\uf0b7",
    "\ufeff",
    "\uf105",
    "\uf0ba",
    "\uf03d",
    "\uf0d8",
    "\uf0fc",
    "\u202c",
]

DEFAULT_STORY_ID_FIELD_CANDIDATES = [
    "story_id",
    "story",
    "cuento_id",
    "cuento",
    "document_id",
    "text_id",
]
DEFAULT_SENTENCE_POSITION_FIELD_CANDIDATES = [
    "sentence_position",
    "sentence_id",
    "trial_position",
    "period_block_start",
    "segment_index",
    "global_word_start",
]


def normalize_scanpath_source(scanpath_source: Optional[str]) -> str:
    source = (scanpath_source or "eyettention").lower()
    if source not in {"eyettention", "measured"}:
        raise ValueError(f"Unsupported scanpath_source: {scanpath_source}")
    return source


def _infer_dataset_loader(measured_scanpath_file: str) -> str:
    lowered = measured_scanpath_file.lower()
    if lowered.endswith(".json") or lowered.endswith(".jsonl"):
        return "json"
    if lowered.endswith(".csv"):
        return "csv"
    raise ValueError(
        "Unsupported measured scanpath file format. Use .json, .jsonl, or .csv: "
        f"{measured_scanpath_file}"
    )


def load_measured_scanpath_dataset(measured_scanpath_file: str):
    dataset_loader = _infer_dataset_loader(measured_scanpath_file)
    return load_dataset(dataset_loader, data_files={"train": measured_scanpath_file})


def load_measured_scanpath_train_eval_dataset(train_file: str, eval_file: str):
    train_loader = _infer_dataset_loader(train_file)
    eval_loader = _infer_dataset_loader(eval_file)
    if train_loader != eval_loader:
        raise ValueError(
            "Train and eval files must use the same dataset loader format. "
            f"Received train={train_loader!r}, eval={eval_loader!r}."
        )
    return load_dataset(train_loader, data_files={"train": train_file, "validation": eval_file})


def _resolve_dataset_field(dataset, requested_field: Optional[str], candidates: Sequence[str], label: str) -> str:
    column_names = set(dataset.column_names)
    if requested_field and requested_field != "auto":
        if requested_field not in column_names:
            raise ValueError(
                f"Could not find requested {label} field {requested_field!r}. "
                f"Available fields: {sorted(column_names)}"
            )
        return requested_field

    for candidate in candidates:
        if candidate in column_names:
            return candidate

    raise ValueError(
        f"Could not find any reasonable {label} field. Tried: {list(candidates)}. "
        f"Available fields: {sorted(column_names)}"
    )


def _parse_integer_position(value: Any, field_name: str, row_index: int) -> int:
    if value is None:
        raise ValueError(f"Missing sentence position field {field_name!r} at row {row_index}.")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Could not parse sentence position field {field_name!r}={value!r} at row {row_index}."
        ) from exc


def _detect_position_base(positions: Sequence[int]) -> str:
    if not positions:
        return "unknown"
    min_position = min(positions)
    if min_position == 0:
        return "0-based"
    if min_position == 1:
        return "1-based"
    return f"min_position={min_position}"


def _deterministic_truncate_indices(indices: List[int], max_samples: Optional[int], seed: int) -> Tuple[List[int], bool]:
    if max_samples is None or max_samples < 0 or max_samples >= len(indices):
        return list(indices), False
    rng = random.Random(seed)
    selected = list(indices)
    rng.shuffle(selected)
    return sorted(selected[:max_samples]), True


def _sample_split_rows(dataset, indices: Sequence[int], story_id_field: str, sentence_position_field: str, text_field: str):
    samples = []
    for row_index in list(indices)[:5]:
        row = dataset[int(row_index)]
        samples.append(
            {
                "row_index": int(row_index),
                "story_id": row[story_id_field],
                "sentence_position": _parse_integer_position(row[sentence_position_field], sentence_position_field, int(row_index)),
                "text": row[text_field],
            }
        )
    return samples


def _build_split_report(
    dataset,
    train_indices: Sequence[int],
    eval_indices: Sequence[int],
    pre_truncation_train_indices: Sequence[int],
    pre_truncation_eval_indices: Sequence[int],
    split_strategy: str,
    story_id_field: str,
    sentence_position_field: str,
    text_field: str,
    eval_sentence_position_mod: int,
    eval_sentence_position_remainder: int,
    seed: int,
    max_train_samples: Optional[int],
    max_eval_samples: Optional[int],
    train_truncated: bool,
    eval_truncated: bool,
) -> Dict[str, Any]:
    train_stories = set()
    eval_stories = set()
    train_texts = set()
    eval_texts = set()
    train_story_sentence = set()
    eval_story_sentence = set()
    train_positions = set()
    eval_positions = set()
    all_positions = []

    for row_index in train_indices:
        row = dataset[int(row_index)]
        story_id = row[story_id_field]
        sentence_position = _parse_integer_position(row[sentence_position_field], sentence_position_field, int(row_index))
        text = row[text_field]
        train_stories.add(story_id)
        train_texts.add(text)
        train_story_sentence.add((story_id, sentence_position))
        train_positions.add(sentence_position)
        all_positions.append(sentence_position)

    for row_index in eval_indices:
        row = dataset[int(row_index)]
        story_id = row[story_id_field]
        sentence_position = _parse_integer_position(row[sentence_position_field], sentence_position_field, int(row_index))
        text = row[text_field]
        eval_stories.add(story_id)
        eval_texts.add(text)
        eval_story_sentence.add((story_id, sentence_position))
        eval_positions.add(sentence_position)
        all_positions.append(sentence_position)

    exact_text_overlap = sorted(train_texts.intersection(eval_texts))
    story_sentence_overlap = sorted(train_story_sentence.intersection(eval_story_sentence))

    return {
        "split_strategy": split_strategy,
        "story_id_field": story_id_field,
        "sentence_position_field": sentence_position_field,
        "text_field": text_field,
        "sentence_position_index_base": _detect_position_base(all_positions),
        "eval_sentence_position_mod": int(eval_sentence_position_mod),
        "eval_sentence_position_remainder": int(eval_sentence_position_remainder),
        "seed": int(seed),
        "max_train_samples": max_train_samples,
        "max_eval_samples": max_eval_samples,
        "train_truncated_after_clean_split": train_truncated,
        "eval_truncated_after_clean_split": eval_truncated,
        "total_rows": len(dataset),
        "pre_truncation_train_rows": len(pre_truncation_train_indices),
        "pre_truncation_eval_rows": len(pre_truncation_eval_indices),
        "train_rows": len(train_indices),
        "eval_rows": len(eval_indices),
        "number_of_stories_train": len(train_stories),
        "number_of_stories_eval": len(eval_stories),
        "number_of_unique_text_train": len(train_texts),
        "number_of_unique_text_eval": len(eval_texts),
        "exact_text_overlap_count": len(exact_text_overlap),
        "story_sentence_overlap_count": len(story_sentence_overlap),
        "exact_text_overlap_examples": exact_text_overlap[:10],
        "story_sentence_overlap_examples": [
            {"story_id": story_id, "sentence_position": sentence_position}
            for story_id, sentence_position in story_sentence_overlap[:10]
        ],
        "eval_sentence_positions_used": sorted(eval_positions),
        "train_sentence_positions_used": sorted(train_positions),
        "train_examples": _sample_split_rows(dataset, train_indices, story_id_field, sentence_position_field, text_field),
        "eval_examples": _sample_split_rows(dataset, eval_indices, story_id_field, sentence_position_field, text_field),
    }


def print_split_report(report: Dict[str, Any]):
    keys_to_print = [
        "split_strategy",
        "train_file",
        "eval_file",
        "total_rows",
        "train_rows",
        "eval_rows",
        "number_of_stories_train",
        "number_of_stories_eval",
        "story_overlap_count",
        "number_of_unique_text_train",
        "number_of_unique_text_eval",
        "exact_text_overlap_count",
        "story_sentence_overlap_count",
        "eval_sentence_positions_used",
        "train_sentence_positions_used",
        "train_examples",
        "eval_examples",
    ]
    printable_report = {key: report[key] for key in keys_to_print if key in report}
    print(json.dumps(printable_report, ensure_ascii=False, indent=2), flush=True)


def save_split_report(report: Dict[str, Any], split_report_path: str):
    target_path = Path(split_report_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def validate_clean_split_report(report: Dict[str, Any]):
    if report["train_rows"] == 0 or report["eval_rows"] == 0:
        raise ValueError(
            "Clean sentence-position split produced an empty split: "
            f"train_rows={report['train_rows']}, eval_rows={report['eval_rows']}."
        )
    if report["exact_text_overlap_count"] > 0:
        raise ValueError(
            "Clean sentence-position split still has exact text overlap between train and eval: "
            f"{report['exact_text_overlap_count']} overlapping texts. "
            "See split_report.json for examples."
        )
    if report["story_sentence_overlap_count"] > 0:
        raise ValueError(
            "Clean sentence-position split has overlapping (story_id, sentence_position) pairs: "
            f"{report['story_sentence_overlap_count']} overlaps. "
            "See split_report.json for examples."
        )


def _sample_precomputed_split_rows(dataset, indices: Sequence[int], story_id_field: str, text_field: str):
    samples = []
    for row_index in list(indices)[:5]:
        row = dataset[int(row_index)]
        samples.append(
            {
                "row_index": int(row_index),
                "story_id": row[story_id_field],
                "split": row.get("split"),
                "source_file": row.get("source_file"),
                "trial_id": row.get("trial_id"),
                "n_words": row.get("n_words"),
                "text": row[text_field],
            }
        )
    return samples


def _collect_precomputed_split_info(dataset, indices: Sequence[int], story_id_field: str, text_field: str, expected_split: str):
    stories = set()
    texts = set()
    split_mismatches = []
    missing_text_rows = []
    for row_index in indices:
        row = dataset[int(row_index)]
        if text_field not in row or row[text_field] is None:
            missing_text_rows.append(int(row_index))
            continue
        stories.add(row[story_id_field])
        texts.add(row[text_field])
        if "split" in row and row["split"] is not None and str(row["split"]).lower() != expected_split:
            split_mismatches.append({"row_index": int(row_index), "split": row["split"]})
    return stories, texts, split_mismatches, missing_text_rows


def _build_precomputed_file_split_report(
    train_dataset,
    eval_dataset,
    train_indices: Sequence[int],
    eval_indices: Sequence[int],
    pre_truncation_train_indices: Sequence[int],
    pre_truncation_eval_indices: Sequence[int],
    train_file: str,
    eval_file: str,
    story_id_field: str,
    text_field: str,
    seed: int,
    max_train_samples: Optional[int],
    max_eval_samples: Optional[int],
    train_truncated: bool,
    eval_truncated: bool,
) -> Dict[str, Any]:
    train_stories, train_texts, train_split_mismatches, train_missing_text_rows = _collect_precomputed_split_info(
        train_dataset,
        train_indices,
        story_id_field,
        text_field,
        "train",
    )
    eval_stories, eval_texts, eval_split_mismatches, eval_missing_text_rows = _collect_precomputed_split_info(
        eval_dataset,
        eval_indices,
        story_id_field,
        text_field,
        "test",
    )
    story_overlap = sorted(train_stories.intersection(eval_stories))
    exact_text_overlap = sorted(train_texts.intersection(eval_texts))
    total_rows = len(train_indices) + len(eval_indices)
    return {
        "split_strategy": "precomputed_files",
        "train_file": str(train_file),
        "eval_file": str(eval_file),
        "story_id_field": story_id_field,
        "text_field": text_field,
        "seed": int(seed),
        "max_train_samples": max_train_samples,
        "max_eval_samples": max_eval_samples,
        "train_truncated_after_clean_split": train_truncated,
        "eval_truncated_after_clean_split": eval_truncated,
        "total_rows": total_rows,
        "pre_truncation_train_rows": len(pre_truncation_train_indices),
        "pre_truncation_eval_rows": len(pre_truncation_eval_indices),
        "train_rows": len(train_indices),
        "eval_rows": len(eval_indices),
        "train_percentage_rows": round((len(train_indices) / total_rows) * 100, 4) if total_rows else 0.0,
        "eval_percentage_rows": round((len(eval_indices) / total_rows) * 100, 4) if total_rows else 0.0,
        "number_of_stories_train": len(train_stories),
        "number_of_stories_eval": len(eval_stories),
        "train_stories": sorted(train_stories),
        "eval_stories": sorted(eval_stories),
        "story_overlap_count": len(story_overlap),
        "story_overlap_examples": story_overlap[:10],
        "number_of_unique_text_train": len(train_texts),
        "number_of_unique_text_eval": len(eval_texts),
        "exact_text_overlap_count": len(exact_text_overlap),
        "exact_text_overlap_examples": exact_text_overlap[:10],
        "split_label_mismatch_count": len(train_split_mismatches) + len(eval_split_mismatches),
        "split_label_mismatch_examples": (train_split_mismatches + eval_split_mismatches)[:10],
        "missing_text_count": len(train_missing_text_rows) + len(eval_missing_text_rows),
        "missing_text_examples": {
            "train": train_missing_text_rows[:10],
            "eval": eval_missing_text_rows[:10],
        },
        "train_examples": _sample_precomputed_split_rows(train_dataset, train_indices, story_id_field, text_field),
        "eval_examples": _sample_precomputed_split_rows(eval_dataset, eval_indices, story_id_field, text_field),
    }


def validate_precomputed_file_split_report(report: Dict[str, Any]):
    if report["train_rows"] == 0 or report["eval_rows"] == 0:
        raise ValueError(
            "Precomputed Train/Test split produced an empty split: "
            f"train_rows={report['train_rows']}, eval_rows={report['eval_rows']}."
        )
    if report["story_overlap_count"] > 0:
        raise ValueError(
            "Precomputed Train/Test split has cuento leakage: "
            f"{report['story_overlap_count']} stories appear in both splits. "
            "See split_report.json for examples."
        )
    if report["missing_text_count"] > 0:
        raise ValueError(
            "Precomputed Train/Test split has rows without text: "
            f"{report['missing_text_count']} rows. See split_report.json for examples."
        )
    if report["split_label_mismatch_count"] > 0:
        raise ValueError(
            "Precomputed Train/Test split has rows whose internal split label does not match the file: "
            f"{report['split_label_mismatch_count']} rows. See split_report.json for examples."
        )


def build_precomputed_file_train_eval_datasets(
    train_file: str,
    eval_file: str,
    max_train_samples: Optional[int],
    max_eval_samples: Optional[int],
    seed: int,
    text_field: str,
    split_report_path: str,
    story_id_field: Optional[str] = "auto",
):
    if not train_file or not eval_file:
        raise ValueError("split_strategy=precomputed_files requires both --train_file and --eval_file.")
    raw_datasets = load_measured_scanpath_train_eval_dataset(train_file, eval_file)
    train_dataset_full = raw_datasets["train"]
    eval_dataset_full = raw_datasets["validation"]
    if text_field not in train_dataset_full.column_names or text_field not in eval_dataset_full.column_names:
        raise ValueError(
            f"Could not find text field {text_field!r} in both precomputed files. "
            f"Train fields: {train_dataset_full.column_names}; eval fields: {eval_dataset_full.column_names}"
        )

    resolved_story_id_field = _resolve_dataset_field(
        train_dataset_full,
        story_id_field,
        DEFAULT_STORY_ID_FIELD_CANDIDATES,
        "story_id",
    )
    if resolved_story_id_field not in eval_dataset_full.column_names:
        raise ValueError(
            f"Could not find story field {resolved_story_id_field!r} in eval file. "
            f"Eval fields: {eval_dataset_full.column_names}"
        )

    train_indices = list(range(len(train_dataset_full)))
    eval_indices = list(range(len(eval_dataset_full)))
    pre_truncation_train_indices = list(train_indices)
    pre_truncation_eval_indices = list(eval_indices)
    train_indices, train_truncated = _deterministic_truncate_indices(train_indices, max_train_samples, seed)
    eval_indices, eval_truncated = _deterministic_truncate_indices(eval_indices, max_eval_samples, seed)

    report = _build_precomputed_file_split_report(
        train_dataset=train_dataset_full,
        eval_dataset=eval_dataset_full,
        train_indices=train_indices,
        eval_indices=eval_indices,
        pre_truncation_train_indices=pre_truncation_train_indices,
        pre_truncation_eval_indices=pre_truncation_eval_indices,
        train_file=train_file,
        eval_file=eval_file,
        story_id_field=resolved_story_id_field,
        text_field=text_field,
        seed=seed,
        max_train_samples=max_train_samples,
        max_eval_samples=max_eval_samples,
        train_truncated=train_truncated,
        eval_truncated=eval_truncated,
    )
    save_split_report(report, split_report_path)
    print_split_report(report)
    validate_precomputed_file_split_report(report)

    return train_dataset_full.select(train_indices), eval_dataset_full.select(eval_indices), report


def build_contiguous_train_eval_datasets(raw_dataset, max_train_samples: Optional[int], max_eval_samples: Optional[int]):
    total_available = len(raw_dataset)
    train_count = total_available if max_train_samples is None or max_train_samples < 0 else min(max_train_samples, total_available)
    remaining = max(0, total_available - train_count)
    eval_limit = remaining if max_eval_samples is None or max_eval_samples < 0 else max_eval_samples
    eval_count = min(eval_limit, remaining)

    train_dataset = raw_dataset.select(range(train_count))
    if eval_count > 0:
        eval_dataset = raw_dataset.select(range(train_count, train_count + eval_count))
    else:
        eval_dataset = None

    return train_dataset, eval_dataset, None


def build_sentence_position_train_eval_datasets(
    raw_dataset,
    max_train_samples: Optional[int],
    max_eval_samples: Optional[int],
    seed: int,
    text_field: str,
    eval_sentence_position_mod: int,
    eval_sentence_position_remainder: int,
    split_report_path: str,
    story_id_field: Optional[str] = "auto",
    sentence_position_field: Optional[str] = "auto",
):
    if eval_sentence_position_mod <= 0:
        raise ValueError("--eval_sentence_position_mod must be greater than 0.")
    if not 0 <= eval_sentence_position_remainder < eval_sentence_position_mod:
        raise ValueError(
            "--eval_sentence_position_remainder must satisfy "
            "0 <= remainder < eval_sentence_position_mod."
        )
    if text_field not in raw_dataset.column_names:
        raise ValueError(f"Could not find text field {text_field!r}. Available fields: {raw_dataset.column_names}")

    resolved_story_id_field = _resolve_dataset_field(
        raw_dataset,
        story_id_field,
        DEFAULT_STORY_ID_FIELD_CANDIDATES,
        "story_id",
    )
    resolved_sentence_position_field = _resolve_dataset_field(
        raw_dataset,
        sentence_position_field,
        DEFAULT_SENTENCE_POSITION_FIELD_CANDIDATES,
        "sentence_position",
    )

    train_indices = []
    eval_indices = []
    for row_index, row in enumerate(raw_dataset):
        sentence_position = _parse_integer_position(
            row[resolved_sentence_position_field],
            resolved_sentence_position_field,
            row_index,
        )
        if sentence_position % eval_sentence_position_mod == eval_sentence_position_remainder:
            eval_indices.append(row_index)
        else:
            train_indices.append(row_index)

    pre_truncation_train_indices = list(train_indices)
    pre_truncation_eval_indices = list(eval_indices)
    train_indices, train_truncated = _deterministic_truncate_indices(train_indices, max_train_samples, seed)
    eval_indices, eval_truncated = _deterministic_truncate_indices(eval_indices, max_eval_samples, seed)

    report = _build_split_report(
        raw_dataset,
        train_indices,
        eval_indices,
        pre_truncation_train_indices,
        pre_truncation_eval_indices,
        "sentence_position",
        resolved_story_id_field,
        resolved_sentence_position_field,
        text_field,
        eval_sentence_position_mod,
        eval_sentence_position_remainder,
        seed,
        max_train_samples,
        max_eval_samples,
        train_truncated,
        eval_truncated,
    )
    save_split_report(report, split_report_path)
    print_split_report(report)
    validate_clean_split_report(report)

    return raw_dataset.select(train_indices), raw_dataset.select(eval_indices), report


def build_mlm_train_eval_datasets(
    raw_dataset,
    split_strategy: str,
    max_train_samples: Optional[int],
    max_eval_samples: Optional[int],
    seed: int,
    text_field: str,
    eval_sentence_position_mod: int,
    eval_sentence_position_remainder: int,
    split_report_path: str,
    story_id_field: Optional[str] = "auto",
    sentence_position_field: Optional[str] = "auto",
):
    if split_strategy == "contiguous":
        return build_contiguous_train_eval_datasets(raw_dataset, max_train_samples, max_eval_samples)
    if split_strategy == "sentence_position":
        return build_sentence_position_train_eval_datasets(
            raw_dataset=raw_dataset,
            max_train_samples=max_train_samples,
            max_eval_samples=max_eval_samples,
            seed=seed,
            text_field=text_field,
            eval_sentence_position_mod=eval_sentence_position_mod,
            eval_sentence_position_remainder=eval_sentence_position_remainder,
            split_report_path=split_report_path,
            story_id_field=story_id_field,
            sentence_position_field=sentence_position_field,
        )
    raise ValueError(f"Unsupported split_strategy={split_strategy!r}. Use 'sentence_position' or 'contiguous'.")


def infer_custom_label_schema(raw_datasets, label_name: str):
    for split_name in ("train", "validation", "test"):
        if split_name not in raw_datasets:
            continue
        if label_name not in raw_datasets[split_name].column_names:
            continue

        label_values = [value for value in raw_datasets[split_name][label_name] if value is not None]
        if not label_values:
            return False, False, None

        first_value = label_values[0]
        if isinstance(first_value, float):
            return True, True, None

        return True, False, sorted(set(label_values))

    return False, False, None


def compute_inverse_word_length(words):
    word_lengths = [len(token) for token in words]
    arr = np.array(word_lengths, dtype=np.float64)
    arr[arr == 0] = 1 / (0 + 0.5)
    arr[arr != 0] = 1 / arr[arr != 0]
    return arr.tolist()


def split_text_to_words(text: str, remove_punctuation_space: bool):
    normalized_text = remove_punctuation_split(text) if remove_punctuation_space else text
    return [word for word in normalized_text.split() if word not in IGNORE_TOKEN_BY_TOKENIZER]


def _parse_word_id_sequence(word_id_value: Any):
    if word_id_value is None:
        return []

    if isinstance(word_id_value, str):
        stripped_value = word_id_value.strip()
        if not stripped_value:
            return []
        for parser in (ast.literal_eval,):
            try:
                word_id_value = parser(stripped_value)
                break
            except (SyntaxError, ValueError):
                continue
        else:
            raise ValueError(f"Could not parse measured word_id sequence: {word_id_value}")

    if isinstance(word_id_value, np.ndarray):
        word_id_value = word_id_value.tolist()
    elif isinstance(word_id_value, tuple):
        word_id_value = list(word_id_value)

    if not isinstance(word_id_value, list):
        raise TypeError(f"Measured word_id must be a list-like value, got: {type(word_id_value)}")

    parsed_positions = []
    for position in word_id_value:
        if position is None:
            continue
        parsed_positions.append(int(position))
    return parsed_positions


def parse_gaze_feature_names(gaze_feature_names: Optional[str]) -> List[str]:
    if gaze_feature_names is None:
        return DEFAULT_GAZE_FEATURE_NAMES.copy()
    if isinstance(gaze_feature_names, str):
        names = [name.strip() for name in gaze_feature_names.split(",") if name.strip()]
    else:
        names = list(gaze_feature_names)
    return names or DEFAULT_GAZE_FEATURE_NAMES.copy()


def _parse_nested_list(value: Any):
    if value is None:
        return None
    if isinstance(value, str):
        stripped_value = value.strip()
        if not stripped_value:
            return None
        value = ast.literal_eval(stripped_value)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return value


def _resolve_gaze_feature_mapping(
    feature_names: Sequence[str],
    requested_feature_names: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    feature_names = list(feature_names or [])
    mapping = {}
    for requested_name in requested_feature_names:
        aliases = GAZE_FEATURE_ALIASES.get(requested_name, [requested_name])
        for alias in aliases:
            if alias in feature_names:
                mapping[requested_name] = {
                    "source_name": alias,
                    "source_index": feature_names.index(alias),
                }
                break
        else:
            mapping[requested_name] = {
                "source_name": None,
                "source_index": None,
            }
    return mapping


def _sanitize_float(value: Any) -> Tuple[float, bool]:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return 0.0, True
    if not math.isfinite(numeric_value):
        return 0.0, True
    return numeric_value, False


def extract_measured_gaze_features(
    feature_names: Sequence[str],
    reading_features_by_fixation: Any,
    reading_features_mask_by_fixation: Any = None,
    requested_feature_names: Optional[Sequence[str]] = None,
    norm_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    requested_feature_names = list(requested_feature_names or DEFAULT_GAZE_FEATURE_NAMES)
    mapping = _resolve_gaze_feature_mapping(feature_names, requested_feature_names)
    rows = _parse_nested_list(reading_features_by_fixation) or []
    mask_rows = _parse_nested_list(reading_features_mask_by_fixation)

    features = []
    masks = []
    replaced_nonfinite = 0
    for row_index, row in enumerate(rows):
        feature_row = []
        mask_row = []
        row = row or []
        raw_mask_row = mask_rows[row_index] if mask_rows is not None and row_index < len(mask_rows) else None
        for requested_name in requested_feature_names:
            source_index = mapping[requested_name]["source_index"]
            value = 0.0
            valid = False
            if source_index is not None and source_index < len(row):
                value, replaced = _sanitize_float(row[source_index])
                replaced_nonfinite += int(replaced)
                if raw_mask_row is None:
                    valid = not replaced
                elif source_index < len(raw_mask_row):
                    valid = bool(raw_mask_row[source_index]) and not replaced
            feature_row.append(value)
            mask_row.append(1 if valid else 0)
        features.append(feature_row)
        masks.append(mask_row)

    if norm_stats is not None and features:
        means = np.array(norm_stats["mean"], dtype=np.float64)
        stds = np.array(norm_stats["std"], dtype=np.float64)
        arr = np.array(features, dtype=np.float64)
        mask_arr = np.array(masks, dtype=np.float64)
        arr = (arr - means) / stds
        arr = np.where(mask_arr > 0, arr, 0.0)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        features = arr.tolist()

    return {
        "features": features,
        "mask": masks,
        "mapping": mapping,
        "replaced_nonfinite": replaced_nonfinite,
    }


def measured_gaze_alignment_ok(example: Dict[str, Any]) -> bool:
    word_ids = _parse_word_id_sequence(example.get("word_id"))
    fixation_features = _parse_nested_list(example.get("reading_features_by_fixation")) or []
    return len(word_ids) == len(fixation_features)


def compute_gaze_feature_norm_stats(
    examples: Iterable[Dict[str, Any]],
    requested_feature_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    requested_feature_names = list(requested_feature_names or DEFAULT_GAZE_FEATURE_NAMES)
    values_by_feature = [[] for _ in requested_feature_names]
    feature_names_seen = set()
    mappings_seen = {}
    total_examples = 0
    total_values = 0
    replaced_nonfinite = 0

    for example in examples:
        total_examples += 1
        feature_names = example.get("feature_names") or []
        feature_names_seen.update(feature_names)
        extracted = extract_measured_gaze_features(
            feature_names=feature_names,
            reading_features_by_fixation=example.get("reading_features_by_fixation"),
            reading_features_mask_by_fixation=example.get("reading_features_mask_by_fixation"),
            requested_feature_names=requested_feature_names,
            norm_stats=None,
        )
        replaced_nonfinite += extracted["replaced_nonfinite"]
        mappings_seen[json.dumps(extracted["mapping"], sort_keys=True)] = extracted["mapping"]
        for row, mask_row in zip(extracted["features"], extracted["mask"]):
            for feature_index, (value, valid) in enumerate(zip(row, mask_row)):
                total_values += 1
                if valid:
                    values_by_feature[feature_index].append(value)

    means = []
    stds = []
    valid_counts = []
    for values in values_by_feature:
        valid_counts.append(len(values))
        if values:
            arr = np.array(values, dtype=np.float64)
            means.append(float(arr.mean()))
            std = float(arr.std())
            stds.append(std if std > 1e-8 else 1.0)
        else:
            means.append(0.0)
            stds.append(1.0)

    return {
        "feature_names": requested_feature_names,
        "mean": means,
        "std": stds,
        "valid_counts": valid_counts,
        "total_examples": total_examples,
        "total_values": total_values,
        "replaced_nonfinite": replaced_nonfinite,
        "feature_names_seen": sorted(feature_names_seen),
        "mappings_seen": list(mappings_seen.values()),
    }


def load_gaze_feature_norm_stats(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str, payload: Dict[str, Any]):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def build_measured_scanpath(word_id_value: Any, sentence_word_count: int):
    lexical_positions = _parse_word_id_sequence(word_id_value)
    lexical_positions = [position for position in lexical_positions if 1 <= position <= sentence_word_count]

    # The aligned dataset stores word_id over lexical words only.
    # Here we add the start and end sentinels expected by the original
    # scanpath encoder: 0 -> synthetic CLS/start, sn_len + 1 -> SEP/end.
    measured_word_ids = [0] + lexical_positions + [sentence_word_count + 1]
    measured_sp_len = len(measured_word_ids)
    return measured_word_ids, measured_sp_len


def build_measured_single_sentence_features(
    text: str,
    word_id_value: Any,
    tokenizer,
    max_seq_length: int,
    remove_punctuation_space: bool,
    feature_names: Optional[Sequence[str]] = None,
    reading_features_by_fixation: Any = None,
    reading_features_mask_by_fixation: Any = None,
    requested_gaze_feature_names: Optional[Sequence[str]] = None,
    gaze_feature_norm_stats: Optional[Dict[str, Any]] = None,
    use_gaze_features: bool = False,
):
    # In measured-scanpath mode the aligned dataset text becomes the exact LM input.
    split_words = split_text_to_words(text, remove_punctuation_space=remove_punctuation_space)
    inverse_word_length = compute_inverse_word_length(split_words)

    encoded = tokenizer(
        split_words,
        padding=False,
        max_length=max_seq_length,
        truncation=True,
        is_split_into_words=True,
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    token_type_ids = encoded.get("token_type_ids")
    if token_type_ids is None:
        token_type_ids = [0] * len(input_ids)

    lm_word_ids = encoded.word_ids()
    lm_word_ids = [value if value is not None else np.nan for value in lm_word_ids]
    nan_positions = np.where(np.isnan(lm_word_ids))[0]
    if nan_positions.size != 2:
        raise ValueError(
            "Measured scanpath mode currently supports single-sentence examples only. "
            f"Unexpected special-token pattern: {nan_positions.tolist()}"
        )

    lm_word_ids[nan_positions[0]] = -1
    lm_word_ids[nan_positions[1]] = lm_word_ids[nan_positions[1] - 1] + 1
    lm_word_ids = [value + 1 for value in lm_word_ids]
    sentence_word_count = int(max(lm_word_ids) - 1)

    measured_word_ids, measured_sp_len = build_measured_scanpath(
        word_id_value=word_id_value,
        sentence_word_count=sentence_word_count,
    )

    output = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "word_ids": lm_word_ids,
        "ET_input_ids": [input_ids],
        "ET_attention_mask": [attention_mask],
        "ET_token_type_ids": [token_type_ids],
        "ET_word_ids": [lm_word_ids],
        "ET_word_len": [[np.nan] + inverse_word_length[:sentence_word_count] + [np.nan]],
        "measured_word_ids": [measured_word_ids],
        "measured_sp_len": [measured_sp_len],
    }

    if use_gaze_features:
        original_lexical_positions = _parse_word_id_sequence(word_id_value)
        kept_fixation_indices = [
            fixation_index
            for fixation_index, position in enumerate(original_lexical_positions)
            if 1 <= position <= sentence_word_count
        ]
        extracted = extract_measured_gaze_features(
            feature_names=feature_names or [],
            reading_features_by_fixation=reading_features_by_fixation,
            reading_features_mask_by_fixation=reading_features_mask_by_fixation,
            requested_feature_names=requested_gaze_feature_names,
            norm_stats=gaze_feature_norm_stats,
        )
        output["measured_gaze_features"] = [[
            extracted["features"][fixation_index]
            for fixation_index in kept_fixation_indices
            if fixation_index < len(extracted["features"])
        ]]
        output["measured_gaze_feature_mask"] = [[
            extracted["mask"][fixation_index]
            for fixation_index in kept_fixation_indices
            if fixation_index < len(extracted["mask"])
        ]]

    return output
