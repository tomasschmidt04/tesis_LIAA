import ast
from typing import Any, Optional

import numpy as np
from datasets import load_dataset

from utils import remove_punctuation_split

CUSTOM_ALIGNED_SCANPATH_TASK = "custom_aligned_scanpath"

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

    return {
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
