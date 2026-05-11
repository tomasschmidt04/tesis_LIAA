from __future__ import annotations

import argparse
import csv
import json
import math
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SUPPORTED_MEASURE_SUFFIXES = {".csv", ".tsv", ".json", ".jsonl", ".pkl", ".pickle"}
SUPPORTED_ALIGNED_SUFFIXES = {".json", ".jsonl"}

STORY_ALIASES = ("story_id", "text_id", "item", "item_name", "story")
PARTICIPANT_ALIASES = ("participant_id", "subject_id", "sub_id", "subj", "subject")
WORD_ALIASES = ("word", "token")
WORD_INDEX_ALIASES = ("word_idx", "word_index", "global_word_idx", "global_word_id", "word_id")

DEFAULT_EXCLUDED_COLUMNS = {
    "trial_id",
    "subject_id",
    "sub_id",
    "subj",
    "subject",
    "participant_id",
    "text_id",
    "story_id",
    "story",
    "item",
    "item_name",
    "segment_index",
    "word",
    "token",
    "clean_word",
    "word_pos",
    "word_id",
    "word_ids",
    "word_idx",
    "word_index",
    "global_word_idx",
    "global_word_id",
    "screen",
    "line",
    "sentence_id",
    "sentence_idx",
    "sentence_pos",
    "screen_pos",
    "char_start",
    "char_end",
    "x",
    "trial_fix",
    "screen_fix",
    "fixation_index",
    "n_fixations",
    "excluded",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add all available word-level eye-tracking measures to aligned scanpath JSONL examples."
        )
    )
    parser.add_argument("--aligned_path", type=str, help="Aligned JSON/JSONL file or directory.")
    parser.add_argument("--measures_path", type=str, help="Measures file or directory.")
    parser.add_argument("--output_path", type=str, help="Output enriched JSONL path.")
    parser.add_argument(
        "--fill_missing",
        choices=("nan", "zero", "mean"),
        default="nan",
        help="How to fill missing metric values.",
    )
    parser.add_argument(
        "--include_columns",
        nargs="*",
        default=None,
        help="Use only these metric columns. Overrides automatic metric detection.",
    )
    parser.add_argument(
        "--exclude_columns",
        nargs="*",
        default=[],
        help="Additional columns to exclude from automatic metric detection.",
    )
    parser.add_argument(
        "--print_examples",
        type=int,
        default=3,
        help="Print this many enriched examples as a compact preview.",
    )
    parser.add_argument(
        "--run_synthetic_test",
        action="store_true",
        help="Run the built-in synthetic test and exit unless paths are also provided.",
    )
    return parser.parse_args()


def normalize_token(token: Any) -> str:
    text = unicodedata.normalize("NFKD", str(token)).lower()
    return "".join(ch for ch in text if ch.isalnum())


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "nan", "na", "none", "null"}:
        return True
    if isinstance(value, (list, dict, tuple, set)):
        return False
    try:
        return bool(value != value)
    except Exception:
        return False


def to_float(value: Any) -> float | None:
    if is_missing(value):
        return None
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def read_json_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        records = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL in {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected JSON object in {path}:{line_number}")
            records.append(record)
        return records

    if isinstance(payload, list):
        if not all(isinstance(row, dict) for row in payload):
            raise ValueError(f"Expected a list of objects in {path}")
        return payload
    if isinstance(payload, dict):
        for key in ("records", "rows", "data"):
            rows = payload.get(key)
            if isinstance(rows, list) and all(isinstance(row, dict) for row in rows):
                return rows
        return [payload]
    raise ValueError(f"Expected JSON object, JSON array, or JSONL in {path}")


def read_table_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        return read_json_records(path)
    if suffix in {".pkl", ".pickle"}:
        import pandas as pd

        payload = pd.read_pickle(path)
        if isinstance(payload, pd.DataFrame):
            return payload.to_dict(orient="records")
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        if isinstance(payload, dict):
            return [payload]
        raise ValueError(f"Unsupported pickle payload in {path}: {type(payload).__name__}")
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle, delimiter=delimiter))
    raise ValueError(f"Unsupported table suffix: {path}")


def iter_input_files(path: Path, suffixes: set[str]) -> Iterable[Path]:
    if path.is_file():
        if path.suffix.lower() in suffixes:
            yield path
        return

    for candidate in sorted(path.rglob("*")):
        if candidate.is_file() and candidate.suffix.lower() in suffixes:
            yield candidate


def first_present(row: dict[str, Any], aliases: Iterable[str]) -> Any:
    for alias in aliases:
        if alias in row and not is_missing(row[alias]):
            return row[alias]
    return None


def infer_story_participant_from_aligned(record: dict[str, Any], path: Path) -> tuple[str | None, str | None]:
    story_id = first_present(record, STORY_ALIASES)
    participant_id = first_present(record, PARTICIPANT_ALIASES)

    trial_id = record.get("trial_id")
    if isinstance(trial_id, str):
        parts = trial_id.split("::")
        if len(parts) >= 2:
            story_id = story_id or parts[0]
            participant_id = participant_id or parts[1]

    if story_id is None:
        story_id = path.parent.name if path.parent.name else None
    if participant_id is None:
        participant_id = path.stem if path.stem else None
    return str(story_id) if story_id is not None else None, str(participant_id) if participant_id is not None else None


def infer_story_participant_from_measures(row: dict[str, Any], path: Path, root: Path) -> tuple[str | None, str | None]:
    story_id = first_present(row, STORY_ALIASES)
    participant_id = first_present(row, PARTICIPANT_ALIASES)

    if path != root and path.parent != root:
        story_id = story_id or path.parent.name
    if path != root:
        participant_id = participant_id or path.stem

    return str(story_id) if story_id is not None else None, str(participant_id) if participant_id is not None else None


def looks_like_identifier(column: str) -> bool:
    lower = column.lower()
    if lower in DEFAULT_EXCLUDED_COLUMNS:
        return True
    if lower.endswith(("_id", "_ids", "_idx", "_index")):
        return True
    if lower.startswith(("id_", "idx_")):
        return True
    return False


def detect_metric_columns(
    rows: list[dict[str, Any]],
    include_columns: list[str] | None,
    exclude_columns: list[str],
) -> list[str]:
    if include_columns is not None and len(include_columns) > 0:
        available = {column for row in rows for column in row}
        missing = [column for column in include_columns if column not in available]
        if missing:
            raise ValueError(f"Requested include_columns not found in measures: {missing}")
        non_numeric = []
        for column in include_columns:
            values = [row.get(column) for row in rows if not is_missing(row.get(column))]
            if values and not all(to_float(value) is not None for value in values):
                non_numeric.append(column)
        if non_numeric:
            raise ValueError(f"Requested include_columns are not numeric: {non_numeric}")
        return include_columns

    excluded = {column.lower() for column in exclude_columns}
    columns = []
    seen = set()
    for row in rows:
        for column in row:
            if column not in seen:
                columns.append(column)
                seen.add(column)

    metrics = []
    for column in columns:
        lower = column.lower()
        if lower in excluded or looks_like_identifier(column):
            continue
        values = [row.get(column) for row in rows if not is_missing(row.get(column))]
        if not values:
            continue
        if all(to_float(value) is not None for value in values):
            metrics.append(column)
    return metrics


def metric_means(rows: list[dict[str, Any]], feature_names: list[str]) -> dict[str, float]:
    means = {}
    for feature in feature_names:
        values = [to_float(row.get(feature)) for row in rows]
        values = [value for value in values if value is not None]
        means[feature] = sum(values) / len(values) if values else float("nan")
    return means


def sort_measure_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index_column = next((column for column in WORD_INDEX_ALIASES if any(column in row for row in rows)), None)
    if index_column is None:
        return rows

    def key(row: dict[str, Any]) -> tuple[int, float]:
        value = to_float(row.get(index_column))
        return (1, 0.0) if value is None else (0, value)

    return sorted(rows, key=key)


def build_measure_groups(
    measures_path: Path,
) -> tuple[list[dict[str, Any]], dict[tuple[str | None, str | None], list[dict[str, Any]]]]:
    rows: list[dict[str, Any]] = []
    groups: dict[tuple[str | None, str | None], list[dict[str, Any]]] = defaultdict(list)

    for file_path in iter_input_files(measures_path, SUPPORTED_MEASURE_SUFFIXES):
        for raw_row in read_table_records(file_path):
            row = dict(raw_row)
            story_id, participant_id = infer_story_participant_from_measures(row, file_path, measures_path)
            row.setdefault("_story_id", story_id)
            row.setdefault("_participant_id", participant_id)
            row.setdefault("_source_path", str(file_path))
            rows.append(row)
            groups[(story_id, participant_id)].append(row)

    for key, group_rows in list(groups.items()):
        groups[key] = sort_measure_rows(group_rows)
    return rows, groups


def candidate_measure_groups(
    groups: dict[tuple[str | None, str | None], list[dict[str, Any]]],
    story_id: str | None,
    participant_id: str | None,
) -> list[tuple[str, list[dict[str, Any]]]]:
    candidates = [
        ((story_id, participant_id), "exact_story_participant"),
        ((story_id, None), "story_only"),
        ((None, participant_id), "participant_only"),
        ((None, None), "global"),
    ]
    found = []
    for key, label in candidates:
        rows = groups.get(key)
        if rows:
            found.append((label, rows))
    if not found and len(groups) == 1:
        only_key = next(iter(groups))
        found.append((f"only_available_group:{only_key}", groups[only_key]))
    return found


def row_word(row: dict[str, Any]) -> str | None:
    value = first_present(row, WORD_ALIASES)
    return str(value) if value is not None else None


def find_token_window(
    rows: list[dict[str, Any]],
    text_tokens: list[str],
    start: int = 0,
) -> int | None:
    normalized_text = [normalize_token(token) for token in text_tokens]
    normalized_rows = [normalize_token(row_word(row) or "") for row in rows]
    n = len(normalized_text)
    if n == 0:
        return 0

    for search_start in (start, 0):
        max_start = len(normalized_rows) - n
        if max_start < 0:
            return None
        for idx in range(search_start, max_start + 1):
            if normalized_rows[idx : idx + n] == normalized_text:
                return idx
    return None


def index_base(rows: list[dict[str, Any]], index_column: str) -> int:
    values = [to_float(row.get(index_column)) for row in rows]
    values = [value for value in values if value is not None]
    return 0 if values and min(values) == 0 else 1


def rows_from_global_start(
    rows: list[dict[str, Any]],
    text_tokens: list[str],
    global_word_start: Any,
) -> list[dict[str, Any] | None] | None:
    start = to_float(global_word_start)
    if start is None:
        return None
    index_column = next((column for column in WORD_INDEX_ALIASES if any(column in row for row in rows)), None)
    if index_column is None:
        return None

    base = index_base(rows, index_column)
    by_index = {}
    for row in rows:
        value = to_float(row.get(index_column))
        if value is not None:
            by_index[int(value) - base] = row
    return [by_index.get(int(start) + offset) for offset in range(len(text_tokens))]


def match_measure_rows(
    record: dict[str, Any],
    rows: list[dict[str, Any]],
    state: dict[int, int],
) -> tuple[list[dict[str, Any] | None], list[str]]:
    text_tokens = record.get("text_tokens") or str(record.get("text", "")).split()
    warnings = []

    matched = rows_from_global_start(rows, text_tokens, record.get("global_word_start"))
    if matched is not None:
        return matched, warnings

    if any(row_word(row) is not None for row in rows):
        group_id = id(rows)
        window_start = find_token_window(rows, text_tokens, state.get(group_id, 0))
        if window_start is not None:
            state[group_id] = window_start + len(text_tokens)
            return rows[window_start : window_start + len(text_tokens)], warnings
        warnings.append("could_not_find_text_tokens_window_in_measure_words")

    group_id = id(rows)
    cursor = state.get(group_id, 0)
    if cursor + len(text_tokens) <= len(rows):
        state[group_id] = cursor + len(text_tokens)
        warnings.append("using_sequential_measure_rows_without_token_window")
        return rows[cursor : cursor + len(text_tokens)], warnings

    warnings.append("not_enough_measure_rows_for_segment")
    return [None] * len(text_tokens), warnings


def fill_value(raw_value: Any, feature: str, fill_missing: str, means: dict[str, float]) -> tuple[float, int]:
    value = to_float(raw_value)
    if value is not None:
        return value, 1
    if fill_missing == "zero":
        return 0.0, 0
    if fill_missing == "mean":
        return means.get(feature, float("nan")), 0
    return float("nan"), 0


def build_word_features(
    matched_rows: list[dict[str, Any] | None],
    feature_names: list[str],
    fill_missing: str,
    means: dict[str, float],
) -> tuple[list[list[float]], list[list[int]]]:
    features = []
    masks = []
    for row in matched_rows:
        row_features = []
        row_mask = []
        for feature in feature_names:
            raw_value = None if row is None else row.get(feature)
            value, mask = fill_value(raw_value, feature, fill_missing, means)
            row_features.append(value)
            row_mask.append(mask)
        features.append(row_features)
        masks.append(row_mask)
    return features, masks


def compare_scanpath_tokens(record: dict[str, Any]) -> tuple[int, int, list[str]]:
    text_tokens = record.get("text_tokens") or str(record.get("text", "")).split()
    scanpath_tokens = record.get("scanpath_tokens") or str(record.get("scanpath_text", "")).split()
    word_ids = record.get("word_id") or record.get("word_ids") or []
    mismatches = 0
    warnings = []

    for idx, (scan_token, word_id) in enumerate(zip(scanpath_tokens, word_ids)):
        wid = int(word_id)
        text_token = text_tokens[wid - 1]
        if normalize_token(text_token) != normalize_token(scan_token):
            mismatches += 1
            if len(warnings) < 10:
                warnings.append(
                    f"token_mismatch[{idx}]: scanpath={scan_token!r} text={text_token!r} word_id={wid}"
                )
    return mismatches, len(word_ids), warnings


def validate_enriched(record: dict[str, Any]) -> None:
    text_tokens = record["text_tokens"]
    scanpath_tokens = record["scanpath_tokens"]
    word_ids = record["word_id"]
    feature_names = record["feature_names"]
    by_word = record["reading_features_by_word"]
    by_fixation = record["reading_features_by_fixation"]

    assert len(scanpath_tokens) == len(word_ids)
    assert len(by_word) == len(text_tokens)
    assert len(by_fixation) == len(word_ids)
    if by_word:
        assert len(feature_names) == len(by_word[0])
    assert all(1 <= int(wid) <= len(text_tokens) for wid in word_ids)


def enrich_record(
    record: dict[str, Any],
    source_path: Path,
    groups: dict[tuple[str | None, str | None], list[dict[str, Any]]],
    feature_names: list[str],
    fill_missing: str,
    means: dict[str, float],
    match_state: dict[int, int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    enriched = dict(record)
    enriched["text_tokens"] = enriched.get("text_tokens") or str(enriched.get("text", "")).split()
    enriched["scanpath_tokens"] = enriched.get("scanpath_tokens") or str(enriched.get("scanpath_text", "")).split()
    enriched["word_id"] = enriched.get("word_id") or enriched.get("word_ids") or []

    story_id, participant_id = infer_story_participant_from_aligned(enriched, source_path)
    warnings = []
    group_label = None
    candidate_rows = None
    for label, rows in candidate_measure_groups(groups, story_id, participant_id):
        group_label = label
        candidate_rows = rows
        break

    if candidate_rows is None:
        matched_rows = [None] * len(enriched["text_tokens"])
        warnings.append(f"no_measure_group_for story_id={story_id!r} participant_id={participant_id!r}")
    else:
        matched_rows, match_warnings = match_measure_rows(enriched, candidate_rows, match_state)
        warnings.extend(match_warnings)

    by_word, mask_by_word = build_word_features(matched_rows, feature_names, fill_missing, means)
    by_fixation = [by_word[int(word_id) - 1] for word_id in enriched["word_id"]]
    mask_by_fixation = [mask_by_word[int(word_id) - 1] for word_id in enriched["word_id"]]

    token_mismatches, token_comparisons, token_warnings = compare_scanpath_tokens(enriched)
    warnings.extend(token_warnings)

    enriched["feature_names"] = feature_names
    enriched["reading_features_by_word"] = by_word
    enriched["reading_features_by_fixation"] = by_fixation
    enriched["reading_features_mask_by_word"] = mask_by_word
    enriched["reading_features_mask_by_fixation"] = mask_by_fixation
    enriched["num_text_tokens"] = len(enriched["text_tokens"])
    enriched["num_fixations"] = len(enriched["word_id"])
    if warnings:
        enriched["reading_feature_warnings"] = warnings

    validate_enriched(enriched)

    matched_word_rows = sum(1 for row in matched_rows if row is not None)
    return enriched, {
        "measure_group_label": group_label,
        "matched_word_rows": matched_word_rows,
        "complete_measure_match": matched_word_rows == len(enriched["text_tokens"]),
        "has_any_measure_match": matched_word_rows > 0,
        "token_mismatches": token_mismatches,
        "token_comparisons": token_comparisons,
        "mask_by_word": mask_by_word,
        "features_by_word": by_word,
    }


def accumulate_metric_stats(
    metric_values: dict[str, list[float]],
    metric_missing: dict[str, int],
    feature_names: list[str],
    features_by_word: list[list[float]],
    mask_by_word: list[list[int]],
) -> None:
    for row_values, row_mask in zip(features_by_word, mask_by_word):
        for feature, value, mask in zip(feature_names, row_values, row_mask):
            if mask:
                metric_values[feature].append(value)
            else:
                metric_missing[feature] += 1


def finalize_stats(
    stats: dict[str, Any],
    feature_names: list[str],
    metric_values: dict[str, list[float]],
    metric_missing: dict[str, int],
    total_word_slots: int,
) -> dict[str, Any]:
    missing_rate = {}
    mean_std = {}
    for feature in feature_names:
        values = [value for value in metric_values[feature] if not math.isnan(value)]
        missing_rate[feature] = metric_missing[feature] / total_word_slots if total_word_slots else 0.0
        if values:
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            mean_std[feature] = {"mean": mean, "std": math.sqrt(variance)}
        else:
            mean_std[feature] = {"mean": float("nan"), "std": float("nan")}

    token_comparisons = stats.pop("_token_comparisons")
    token_mismatches = stats.pop("_token_mismatches")
    stats["metrics_detected"] = feature_names
    stats["missing_rate"] = missing_rate
    stats["mean_std"] = mean_std
    stats["mismatch_rate"] = token_mismatches / token_comparisons if token_comparisons else 0.0
    return stats


def enrich_file(
    aligned_path: Path,
    measures_path: Path,
    output_path: Path,
    fill_missing: str,
    include_columns: list[str] | None,
    exclude_columns: list[str],
    print_examples: int,
) -> dict[str, Any]:
    measure_rows, groups = build_measure_groups(measures_path)
    if not measure_rows:
        raise ValueError(f"No measure rows found in {measures_path}")

    feature_names = detect_metric_columns(measure_rows, include_columns, exclude_columns)
    means = metric_means(measure_rows, feature_names)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    match_state: dict[int, int] = {}
    metric_values: dict[str, list[float]] = defaultdict(list)
    metric_missing: dict[str, int] = defaultdict(int)
    total_word_slots = 0
    previews = []

    stats: dict[str, Any] = {
        "examples_processed": 0,
        "examples_with_measure_match": 0,
        "examples_without_measure_match": 0,
        "examples_with_complete_measure_match": 0,
        "_token_mismatches": 0,
        "_token_comparisons": 0,
        "aligned_path": str(aligned_path),
        "measures_path": str(measures_path),
        "output_path": str(output_path),
        "fill_missing": fill_missing,
    }

    with output_path.open("w", encoding="utf-8") as out_handle:
        for aligned_file in iter_input_files(aligned_path, SUPPORTED_ALIGNED_SUFFIXES):
            for record in read_json_records(aligned_file):
                enriched, record_stats = enrich_record(
                    record=record,
                    source_path=aligned_file,
                    groups=groups,
                    feature_names=feature_names,
                    fill_missing=fill_missing,
                    means=means,
                    match_state=match_state,
                )
                out_handle.write(json.dumps(enriched, ensure_ascii=False))
                out_handle.write("\n")

                stats["examples_processed"] += 1
                stats["examples_with_measure_match"] += int(record_stats["has_any_measure_match"])
                stats["examples_without_measure_match"] += int(not record_stats["has_any_measure_match"])
                stats["examples_with_complete_measure_match"] += int(record_stats["complete_measure_match"])
                stats["_token_mismatches"] += record_stats["token_mismatches"]
                stats["_token_comparisons"] += record_stats["token_comparisons"]
                total_word_slots += len(enriched["text_tokens"])
                accumulate_metric_stats(
                    metric_values,
                    metric_missing,
                    feature_names,
                    record_stats["features_by_word"],
                    record_stats["mask_by_word"],
                )
                if len(previews) < print_examples:
                    previews.append(enriched)

    stats = finalize_stats(stats, feature_names, metric_values, metric_missing, total_word_slots)
    stats_path = Path(str(output_path) + ".stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    if print_examples:
        for idx, preview in enumerate(previews, start=1):
            compact = {
                "trial_id": preview.get("trial_id"),
                "segment_index": preview.get("segment_index"),
                "feature_names": preview.get("feature_names"),
                "num_text_tokens": preview.get("num_text_tokens"),
                "num_fixations": preview.get("num_fixations"),
                "first_word_features": preview.get("reading_features_by_word", [])[:1],
            }
            print(f"PREVIEW {idx}: {json.dumps(compact, ensure_ascii=False)}")

    print(f"Wrote enriched JSONL: {output_path}")
    print(f"Wrote stats JSON: {stats_path}")
    return stats


def run_synthetic_test() -> None:
    aligned = {
        "text": "El perro come",
        "text_tokens": ["El", "perro", "come"],
        "scanpath_tokens": ["perro", "perro", "El", "come"],
        "word_id": [2, 2, 1, 3],
        "trial_id": "synthetic::sub-001::seg_0000",
        "segment_index": 0,
        "match_quality": "high",
        "coverage": 1.0,
    }
    measure_rows = [
        {"story_id": "synthetic", "participant_id": "sub-001", "word_idx": 0, "word": "El", "FFD": 10, "TFD": 20, "FC": 1, "RC": 0},
        {"story_id": "synthetic", "participant_id": "sub-001", "word_idx": 1, "word": "perro", "FFD": 30, "TFD": 40, "FC": 2, "RC": 1},
        {"story_id": "synthetic", "participant_id": "sub-001", "word_idx": 2, "word": "come", "FFD": 50, "TFD": 60, "FC": 1, "RC": 0},
    ]
    groups = {("synthetic", "sub-001"): measure_rows}
    feature_names = detect_metric_columns(measure_rows, include_columns=None, exclude_columns=[])
    means = metric_means(measure_rows, feature_names)
    enriched, _ = enrich_record(
        record=aligned,
        source_path=Path("synthetic.jsonl"),
        groups=groups,
        feature_names=feature_names,
        fill_missing="nan",
        means=means,
        match_state={},
    )

    expected_by_word = [
        [10.0, 20.0, 1.0, 0.0],
        [30.0, 40.0, 2.0, 1.0],
        [50.0, 60.0, 1.0, 0.0],
    ]
    expected_by_fixation = [
        [30.0, 40.0, 2.0, 1.0],
        [30.0, 40.0, 2.0, 1.0],
        [10.0, 20.0, 1.0, 0.0],
        [50.0, 60.0, 1.0, 0.0],
    ]
    assert enriched["feature_names"] == ["FFD", "TFD", "FC", "RC"]
    assert enriched["reading_features_by_word"] == expected_by_word
    assert enriched["reading_features_by_fixation"] == expected_by_fixation
    assert enriched["text"] == aligned["text"]
    assert enriched["text_tokens"] == aligned["text_tokens"]
    assert enriched["scanpath_tokens"] == aligned["scanpath_tokens"]
    assert enriched["word_id"] == aligned["word_id"]
    assert enriched["trial_id"] == aligned["trial_id"]
    print("Synthetic test passed.")


def main() -> None:
    args = parse_args()
    if args.run_synthetic_test:
        run_synthetic_test()
        if not (args.aligned_path and args.measures_path and args.output_path):
            return

    missing_args = [
        name
        for name in ("aligned_path", "measures_path", "output_path")
        if getattr(args, name) is None
    ]
    if missing_args:
        raise SystemExit(f"Missing required arguments unless --run_synthetic_test is used: {missing_args}")

    enrich_file(
        aligned_path=Path(args.aligned_path),
        measures_path=Path(args.measures_path),
        output_path=Path(args.output_path),
        fill_missing=args.fill_missing,
        include_columns=args.include_columns,
        exclude_columns=args.exclude_columns,
        print_examples=args.print_examples,
    )


if __name__ == "__main__":
    main()
