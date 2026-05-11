#!/usr/bin/env python3
"""Simple token/input analysis for aligned scanpath files.

Reads .json/.jsonl files containing either JSON Lines or a JSON array and writes:
- per_example_token_stats.csv
- summary_stats.csv
- dataset_summary.json
- diversity_summary.json
- vocab_summary.json
- text_word_token_frequencies.csv
- scanpath_word_token_frequencies.csv
- text_lm_token_frequencies.csv
- scanpath_lm_token_frequencies.csv
- plots/*.png
"""

from __future__ import annotations

import argparse
import json
import math
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import AutoTokenizer


BASIC_EDGE_PUNCTUATION = string.punctuation + "¡¿“”‘’«»…"
SUBJECT_RE = re.compile(r"(sub-[A-Za-z0-9]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze input and token distributions in aligned scanpath JSON/JSONL files."
    )
    parser.add_argument(
        "--aligned_dir",
        required=True,
        type=Path,
        help="Directory containing aligned .json/.jsonl files, e.g. results_all_alligned.",
    )
    parser.add_argument(
        "--output_dir",
        default=Path("analysis/simple_input_token_analysis"),
        type=Path,
        help="Directory where CSV, JSON, and plot outputs will be written.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="dccuchile/bert-base-spanish-wwm-cased",
        help="Tokenizer model name or local path.",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="Maximum sequence length used to flag examples that would be truncated.",
    )
    return parser.parse_args()


def warn(message: str) -> None:
    print(f"WARNING: {message}")


def iter_aligned_files(aligned_dir: Path) -> List[Path]:
    files = sorted(
        path
        for path in aligned_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}
    )
    return files


def load_json_array_or_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield examples from a JSON array file or a JSON Lines file.

    The function first tries to parse the whole file as JSON. If that fails, it
    falls back to line-by-line parsing and skips malformed lines with a warning.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raw = path.read_text(encoding="utf-8-sig")

    stripped = raw.strip()
    if not stripped:
        return

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, list):
        for idx, item in enumerate(parsed, start=1):
            if isinstance(item, dict):
                yield item
            else:
                warn(f"{path}: array item {idx} is not an object; skipping.")
        return

    if isinstance(parsed, dict):
        yield parsed
        return

    for line_number, line in enumerate(raw.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            warn(f"{path}:{line_number}: malformed JSON line skipped ({exc}).")
            continue
        if not isinstance(item, dict):
            warn(f"{path}:{line_number}: JSON value is not an object; skipping.")
            continue
        yield item


def infer_subject_id(path: Path) -> str:
    match = SUBJECT_RE.search(path.stem)
    if match:
        return match.group(1)
    return ""


def as_token_list(value: Any, fallback_text: Any = "") -> List[str]:
    if isinstance(value, list):
        return [str(token) for token in value]
    if isinstance(value, str):
        return value.split()
    if isinstance(fallback_text, str):
        return fallback_text.split()
    return []


def normalize_token(token: str) -> str:
    return token.lower().strip(BASIC_EDGE_PUNCTUATION)


def add_normalized_tokens(tokens: Iterable[str], vocab: Set[str]) -> int:
    total = 0
    for token in tokens:
        normalized = normalize_token(str(token))
        if not normalized:
            continue
        vocab.add(normalized)
        total += 1
    return total


def normalized_tokens(tokens: Iterable[str]) -> List[str]:
    clean_tokens = []
    for token in tokens:
        normalized = normalize_token(str(token))
        if normalized:
            clean_tokens.append(normalized)
    return clean_tokens


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return numerator / denominator


def lm_token_count(tokenizer: Any, text: str, add_special_tokens: bool) -> int:
    return len(
        tokenizer.encode(
            text or "",
            add_special_tokens=add_special_tokens,
            truncation=False,
        )
    )


def lm_tokens(tokenizer: Any, text: str) -> List[str]:
    return tokenizer.tokenize(text or "")


def frequency_counter_to_df(counter: Counter[str]) -> pd.DataFrame:
    rows = []
    total = sum(counter.values())
    for rank, (token, count) in enumerate(counter.most_common(), start=1):
        rows.append(
            {
                "rank": rank,
                "token": token,
                "count": count,
                "relative_frequency": safe_ratio(count, total),
            }
        )
    return pd.DataFrame(rows, columns=["rank", "token", "count", "relative_frequency"])


def write_frequency_csv(counter: Counter[str], output_path: Path) -> None:
    frequency_counter_to_df(counter).to_csv(output_path, index=False)


def compute_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    numeric_df = df.select_dtypes(include=[np.number, "bool"]).copy()
    rows: List[Dict[str, Any]] = []
    for column in numeric_df.columns:
        values = pd.to_numeric(numeric_df[column], errors="coerce").dropna().astype(float)
        if values.empty:
            rows.append({"column": column, "count": 0})
            continue
        rows.append(
            {
                "column": column,
                "count": int(values.count()),
                "mean": values.mean(),
                "std": values.std(ddof=1),
                "min": values.min(),
                "p25": values.quantile(0.25),
                "median": values.quantile(0.50),
                "p75": values.quantile(0.75),
                "p90": values.quantile(0.90),
                "p95": values.quantile(0.95),
                "max": values.max(),
            }
        )
    return pd.DataFrame(rows)


def plot_hist(series: pd.Series, title: str, xlabel: str, output_path: Path) -> None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    plt.figure(figsize=(8, 5))
    if values.empty:
        plt.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        bins = min(50, max(10, int(math.sqrt(len(values)))))
        plt.hist(values, bins=bins, color="#3b6ea8", edgecolor="white", alpha=0.9)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Examples")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_top_token_counts(
    counter: Counter[str],
    title: str,
    output_path: Path,
    top_n: int = 50,
) -> None:
    top_items = counter.most_common(top_n)
    plt.figure(figsize=(12, 7))
    if not top_items:
        plt.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        tokens = [token for token, _ in reversed(top_items)]
        counts = [count for _, count in reversed(top_items)]
        plt.barh(tokens, counts, color="#3b6ea8", alpha=0.9)
    plt.title(title)
    plt.xlabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_boxplot(df: pd.DataFrame, output_path: Path) -> None:
    values = df[["text_lm_token_count", "scanpath_lm_token_count"]].apply(
        pd.to_numeric, errors="coerce"
    )
    plt.figure(figsize=(7, 5))
    if values.dropna(how="all").empty:
        plt.text(0.5, 0.5, "No data", ha="center", va="center")
    else:
        plt.boxplot(
            [values["text_lm_token_count"].dropna(), values["scanpath_lm_token_count"].dropna()],
            showfliers=False,
        )
        plt.xticks([1, 2], ["Text", "Scanpath"])
    plt.title("LM Token Counts: Text vs Scanpath")
    plt.ylabel("LM tokens")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def write_plots(df: pd.DataFrame, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_hist(
        df["text_lm_token_count"],
        "Text LM Token Count",
        "LM tokens without special tokens",
        plots_dir / "hist_text_lm_token_count.png",
    )
    plot_hist(
        df["scanpath_lm_token_count"],
        "Scanpath LM Token Count",
        "LM tokens without special tokens",
        plots_dir / "hist_scanpath_lm_token_count.png",
    )
    plot_hist(
        df["scanpath_vs_text_lm_token_ratio"],
        "Scanpath/Text LM Token Ratio",
        "scanpath_lm_token_count / text_lm_token_count",
        plots_dir / "hist_text_vs_scanpath_lm_token_ratio.png",
    )
    plot_boxplot(df, plots_dir / "boxplot_text_vs_scanpath_lm_tokens.png")


def write_frequency_plots(
    text_word_counter: Counter[str],
    scanpath_word_counter: Counter[str],
    text_lm_counter: Counter[str],
    scanpath_lm_counter: Counter[str],
    plots_dir: Path,
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_top_token_counts(
        text_word_counter,
        "Top 50 Text Word Tokens",
        plots_dir / "top50_text_word_token_frequencies.png",
    )
    plot_top_token_counts(
        scanpath_word_counter,
        "Top 50 Scanpath Word Tokens",
        plots_dir / "top50_scanpath_word_token_frequencies.png",
    )
    plot_top_token_counts(
        text_lm_counter,
        "Top 50 Text BETO/LM Tokens",
        plots_dir / "top50_text_lm_token_frequencies.png",
    )
    plot_top_token_counts(
        scanpath_lm_counter,
        "Top 50 Scanpath BETO/LM Tokens",
        plots_dir / "top50_scanpath_lm_token_frequencies.png",
    )


def count_types_with_frequency_at_most(counter: Counter[str], max_frequency: int) -> int:
    return sum(1 for count in counter.values() if count <= max_frequency)


def tokenizer_vocab_summary(tokenizer: Any) -> Dict[str, Any]:
    vocab = tokenizer.get_vocab()
    special_tokens = set(tokenizer.all_special_tokens)
    non_special_tokens = [token for token in vocab if token not in special_tokens]
    continuation_subwords = [
        token for token in non_special_tokens if token.startswith("##")
    ]
    word_start_tokens = [
        token for token in non_special_tokens if not token.startswith("##")
    ]
    return {
        "beto_vocab_size": len(vocab),
        "beto_special_token_count": len(special_tokens),
        "beto_non_special_token_count": len(non_special_tokens),
        "beto_word_start_token_count": len(word_start_tokens),
        "beto_continuation_subword_token_count": len(continuation_subwords),
    }


def pct_true(series: pd.Series) -> float:
    if len(series) == 0:
        return float("nan")
    return 100.0 * pd.to_numeric(series, errors="coerce").fillna(0).mean()


def print_console_summary(
    df: pd.DataFrame,
    total_files: int,
    total_subjects: int,
    diversity: Dict[str, Any],
) -> None:
    text_counts = pd.to_numeric(df["text_lm_token_count"], errors="coerce")
    scan_counts = pd.to_numeric(df["scanpath_lm_token_count"], errors="coerce")
    ratios = pd.to_numeric(df["scanpath_vs_text_lm_token_ratio"], errors="coerce")

    print("\nSimple input/token analysis summary")
    print(f"total files: {total_files}")
    print(f"total examples: {len(df)}")
    print(f"total inputs/oraciones: {len(df)}")
    print(f"total subjects: {total_subjects}")
    print(f"promedio tokens LM texto directo: {text_counts.mean():.3f}")
    print(f"mediana tokens LM texto directo: {text_counts.median():.3f}")
    print(f"p95 tokens LM texto directo: {text_counts.quantile(0.95):.3f}")
    print(f"porcentaje truncado texto directo: {pct_true(df['text_was_truncated']):.2f}%")
    print(f"promedio tokens LM scanpath: {scan_counts.mean():.3f}")
    print(f"mediana tokens LM scanpath: {scan_counts.median():.3f}")
    print(f"p95 tokens LM scanpath: {scan_counts.quantile(0.95):.3f}")
    print(f"porcentaje truncado scanpath: {pct_true(df['scanpath_was_truncated']):.2f}%")
    print(f"ratio promedio scanpath/texto: {ratios.mean():.3f}")
    print(f"unique tokens texto: {diversity['unique_text_word_tokens']}")
    print(f"unique tokens scanpath: {diversity['unique_scanpath_tokens']}")
    print(f"vocabulario BETO total: {diversity['beto_vocab_size']}")
    print(
        "tokens BETO observados en texto directo: "
        f"{diversity['observed_text_lm_token_types']} "
        f"({100 * diversity['observed_text_lm_token_ratio_over_beto_vocab']:.2f}% del vocab)"
    )
    print(
        "tokens BETO observados en scanpath: "
        f"{diversity['observed_scanpath_lm_token_types']} "
        f"({100 * diversity['observed_scanpath_lm_token_ratio_over_beto_vocab']:.2f}% del vocab)"
    )
    print(f"type-token ratio texto: {diversity['text_type_token_ratio']:.6f}")
    print(f"type-token ratio scanpath: {diversity['scanpath_type_token_ratio']:.6f}")
    print(
        "overlap vocabulario texto/scanpath: "
        f"{diversity['common_tokens_text_and_scanpath']} common, "
        f"over_text={diversity['vocab_overlap_ratio_over_text']:.6f}, "
        f"over_scanpath={diversity['vocab_overlap_ratio_over_scanpath']:.6f}"
    )


def main() -> None:
    args = parse_args()
    aligned_dir = args.aligned_dir
    output_dir = args.output_dir

    if not aligned_dir.exists() or not aligned_dir.is_dir():
        raise SystemExit(f"aligned_dir does not exist or is not a directory: {aligned_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    files = iter_aligned_files(aligned_dir)
    if not files:
        warn(f"No .json/.jsonl files found in {aligned_dir}")

    print(f"Loading tokenizer: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    rows: List[Dict[str, Any]] = []
    subject_ids: Set[str] = set()
    text_vocab: Set[str] = set()
    scanpath_vocab: Set[str] = set()
    text_word_counter: Counter[str] = Counter()
    scanpath_word_counter: Counter[str] = Counter()
    text_lm_counter: Counter[str] = Counter()
    scanpath_lm_counter: Counter[str] = Counter()
    total_text_word_tokens = 0
    total_scanpath_tokens = 0

    for path in files:
        subject_id = infer_subject_id(path)
        if subject_id:
            subject_ids.add(subject_id)

        for example in load_json_array_or_jsonl(path):
            text = str(example.get("text") or "")
            text_tokens = as_token_list(example.get("text_tokens"), text)
            scanpath_tokens = as_token_list(
                example.get("scanpath_tokens"), example.get("scanpath_text", "")
            )
            scanpath_text_for_lm = " ".join(scanpath_tokens)

            text_word_count = len(text_tokens)
            scanpath_token_count = len(scanpath_tokens)
            scanpath_unique_token_count = len(set(scanpath_tokens))
            scanpath_repetition_ratio = 1.0 - safe_ratio(
                scanpath_unique_token_count, scanpath_token_count
            )

            text_lm_token_count = lm_token_count(tokenizer, text, add_special_tokens=False)
            text_lm_token_count_with_special_tokens = lm_token_count(
                tokenizer, text, add_special_tokens=True
            )
            text_lm_token_list = lm_tokens(tokenizer, text)
            scanpath_lm_token_count = lm_token_count(
                tokenizer, scanpath_text_for_lm, add_special_tokens=False
            )
            scanpath_lm_token_count_with_special_tokens = lm_token_count(
                tokenizer, scanpath_text_for_lm, add_special_tokens=True
            )
            scanpath_lm_token_list = lm_tokens(tokenizer, scanpath_text_for_lm)

            total_text_word_tokens += add_normalized_tokens(text_tokens, text_vocab)
            total_scanpath_tokens += add_normalized_tokens(scanpath_tokens, scanpath_vocab)
            text_word_counter.update(normalized_tokens(text_tokens))
            scanpath_word_counter.update(normalized_tokens(scanpath_tokens))
            text_lm_counter.update(text_lm_token_list)
            scanpath_lm_counter.update(scanpath_lm_token_list)

            rows.append(
                {
                    "source_file": path.name,
                    "subject_id": subject_id,
                    "trial_id": example.get("trial_id", ""),
                    "segment_index": example.get("segment_index", ""),
                    "text_word_count": text_word_count,
                    "text_lm_token_count": text_lm_token_count,
                    "text_lm_token_count_with_special_tokens": text_lm_token_count_with_special_tokens,
                    "text_was_truncated": text_lm_token_count_with_special_tokens > args.max_seq_length,
                    "scanpath_token_count": scanpath_token_count,
                    "scanpath_unique_token_count": scanpath_unique_token_count,
                    "scanpath_repetition_ratio": scanpath_repetition_ratio,
                    "scanpath_lm_token_count": scanpath_lm_token_count,
                    "scanpath_lm_token_count_with_special_tokens": scanpath_lm_token_count_with_special_tokens,
                    "scanpath_was_truncated": scanpath_lm_token_count_with_special_tokens > args.max_seq_length,
                    "scanpath_vs_text_word_ratio": safe_ratio(
                        scanpath_token_count, text_word_count
                    ),
                    "scanpath_vs_text_lm_token_ratio": safe_ratio(
                        scanpath_lm_token_count, text_lm_token_count
                    ),
                }
            )

    df = pd.DataFrame(rows)
    expected_columns = [
        "source_file",
        "subject_id",
        "trial_id",
        "segment_index",
        "text_word_count",
        "text_lm_token_count",
        "text_lm_token_count_with_special_tokens",
        "text_was_truncated",
        "scanpath_token_count",
        "scanpath_unique_token_count",
        "scanpath_repetition_ratio",
        "scanpath_lm_token_count",
        "scanpath_lm_token_count_with_special_tokens",
        "scanpath_was_truncated",
        "scanpath_vs_text_word_ratio",
        "scanpath_vs_text_lm_token_ratio",
    ]
    df = df.reindex(columns=expected_columns)
    total_examples = len(df)
    total_subjects = len(subject_ids)

    df.to_csv(output_dir / "per_example_token_stats.csv", index=False)
    compute_summary_stats(df).to_csv(output_dir / "summary_stats.csv", index=False)
    write_frequency_csv(text_word_counter, output_dir / "text_word_token_frequencies.csv")
    write_frequency_csv(
        scanpath_word_counter, output_dir / "scanpath_word_token_frequencies.csv"
    )
    write_frequency_csv(text_lm_counter, output_dir / "text_lm_token_frequencies.csv")
    write_frequency_csv(
        scanpath_lm_counter, output_dir / "scanpath_lm_token_frequencies.csv"
    )

    common_tokens = text_vocab & scanpath_vocab
    tokenizer_vocab = tokenizer_vocab_summary(tokenizer)
    observed_any_lm_tokens = set(text_lm_counter) | set(scanpath_lm_counter)
    dataset_summary = {
        "total_files": len(files),
        "total_examples": total_examples,
        "total_inputs": total_examples,
        "total_sentences": total_examples,
        "total_subjects": total_subjects,
    }
    diversity = {
        **dataset_summary,
        **tokenizer_vocab,
        "total_text_word_tokens": total_text_word_tokens,
        "unique_text_word_tokens": len(text_vocab),
        "text_type_token_ratio": safe_ratio(len(text_vocab), total_text_word_tokens),
        "text_word_token_types_appearing_once": count_types_with_frequency_at_most(
            text_word_counter, 1
        ),
        "text_word_token_types_appearing_5_or_fewer": count_types_with_frequency_at_most(
            text_word_counter, 5
        ),
        "total_scanpath_tokens": total_scanpath_tokens,
        "unique_scanpath_tokens": len(scanpath_vocab),
        "scanpath_type_token_ratio": safe_ratio(len(scanpath_vocab), total_scanpath_tokens),
        "scanpath_word_token_types_appearing_once": count_types_with_frequency_at_most(
            scanpath_word_counter, 1
        ),
        "scanpath_word_token_types_appearing_5_or_fewer": count_types_with_frequency_at_most(
            scanpath_word_counter, 5
        ),
        "common_tokens_text_and_scanpath": len(common_tokens),
        "text_vocab_size": len(text_vocab),
        "scanpath_vocab_size": len(scanpath_vocab),
        "vocab_overlap_ratio_over_text": safe_ratio(len(common_tokens), len(text_vocab)),
        "vocab_overlap_ratio_over_scanpath": safe_ratio(len(common_tokens), len(scanpath_vocab)),
        "observed_text_lm_token_total": sum(text_lm_counter.values()),
        "observed_text_lm_token_types": len(text_lm_counter),
        "observed_text_lm_token_ratio_over_beto_vocab": safe_ratio(
            len(text_lm_counter), tokenizer_vocab["beto_vocab_size"]
        ),
        "text_lm_token_types_appearing_once": count_types_with_frequency_at_most(
            text_lm_counter, 1
        ),
        "text_lm_token_types_appearing_5_or_fewer": count_types_with_frequency_at_most(
            text_lm_counter, 5
        ),
        "observed_scanpath_lm_token_total": sum(scanpath_lm_counter.values()),
        "observed_scanpath_lm_token_types": len(scanpath_lm_counter),
        "observed_scanpath_lm_token_ratio_over_beto_vocab": safe_ratio(
            len(scanpath_lm_counter), tokenizer_vocab["beto_vocab_size"]
        ),
        "scanpath_lm_token_types_appearing_once": count_types_with_frequency_at_most(
            scanpath_lm_counter, 1
        ),
        "scanpath_lm_token_types_appearing_5_or_fewer": count_types_with_frequency_at_most(
            scanpath_lm_counter, 5
        ),
        "observed_any_lm_token_types": len(observed_any_lm_tokens),
        "observed_any_lm_token_ratio_over_beto_vocab": safe_ratio(
            len(observed_any_lm_tokens), tokenizer_vocab["beto_vocab_size"]
        ),
    }
    vocab_summary = {
        **tokenizer_vocab,
        "observed_text_lm_token_total": diversity["observed_text_lm_token_total"],
        "observed_text_lm_token_types": diversity["observed_text_lm_token_types"],
        "observed_text_lm_token_ratio_over_beto_vocab": diversity[
            "observed_text_lm_token_ratio_over_beto_vocab"
        ],
        "observed_scanpath_lm_token_total": diversity["observed_scanpath_lm_token_total"],
        "observed_scanpath_lm_token_types": diversity["observed_scanpath_lm_token_types"],
        "observed_scanpath_lm_token_ratio_over_beto_vocab": diversity[
            "observed_scanpath_lm_token_ratio_over_beto_vocab"
        ],
        "observed_any_lm_token_types": diversity["observed_any_lm_token_types"],
        "observed_any_lm_token_ratio_over_beto_vocab": diversity[
            "observed_any_lm_token_ratio_over_beto_vocab"
        ],
        "note": (
            "BETO uses WordPiece tokens. beto_vocab_size counts tokenizer tokens, "
            "not necessarily complete words."
        ),
    }
    with (output_dir / "dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_summary, f, ensure_ascii=False, indent=2)

    with (output_dir / "diversity_summary.json").open("w", encoding="utf-8") as f:
        json.dump(diversity, f, ensure_ascii=False, indent=2)

    with (output_dir / "vocab_summary.json").open("w", encoding="utf-8") as f:
        json.dump(vocab_summary, f, ensure_ascii=False, indent=2)

    write_plots(df, output_dir / "plots")
    write_frequency_plots(
        text_word_counter,
        scanpath_word_counter,
        text_lm_counter,
        scanpath_lm_counter,
        output_dir / "plots",
    )
    print_console_summary(df, len(files), total_subjects, diversity)


if __name__ == "__main__":
    main()
