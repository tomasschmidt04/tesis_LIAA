#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

ALIASES = {
    "FFD": ["FFD"],
    "TRT": ["TRT", "TFD"],
    "nFix": ["nFix", "FC"],
}


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def resolve_mapping(feature_names, requested):
    mapping = {}
    for name in requested:
        for alias in ALIASES.get(name, [name]):
            if alias in feature_names:
                mapping[name] = {"source_name": alias, "source_index": feature_names.index(alias)}
                break
        else:
            mapping[name] = {"source_name": None, "source_index": None}
    return mapping


def clean_value(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0, True
    if not math.isfinite(value):
        return 0.0, True
    return value, False


def extract(example, requested):
    feature_names = example.get("feature_names") or []
    mapping = resolve_mapping(feature_names, requested)
    rows = example.get("reading_features_by_fixation") or []
    masks = example.get("reading_features_mask_by_fixation") or []
    features = []
    feature_mask = []
    replaced = 0
    for row_index, row in enumerate(rows):
        out_row = []
        mask_row = []
        source_mask = masks[row_index] if row_index < len(masks) else None
        for name in requested:
            source_index = mapping[name]["source_index"]
            value = 0.0
            valid = False
            if source_index is not None and source_index < len(row):
                value, was_replaced = clean_value(row[source_index])
                replaced += int(was_replaced)
                if source_mask is None:
                    valid = not was_replaced
                elif source_index < len(source_mask):
                    valid = bool(source_mask[source_index]) and not was_replaced
            out_row.append(value)
            mask_row.append(1 if valid else 0)
        features.append(out_row)
        feature_mask.append(mask_row)
    return features, feature_mask, mapping, replaced


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--feature_names", default="FFD,TRT,nFix")
    parser.add_argument("--max_samples", type=int, default=200)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    requested = [name.strip() for name in args.feature_names.split(",") if name.strip()]

    values = [[] for _ in requested]
    feature_names_seen = set()
    mapping_used = None
    total = 0
    mismatches = 0
    replaced_total = 0

    sample_path = output_dir / "sample_examples.jsonl"
    mismatch_path = output_dir / "mismatches.jsonl"
    sample_path.write_text("", encoding="utf-8")
    mismatch_path.write_text("", encoding="utf-8")

    for example in load_jsonl(args.input):
        if total >= args.max_samples:
            break
        total += 1
        word_id = example.get("word_id") or []
        by_fixation = example.get("reading_features_by_fixation") or []
        if len(word_id) != len(by_fixation):
            mismatches += 1
            with mismatch_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "trial_id": example.get("trial_id"),
                    "word_id_len": len(word_id),
                    "reading_features_by_fixation_len": len(by_fixation),
                }, ensure_ascii=False) + "\n")
            continue

        feature_names_seen.update(example.get("feature_names") or [])
        features, mask, mapping, replaced = extract(example, requested)
        mapping_used = mapping_used or mapping
        replaced_total += replaced
        for row, mask_row in zip(features, mask):
            for feature_index, (value, valid) in enumerate(zip(row, mask_row)):
                if valid:
                    values[feature_index].append(value)

        if total <= 5:
            with sample_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "trial_id": example.get("trial_id"),
                    "text": example.get("text"),
                    "feature_names": example.get("feature_names"),
                    "mapping": mapping,
                    "word_id_len": len(word_id),
                    "measured_gaze_features_shape": [len(features), len(features[0]) if features else 0],
                    "measured_gaze_features_first_rows": features[:2],
                }, ensure_ascii=False) + "\n")

    means = []
    stds = []
    valid_counts = []
    for feature_values in values:
        valid_counts.append(len(feature_values))
        if feature_values:
            mean = sum(feature_values) / len(feature_values)
            variance = sum((value - mean) ** 2 for value in feature_values) / len(feature_values)
            std = math.sqrt(variance)
            means.append(mean)
            stds.append(std if std > 1e-8 else 1.0)
        else:
            means.append(0.0)
            stds.append(1.0)

    norm = {
        "feature_names": requested,
        "mean": means,
        "std": stds,
        "valid_counts": valid_counts,
        "total_examples_inspected": total,
        "replaced_nonfinite": replaced_total,
        "feature_names_seen": sorted(feature_names_seen),
    }
    (output_dir / "gaze_feature_norm.json").write_text(json.dumps(norm, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "gaze_feature_mapping.json").write_text(json.dumps({
        "requested_features": requested,
        "mapping": mapping_used,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "dataset_metricas_summary.json").write_text(json.dumps({
        "input": args.input,
        "feature_names_seen": sorted(feature_names_seen),
        "mapping": mapping_used,
        "nan_inf_replaced": replaced_total,
        "mismatches_discarded": mismatches,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("feature_names encontrados:", sorted(feature_names_seen))
    print("mapeo usado:", mapping_used)
    print("NaN/inf reemplazados:", replaced_total)
    print("mismatches descartados:", mismatches)


if __name__ == "__main__":
    main()
