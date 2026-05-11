#!/usr/bin/env python3
import argparse
import json
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


def mapping(feature_names, requested):
    out = {}
    for name in requested:
        for alias in ALIASES.get(name, [name]):
            if alias in feature_names:
                out[name] = feature_names.index(alias)
                break
    return out


def select_features(example, requested):
    feature_names = example.get("feature_names") or []
    selected = []
    selected_mask = []
    idx_map = mapping(feature_names, requested)
    rows = example.get("reading_features_by_fixation") or []
    masks = example.get("reading_features_mask_by_fixation") or []
    for row_index, row in enumerate(rows):
        mask_row = masks[row_index] if row_index < len(masks) else None
        selected.append([float(row[idx_map[name]]) if name in idx_map else 0.0 for name in requested])
        selected_mask.append([
            int(mask_row[idx_map[name]]) if mask_row is not None and name in idx_map else 0
            for name in requested
        ])
    return selected, selected_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--feature_names", default="FFD,TRT,nFix")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    requested = [name.strip() for name in args.feature_names.split(",") if name.strip()]
    examples = []
    for example in load_jsonl(args.input):
        if len(examples) >= args.batch_size:
            break
        features, masks = select_features(example, requested)
        measured_word_ids = [0] + (example.get("word_id") or []) + [len(example.get("text_tokens") or []) + 1]
        examples.append({
            "measured_word_ids": measured_word_ids,
            "measured_sp_len": len(measured_word_ids),
            "measured_gaze_features": features,
            "measured_gaze_feature_mask": masks,
        })

    batch_size = len(examples)
    max_word_ids_len = max(len(example["measured_word_ids"]) for example in examples)
    max_scanpath_len = max(len(example["measured_gaze_features"]) for example in examples)
    feature_dim = len(requested)

    shapes = {
        "batch_size": batch_size,
        "max_scanpath_len": max_scanpath_len,
        "measured_word_ids.shape": [batch_size, 1, max_word_ids_len],
        "measured_sp_len.shape": [batch_size, 1],
        "measured_gaze_features.shape": [batch_size, 1, max_scanpath_len, feature_dim],
        "measured_gaze_feature_mask.shape": [batch_size, 1, max_scanpath_len, feature_dim],
    }

    (output_dir / "batch_shapes.txt").write_text("\n".join(f"{key}: {value}" for key, value in shapes.items()) + "\n", encoding="utf-8")
    (output_dir / "sample_batch_summary.json").write_text(json.dumps({
        **shapes,
        "first_example_features_before_padding": examples[0]["measured_gaze_features"][:2],
        "first_example_features_after_padding": examples[0]["measured_gaze_features"][:2],
        "note": "Inspector liviano; el collator real escribe el mismo tipo de archivo con --debug_gaze_features True.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(shapes, ensure_ascii=False))


if __name__ == "__main__":
    main()
