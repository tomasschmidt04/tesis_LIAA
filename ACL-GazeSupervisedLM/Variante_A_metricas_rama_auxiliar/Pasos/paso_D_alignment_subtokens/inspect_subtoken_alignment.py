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


def simple_subtokens(token):
    token = str(token)
    if len(token) <= 6:
        return [token]
    return [token[:4], "##" + token[4:]]


def select_gaze_features(example, requested):
    feature_names = example.get("feature_names") or []
    rows = example.get("reading_features_by_fixation") or []
    mapping = {}
    for name in requested:
        for alias in ALIASES.get(name, [name]):
            if alias in feature_names:
                mapping[name] = feature_names.index(alias)
                break
    selected = []
    for row in rows:
        selected.append([row[mapping[name]] if name in mapping else 0.0 for name in requested])
    return selected, {name: {"source_name": feature_names[mapping[name]], "source_index": mapping[name]} for name in mapping}


def beto_tokenization(text_tokens, model_name_or_path):
    try:
        from transformers import AutoTokenizer
    except Exception:
        return None
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    encoded = tokenizer(text_tokens, is_split_into_words=True, truncation=True)
    word_ids = encoded.word_ids()
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
    by_word = [[] for _ in text_tokens]
    for token, word_id in zip(tokens, word_ids):
        if word_id is not None and 0 <= word_id < len(by_word):
            by_word[word_id].append(token)
    return by_word


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", default=".")
    parser.add_argument("--max_examples", type=int, default=5)
    parser.add_argument("--model_name_or_path", default="dccuchile/bert-base-spanish-wwm-cased")
    parser.add_argument("--feature_names", default="FFD,TRT,nFix")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    examples_path = output_dir / "subtoken_alignment_examples.jsonl"
    examples_path.write_text("", encoding="utf-8")
    requested = [name.strip() for name in args.feature_names.split(",") if name.strip()]

    total = 0
    warnings = 0
    for example in load_jsonl(args.input):
        if total >= args.max_examples:
            break
        word_id = example.get("word_id") or []
        text_tokens = example.get("text_tokens") or str(example.get("text") or "").split()
        features, mapping = select_gaze_features(example, requested)
        if len(word_id) != len(features):
            warnings += 1
            continue

        tokenized_words = beto_tokenization(text_tokens, args.model_name_or_path)
        tokenizer_used = args.model_name_or_path if tokenized_words is not None else "fallback_aproximado"
        if tokenized_words is None:
            tokenized_words = [simple_subtokens(token) for token in text_tokens]
        expanded_features = []
        gaze_token_pos = []
        for fixation_index, word_position in enumerate(word_id):
            token_index = int(word_position) - 1
            subtokens = tokenized_words[token_index] if 0 <= token_index < len(tokenized_words) else []
            for subtoken_index, _ in enumerate(subtokens):
                gaze_token_pos.append([token_index, subtoken_index])
                expanded_features.append(features[fixation_index])

        payload = {
            "text": example.get("text"),
            "word_id_original": word_id,
            "scanpath_tokens": example.get("scanpath_tokens"),
            "feature_mapping": mapping,
            "tokenizer_used": tokenizer_used,
            "tokenizacion_BETO": tokenized_words,
            "gaze_token_pos": gaze_token_pos,
            "measured_gaze_features_por_fijacion": features,
            "measured_gaze_features_expandida_a_subtokens": expanded_features,
            "length_check": {
                "expanded_positions": len(gaze_token_pos),
                "expanded_features": len(expanded_features),
                "ok": len(gaze_token_pos) == len(expanded_features),
            },
        }
        with examples_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        total += 1

    (output_dir / "alignment_summary.json").write_text(json.dumps({
        "examples_written": total,
        "warnings": warnings,
        "note": "Usa BETO si transformers esta instalado; si no, cae a tokenizacion aproximada. El modelo usa word_ids reales del tokenizer.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "alignment_warnings.jsonl").touch()
    print(f"examples_written={total}")
    print(f"warnings={warnings}")


if __name__ == "__main__":
    main()
