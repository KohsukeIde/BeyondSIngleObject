#!/usr/bin/env python3
"""Evaluate ModelNet-style classification outputs with CLIP text similarity.

This mirrors the PointLLM evaluation convention: encode the generated text and
each class name with CLIP's text encoder, then take the nearest class as the
prediction. It supports both single-object outputs (`label_name`) and
multi-object outputs (`object_details` + top-level `target_position`).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


DEFAULT_CATEGORIES = Path("configs/eval/modelnet40_shape_names_modified.txt")


def load_json(path: Path):
    with path.open(encoding="utf-8") as fp:
        return json.load(fp)


def load_categories(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as fp:
        return [line.strip() for line in fp if line.strip()]


def get_results(payload) -> list[dict]:
    if isinstance(payload, dict):
        return payload.get("results", [])
    if isinstance(payload, list):
        return payload
    raise TypeError(f"Unsupported JSON root: {type(payload).__name__}")


def get_model_output(sample: dict) -> str:
    for key in ("model_output", "prediction", "answer"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def get_true_category(sample: dict, target_position: int | None) -> str | None:
    label_name = sample.get("label_name")
    if isinstance(label_name, str) and label_name:
        return label_name

    details = sample.get("object_details") or []
    if details and target_position is not None:
        for obj in details:
            if obj.get("position") == target_position:
                category = obj.get("category")
                if isinstance(category, list):
                    return category[0] if category else None
                return category

    if details:
        category = details[0].get("category")
        if isinstance(category, list):
            return category[0] if category else None
        return category
    return None


def encode_texts(model, processor, texts: list[str], device: str, batch_size: int) -> torch.Tensor:
    encoded = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding text"):
        batch = texts[start : start + batch_size]
        with torch.no_grad():
            inputs = processor(text=batch, return_tensors="pt", padding=True, truncation=True)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            features = model.get_text_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
            encoded.append(features.cpu())
    return torch.cat(encoded, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_json", type=Path)
    parser.add_argument("--categories", type=Path, default=DEFAULT_CATEGORIES)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--clip_model",
        default="openai/clip-vit-large-patch14",
        help="CLIP text encoder. Historical PointLLM/Beyond classification scoring used ViT-L/14.",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--category_prompt",
        default="This is a {}",
        help="Prompt template used for category text embeddings.",
    )
    args = parser.parse_args()

    payload = load_json(args.result_json)
    results = get_results(payload)
    target_position = payload.get("target_position") if isinstance(payload, dict) else None
    categories = load_categories(args.categories)

    y_true = []
    texts = []
    kept_samples = []
    skipped = Counter()
    for sample in results:
        true_category = get_true_category(sample, target_position)
        output = get_model_output(sample)
        if not true_category:
            skipped["missing_true_category"] += 1
            continue
        if not output:
            skipped["missing_model_output"] += 1
            continue
        y_true.append(true_category)
        texts.append(output)
        kept_samples.append(sample)

    if not texts:
        raise ValueError("No evaluable samples found. Expected model_output/prediction plus label_name/object_details.")

    print(f"Loading CLIP model: {args.clip_model}")
    model = CLIPModel.from_pretrained(args.clip_model).to(args.device)
    processor = CLIPProcessor.from_pretrained(args.clip_model)
    model.eval()

    category_texts = [args.category_prompt.format(category) for category in categories]
    category_embeddings = encode_texts(model, processor, category_texts, args.device, args.batch_size)
    output_embeddings = encode_texts(model, processor, texts, args.device, args.batch_size)

    similarities = output_embeddings @ category_embeddings.T
    pred_indices = similarities.argmax(dim=1).numpy()
    confidences = similarities.max(dim=1).values.numpy()
    y_pred = [categories[idx] for idx in pred_indices]

    correct = [truth == pred for truth, pred in zip(y_true, y_pred)]
    accuracy = accuracy_score(y_true, y_pred)
    labels_in_report = sorted(set(y_true) | set(y_pred))
    report = classification_report(y_true, y_pred, labels=labels_in_report, output_dict=True, zero_division=0)

    details = []
    for sample, truth, pred, conf, ok in zip(kept_samples, y_true, y_pred, confidences, correct):
        details.append(
            {
                "object_id": sample.get("object_id", sample.get("sample_index")),
                "true": truth,
                "pred": pred,
                "correct": bool(ok),
                "confidence": float(conf),
                "model_output": get_model_output(sample),
            }
        )

    output_payload = {
        "input": str(args.result_json),
        "categories": str(args.categories),
        "clip_model": args.clip_model,
        "target_position": target_position,
        "num_samples": len(details),
        "accuracy": float(accuracy),
        "correct": int(sum(correct)),
        "skipped": dict(skipped),
        "confidence": {
            "mean": float(np.mean(confidences)),
            "min": float(np.min(confidences)),
            "max": float(np.max(confidences)),
        },
        "classification_report": report,
        "details": details,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as fp:
            json.dump(output_payload, fp, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in output_payload.items() if k != "details"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
