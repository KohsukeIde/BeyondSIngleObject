#!/usr/bin/env python3
"""Compute lightweight NLP overlap metrics for PointLLM-style inference JSON.

The script is intentionally separate from GPT-based evaluation. It reads the
`results` array produced by `pointllm.eval.cvpr.eval_cvpr_patch` and reports
surface metrics for prediction/reference text. For multi-turn Shape Mating
outputs, pass the original annotation JSON to evaluate the reasoning turn.

These scores are supplemental sanity checks for MO3D / Shape Mating / Change
Captioning. Paper metrics remain GPT-based MO3D, Shape Mating S/R, Change
Captioning B/R/M, and CLIP-based ModelNet classification.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer


def normalize(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"^\s*(answer|reason)\s*:\s*", "", text, flags=re.IGNORECASE)
    return " ".join(text.strip().split())


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", normalize(text).lower())


def load_json(path: Path):
    with path.open(encoding="utf-8") as fp:
        return json.load(fp)


def load_results(path: Path) -> list[dict]:
    payload = load_json(path)
    if isinstance(payload, dict):
        return payload.get("results", [])
    if isinstance(payload, list):
        return payload
    raise TypeError(f"Unsupported JSON root in {path}: {type(payload).__name__}")


def load_reason_refs(path: Path | None) -> dict[int, str]:
    if path is None:
        return {}
    samples = load_json(path)
    refs = {}
    for idx, sample in enumerate(samples):
        conversations = sample.get("conversations") or []
        if len(conversations) >= 4:
            refs[idx] = normalize(conversations[3].get("value", ""))
    return refs


def meteor_score_safe(reference_tokens: list[str], prediction_tokens: list[str]) -> float | None:
    try:
        from nltk.translate.meteor_score import meteor_score

        if not reference_tokens or not prediction_tokens:
            return 0.0
        return float(meteor_score([reference_tokens], prediction_tokens))
    except Exception:
        return None


def exact_f1(reference_tokens: list[str], prediction_tokens: list[str]) -> float:
    if not reference_tokens and not prediction_tokens:
        return 1.0
    if not reference_tokens or not prediction_tokens:
        return 0.0
    ref_counts = Counter(reference_tokens)
    pred_counts = Counter(prediction_tokens)
    overlap = sum((ref_counts & pred_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_metrics(pairs: Iterable[tuple[str, str]]) -> dict:
    pairs = [(normalize(ref), normalize(pred)) for ref, pred in pairs]
    if not pairs:
        return {"count": 0}

    smooth = SmoothingFunction().method1
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    accum: dict[str, list[float]] = defaultdict(list)
    meteor_values = []
    meteor_available = True

    for reference, prediction in pairs:
        ref_tokens = tokenize(reference)
        pred_tokens = tokenize(prediction)
        refs = [ref_tokens]
        weights = {
            "bleu1": (1.0, 0.0, 0.0, 0.0),
            "bleu2": (0.5, 0.5, 0.0, 0.0),
            "bleu3": (1 / 3, 1 / 3, 1 / 3, 0.0),
            "bleu4": (0.25, 0.25, 0.25, 0.25),
        }
        for name, weight in weights.items():
            score = 0.0
            if ref_tokens and pred_tokens:
                score = sentence_bleu(refs, pred_tokens, weights=weight, smoothing_function=smooth)
            accum[name].append(float(score))

        rouge_l = rouge.score(reference, prediction)["rougeL"].fmeasure if reference or prediction else 1.0
        accum["rougeL"].append(float(rouge_l))
        accum["token_f1"].append(exact_f1(ref_tokens, pred_tokens))

        meteor = meteor_score_safe(ref_tokens, pred_tokens)
        if meteor is None:
            meteor_available = False
        else:
            meteor_values.append(meteor)

        accum["ref_len"].append(float(len(ref_tokens)))
        accum["pred_len"].append(float(len(pred_tokens)))

    out = {"count": len(pairs)}
    for name, values in sorted(accum.items()):
        out[name] = sum(values) / len(values)
    if meteor_available and meteor_values:
        out["meteor"] = sum(meteor_values) / len(meteor_values)
    else:
        out["meteor"] = None
    return out


def is_shape_mating_result(result: dict) -> bool:
    task = result.get("metadata", {}).get("task_type") or result.get("metadata", {}).get("task")
    if task and str(task) != "shape_mating":
        return False
    if "is_shape_mating" in result:
        return bool(result.get("is_shape_mating"))
    question = str(result.get("question", "")).lower()
    return "options: a, b, c, d" in question


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inference_json", type=Path)
    parser.add_argument("--annotation_json", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--task",
        default="auto",
        choices=["auto", "mo3d", "shape_mating", "change_captioning"],
        help="Evaluation policy. Change Captioning verify and Shape Mating answer turns are skipped by default.",
    )
    parser.add_argument(
        "--include_answer_turn",
        action="store_true",
        help="Also report answer-turn metrics for Shape Mating. By default answer-only SM metrics are skipped.",
    )
    parser.add_argument(
        "--include_verify",
        action="store_true",
        help="Also report lexical metrics for Change Captioning verify Yes/No turns. Usually not meaningful.",
    )
    args = parser.parse_args()

    results = load_results(args.inference_json)
    reason_refs = load_reason_refs(args.annotation_json)

    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    skipped = Counter()

    for result in results:
        idx = int(result.get("sample_index", 0))
        task = result.get("metadata", {}).get("task_type") or result.get("metadata", {}).get("task")
        if not task:
            task = "shape_mating" if is_shape_mating_result(result) else "unknown"
        task = str(task)

        if is_shape_mating_result(result):
            if args.include_answer_turn:
                groups["shape_mating_answer"].append((result.get("ground_truth", ""), result.get("prediction", "")))
            else:
                skipped["shape_mating_answer_turn"] += 1
            if "reasoning" in result and idx in reason_refs:
                groups["shape_mating_reason"].append((reason_refs[idx], result.get("reasoning", "")))
            else:
                skipped["shape_mating_reason_missing_ref"] += int("reasoning" in result)
            continue

        if task == "verify" and not args.include_verify:
            skipped["change_captioning_verify_turn"] += 1
            continue

        groups[task].append((result.get("ground_truth", ""), result.get("prediction", "")))

    payload = {
        "input": str(args.inference_json),
        "annotation_json": str(args.annotation_json) if args.annotation_json else None,
        "task": args.task,
        "metric_policy": (
            "Supplemental lexical sanity check only. Do not use these as the primary "
            "paper metrics for MO3D, Shape Mating, or Change Captioning."
        ),
        "skipped": dict(skipped),
        "metrics": {name: compute_metrics(pairs) for name, pairs in sorted(groups.items())},
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, ensure_ascii=False)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
