#!/usr/bin/env python3
"""Run ModelNet classification prompts with a Multi-3DLLM checkpoint.

The output schema intentionally matches the historical PointLLM ModelNet
classification JSON (`model_output`, `label_name`, `object_details`,
`target_position`) so it can be scored by `tools/evaluate_clip_classification.py`.
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from pointllm.data.utils import pc_norm
from pointllm.eval.cvpr.eval_cvpr_patch import generate_answer, prepare_model


DEFAULT_MODELNET_DATA = Path("data/modelnet40_data/modelnet40_test_8192pts_fps.dat")
DEFAULT_CATEGORIES = Path("configs/eval/modelnet40_shape_names_modified.txt")

PROMPTS = {
    "pointllm_cls": {
        1: {
            1: "<point>\nWhat is this?",
        },
    },
    "paper": {
        1: {
            1: "<point>\nWhat is this object?",
        },
        2: {
            1: "<point> <point>\nWhat is the first object?",
            2: "<point> <point>\nWhat is the second object?",
        },
        3: {
            1: "<point> <point> <point>\nWhat is the first object?",
            2: "<point> <point> <point>\nWhat is the second object?",
            3: "<point> <point> <point>\nWhat is the third object?",
        },
    },
}
PROMPTS["pointllm_multi"] = PROMPTS["paper"]


def build_question(num_objects: int, target_position: int, prompt_mode: str) -> str:
    if num_objects > 1:
        prompt_mode = "paper"
    return PROMPTS[prompt_mode][num_objects][target_position]


def load_categories(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as fp:
        return [line.strip() for line in fp if line.strip()]


def load_modelnet(path: Path):
    with path.open("rb") as fp:
        points, labels = pickle.load(fp)
    return points, labels


def normalize_points(points: np.ndarray, pointnum: int, use_color: bool) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if pointnum and points.shape[0] != pointnum:
        replace = points.shape[0] < pointnum
        idx = np.random.choice(points.shape[0], pointnum, replace=replace)
        points = points[idx]
    points = pc_norm(points[:, :6] if points.shape[1] >= 6 else points[:, :3])
    if use_color:
        if points.shape[1] < 6:
            points = np.concatenate([points[:, :3], np.zeros_like(points[:, :3])], axis=1)
        else:
            points = points[:, :6]
    else:
        points = points[:, :3]
    return points.astype(np.float32)


def build_category_index(labels: list, categories: list[str]) -> dict[str, list[int]]:
    by_category: dict[str, list[int]] = defaultdict(list)
    for idx, label in enumerate(labels):
        label_idx = int(label.item() if hasattr(label, "item") else label)
        by_category[categories[label_idx]].append(idx)
    return by_category


def build_eval_indices(
    labels: list,
    categories: list[str],
    by_category: dict[str, list[int]],
    limit: int,
    sampling: str,
    seed: int,
) -> list[int]:
    all_indices = list(range(len(labels)))
    if limit <= 0 or limit >= len(all_indices):
        return all_indices

    rng = random.Random(seed)
    if sampling == "sequential":
        return all_indices[:limit]
    if sampling == "random":
        rng.shuffle(all_indices)
        return all_indices[:limit]

    per_category = {category: list(by_category.get(category, [])) for category in categories}
    for indices in per_category.values():
        rng.shuffle(indices)

    selected = []
    cursor = 0
    while len(selected) < limit:
        progressed = False
        for category in categories:
            indices = per_category[category]
            if cursor < len(indices):
                selected.append(indices[cursor])
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break
        cursor += 1
    return selected


def build_sample_indices(
    base_idx: int,
    labels: list,
    categories: list[str],
    by_category: dict[str, list[int]],
    num_objects: int,
) -> list[int]:
    selected = [base_idx]
    if num_objects == 1:
        return selected

    base_label = int(labels[base_idx].item() if hasattr(labels[base_idx], "item") else labels[base_idx])
    base_category = categories[base_label]
    available = [category for category in categories if category != base_category and by_category.get(category)]
    rng = random.Random(42 + base_idx)
    for _ in range(1, num_objects):
        category = rng.choice(available)
        selected.append(rng.choice(by_category[category]))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", type=Path, default=DEFAULT_MODELNET_DATA)
    parser.add_argument("--categories", type=Path, default=DEFAULT_CATEGORIES)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num_objects", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--target_position", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument(
        "--prompt_mode",
        default="paper",
        choices=["pointllm_cls", "paper", "pointllm_multi"],
        help="Use the paper table prompt set by default; pointllm_cls keeps the original single-object 'What is this?' baseline.",
    )
    parser.add_argument("--limit", type=int, default=0, help="0 or negative evaluates the full ModelNet40 test split.")
    parser.add_argument("--sampling", default="balanced", choices=["balanced", "random", "sequential"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pointnum", type=int, default=8192)
    parser.add_argument("--use_color", action="store_true", default=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--relation_mode", default="patch", choices=["object", "patch", "micro", "fast_patch"])
    parser.add_argument("--model_variant", default="auto", choices=["auto", "cvpr", "original"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.target_position > args.num_objects:
        raise ValueError("--target_position cannot exceed --num_objects")

    categories = load_categories(args.categories)
    points, labels = load_modelnet(args.data_path)
    by_category = build_category_index(labels, categories)
    eval_indices = build_eval_indices(labels, categories, by_category, args.limit, args.sampling, args.seed)
    total = len(eval_indices)
    question = build_question(args.num_objects, args.target_position, args.prompt_mode)

    device = torch.device(args.device)
    model, tokenizer, conv, stop_str, point_cfg, point_dtype, model_variant = prepare_model(
        model_path=args.model_path,
        device=device,
        relation_mode=args.relation_mode,
        model_variant=args.model_variant,
        debug=args.debug,
    )

    results = []
    np.random.seed(args.seed)
    random.seed(args.seed)
    for base_idx in tqdm(eval_indices, desc="ModelNet classification"):
        indices = build_sample_indices(base_idx, labels, categories, by_category, args.num_objects)
        point_tensors = []
        object_details = []
        for position, sample_idx in enumerate(indices, start=1):
            label_idx = int(labels[sample_idx].item() if hasattr(labels[sample_idx], "item") else labels[sample_idx])
            point_array = normalize_points(points[sample_idx], args.pointnum, args.use_color)
            point_tensors.append(torch.from_numpy(point_array).to(device=device, dtype=point_dtype))
            object_details.append(
                {
                    "position": position,
                    "category": categories[label_idx],
                    "label": label_idx,
                    "original_dataset_index": sample_idx,
                }
            )

        prediction = generate_answer(
            model=model,
            tokenizer=tokenizer,
            conv_template=conv,
            stop_str=stop_str,
            point_backbone_config=point_cfg,
            question=question,
            point_clouds=[point_tensors],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=1.0,
            no_repeat_ngram_size=0,
            debug=args.debug and base_idx < 3,
        )

        target = object_details[args.target_position - 1]
        results.append(
            {
                "object_id": base_idx,
                "sample_index": base_idx,
                "object_details": object_details,
                "ground_truth": target["label"],
                "label_name": target["category"] if args.num_objects == 1 else None,
                "model_output": prediction,
            }
        )

    payload = {
        "model_path": args.model_path,
        "data_path": str(args.data_path),
        "categories": str(args.categories),
        "model_variant": model_variant,
        "prompt": question,
        "prompt_mode": args.prompt_mode,
        "num_objects": args.num_objects,
        "target_position": args.target_position,
        "limit": total,
        "sampling": args.sampling,
        "seed": args.seed,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2, ensure_ascii=False)
    print(f"Saved ModelNet classification outputs to {args.output}")


if __name__ == "__main__":
    main()
