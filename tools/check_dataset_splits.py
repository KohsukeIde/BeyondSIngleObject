#!/usr/bin/env python3
"""Check object/path overlap between train and test annotation files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: Path):
    with path.open(encoding="utf-8") as fp:
        return json.load(fp)


def sample_paths(sample: dict) -> tuple[str, ...]:
    if isinstance(sample.get("objects"), list):
        paths = [obj.get("pc_path") or obj.get("id") for obj in sample["objects"]]
        return tuple(str(path) for path in paths if path)
    if isinstance(sample.get("object_ids"), list):
        return tuple(str(item) for item in sample["object_ids"])
    if sample.get("object_id"):
        return (str(sample["object_id"]),)
    return ()


def collect(path: Path):
    samples = load_json(path)
    object_keys: set[str] = set()
    tuple_keys: set[tuple[str, ...]] = set()
    sample_ids: set[str] = set()

    for sample in samples:
        paths = sample_paths(sample)
        if paths:
            tuple_keys.add(paths)
            object_keys.update(paths)
        if sample.get("id"):
            sample_ids.add(str(sample["id"]))

    return {
        "num_samples": len(samples),
        "objects": object_keys,
        "tuples": tuple_keys,
        "sample_ids": sample_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--fail_on_tuple_overlap", action="store_true")
    args = parser.parse_args()

    train = collect(args.train)
    test = collect(args.test)

    object_overlap = train["objects"] & test["objects"]
    tuple_overlap = train["tuples"] & test["tuples"]
    sample_id_overlap = train["sample_ids"] & test["sample_ids"]

    print(f"train_samples: {train['num_samples']}")
    print(f"test_samples: {test['num_samples']}")
    print(f"train_objects: {len(train['objects'])}")
    print(f"test_objects: {len(test['objects'])}")
    print(f"object_overlap: {len(object_overlap)}")
    print(f"tuple_overlap: {len(tuple_overlap)}")
    print(f"sample_id_overlap: {len(sample_id_overlap)}")

    if args.fail_on_tuple_overlap and tuple_overlap:
        raise SystemExit("tuple overlap detected")


if __name__ == "__main__":
    main()
