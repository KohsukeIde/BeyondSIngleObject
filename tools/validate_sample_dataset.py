#!/usr/bin/env python3
"""Validate the public sample multi-object annotation without importing torch."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_csv_point_cloud(path: Path) -> tuple[int, int]:
    rows = []
    with path.open(newline="") as fp:
        for row in csv.reader(fp):
            if not row:
                continue
            rows.append([float(value) for value in row])
    if not rows:
        raise ValueError(f"empty point cloud: {path}")
    width = len(rows[0])
    if width < 3:
        raise ValueError(f"point cloud must have at least XYZ columns: {path}")
    for row in rows:
        if len(row) != width:
            raise ValueError(f"inconsistent column count in {path}")
    return len(rows), width


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anno_path", default="examples/sample_data/sample_mo3d.json")
    parser.add_argument("--data_path", default="examples/sample_data/sample_point_clouds")
    parser.add_argument("--pointnum", type=int, default=8)
    args = parser.parse_args()

    anno_path = Path(args.anno_path)
    data_path = Path(args.data_path)
    samples = json.loads(anno_path.read_text())
    if not isinstance(samples, list) or not samples:
        raise ValueError("annotation must be a non-empty JSON list")

    total_clouds = 0
    for index, sample in enumerate(samples):
        conversations = sample.get("conversations") or []
        if not conversations or "<point>" not in conversations[0].get("value", ""):
            raise ValueError(f"sample {index} is missing a <point> prompt")
        object_ids = sample.get("object_ids")
        if not isinstance(object_ids, list) or len(object_ids) < 1:
            raise ValueError(f"sample {index} must contain object_ids")
        for object_id in object_ids:
            pc_path = data_path / object_id
            if not pc_path.exists():
                raise FileNotFoundError(pc_path)
            rows, cols = read_csv_point_cloud(pc_path)
            if rows != args.pointnum:
                raise ValueError(f"{pc_path} has {rows} points; expected {args.pointnum}")
            if cols not in (3, 6):
                raise ValueError(f"{pc_path} should have 3 or 6 columns, got {cols}")
            total_clouds += 1

    print(f"validated {len(samples)} samples and {total_clouds} point clouds")


if __name__ == "__main__":
    main()
