from __future__ import annotations

import copy
import json
import os
import random
from bisect import bisect_right
from pathlib import Path
from typing import Dict, List, Optional

import torch
import transformers
from torch.utils.data import Dataset

from .utils import (
    DataCollatorForPointTextDataset,
    load_objaverse_point_cloud,
    load_point_cloud,
    preprocess_multimodal_point_cloud,
    preprocess_v1,
)


def make_object_point_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
) -> Dict:
    data_collator = DataCollatorForPointTextDataset(tokenizer=tokenizer)
    train_dataset = ObjectPointCloudDataset(
        split="train",
        data_path=data_args.data_path,
        anno_path=data_args.anno_path,
        pointnum=data_args.pointnum,
        conversation_types=data_args.conversation_types,
        tokenizer=tokenizer,
        use_color=data_args.use_color,
        data_args=data_args,
    )

    if data_args.split_train_val:
        val_dataset = (
            train_dataset
            if data_args.data_debug_num > 0
            else ObjectPointCloudDataset(
                split="val",
                data_path=data_args.data_path,
                anno_path=data_args.anno_path,
                pointnum=data_args.pointnum,
                conversation_types=data_args.conversation_types,
                tokenizer=tokenizer,
                use_color=data_args.use_color,
                data_args=data_args,
            )
        )
    else:
        val_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=val_dataset, data_collator=data_collator)


def make_multitask_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
) -> Dict:
    data_collator = DataCollatorForPointTextDataset(tokenizer=tokenizer)
    multitask_config = getattr(data_args, "multitask_config", None) or {}
    dataset_configs = multitask_config.get("datasets", [])
    datasets = []
    for config in dataset_configs:
        anno_path = config.get("anno_path")
        data_path = config.get("data_path")
        if not anno_path or not data_path:
            raise ValueError(f"multitask dataset is missing paths: {config}")
        datasets.append(
            ObjectPointCloudDataset(
                split="train",
                data_path=data_path,
                anno_path=anno_path,
                pointnum=data_args.pointnum,
                conversation_types=data_args.conversation_types,
                tokenizer=tokenizer,
                use_color=data_args.use_color,
                data_args=data_args,
                dataset_name=config.get("name"),
            )
        )
    if not datasets:
        raise ValueError("POINTLLM_MULTITASK is enabled but no datasets were configured.")
    fullmix_probs = multitask_config.get("fullmix_probs") or []
    epoch_size = int(multitask_config.get("epoch_size") or 0)
    seed = int(multitask_config.get("seed") or 42)
    if fullmix_probs and len(fullmix_probs) == len(datasets):
        train_dataset = WeightedConcatPointCloudDataset(
            datasets,
            probs=fullmix_probs,
            epoch_size=epoch_size,
            seed=seed,
        )
    else:
        train_dataset = ConcatPointCloudDataset(datasets)
    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )


class ConcatPointCloudDataset(Dataset):
    def __init__(self, datasets: List[Dataset]):
        self.datasets = datasets
        self.cumulative_sizes = []
        total = 0
        for dataset in datasets:
            total += len(dataset)
            self.cumulative_sizes.append(total)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, index):
        dataset_idx = bisect_right(self.cumulative_sizes, index)
        sample_idx = index
        if dataset_idx > 0:
            sample_idx -= self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]


class WeightedConcatPointCloudDataset(Dataset):
    def __init__(self, datasets: List[Dataset], probs: List[float], epoch_size: int = 0, seed: int = 42):
        if len(datasets) != len(probs):
            raise ValueError("datasets and probs must have the same length")
        if any(len(dataset) == 0 for dataset in datasets):
            raise ValueError("all multitask datasets must be non-empty")
        prob_sum = sum(max(0.0, float(prob)) for prob in probs)
        if prob_sum <= 0:
            raise ValueError("multitask sampling probabilities must sum to a positive value")

        self.datasets = datasets
        self.probs = [max(0.0, float(prob)) / prob_sum for prob in probs]
        total = int(epoch_size) if epoch_size and epoch_size > 0 else sum(len(dataset) for dataset in datasets)
        counts = [max(1, int(round(prob * total))) for prob in self.probs]
        diff = total - sum(counts)
        # Keep the virtual epoch length exact after rounding.
        order = sorted(range(len(counts)), key=lambda idx: self.probs[idx], reverse=True)
        cursor = 0
        while diff != 0 and order:
            idx = order[cursor % len(order)]
            if diff > 0:
                counts[idx] += 1
                diff -= 1
            elif counts[idx] > 1:
                counts[idx] -= 1
                diff += 1
            cursor += 1

        rng = random.Random(seed)
        self.mapping = []
        self.dataset_counts = counts
        for dataset_idx, (dataset, count) in enumerate(zip(datasets, counts)):
            indices = []
            base_indices = list(range(len(dataset)))
            while len(indices) < count:
                cycle = base_indices[:]
                rng.shuffle(cycle)
                indices.extend(cycle)
            self.mapping.extend((dataset_idx, sample_idx) for sample_idx in indices[:count])
        rng.shuffle(self.mapping)

        names = [getattr(dataset, "dataset_name", None) or f"dataset_{idx}" for idx, dataset in enumerate(datasets)]
        summary = ", ".join(
            f"{name}:{count}/{len(dataset)}"
            for name, count, dataset in zip(names, counts, datasets)
        )
        print(f"Weighted multitask virtual epoch size: {len(self.mapping)} ({summary})")

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, index):
        dataset_idx, sample_idx = self.mapping[index]
        return self.datasets[dataset_idx][sample_idx]


class ObjectPointCloudDataset(Dataset):
    def __init__(
        self,
        data_path=None,
        anno_path=None,
        tokenizer=None,
        pointnum=8192,
        split="train",
        conversation_types=None,
        use_color=True,
        data_args=None,
        dataset_name: Optional[str] = None,
    ):
        super().__init__()

        self.data_path = Path(data_path)
        self.anno_path = anno_path
        self.tokenizer = tokenizer
        self.split = split
        self.conversation_types = conversation_types or ("simple_description",)
        self.data_args = data_args
        self.normalize_pc = True
        self.use_color = use_color
        self.pointnum = pointnum
        self.point_backbone_config = data_args.point_backbone_config if data_args is not None else None
        self.point_indicator = "<point>"
        self.dataset_name = dataset_name

        print(f"Loading anno file from {anno_path}.")
        with open(anno_path, "r") as json_file:
            self.list_data_dict = json.load(json_file)

        print(f"Using conversation_type: {self.conversation_types}")
        print(f"Before filtering, the dataset size is: {len(self.list_data_dict)}.")
        has_object_pc_paths = any(
            isinstance(data.get("objects"), list)
            and data["objects"]
            and isinstance(data["objects"][0], dict)
            and "pc_path" in data["objects"][0]
            for data in self.list_data_dict[:20]
        )
        disable_convtype_filter = has_object_pc_paths
        if disable_convtype_filter:
            print("Conversation type filtering disabled for task-specific annotation data.")
        else:
            self.list_data_dict = [
                data
                for data in self.list_data_dict
                if data.get("conversation_type", "simple_description") in self.conversation_types
            ]
        print(f"After filtering, the dataset size is: {len(self.list_data_dict)}.")

        if self.data_args is not None and self.data_args.data_debug_num > 0:
            self.list_data_dict = self.list_data_dict[: self.data_args.data_debug_num]
        elif self.data_args is not None and self.data_args.split_train_val:
            split_idx = int(self.data_args.split_ratio * len(self.list_data_dict))
            if self.split == "train":
                self.list_data_dict = self.list_data_dict[:split_idx]
            else:
                self.list_data_dict = self.list_data_dict[split_idx:]
        self.max_point_clouds = max(
            (len(self._get_object_ids(entry)) for entry in self.list_data_dict),
            default=1,
        )

    def __len__(self):
        return len(self.list_data_dict)

    def _detect_dataset_format(self, entry: dict) -> str:
        if "objects" in entry and isinstance(entry["objects"], list):
            if entry["objects"] and "pc_path" in entry["objects"][0]:
                return "shape_mating"
        object_ids = entry.get("object_ids") or []
        if object_ids:
            first = str(object_ids[0])
            if "/" in first or first.endswith((".npz", ".npy", ".csv")):
                return "path_list"
        return "objaverse"

    def _get_object_ids(self, entry: dict) -> List[str]:
        order = entry.get("gpt_input_order") or entry.get("metadata", {}).get("gpt_input_order")
        if order:
            return list(order)
        if "object_ids" in entry and isinstance(entry["object_ids"], list):
            return list(entry["object_ids"])
        if "object_id" in entry:
            return [entry["object_id"]]
        if "objects" in entry and isinstance(entry["objects"], list):
            return [obj.get("id") or obj.get("pc_path") for obj in entry["objects"]]
        raise ValueError("sample must contain object_id, object_ids, or objects[*].pc_path")

    def _resolve_pointcloud_path(self, object_id: str, entry: dict, obj_index: int) -> Path:
        dataset_format = self._detect_dataset_format(entry)
        candidates: List[Path] = []
        if dataset_format == "shape_mating":
            objects = entry.get("objects", [])
            pc_path = objects[obj_index].get("pc_path")
            if not pc_path:
                raise ValueError(f"objects[{obj_index}] is missing pc_path")
            pc_path = str(pc_path)
            candidates.append(self.data_path / pc_path)
            source = objects[obj_index].get("source", {})
            dataset_name = str(source.get("dataset", "")).strip()
            if pc_path.startswith("shapemating/thingi10k/"):
                candidates.append(self.data_path / pc_path[len("shapemating/thingi10k/") :])
            if pc_path.startswith("shapemating/"):
                stripped = pc_path[len("shapemating/") :]
                candidates.append(self.data_path / stripped)
                if dataset_name and not stripped.startswith(f"{dataset_name}/"):
                    candidates.append(self.data_path / "shapemating" / dataset_name / stripped)
                    candidates.append(self.data_path / dataset_name / stripped)
            return next((candidate for candidate in candidates if candidate.exists()), candidates[0])

        object_id = str(object_id)
        if dataset_format == "path_list":
            candidates.append(self.data_path / object_id)
            if object_id.startswith("point_clouds/"):
                candidates.append(self.data_path / object_id[len("point_clouds/") :])
            if object_id.startswith("point_clouds/scaled_to_align_rendering/"):
                candidates.append(
                    self.data_path / object_id[len("point_clouds/scaled_to_align_rendering/") :]
                )
            return next((candidate for candidate in candidates if candidate.exists()), candidates[0])

        direct = self.data_path / object_id
        if direct.exists():
            return direct
        filename = f"{object_id}_{self.pointnum}.npy"
        candidates = [
            self.data_path / filename,
            self.data_path / f"{self.pointnum}_npy" / filename,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[-1]

    def _load_point_clouds(self, entry: dict) -> tuple[List[torch.Tensor], List[str]]:
        object_ids = self._get_object_ids(entry)
        point_clouds = []
        dataset_format = self._detect_dataset_format(entry)
        for obj_index, object_id in enumerate(object_ids):
            path = self._resolve_pointcloud_path(object_id, entry, obj_index)
            if path.exists():
                pc = load_point_cloud(path, pointnum=self.pointnum, use_color=self.use_color)
            elif dataset_format == "objaverse":
                pc = load_objaverse_point_cloud(
                    self.data_path,
                    object_id,
                    pointnum=self.pointnum,
                    use_color=self.use_color,
                )
            else:
                raise FileNotFoundError(f"could not resolve point cloud for {object_id} under {self.data_path}")
            point_clouds.append(torch.from_numpy(pc))
        return point_clouds, object_ids

    def _pad_point_clouds_for_distributed(self, point_clouds: List[torch.Tensor]) -> tuple[List[torch.Tensor], int]:
        num_valid = len(point_clouds)
        try:
            world_size = int(os.environ.get("WORLD_SIZE", "1"))
        except ValueError:
            world_size = 1
        if world_size <= 1 or not point_clouds:
            return point_clouds, num_valid

        # FSDP requires every rank to execute the same number of point-backbone
        # forwards. MO3D mixes two- and three-object samples, so pad each sample
        # to the dataset/task maximum and mask the dummy features later.
        target_count = max(3, self.max_point_clouds, num_valid)
        padded = list(point_clouds)
        while len(padded) < target_count:
            padded.append(torch.zeros_like(point_clouds[0]))
        return padded, num_valid

    def __getitem__(self, index):
        entry = self.list_data_dict[index]
        sources = [entry] if isinstance(index, int) else entry
        assert len(sources) == 1, "sources should be a list"
        entry = sources[0]

        has_point = self.point_indicator in entry["conversations"][0]["value"]
        point_clouds = None
        object_ids = None
        num_point_clouds_valid = None
        if has_point:
            point_clouds, object_ids = self._load_point_clouds(entry)
            point_clouds, num_point_clouds_valid = self._pad_point_clouds_for_distributed(point_clouds)

        if self.tokenizer is None:
            data_dict = dict(point_clouds=point_clouds, object_ids=object_ids)
            if num_point_clouds_valid is not None:
                data_dict["num_point_clouds_valid"] = num_point_clouds_valid
            return data_dict

        conversations = copy.deepcopy([entry["conversations"]])
        if has_point:
            text_point_cloud_count = num_point_clouds_valid or len(point_clouds)
            conversations = preprocess_multimodal_point_cloud(
                conversations,
                self.point_backbone_config,
                point_indicator=self.point_indicator,
                num_point_clouds=text_point_cloud_count,
            )

        data_dict = preprocess_v1(conversations, self.tokenizer)
        data_dict = dict(
            input_ids=data_dict["input_ids"][0],
            labels=data_dict["labels"][0],
        )
        if has_point:
            data_dict["point_clouds"] = point_clouds
            data_dict["object_ids"] = object_ids
            if num_point_clouds_valid is not None:
                data_dict["num_point_clouds_valid"] = num_point_clouds_valid
        return data_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/point_clouds", type=str)
    parser.add_argument("--anno_path", required=True, type=str)
    parser.add_argument("--pointnum", default=8192, type=int)
    parser.add_argument("--tokenizer_path", required=True, type=str)
    args = parser.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer_path)
    args.point_backbone_config = None
    args.conversation_types = ["simple_description"]
    args.use_color = True
    args.data_debug_num = 0
    args.split_train_val = False
    args.split_ratio = 0.9

    dataset = ObjectPointCloudDataset(
        data_path=args.data_path,
        anno_path=args.anno_path,
        pointnum=args.pointnum,
        tokenizer=tokenizer,
        data_args=args,
    )
    print(f"Dataset length: {len(dataset)}")
