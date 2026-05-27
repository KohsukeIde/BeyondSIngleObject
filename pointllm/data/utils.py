from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch
import transformers

from pointllm import conversation as conversation_lib


IGNORE_INDEX = -100


class LRUCache:
    def __init__(self, capacity, max_access_count):
        self.cache = OrderedDict()
        self.access_count = defaultdict(int)
        self.capacity = capacity
        self.max_access_count = max_access_count

    def get(self, key):
        if key not in self.cache:
            return None
        value = self.cache.pop(key)
        self.cache[key] = value
        self.access_count[key] += 1
        return value

    def put(self, key, value):
        if key in self.cache:
            self.cache.pop(key)
        elif len(self.cache) == self.capacity:
            oldest_key = next(iter(self.cache))
            self.cache.popitem(last=False)
            del self.access_count[oldest_key]
        self.cache[key] = value
        self.access_count[key] = 1

    def get_access_count(self, key):
        return self.access_count.get(key, 0)

    def reset_access_count(self, key):
        self.access_count[key] = 0


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for rou in rounds:
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length and cur_len != total_len:
            target[:] = IGNORE_INDEX
            print(
                f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. "
                "(ignored)"
            )

    return dict(input_ids=input_ids, labels=targets)


def _single_point_token_region(point_backbone_config: dict) -> str:
    point_token_len = point_backbone_config["point_token_len"]
    default_point_patch_token = point_backbone_config["default_point_patch_token"]
    region = default_point_patch_token * point_token_len
    if point_backbone_config["mm_use_point_start_end"]:
        region = (
            point_backbone_config["default_point_start_token"]
            + region
            + point_backbone_config["default_point_end_token"]
        )
    return region


def build_point_token_sequence(point_backbone_config: dict, num_point_clouds: int = 1) -> str:
    num_point_clouds = max(1, int(num_point_clouds))
    return _single_point_token_region(point_backbone_config) * num_point_clouds


def preprocess_multimodal_point_cloud(
    sources: Sequence[str],
    point_backbone_config: dict,
    point_indicator: str = "<point>",
    num_point_clouds: int = 1,
) -> Dict:
    single_region = _single_point_token_region(point_backbone_config)
    multi_region = build_point_token_sequence(point_backbone_config, num_point_clouds)

    for source in sources:
        for sentence in source:
            value = sentence["value"]
            indicator_count = value.count(point_indicator)
            if indicator_count == 0:
                continue
            if indicator_count == 1:
                sentence["value"] = value.replace(point_indicator, multi_region)
            else:
                sentence["value"] = value.replace(point_indicator, single_region)

    return sources


def pc_norm(pc):
    pc = np.asarray(pc)
    xyz = pc[:, :3]
    other_feature = pc[:, 3:]

    centroid = np.mean(xyz, axis=0)
    xyz = xyz - centroid
    radius = np.max(np.sqrt(np.sum(xyz**2, axis=1)))
    if radius > 0:
        xyz = xyz / radius

    return np.concatenate((xyz, other_feature), axis=1)


def _load_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        with np.load(path) as data:
            for key in ("points", "xyz", "arr_0"):
                if key in data:
                    return data[key]
            keys = list(data.keys())
            if not keys:
                raise ValueError(f"empty npz point cloud: {path}")
            return data[keys[0]]
    if suffix == ".csv":
        return np.loadtxt(path, delimiter=",")
    return np.load(path)


def load_point_cloud(path: str | Path, pointnum: int = 8192, use_color: bool = True) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    pc = np.asarray(_load_array(path), dtype=np.float32)
    if pc.ndim != 2 or pc.shape[1] < 3:
        raise ValueError(f"point cloud must be an NxC array with C>=3: {path}")

    if pointnum and pc.shape[0] != pointnum:
        replace = pc.shape[0] < pointnum
        indices = np.random.choice(pc.shape[0], pointnum, replace=replace)
        pc = pc[indices]

    pc = pc_norm(pc)
    if not use_color:
        pc = pc[:, :3]
    elif pc.shape[1] < 6:
        rgb = np.full((pc.shape[0], 3), 0.5, dtype=pc.dtype)
        pc = np.concatenate([pc[:, :3], rgb], axis=1)
    else:
        pc = pc[:, :6]

    return pc.astype(np.float32)


def load_objaverse_point_cloud(data_path, object_id, pointnum=8192, use_color=False):
    data_path = Path(data_path)
    filename = f"{object_id}_{pointnum}.npy"
    candidates = [
        data_path / filename,
        data_path / f"{pointnum}_npy" / filename,
        data_path / str(object_id),
    ]
    for candidate in candidates:
        if candidate.exists():
            return load_point_cloud(candidate, pointnum=pointnum, use_color=use_color)
    raise FileNotFoundError(f"could not resolve point cloud for {object_id} under {data_path}")


@dataclass
class DataCollatorForPointTextDataset:
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if "point_clouds" in instances[0]:
            point_clouds = [instance["point_clouds"] for instance in instances]
            if any(isinstance(item, list) for item in point_clouds):
                normalized = [item if isinstance(item, list) else [item] for item in point_clouds]
                batch["point_clouds"] = normalized
                valid_counts = [instance.get("num_point_clouds_valid") for instance in instances]
                if any(count is not None for count in valid_counts):
                    batch["num_point_clouds_valid"] = [
                        int(count) if count is not None else len(normalized[idx])
                        for idx, count in enumerate(valid_counts)
                    ]
                else:
                    batch["num_point_clouds_valid"] = [len(item) for item in normalized]
            elif all(x is not None and x.shape == point_clouds[0].shape for x in point_clouds):
                batch["point_clouds"] = torch.stack(point_clouds)
            else:
                batch["point_clouds"] = point_clouds

        return batch


def farthest_point_sample(point, npoint):
    n, _ = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((n,)) * 1e10
    farthest = np.random.randint(0, n)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    return point[centroids.astype(np.int32)]


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    radius = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    if radius > 0:
        pc = pc / radius
    return pc
