#!/usr/bin/env python3
"""Build rule-normalized Change Captioning annotations from ShapeTalk.

This builder derives labels from ShapeTalk structure:

- verify positive: anchor + positive target -> Answer: Yes
- verify negative: anchor + same-anchor negative target -> Answer: No
- delta caption: anchor + positive target -> geometric edit text

It keeps the public task surface aligned with the paper: verify and delta
captioning.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable


DEFAULT_SOURCE_DIR = Path("data/shapetalk/output/full")
DEFAULT_OUTPUT_DIR = Path("data/change_captioning")
DEFAULT_POINT_CLOUD_ROOT = Path("data/shapetalk")
DATASET_VERSION = "change_captioning"

VERIFY_QUESTION = """<point>
Object 1 is the source shape.

<point>
Object 2 is a candidate target shape.

The instruction describes the intended target relative to Object 1.
Target requirements:
{requirements}

Is Object 2 the intended target?
Return exactly one line in this format:
Answer: <decision>"""

VERIFY_REASON_PROMPT = "Briefly explain in ONE sentence. Do not change the answer."

VERIFY_OBSERVE_PROMPT = """{prefix}

First describe the actual geometric edits from Object 1 to Object 2.
Then decide whether Object 2 satisfies the target requirements.
End with exactly one final line in this format:
Answer: <Yes/No>"""

DELTA_QUESTION = """<point>
Object 1 is the source shape.

<point>
Object 2 is the target shape.

Describe only the geometric edits needed to change Object 1 into Object 2.
Do not describe pose, position, color, material, or texture.
Write 3 to 6 semicolon-separated edits, and do not repeat the same detail."""

TRIM_PREFIXES = (
    "candidate satisfies the instruction:",
    "instruction not satisfied; candidate instead:",
    "candidate instead:",
    "reason:",
)

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "candidate",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "object",
    "of",
    "on",
    "or",
    "shape",
    "target",
    "targets",
    "that",
    "the",
    "there",
    "this",
    "to",
    "with",
}

DELTA_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "candidate",
    "does",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "object",
    "of",
    "on",
    "or",
    "shape",
    "target",
    "targets",
    "that",
    "the",
    "there",
    "this",
    "to",
    "with",
}

NEGATION_WORDS = {"no", "not", "without", "none", "never"}


@dataclass
class PairInfo:
    cls: str
    anchor_uid: str
    pos_uid: str
    split: str
    source_splits: Counter = field(default_factory=Counter)
    index_ids: list[int] = field(default_factory=list)
    instructions: list[str] = field(default_factory=list)
    instruction_norms: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class NegativeInfo:
    neg_uid: str
    neg_ref: str
    row_id: int
    overlap: float


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize_text(text: str) -> str:
    text = str(text or "").replace("<dia>", ";")
    text = re.sub(r"\.T\b", ".", text)
    text = re.sub(r"\b[Tt]he\s+[Tt]argets\b", "the target's", text)
    text = re.sub(r"\b[Tt]argets\b", "target's", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r";\s*;", ";", text)
    text = text.strip(" ;")
    lowered = text.lower()
    for prefix in TRIM_PREFIXES:
        if lowered.startswith(prefix):
            text = text[len(prefix) :].strip(" .;:")
            break
    return text.strip(" ;")


def normalize_key(text: str) -> str:
    text = normalize_text(text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def split_requirements(text: str) -> list[str]:
    if "<dia>" in str(text):
        raw_parts = str(text).split("<dia>")
    else:
        raw_parts = re.split(r"\s*;\s*", str(text))
    parts = []
    seen = set()
    for part in raw_parts:
        cleaned = normalize_text(part).rstrip(".")
        key = normalize_key(cleaned)
        if cleaned and key and key not in seen:
            parts.append(cleaned)
            seen.add(key)
    return parts


def requirements_block(requirements: list[str]) -> str:
    if not requirements:
        return "- the intended geometric target changes"
    return "\n".join(f"- {req.rstrip('.')}" for req in requirements)


def strip_verify_return_instruction(question: str) -> str:
    return re.sub(
        r"\n+Return exactly one line in this format:\s*\n?Answer:\s*<decision>\s*$",
        "",
        question.strip(),
        flags=re.IGNORECASE,
    ).strip()


def truncate_words(text: str, max_words: int) -> str:
    words = normalize_text(text).split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]).rstrip(" ,;:.") + "..."


def content_tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+(?:'[a-z]+)?", normalize_text(text).lower())
    return {word for word in words if word not in STOPWORDS and len(word) > 2}


def token_overlap(a: str, b: str) -> float:
    a_tokens = content_tokens(a)
    b_tokens = content_tokens(b)
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / max(1, min(len(a_tokens), len(b_tokens)))


def delta_tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+(?:'[a-z]+)?", normalize_text(text).lower())
    tokens = set()
    for word in words:
        if word in NEGATION_WORDS or word.endswith("n't"):
            tokens.add("NEG")
            continue
        if word not in DELTA_STOPWORDS and len(word) > 2:
            tokens.add(word)
    return tokens


def delta_clause_key(text: str) -> str:
    return " ".join(sorted(delta_tokens(text)))


def delta_clause_similarity(a: str, b: str) -> float:
    a_tokens = delta_tokens(a)
    b_tokens = delta_tokens(b)
    if not a_tokens or not b_tokens:
        return 0.0
    token_score = len(a_tokens & b_tokens) / max(1, min(len(a_tokens), len(b_tokens)))
    sequence_score = SequenceMatcher(None, delta_clause_key(a), delta_clause_key(b)).ratio()
    return max(token_score, sequence_score)


def split_delta_fragments(text: str) -> list[str]:
    text = normalize_text(text)
    text = re.sub(r"([a-z])\.([A-Z])", r"\1; \2", text)
    text = re.sub(r"([a-z])\.([a-z])", r"\1 \2", text)
    return [
        fragment
        for fragment in re.split(r"\s*;\s*|\.\s+", text)
        if fragment.strip(" .;:,")
    ]


def clean_delta_clauses(
    requirements: list[str],
    max_clauses: int,
    duplicate_overlap: float,
) -> list[str]:
    """Remove near-duplicate ShapeTalk clauses while preserving concise edits."""
    kept: list[str] = []
    kept_keys: list[str] = []
    for raw in requirements:
        for fragment in split_delta_fragments(raw):
            clause = normalize_text(fragment).rstrip(" .;:,")
            if not clause:
                continue
            key = delta_clause_key(clause)
            if not key:
                continue

            duplicate_idx = None
            for idx, existing in enumerate(kept):
                if key == kept_keys[idx] or delta_clause_similarity(clause, existing) >= duplicate_overlap:
                    duplicate_idx = idx
                    break
            if duplicate_idx is None:
                kept.append(clause)
                kept_keys.append(key)
            else:
                # Prefer the more informative wording when two clauses describe the
                # same edit, e.g. "There are drawers" vs. "It has three drawers".
                if len(delta_tokens(clause)) > len(delta_tokens(kept[duplicate_idx])):
                    kept[duplicate_idx] = clause
                    kept_keys[duplicate_idx] = key

    if max_clauses > 0:
        kept = kept[:max_clauses]
    return kept


def format_delta_caption(clauses: list[str]) -> str:
    return "; ".join(clause.rstrip(" .;:,") for clause in clauses)


def clean_observed_edit(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"^\s*(reason|answer|observed edits?)\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*the candidate (reflects|matches)[^:]*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*candidate does not satisfy the requirement\.?\s*$", "", text, flags=re.IGNORECASE)
    return " ".join(text.replace("\n", " ").split()).strip(" ;.")


def uid_to_npz(uid: str) -> str:
    return f"point_clouds/scaled_to_align_rendering/{uid}.npz"


def load_split_map(path: Path) -> dict[str, str]:
    split_map = {}
    with path.open(encoding="utf-8", newline="") as fp:
        for row in csv.DictReader(fp):
            uid = row.get("model_uid")
            split = row.get("split")
            if uid and split:
                split_map[uid] = split
    return split_map


def class_from_uid(uid: str) -> str:
    return uid.split("/", 1)[0]


def same_known_split(uids: Iterable[str], split_map: dict[str, str]) -> str | None:
    splits = [split_map.get(uid) for uid in uids]
    if not all(splits):
        return None
    return splits[0] if len(set(splits)) == 1 else None


def build_pairs(index_path: Path, split_map: dict[str, str]) -> tuple[dict[tuple[str, str, str], PairInfo], dict]:
    pairs: dict[tuple[str, str, str], PairInfo] = {}
    stats = Counter()
    for row in read_jsonl(index_path):
        stats["rows"] += 1
        if row.get("split") == "ignore":
            stats["drop_source_ignore"] += 1
            continue
        cls = row["class"]
        anchor_uid = row["anchor_uid"]
        pos_uid = row["pos_uid"]
        obj_split = same_known_split((anchor_uid, pos_uid), split_map)
        if obj_split is None:
            stats["drop_missing_or_mixed_anchor_pos_split"] += 1
            continue
        if class_from_uid(anchor_uid) != cls or class_from_uid(pos_uid) != cls:
            stats["drop_class_mismatch"] += 1
            continue

        key = (cls, anchor_uid, pos_uid)
        if key not in pairs:
            pairs[key] = PairInfo(cls=cls, anchor_uid=anchor_uid, pos_uid=pos_uid, split=obj_split)
        pair = pairs[key]
        pair.source_splits[row.get("split")] += 1
        pair.index_ids.append(int(row["id"]))
        for req in split_requirements(row.get("instruction", "")):
            req_key = normalize_key(req)
            if req_key not in pair.instruction_norms:
                pair.instructions.append(req)
                pair.instruction_norms.add(req_key)

    for key in list(pairs):
        if not pairs[key].instructions:
            stats["drop_empty_requirements"] += 1
            del pairs[key]

    return pairs, dict(stats)


def build_negatives(
    negative_path: Path,
    pairs: dict[tuple[str, str, str], PairInfo],
    split_map: dict[str, str],
    max_overlap: float,
    min_negative_tokens: int,
    require_negative_ref_index: bool,
) -> tuple[dict[tuple[str, str, str], NegativeInfo], dict]:
    candidates: dict[tuple[str, str, str], list[NegativeInfo]] = defaultdict(list)
    stats = Counter()
    seen_rows = set()
    for row in read_jsonl(negative_path):
        stats["rows"] += 1
        row_key = (
            row.get("class"),
            row.get("split"),
            row.get("anchor_uid"),
            row.get("pos_uid"),
            tuple(row.get("negatives") or []),
            bool(row.get("neg_from_same_anchor")),
            normalize_key(row.get("neg_ref_utterance") or ""),
        )
        if row_key in seen_rows:
            stats["drop_exact_duplicate"] += 1
            continue
        seen_rows.add(row_key)
        if row.get("split") == "ignore":
            stats["drop_source_ignore"] += 1
            continue
        if not row.get("neg_from_same_anchor"):
            stats["drop_not_same_anchor"] += 1
            continue
        negatives = row.get("negatives") or []
        if len(negatives) != 1:
            stats["drop_negative_count"] += 1
            continue

        cls = row["class"]
        anchor_uid = row["anchor_uid"]
        pos_uid = row["pos_uid"]
        neg_uid = negatives[0]
        if len({anchor_uid, pos_uid, neg_uid}) != 3:
            stats["drop_duplicate_uid"] += 1
            continue
        if any(class_from_uid(uid) != cls for uid in (anchor_uid, pos_uid, neg_uid)):
            stats["drop_class_mismatch"] += 1
            continue
        obj_split = same_known_split((anchor_uid, pos_uid, neg_uid), split_map)
        if obj_split is None:
            stats["drop_missing_or_mixed_triplet_split"] += 1
            continue

        key = (cls, anchor_uid, pos_uid)
        pair = pairs.get(key)
        if pair is None:
            stats["drop_missing_pair"] += 1
            continue

        neg_ref = normalize_text(row.get("neg_ref_utterance", "")).rstrip(".")
        if len(content_tokens(neg_ref)) < min_negative_tokens:
            stats["drop_short_negative_ref"] += 1
            continue
        neg_ref_key = normalize_key(neg_ref)
        if require_negative_ref_index:
            neg_pair = pairs.get((cls, anchor_uid, neg_uid))
            if neg_pair is None or neg_ref_key not in neg_pair.instruction_norms:
                stats["drop_negative_ref_not_in_index"] += 1
                continue
        if normalize_key(neg_ref) in pair.instruction_norms:
            stats["drop_negative_ref_exact_requirement"] += 1
            continue
        overlap = max(token_overlap(neg_ref, req) for req in pair.instructions)
        if overlap >= max_overlap:
            stats["drop_negative_ref_high_overlap"] += 1
            continue

        candidates[key].append(
            NegativeInfo(
                neg_uid=neg_uid,
                neg_ref=neg_ref,
                row_id=int(row["id"]),
                overlap=overlap,
            )
        )

    selected = {}
    for key, options in candidates.items():
        selected[key] = sorted(options, key=lambda item: (item.overlap, item.row_id, item.neg_uid))[0]

    stats["pairs_with_negative"] = len(selected)
    return selected, dict(stats)


def make_verify_sample(
    pair: PairInfo,
    sample_id: str,
    answer: str,
    candidate_uid: str,
    cand_label: str,
    reason: str,
    negative: NegativeInfo | None = None,
    training_view: str = "verify_reason",
) -> dict:
    conversations = [
        {
            "from": "human",
            "value": VERIFY_QUESTION.format(requirements=requirements_block(pair.instructions)),
        },
        {"from": "gpt", "value": f"Answer: {answer}"},
    ]
    if training_view != "verify_answer":
        conversations.extend(
            [
                {"from": "human", "value": VERIFY_REASON_PROMPT},
                {"from": "gpt", "value": f"Reason: {reason}"},
            ]
        )

    triplet_id = f"{pair.cls}|{pair.anchor_uid}|{pair.pos_uid}|{negative.neg_uid if negative else candidate_uid}"
    metadata = {
        "task": "verify",
        "version": DATASET_VERSION,
        "training_view": training_view,
        "class": pair.cls,
        "split": pair.split,
        "pair_id": f"{pair.cls}|{pair.anchor_uid}|{pair.pos_uid}",
        "triplet_id": triplet_id,
        "anchor_npz": uid_to_npz(pair.anchor_uid),
        "candidate_npz": uid_to_npz(candidate_uid),
        "positive_npz": uid_to_npz(pair.pos_uid),
        "negative_npz": uid_to_npz(negative.neg_uid) if negative else None,
        "anchor_uid": pair.anchor_uid,
        "candidate_uid": candidate_uid,
        "positive_uid": pair.pos_uid,
        "negative_uid": negative.neg_uid if negative else None,
        "cand_label": cand_label,
        "requirements_text": "; ".join(pair.instructions),
        "requirements": pair.instructions,
        "answer": answer,
        "reason_gt": reason,
        "source_index_ids": pair.index_ids,
        "source_splits": dict(pair.source_splits),
    }
    if negative:
        metadata.update(
            {
                "negative_bank_id": negative.row_id,
                "negative_ref_utterance": negative.neg_ref,
                "negative_ref_overlap": round(negative.overlap, 4),
            }
        )

    return {
        "id": sample_id,
        "object_ids": [uid_to_npz(pair.anchor_uid), uid_to_npz(candidate_uid)],
        "conversation_type": "simple_description",
        "conversations": conversations,
        "metadata": metadata,
    }


def observed_edits_for_verify(sample: dict, max_clauses: int) -> str:
    meta = sample.get("metadata") or {}
    answer = meta.get("answer")
    if answer == "No":
        observed = clean_observed_edit(meta.get("negative_ref_utterance") or meta.get("reason_gt"))
    else:
        requirements = meta.get("requirements") or []
        if not requirements and meta.get("requirements_text"):
            requirements = [part.strip() for part in str(meta["requirements_text"]).split(";") if part.strip()]
        observed = "; ".join(
            clean_observed_edit(req)
            for req in requirements[:max_clauses]
            if clean_observed_edit(req)
        )
    if not observed:
        observed = clean_observed_edit(meta.get("requirements_text") or "the candidate matches the target requirements")
    return observed


def make_verify_observe_answer_sample(sample: dict, max_clauses: int) -> dict:
    out = copy.deepcopy(sample)
    meta = out.setdefault("metadata", {})
    answer = meta.get("answer")
    if answer not in {"Yes", "No"}:
        raise ValueError(f"verify sample missing Yes/No metadata.answer: {sample.get('id')}")
    question = strip_verify_return_instruction(out["conversations"][0]["value"])
    observed = observed_edits_for_verify(out, max_clauses=max_clauses)
    out["conversations"] = [
        {"from": "human", "value": VERIFY_OBSERVE_PROMPT.format(prefix=question)},
        {"from": "gpt", "value": f"Observed edits: {observed}.\nAnswer: {answer}"},
    ]
    meta["training_view"] = "verify_observe_answer"
    meta["observed_edits_gt"] = observed
    out["id"] = f"{out.get('id', 'sample')}_observe_answer"
    return out


def make_delta_sample(
    pair: PairInfo,
    sample_id: str,
    max_delta_clauses: int,
    delta_duplicate_overlap: float,
) -> dict:
    cleaned_requirements = clean_delta_clauses(
        pair.instructions,
        max_clauses=max_delta_clauses,
        duplicate_overlap=delta_duplicate_overlap,
    )
    delta = format_delta_caption(cleaned_requirements)
    return {
        "id": sample_id,
        "object_ids": [uid_to_npz(pair.anchor_uid), uid_to_npz(pair.pos_uid)],
        "conversation_type": "simple_description",
        "conversations": [
            {"from": "human", "value": DELTA_QUESTION},
            {"from": "gpt", "value": delta},
        ],
        "metadata": {
            "task": "delta_caption",
            "version": DATASET_VERSION,
            "training_view": "delta_caption",
            "class": pair.cls,
            "split": pair.split,
            "pair_id": f"{pair.cls}|{pair.anchor_uid}|{pair.pos_uid}",
            "anchor_npz": uid_to_npz(pair.anchor_uid),
            "candidate_npz": uid_to_npz(pair.pos_uid),
            "anchor_uid": pair.anchor_uid,
            "candidate_uid": pair.pos_uid,
            "positive_uid": pair.pos_uid,
            "cand_label": "positive",
            "requirements_text": "; ".join(pair.instructions),
            "requirements": cleaned_requirements,
            "source_requirements": pair.instructions,
            "delta_gt": delta,
            "delta_clause_count": len(cleaned_requirements),
            "delta_source_clause_count": len(pair.instructions),
            "source_index_ids": pair.index_ids,
            "source_splits": dict(pair.source_splits),
        },
    }


def reason_for_positive(pair: PairInfo) -> str:
    req = truncate_words(pair.instructions[0], 12) if pair.instructions else "the requested target change"
    return f"The candidate matches the requested target changes, including: {req}."


def reason_for_negative(negative: NegativeInfo) -> str:
    ref = truncate_words(negative.neg_ref, 14)
    return f"The candidate reflects a different edit: {ref}."


def sample_items(items: list, limit: int, rng: random.Random) -> list:
    items = list(items)
    rng.shuffle(items)
    if limit and limit > 0:
        items = items[: min(limit, len(items))]
    return items


def repeat_samples(samples: list[dict], repeats: int, repeat_tag: str) -> list[dict]:
    repeated = []
    for repeat_idx in range(max(0, repeats)):
        for sample in samples:
            item = copy.deepcopy(sample)
            item["id"] = f"{sample['id']}__{repeat_tag}{repeat_idx}"
            item.setdefault("metadata", {})["repeat_source_id"] = sample["id"]
            item["metadata"]["repeat_tag"] = repeat_tag
            item["metadata"]["repeat_index"] = repeat_idx
            repeated.append(item)
    return repeated


def sample_items_class_balanced(items: list[tuple[tuple[str, str, str], PairInfo]], limit: int, rng: random.Random) -> list:
    """Sample pair items by round-robin over classes for less biased eval subsets."""
    items = list(items)
    if not limit or limit <= 0 or len(items) <= limit:
        rng.shuffle(items)
        return items

    by_class: dict[str, list[tuple[tuple[str, str, str], PairInfo]]] = defaultdict(list)
    for item in items:
        by_class[item[1].cls].append(item)
    for bucket in by_class.values():
        rng.shuffle(bucket)

    classes = list(by_class)
    rng.shuffle(classes)
    selected = []
    while len(selected) < limit and classes:
        next_classes = []
        for cls in classes:
            bucket = by_class[cls]
            if bucket and len(selected) < limit:
                selected.append(bucket.pop())
            if bucket:
                next_classes.append(cls)
        classes = next_classes
        rng.shuffle(classes)
    return selected


def collect_pair_uids(pairs: dict[tuple[str, str, str], PairInfo], negatives: dict[tuple[str, str, str], NegativeInfo]) -> dict[str, set[str]]:
    by_class: dict[str, set[str]] = defaultdict(set)
    for key, pair in pairs.items():
        by_class[pair.cls].update((pair.anchor_uid, pair.pos_uid))
        negative = negatives.get(key)
        if negative:
            by_class[pair.cls].add(negative.neg_uid)
    return by_class


def build_object_random_split(
    pairs: dict[tuple[str, str, str], PairInfo],
    negatives: dict[tuple[str, str, str], NegativeInfo],
    rng: random.Random,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[dict[str, str], dict]:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if ratio_sum <= 0:
        raise ValueError("object split ratios must sum to a positive value")
    train_ratio /= ratio_sum
    val_ratio /= ratio_sum
    test_ratio /= ratio_sum

    split_map = {}
    stats = {"ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio}, "classes": {}}
    for cls, uids in sorted(collect_pair_uids(pairs, negatives).items()):
        uid_list = sorted(uids)
        rng.shuffle(uid_list)
        n = len(uid_list)
        if n >= 20:
            n_test = max(1, round(n * test_ratio))
            n_val = max(1, round(n * val_ratio))
            if n_test + n_val >= n:
                n_test = max(1, n // 5)
                n_val = max(1, n // 5)
        else:
            n_test = 0
            n_val = 0

        counts = Counter()
        for idx, uid in enumerate(uid_list):
            if idx < n_test:
                split = "test"
            elif idx < n_test + n_val:
                split = "val"
            else:
                split = "train"
            split_map[uid] = split
            counts[split] += 1
        stats["classes"][cls] = dict(sorted(counts.items()))
    return split_map, stats


def same_generated_split(uids: Iterable[str], split_map: dict[str, str]) -> str | None:
    splits = [split_map.get(uid) for uid in uids]
    if not all(splits):
        return None
    return splits[0] if len(set(splits)) == 1 else None


def build_split_items(
    pairs: dict[tuple[str, str, str], PairInfo],
    negatives: dict[tuple[str, str, str], NegativeInfo],
    object_split_map: dict[str, str] | None = None,
) -> tuple[
    dict[str, list[tuple[tuple[str, str, str], PairInfo]]],
    dict[str, list[tuple[tuple[str, str, str], PairInfo]]],
    dict,
]:
    pair_items_by_split: dict[str, list[tuple[tuple[str, str, str], PairInfo]]] = defaultdict(list)
    triplet_items_by_split: dict[str, list[tuple[tuple[str, str, str], PairInfo]]] = defaultdict(list)
    stats = Counter()

    for key, pair in pairs.items():
        if object_split_map is None:
            pair_split = pair.split
        else:
            pair_split = same_generated_split((pair.anchor_uid, pair.pos_uid), object_split_map)
        if pair_split:
            pair.split = pair_split
            pair_items_by_split[pair_split].append((key, pair))
            stats[f"pairs:{pair_split}"] += 1
        else:
            stats["drop_pair_cross_object_split"] += 1

        negative = negatives.get(key)
        if not negative:
            continue
        if object_split_map is None:
            triplet_split = pair.split
        else:
            triplet_split = same_generated_split((pair.anchor_uid, pair.pos_uid, negative.neg_uid), object_split_map)
        if triplet_split:
            pair.split = triplet_split
            triplet_items_by_split[triplet_split].append((key, pair))
            stats[f"triplets:{triplet_split}"] += 1
        else:
            stats["drop_triplet_cross_object_split"] += 1

    return pair_items_by_split, triplet_items_by_split, dict(stats)


def make_verify_samples_for_pairs(
    pair_items: list[tuple[tuple[str, str, str], PairInfo]],
    negatives: dict[tuple[str, str, str], NegativeInfo],
    prefix: str,
    training_view: str,
) -> list[dict]:
    samples = []
    for idx, (key, pair) in enumerate(pair_items):
        negative = negatives[key]
        pos_reason = reason_for_positive(pair)
        neg_reason = reason_for_negative(negative)
        samples.append(
            make_verify_sample(
                pair=pair,
                sample_id=f"{prefix}_verify_yes_{idx:06d}",
                answer="Yes",
                candidate_uid=pair.pos_uid,
                cand_label="positive",
                reason=pos_reason,
                negative=negative,
                training_view=training_view,
            )
        )
        samples.append(
            make_verify_sample(
                pair=pair,
                sample_id=f"{prefix}_verify_no_{idx:06d}",
                answer="No",
                candidate_uid=negative.neg_uid,
                cand_label="negative",
                reason=neg_reason,
                negative=negative,
                training_view=training_view,
            )
        )
    return samples


def validate_samples(samples: list[dict], point_cloud_root: Path | None = None) -> dict:
    stats = Counter()
    missing_paths = []
    for sample in samples:
        stats["samples"] += 1
        conversations = sample.get("conversations") or []
        if len(conversations) not in {2, 4}:
            raise ValueError(f"{sample.get('id')} has unexpected conversation length {len(conversations)}")
        for idx, turn in enumerate(conversations):
            expected = "human" if idx % 2 == 0 else "gpt"
            if turn.get("from") != expected:
                raise ValueError(f"{sample.get('id')} turn {idx} should be {expected}")
        if not sample.get("object_ids"):
            raise ValueError(f"{sample.get('id')} is missing object_ids")
        if sample.get("conversation_type") != "simple_description":
            raise ValueError(f"{sample.get('id')} has unexpected conversation_type")
        task = sample.get("metadata", {}).get("task")
        stats[f"task:{task}"] += 1
        if task == "verify":
            answer = sample["metadata"].get("answer")
            if answer not in {"Yes", "No"}:
                raise ValueError(f"{sample.get('id')} has bad answer {answer!r}")
            stats[f"verify:{answer}"] += 1
        if point_cloud_root:
            for obj_id in sample["object_ids"]:
                path = point_cloud_root / obj_id
                if not path.exists():
                    missing_paths.append(str(path))
                    if len(missing_paths) >= 10:
                        break
        if len(missing_paths) >= 10:
            break
    if missing_paths:
        raise FileNotFoundError("missing point cloud paths, first examples: " + ", ".join(missing_paths))
    return dict(stats)


def summarize(name: str, samples: list[dict]) -> dict:
    task_counts = Counter(sample["metadata"]["task"] for sample in samples)
    answer_counts = Counter(
        sample["metadata"].get("answer")
        for sample in samples
        if sample["metadata"]["task"] == "verify"
    )
    view_counts = Counter(sample["metadata"].get("training_view") for sample in samples)
    class_counts = Counter(sample["metadata"].get("class") for sample in samples)
    object_uids = set()
    delta_clause_counts = []
    delta_source_clause_counts = []
    for sample in samples:
        meta = sample["metadata"]
        if meta.get("task") == "delta_caption":
            if "delta_clause_count" in meta:
                delta_clause_counts.append(meta["delta_clause_count"])
            if "delta_source_clause_count" in meta:
                delta_source_clause_counts.append(meta["delta_source_clause_count"])
        for key in ("anchor_uid", "candidate_uid", "positive_uid", "negative_uid"):
            uid = meta.get(key)
            if uid:
                object_uids.add(uid)
    summary = {
        "name": name,
        "num_samples": len(samples),
        "tasks": dict(sorted(task_counts.items())),
        "verify_answers": dict(sorted((k, v) for k, v in answer_counts.items() if k)),
        "training_views": dict(sorted((k, v) for k, v in view_counts.items() if k)),
        "unique_objects": len(object_uids),
        "top_classes": class_counts.most_common(20),
    }
    if delta_clause_counts:
        summary["delta_clause_count_avg"] = round(sum(delta_clause_counts) / len(delta_clause_counts), 3)
        summary["delta_clause_count_max"] = max(delta_clause_counts)
        summary["delta_source_clause_count_avg"] = round(
            sum(delta_source_clause_counts) / len(delta_source_clause_counts), 3
        )
    return summary


def sample_object_uids(samples: list[dict]) -> set[str]:
    return {
        uid
        for sample in samples
        for uid in (
            sample["metadata"].get("anchor_uid"),
            sample["metadata"].get("candidate_uid"),
            sample["metadata"].get("positive_uid"),
            sample["metadata"].get("negative_uid"),
        )
        if uid
    }


def validate_split_object_disjoint(splits: dict[str, list[dict]]) -> dict[str, int]:
    objects_by_split = {name: sample_object_uids(samples) for name, samples in splits.items()}
    overlap_counts = {}
    names = sorted(objects_by_split)
    for idx, left in enumerate(names):
        for right in names[idx + 1 :]:
            overlap = sorted(objects_by_split[left] & objects_by_split[right])
            overlap_counts[f"{left}/{right}"] = len(overlap)
            if overlap:
                raise ValueError(f"{left}/{right} object overlap detected, first examples: {overlap[:10]}")
    return overlap_counts


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--point_cloud_root", type=Path, default=DEFAULT_POINT_CLOUD_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split_strategy",
        choices=("object_random", "source_map"),
        default="object_random",
        help="object_random builds class-stratified object-disjoint splits; source_map reproduces the ShapeTalk qa_split_map split.",
    )
    parser.add_argument("--train_object_ratio", type=float, default=0.70)
    parser.add_argument("--val_object_ratio", type=float, default=0.15)
    parser.add_argument("--test_object_ratio", type=float, default=0.15)
    parser.add_argument("--train_verify_pairs", type=int, default=9000)
    parser.add_argument("--test_verify_pairs", type=int, default=1000)
    parser.add_argument("--val_verify_pairs", type=int, default=1000)
    parser.add_argument("--train_delta", type=int, default=18000)
    parser.add_argument("--test_delta", type=int, default=2000)
    parser.add_argument("--val_delta", type=int, default=2000)
    parser.add_argument("--eval_verify_pairs", type=int, default=50)
    parser.add_argument("--eval_delta", type=int, default=100)
    parser.add_argument(
        "--max_delta_clauses",
        type=int,
        default=6,
        help="Maximum number of near-deduplicated delta-caption clauses per sample. Use <=0 to keep all.",
    )
    parser.add_argument(
        "--delta_duplicate_overlap",
        type=float,
        default=0.75,
        help="Token/sequence similarity threshold for dropping near-duplicate delta clauses.",
    )
    parser.add_argument(
        "--verify_answer_repeats",
        type=int,
        default=10,
        help="Repeat answer-only verify samples in the training JSON.",
    )
    parser.add_argument(
        "--verify_observe_repeats",
        type=int,
        default=2,
        help="Repeat observe-then-answer verify samples in the training JSON.",
    )
    parser.add_argument(
        "--verify_reason_repeats",
        type=int,
        default=0,
        help="Repeat two-turn verify samples in the training JSON.",
    )
    parser.add_argument(
        "--delta_repeats",
        type=int,
        default=1,
        help="Repeat delta-caption samples in the training JSON.",
    )
    parser.add_argument(
        "--max_observed_clauses",
        type=int,
        default=4,
        help="Maximum number of clauses used in observe-then-answer verify supervision.",
    )
    parser.add_argument("--max_negative_overlap", type=float, default=0.8)
    parser.add_argument("--min_negative_tokens", type=int, default=2)
    parser.add_argument(
        "--allow_unindexed_negative_ref",
        action="store_true",
        help="Do not require neg_ref_utterance to exist as a same-anchor index instruction for the negative UID.",
    )
    parser.add_argument("--skip_point_path_check", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    split_map_path = args.source_dir / "qa_split_map.csv"
    index_path = args.source_dir / "index_shapenet.jsonl"
    negative_path = args.source_dir / "negative_bank_same_anchor.qa.jsonl"

    split_map = load_split_map(split_map_path)
    pairs, pair_stats = build_pairs(index_path, split_map)
    negatives, negative_stats = build_negatives(
        negative_path=negative_path,
        pairs=pairs,
        split_map=split_map,
        max_overlap=args.max_negative_overlap,
        min_negative_tokens=args.min_negative_tokens,
        require_negative_ref_index=not args.allow_unindexed_negative_ref,
    )

    object_split_map = None
    object_split_stats = None
    if args.split_strategy == "object_random":
        object_split_map, object_split_stats = build_object_random_split(
            pairs=pairs,
            negatives=negatives,
            rng=rng,
            train_ratio=args.train_object_ratio,
            val_ratio=args.val_object_ratio,
            test_ratio=args.test_object_ratio,
        )

    pair_items_by_split, triplet_items_by_split, split_item_stats = build_split_items(
        pairs=pairs,
        negatives=negatives,
        object_split_map=object_split_map,
    )

    train_pairs = sample_items(triplet_items_by_split["train"], args.train_verify_pairs, rng)
    val_pairs = sample_items(triplet_items_by_split["val"], args.val_verify_pairs, rng)
    test_pairs = sample_items(triplet_items_by_split["test"], args.test_verify_pairs, rng)

    train_delta_pairs = sample_items(pair_items_by_split["train"], args.train_delta, rng)
    val_delta_pairs = sample_items(pair_items_by_split["val"], args.val_delta, rng)
    test_delta_pairs = sample_items(pair_items_by_split["test"], args.test_delta, rng)

    train_verify_answer = make_verify_samples_for_pairs(
        train_pairs, negatives, prefix="cc_train_answer", training_view="verify_answer"
    )
    train_verify_reason = make_verify_samples_for_pairs(
        train_pairs, negatives, prefix="cc_train_reason", training_view="verify_reason"
    )
    train_verify_observe = [
        make_verify_observe_answer_sample(sample, args.max_observed_clauses)
        for sample in train_verify_reason
    ]
    val_verify_reason = make_verify_samples_for_pairs(
        val_pairs, negatives, prefix="cc_val_reason", training_view="verify_reason"
    )
    test_verify_reason = make_verify_samples_for_pairs(
        test_pairs, negatives, prefix="cc_test_reason", training_view="verify_reason"
    )

    train_delta = [
        make_delta_sample(pair, f"cc_train_delta_{idx:06d}", args.max_delta_clauses, args.delta_duplicate_overlap)
        for idx, (_, pair) in enumerate(train_delta_pairs)
    ]
    val_delta = [
        make_delta_sample(pair, f"cc_val_delta_{idx:06d}", args.max_delta_clauses, args.delta_duplicate_overlap)
        for idx, (_, pair) in enumerate(val_delta_pairs)
    ]
    test_delta = [
        make_delta_sample(pair, f"cc_test_delta_{idx:06d}", args.max_delta_clauses, args.delta_duplicate_overlap)
        for idx, (_, pair) in enumerate(test_delta_pairs)
    ]

    train = (
        repeat_samples(train_verify_answer, args.verify_answer_repeats, "ans")
        + repeat_samples(train_verify_observe, args.verify_observe_repeats, "obs")
        + repeat_samples(train_verify_reason, args.verify_reason_repeats, "reason")
        + repeat_samples(train_delta, args.delta_repeats, "delta")
    )
    val = val_verify_reason + val_delta
    test = test_verify_reason + test_delta
    for split_samples in (train_verify_answer, train_verify_reason, train_verify_observe, train_delta, train, val, test):
        rng.shuffle(split_samples)
    random.Random(args.seed + 7001).shuffle(train)

    eval_pairs = sample_items_class_balanced(test_pairs, args.eval_verify_pairs, rng)
    eval_delta_pairs = sample_items_class_balanced(test_delta_pairs, args.eval_delta, rng)
    eval_subset = make_verify_samples_for_pairs(
        eval_pairs, negatives, prefix="cc_eval_reason", training_view="verify_reason"
    )
    eval_subset += [
        make_delta_sample(pair, f"cc_eval_delta_{idx:06d}", args.max_delta_clauses, args.delta_duplicate_overlap)
        for idx, (_, pair) in enumerate(eval_delta_pairs)
    ]
    rng.shuffle(eval_subset)

    point_cloud_root = None if args.skip_point_path_check else args.point_cloud_root
    validation = {
        "train_verify_answer": validate_samples(train_verify_answer, point_cloud_root),
        "train_verify_observe_answer": validate_samples(train_verify_observe, point_cloud_root),
        "train_verify_reason": validate_samples(train_verify_reason, point_cloud_root),
        "train_delta_caption": validate_samples(train_delta, point_cloud_root),
        "train": validate_samples(train, point_cloud_root),
        "val": validate_samples(val, point_cloud_root),
        "test": validate_samples(test, point_cloud_root),
        "eval_subset": validate_samples(eval_subset, point_cloud_root),
    }

    split_object_overlap = validate_split_object_disjoint({"train": train, "val": val, "test": test})

    write_json(args.output_dir / "train_verify_answer.json", train_verify_answer)
    write_json(args.output_dir / "train_verify_observe_answer.json", train_verify_observe)
    write_json(args.output_dir / "train_verify_reason.json", train_verify_reason)
    write_json(args.output_dir / "train_delta_caption.json", train_delta)
    write_json(args.output_dir / "train.json", train)
    write_json(args.output_dir / "val.json", val)
    write_json(args.output_dir / "test.json", test)
    write_json(args.output_dir / "eval_subset.json", eval_subset)
    write_json(args.output_dir / "all.json", train + val + test)

    stats = {
        "source_dir": str(args.source_dir),
        "output_dir": str(args.output_dir),
        "point_cloud_root": str(args.point_cloud_root),
        "seed": args.seed,
        "dataset_version": DATASET_VERSION,
        "split_strategy": args.split_strategy,
        "object_split_stats": object_split_stats,
        "limits": {
            "train_verify_pairs": args.train_verify_pairs,
            "test_verify_pairs": args.test_verify_pairs,
            "val_verify_pairs": args.val_verify_pairs,
            "train_delta": args.train_delta,
            "test_delta": args.test_delta,
            "val_delta": args.val_delta,
            "eval_verify_pairs": args.eval_verify_pairs,
            "eval_delta": args.eval_delta,
            "max_delta_clauses": args.max_delta_clauses,
            "max_observed_clauses": args.max_observed_clauses,
            "verify_answer_repeats": args.verify_answer_repeats,
            "verify_observe_repeats": args.verify_observe_repeats,
            "verify_reason_repeats": args.verify_reason_repeats,
            "delta_repeats": args.delta_repeats,
        },
        "filters": {
            "max_negative_overlap": args.max_negative_overlap,
            "min_negative_tokens": args.min_negative_tokens,
            "delta_duplicate_overlap": args.delta_duplicate_overlap,
            "require_negative_ref_index": not args.allow_unindexed_negative_ref,
        },
        "split_map_entries": len(split_map),
        "pair_stats": pair_stats,
        "negative_stats": negative_stats,
        "split_item_stats": split_item_stats,
        "available_pairs_by_split": {
            split: len(items) for split, items in sorted(pair_items_by_split.items())
        },
        "available_triplets_by_split": {
            split: len(items) for split, items in sorted(triplet_items_by_split.items())
        },
        "validation": validation,
        "split_object_overlap": split_object_overlap,
        "splits": {
            "train_verify_answer": summarize("train_verify_answer", train_verify_answer),
            "train_verify_observe_answer": summarize("train_verify_observe_answer", train_verify_observe),
            "train_verify_reason": summarize("train_verify_reason", train_verify_reason),
            "train_delta_caption": summarize("train_delta_caption", train_delta),
            "train": summarize("train", train),
            "val": summarize("val", val),
            "test": summarize("test", test),
            "eval_subset": summarize("eval_subset", eval_subset),
        },
    }
    write_json(args.output_dir / "stats.json", stats)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
