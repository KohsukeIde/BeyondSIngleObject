#!/usr/bin/env python3
"""
Evaluate a Multi-3DLLM checkpoint on a QA JSON file.

This script loads the fine-tuned model, runs inference on the provided annotation JSON,
and reports simple text matching metrics (exact / relaxed match) between model outputs
and the reference answers in the dataset.

Supported dataset formats (auto-detected):
1. Objaverse: object_ids are simple IDs, expects {data_path}/{pointnum}_npy/{object_id}_{pointnum}.npy
2. Shape Mating: uses object_ids or objects[*].pc_path entries under --data_path
3. Change Captioning: object_ids contain relative .npz/.npy/.csv point-cloud paths

Examples:
  python -m pointllm.eval.cvpr.eval_cvpr_patch \
    --model_path checkpoints/multi-3dllm \
    --anno_path data/mo3d/test.json \
    --data_path data/point_clouds
"""

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
import torch
from tqdm import tqdm

from pointllm.conversation import conv_templates, SeparatorStyle
from pointllm.utils import disable_torch_init
from pointllm.model_cvpr import PointLLMCVPRLlamaForCausalLM
from pointllm.model import PointLLMLlamaForCausalLM
from pointllm.model.utils import KeywordsStoppingCriteria
from pointllm.data.utils import build_point_token_sequence, pc_norm
from transformers import AutoTokenizer, AutoConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Multi-3DLLM on QA JSON.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned model (e.g. outputs/.../checkpoint).",
    )
    parser.add_argument(
        "--anno_path",
        type=str,
        default="data/mo3d/test.json",
        help="Path to the annotation JSON to evaluate.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/point_clouds",
        help="Root directory that stores point cloud npy files (expects {pointnum}_npy folders).",
    )
    parser.add_argument(
        "--pointnum",
        type=int,
        default=8192,
        help="Number of points per point cloud (used to resolve file names).",
    )
    parser.add_argument(
        "--no_color",
        action="store_true",
        help="Disable RGB channels and use XYZ only.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of tokens to generate per sample.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Use 0 for deterministic decoding.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Nucleus sampling probability.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling parameter.",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.0,
        help="Generation repetition penalty. Values >1 can reduce repeated clauses.",
    )
    parser.add_argument(
        "--no_repeat_ngram_size",
        type=int,
        default=0,
        help="Generation no-repeat ngram size. Use 0 to disable.",
    )
    parser.add_argument(
        "--dedupe_delta_output",
        action="store_true",
        help="Rule-deduplicate semicolon-separated delta-caption outputs after generation.",
    )
    parser.add_argument(
        "--max_delta_output_clauses",
        type=int,
        default=6,
        help="Maximum clauses retained by --dedupe_delta_output. Use <=0 to keep all.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to store evaluation results. Defaults to <model_path>/evaluation.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Optional custom name for the result JSON file.",
    )
    parser.add_argument(
        "--relation_mode",
        type=str,
        default="patch",
        choices=["object", "patch", "micro", "fast_patch"],
        help="Force the relation mode used by the model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Limit the number of evaluation samples (use <=0 for all).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--as_4d",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--reverse_order",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no_pc",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--model_variant",
        type=str,
        default="auto",
        choices=["auto", "cvpr", "original"],
        help="Model variant: auto-detect (default), force Multi-3DLLM, or force original PointLLM.",
    )
    parser.add_argument(
        "--relation_gamma",
        type=float,
        default=None,
        help="Patch relation residual scale (gamma). Only used for relation-aware checkpoints.",
    )
    parser.add_argument(
        "--disable_relation_inference",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--debug_feats",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--select_one_mode",
        action="store_true",
        help="Change 'Select all that apply' to 'Select one' for shape mating tasks.",
    )
    parser.add_argument(
        "--multi_turn",
        action="store_true",
        help="Enable multi-turn inference to capture reasoning after answer.",
    )
    parser.add_argument(
        "--score_verify_options",
        action="store_true",
        help="For verify samples, choose Yes/No by conditional log-likelihood instead of free generation.",
    )
    parser.add_argument(
        "--reasoning_prompt",
        type=str,
        default="Briefly explain in ONE sentence. Do NOT mention other options.",
        help="Prompt to use for reasoning turn in multi-turn mode.",
    )
    parser.add_argument(
        "--verify_reasoning_prompt",
        type=str,
        default="Briefly explain in ONE sentence. Do not change the answer.",
        help="Prompt to use for the reasoning turn on two-stage verify samples.",
    )
    return parser.parse_args()


@dataclass
class Sample:
    index: int
    object_ids: List[str]
    point_clouds: List[torch.Tensor]
    question: str
    answer: str
    metadata: Dict


class CVPRPatchEvalDataset:
    """Lightweight dataset wrapper that keeps JSON metadata while loading point clouds."""

    def __init__(
        self,
        anno_path: str,
        data_path: str,
        pointnum: int = 8192,
        use_color: bool = True,
        debug: bool = False,
        select_one_mode: bool = False,
    ) -> None:
        with open(anno_path, "r") as fp:
            self.samples = json.load(fp)
        self.data_path = data_path
        self.pointnum = pointnum
        self.use_color = use_color
        self.debug = bool(debug)
        self.select_one_mode = bool(select_one_mode)
        
        # Auto-detect dataset format
        self.dataset_format = self._detect_dataset_format()
        if self.debug:
            print(f"[DEBUG] Detected dataset format: {self.dataset_format}")
            if self.select_one_mode:
                print(f"[DEBUG] Select one mode enabled (will rewrite 'Select all' to 'Select one')")

    def __len__(self) -> int:
        return len(self.samples)

    def _detect_dataset_format(self) -> str:
        """
        Detect dataset format from the first sample.
        Returns: 'shape_mating', 'change_captioning', or 'objaverse'
        """
        if not self.samples:
            return "objaverse"
        
        first_sample = self.samples[0]
        
        # Check for shape_mating format (has 'objects' field with 'pc_path')
        if "objects" in first_sample and isinstance(first_sample["objects"], list):
            if len(first_sample["objects"]) > 0 and "pc_path" in first_sample["objects"][0]:
                return "shape_mating"
        
        # Check for path-list formats (change captioning, sample CSVs, or released
        # point clouds stored as relative files under --data_path).
        if "object_ids" in first_sample and isinstance(first_sample["object_ids"], list):
            if len(first_sample["object_ids"]) > 0:
                obj_id = first_sample["object_ids"][0]
                if "/" in obj_id or obj_id.endswith((".npz", ".npy", ".csv")):
                    return "change_captioning"
        
        # Default to objaverse format
        return "objaverse"

    def _resolve_pointcloud_path(self, object_id: str, entry: Optional[Dict] = None, obj_index: int = 0) -> str:
        """
        Resolve point cloud path based on dataset format.
        
        Args:
            object_id: Object ID or path from object_ids
            entry: Full sample entry (needed for shape_mating)
            obj_index: Index in object list (needed for shape_mating)
        """
        if self.dataset_format == "shape_mating":
            # Use pc_path from objects field
            if entry is None:
                raise ValueError("entry is required for shape_mating format")
            objects = entry.get("objects", [])
            if obj_index >= len(objects):
                raise IndexError(f"obj_index {obj_index} out of range for objects list")
            pc_path = objects[obj_index].get("pc_path", "")
            candidates = [os.path.join(self.data_path, pc_path)]
            source = objects[obj_index].get("source", {})
            dataset_name = str(source.get("dataset", "")).strip()
            if pc_path.startswith("shapemating/thingi10k/"):
                candidates.append(os.path.join(self.data_path, pc_path[len("shapemating/thingi10k/"):]))
            if pc_path.startswith("shapemating/"):
                stripped = pc_path[len("shapemating/"):]
                candidates.append(os.path.join(self.data_path, stripped))
                if dataset_name and not stripped.startswith(f"{dataset_name}/"):
                    candidates.append(os.path.join(self.data_path, "shapemating", dataset_name, stripped))
                    candidates.append(os.path.join(self.data_path, dataset_name, stripped))
            return next((candidate for candidate in candidates if os.path.exists(candidate)), candidates[0])
        
        elif self.dataset_format == "change_captioning":
            # object_id is already a relative path like "point_clouds/scaled_to_align_rendering/table/..."
            candidates = [os.path.join(self.data_path, object_id)]
            if object_id.startswith("point_clouds/"):
                candidates.append(os.path.join(self.data_path, object_id[len("point_clouds/"):]))
            if object_id.startswith("point_clouds/scaled_to_align_rendering/"):
                candidates.append(
                    os.path.join(
                        self.data_path,
                        object_id[len("point_clouds/scaled_to_align_rendering/"):],
                    )
                )
            return next((candidate for candidate in candidates if os.path.exists(candidate)), candidates[0])
        
        else:  # objaverse format (default)
            filename = f"{object_id}_{self.pointnum}.npy"
            return os.path.join(self.data_path, f"{self.pointnum}_npy", filename)

    def _load_point_cloud(self, object_id: str, entry: Optional[Dict] = None, obj_index: int = 0) -> np.ndarray:
        npy_path = self._resolve_pointcloud_path(object_id, entry, obj_index)
        if not os.path.exists(npy_path):
            raise FileNotFoundError(f"Point cloud file not found: {npy_path}")
        
        # Handle .npz files (used in change_captioning dataset)
        if npy_path.endswith('.npz'):
            with np.load(npy_path) as data:
                # Try common keys for point cloud data
                if 'points' in data:
                    pc = data['points']
                elif 'xyz' in data:
                    pc = data['xyz']
                elif 'arr_0' in data:
                    pc = data['arr_0']
                else:
                    # Get the first array if key is unknown
                    keys = list(data.keys())
                    if self.debug:
                        print(f"[DEBUG] .npz file keys: {keys}, using first key: {keys[0]}")
                    pc = data[keys[0]]
        elif npy_path.endswith('.csv'):
            pc = np.loadtxt(npy_path, delimiter=",")
        else:
            pc = np.load(npy_path)

        if self.pointnum and pc.shape[0] != self.pointnum:
            seed_bytes = hashlib.sha1(str(npy_path).encode("utf-8")).digest()[:8]
            seed = int.from_bytes(seed_bytes, byteorder="little", signed=False)
            rng = np.random.default_rng(seed)
            replace = pc.shape[0] < self.pointnum
            indices = rng.choice(pc.shape[0], self.pointnum, replace=replace)
            pc = pc[indices]
        
        pc = pc_norm(pc)
        if not self.use_color:
            pc = pc[:, :3]
        elif pc.shape[1] < 6:
            # Padding RGB with 0.5 if color missing.
            num_points = pc.shape[0]
            rgb = np.full((num_points, 3), 0.5, dtype=pc.dtype)
            pc = np.concatenate([pc[:, :3], rgb], axis=1)
        else:
            pc = pc[:, :6]
        return pc.astype(np.float32)

    def _extract_qa(self, conversation: List[Dict]) -> Tuple[str, str]:
        question = None
        answer = None
        for turn in conversation:
            if turn["from"] == "human" and question is None:
                question = turn["value"]
            elif turn["from"] == "gpt" and question is not None:
                answer = turn["value"]
                break
        if question is None or answer is None:
            raise ValueError("Conversation does not contain a single human->gpt pair.")
        return question, answer
    
    def _rewrite_select_one(self, question: str) -> str:
        """Rewrite 'Select all that apply' to 'Select one' for shape mating tasks."""
        import re
        # Pattern to match "Which pairs can mate? Select all that apply."
        # Handle variations in whitespace and case
        pattern = r'(Which\s+pairs?\s+can\s+mate\?)\s*Select\s+all\s+that\s+apply\.'
        replacement = r'\1 Select one.'
        
        rewritten = re.sub(pattern, replacement, question, flags=re.IGNORECASE)
        
        # Also handle "pairs" -> "pair" in the question itself
        if rewritten != question:
            rewritten = re.sub(r'\bWhich\s+pairs\s+can\s+mate\?', 'Which pair can mate?', rewritten, flags=re.IGNORECASE)
        
        if self.debug and rewritten != question:
            print(f"[DEBUG] Rewrote question:")
            print(f"  Before: {question[:100]}...")
            print(f"  After:  {rewritten[:100]}...")
        
        return rewritten

    def __getitem__(self, index: int) -> Sample:
        entry = self.samples[index]
        raw_ids = entry.get("object_ids", []) or []
        # Check both root level and metadata for gpt_input_order
        gpt_order = entry.get("gpt_input_order") or entry.get("metadata", {}).get("gpt_input_order")
        if gpt_order:
            object_ids = list(gpt_order)
            if self.debug and raw_ids and object_ids != raw_ids:
                print(
                    f"[DEBUG] sample {index}: object_ids reordered "
                    "(metadata.gpt_input_order overrides original object_ids)."
                )
        else:
            object_ids = raw_ids
        conversation = entry.get("conversations", [])
        metadata = entry.get("metadata", {})
        
        # Include answer field in metadata if present (for shape_mating evaluation)
        if "answer" in entry:
            metadata["answer"] = entry["answer"]

        question, answer = self._extract_qa(conversation)
        
        # Apply select_one_mode: rewrite "Select all that apply" to "Select one"
        if self.select_one_mode:
            question = self._rewrite_select_one(question)

        # Load point clouds with format-specific handling
        point_clouds = [
            torch.from_numpy(self._load_point_cloud(object_id, entry, obj_idx))
            for obj_idx, object_id in enumerate(object_ids)
        ]

        return Sample(
            index=index,
            object_ids=object_ids,
            point_clouds=point_clouds,
            question=question,
            answer=answer,
            metadata=metadata,
        )


def replace_point_tokens(
    text: str,
    point_backbone_config: Dict,
    num_point_clouds: int = 1,
) -> str:
    """Replace <point> placeholders with the proper special token sequence."""
    single_region = build_point_token_sequence(point_backbone_config, 1)
    multi_region = build_point_token_sequence(point_backbone_config, num_point_clouds)
    placeholder_count = text.count("<point>")
    if placeholder_count == 1:
        return text.replace("<point>", multi_region)
    return text.replace("<point>", single_region)


def count_point_clouds(point_clouds: Optional[Sequence[Sequence[torch.Tensor]]]) -> int:
    if point_clouds is None or len(point_clouds) == 0:
        return 0
    first = point_clouds[0]
    return len(first) if isinstance(first, (list, tuple)) else 1


def normalize_text(text: str) -> str:
    return " ".join(text.strip().lower().split())


DELTA_OUTPUT_STOPWORDS = {
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


def delta_output_tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())
    tokens = set()
    for word in words:
        if word in {"no", "not", "without", "none", "never"} or word.endswith("n't"):
            tokens.add("NEG")
        elif word not in DELTA_OUTPUT_STOPWORDS and len(word) > 2:
            tokens.add(word)
    return tokens


def delta_output_similarity(a: str, b: str) -> float:
    a_tokens = delta_output_tokens(a)
    b_tokens = delta_output_tokens(b)
    if not a_tokens or not b_tokens:
        return 0.0
    token_score = len(a_tokens & b_tokens) / max(1, min(len(a_tokens), len(b_tokens)))
    sequence_score = SequenceMatcher(None, " ".join(sorted(a_tokens)), " ".join(sorted(b_tokens))).ratio()
    return max(token_score, sequence_score)


def dedupe_delta_caption_output(text: str, max_clauses: int = 6, duplicate_overlap: float = 0.75) -> str:
    clauses = [clause.strip(" .;:,") for clause in re.split(r"[.;]\s*", text or "") if clause.strip(" .;:,")]
    kept: list[str] = []
    for clause in clauses:
        duplicate_idx = None
        for idx, existing in enumerate(kept):
            if delta_output_similarity(clause, existing) >= duplicate_overlap:
                duplicate_idx = idx
                break
        if duplicate_idx is None:
            kept.append(clause)
        elif len(delta_output_tokens(clause)) > len(delta_output_tokens(kept[duplicate_idx])):
            kept[duplicate_idx] = clause
    if max_clauses > 0:
        kept = kept[:max_clauses]
    return "; ".join(kept)


def extract_sm_choice(text: str) -> Optional[str]:
    """Extract Shape Mating pair choice from text.
    Returns: '(1,2)', '(1,3)', '(2,3)', 'None', or None if not found.
    """
    import re
    choice_to_pair = {"A": "(1,2)", "B": "(1,3)", "C": "(2,3)", "D": "None"}
    choice_match = re.search(r'\banswer\s*:\s*([ABCD])\b', text, flags=re.IGNORECASE)
    if choice_match:
        return choice_to_pair[choice_match.group(1).upper()]
    choice_match = re.match(r'\s*([ABCD])\b', text, flags=re.IGNORECASE)
    if choice_match:
        return choice_to_pair[choice_match.group(1).upper()]
    # Look for pair patterns like (1,2), (1,3), (2,3) or None
    match = re.search(r'\((\d),(\d)\)|None', text, flags=re.IGNORECASE)
    return match.group(0) if match else None


def is_shape_mating_task(question: str) -> bool:
    """Detect if this is a Shape Mating task based on question content."""
    sm_keywords = [
        "mate", "interlock", "pair", "(1,2)", "(1,3)", "(2,3)",
        "Options:", "which pair"
    ]
    q_lower = question.lower()
    return any(kw.lower() in q_lower for kw in sm_keywords)


def get_task_type(sample: Sample) -> str:
    metadata = sample.metadata or {}
    task = metadata.get("task") or metadata.get("task_type") or "unknown"
    return str(task)


def uses_multi_turn_inference(sample: Sample) -> bool:
    task = get_task_type(sample)
    return task in {"verify", "shape_mating"} or is_shape_mating_task(sample.question)


def _detect_variant(cfg: AutoConfig, requested: str) -> str:
    if requested != "auto":
        return requested
    archs = getattr(cfg, "architectures", []) or []
    arch_hint = any("CVPR" in arch or "PointLLMCVPR" in arch for arch in archs)
    cfg_hint = bool(getattr(cfg, "cvpr_use_relation_module", False) or getattr(cfg, "cvpr_relation_mode", None))
    return "cvpr" if arch_hint or cfg_hint else "original"


def _log_special_tokens(tokenizer, point_backbone_config: Optional[Dict], debug: bool) -> None:
    if not debug or point_backbone_config is None:
        return
    tokens = []
    for key in ("default_point_patch_token", "default_point_start_token", "default_point_end_token"):
        tok = point_backbone_config.get(key, None)
        if tok:
            tokens.append(tok)
    # Include backward-compat keys for clarity
    extra = ["<point>", "<patch>"]
    tokens.extend(extra)
    token_ids = {tok: tokenizer.convert_tokens_to_ids(tok) for tok in tokens}
    print(f"[DEBUG] special token ids (requested): {token_ids}")


def prepare_model(
    model_path: str,
    device: str,
    relation_mode: str = "patch",
    model_variant: str = "auto",
    debug: bool = False,
    relation_gamma: Optional[float] = None,
    disable_relation_inference: bool = False,
) -> Tuple:
    disable_torch_init()
    cfg = AutoConfig.from_pretrained(model_path)
    detected_variant = _detect_variant(cfg, model_variant)
    print(f"[INFO] Detected/selected model_variant={detected_variant}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if detected_variant == "cvpr":
        model = PointLLMCVPRLlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
    else:
        model = PointLLMLlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
    print(f"[DEBUG] model class = {model.__class__.__name__}")
    model = model.to(device)
    model.eval()
    model.config.use_cache = True
    if hasattr(model, "generation_config"):
        model.generation_config.use_cache = True
    model.initialize_tokenizer_point_backbone_config_wo_embedding(tokenizer)
    point_backbone_config = model.get_model().point_backbone_config if hasattr(model, "get_model") else None
    _log_special_tokens(tokenizer, point_backbone_config, debug)
    if detected_variant == "cvpr":
        # Force relation mode to match training setup (defaults to 'patch')
        try:
            model.config.cvpr_relation_mode = relation_mode
            inner = getattr(model, "get_model", lambda: None)()
            if inner is not None and hasattr(inner, "relation_mode"):
                inner.relation_mode = relation_mode
            print(f"[INFO] Using relation_mode={relation_mode}")
        except Exception as e:
            print(f"[WARN] Failed to set relation_mode: {e}")
        # Optionally set relation gamma (patch residual scale)
        try:
            if relation_gamma is not None:
                model.config.cvpr_relation_patch_gamma = float(relation_gamma)
                inner = getattr(model, "get_model", lambda: None)()
                if inner is not None and hasattr(inner, "set_relation_patch_gamma"):
                    inner.set_relation_patch_gamma(float(relation_gamma))
                elif inner is not None and hasattr(inner, "relation_gamma"):
                    # Fallback: set attribute directly
                    inner.relation_gamma = float(relation_gamma)
                print(f"[INFO] Using relation_gamma={relation_gamma}")
        except Exception as e:
            print(f"[WARN] Failed to set relation_gamma: {e}")
        # Optionally disable relation module during inference
        if disable_relation_inference:
            try:
                inner = getattr(model, "get_model", lambda: None)()
                if inner is not None:
                    inner.relation_module = None
                    print("[INFO] Disabled relation module for inference (--disable_relation_inference)")
            except Exception as e:
                print(f"[WARN] Failed to disable relation module: {e}")
    else:
        if relation_mode != "patch":
            print("[INFO] relation_mode is ignored for original PointLLM model.")
        if relation_gamma is not None:
            print("[INFO] relation_gamma is ignored for original PointLLM model.")
        if disable_relation_inference:
            print("[INFO] --disable_relation_inference is only applicable to relation-aware checkpoints.")
    conv = conv_templates["vicuna_v1_1"].copy()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    point_backbone_config = model.get_model().point_backbone_config
    point_backbone = getattr(model.get_model(), "point_backbone", None)
    point_dtype = (
        next(point_backbone.parameters()).dtype if point_backbone is not None else model.dtype
    )
    return model, tokenizer, conv, stop_str, point_backbone_config, point_dtype, detected_variant


def generate_answer(
    model,
    tokenizer,
    conv_template,
    stop_str: str,
    point_backbone_config: Dict,
    question: str,
    point_clouds: Optional[Sequence[Sequence[torch.Tensor]]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    debug: bool = False,
) -> str:
    conv = conv_template.copy()
    # If no point clouds are provided (ablation), do NOT inject point patch tokens
    # to avoid feeding thousands of special tokens without features.
    if point_clouds is None:
        prompt_text = question.replace("<point>", "").strip()
    else:
        prompt_text = replace_point_tokens(
            question,
            point_backbone_config,
            num_point_clouds=count_point_clouds(point_clouds),
        )
    conv.append_message(conv.roles[0], prompt_text)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    inputs = tokenizer([prompt], return_tensors="pt")
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    if debug:
        try:
            ids = input_ids[0]
            max_pos = getattr(model.config, "max_position_embeddings", 2048)
            ps_id = point_backbone_config.get("point_start_token", None)
            pe_id = point_backbone_config.get("point_end_token", None)
            pp_id = point_backbone_config.get("point_patch_token", None)
            n_ps = int((ids == ps_id).sum().item()) if ps_id is not None else -1
            n_pe = int((ids == pe_id).sum().item()) if pe_id is not None else -1
            n_pp = int((ids == pp_id).sum().item()) if pp_id is not None else -1
            print(f"[DEBUG] prompt_len={ids.numel()}  max_pos={max_pos}  <start>={n_ps}  <end>={n_pe}  <patch>={n_pp}")
            starts = (input_ids[0] == ps_id).sum().item() if ps_id is not None else 0
            ends = (input_ids[0] == pe_id).sum().item() if pe_id is not None else 0
            patches = (input_ids[0] == pp_id).sum().item() if pp_id is not None else 0
            print(f"[DEBUG] token counts: <start>={starts}, <end>={ends}, <patch>={patches}")
            if point_clouds is not None and isinstance(point_clouds, list) and len(point_clouds) > 0:
                num_pc = len(point_clouds[0])
                dtype0 = point_clouds[0][0].dtype
                shape0 = tuple(point_clouds[0][0].shape)
            else:
                num_pc, dtype0, shape0 = 0, 'NA', 'NA'
            print(f"[DEBUG] num_point_clouds={num_pc}, dtype={dtype0}, shape0={shape0}")
        except Exception as e:
            print(f"[DEBUG] token/pc debug failed: {e}")

    stop_words = [stop_str]
    if getattr(tokenizer, 'eos_token', None):
        stop_words.append(tokenizer.eos_token)
    stopping_criteria = KeywordsStoppingCriteria(stop_words, tokenizer, input_ids)
    do_sample = temperature > 0

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            point_clouds=point_clouds,
            do_sample=do_sample,
            temperature=max(temperature, 1e-5) if do_sample else None,
            top_p=top_p if do_sample else None,
            top_k=top_k if do_sample else None,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=max(0, no_repeat_ngram_size),
            max_new_tokens=max_new_tokens,
            stopping_criteria=[stopping_criteria],
            use_cache=True,
        )

    generated = tokenizer.decode(
        output_ids[0, input_ids.size(1) :], skip_special_tokens=True
    ).strip()
    if generated.endswith(stop_str):
        generated = generated[: -len(stop_str)].strip()
    return generated


def generate_answer_multi_turn(
    model,
    tokenizer,
    conv_template,
    stop_str: str,
    point_backbone_config: Dict,
    question: str,
    point_clouds: Optional[Sequence[Sequence[torch.Tensor]]],
    reasoning_prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    debug: bool = False,
) -> Tuple[str, str]:
    """
    Generate answer in multi-turn mode: first the answer, then the reasoning.
    Returns (answer, reasoning) tuple.
    """
    conv = conv_template.copy()
    
    # If no point clouds are provided, do NOT inject point patch tokens
    if point_clouds is None:
        prompt_text = question.replace("<point>", "").strip()
    else:
        prompt_text = replace_point_tokens(
            question,
            point_backbone_config,
            num_point_clouds=count_point_clouds(point_clouds),
        )
    
    # Turn 1: Ask question
    conv.append_message(conv.roles[0], prompt_text)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    inputs = tokenizer([prompt], return_tensors="pt")
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    
    stop_words = [stop_str]
    if getattr(tokenizer, 'eos_token', None):
        stop_words.append(tokenizer.eos_token)
    stopping_criteria = KeywordsStoppingCriteria(stop_words, tokenizer, input_ids)
    do_sample = temperature > 0
    
    # Generate first answer
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            point_clouds=point_clouds,
            do_sample=do_sample,
            temperature=max(temperature, 1e-5) if do_sample else None,
            top_p=top_p if do_sample else None,
            top_k=top_k if do_sample else None,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=max(0, no_repeat_ngram_size),
            max_new_tokens=max_new_tokens,
            stopping_criteria=[stopping_criteria],
            use_cache=True,
        )
    
    answer = tokenizer.decode(
        output_ids[0, input_ids.size(1):], skip_special_tokens=True
    ).strip()
    if answer.endswith(stop_str):
        answer = answer[: -len(stop_str)].strip()
    
    # Turn 2: Ask for reasoning
    # Update conversation with the answer
    conv.messages[-1][1] = answer
    conv.append_message(conv.roles[0], reasoning_prompt)
    conv.append_message(conv.roles[1], None)
    prompt2 = conv.get_prompt()
    
    inputs2 = tokenizer([prompt2], return_tensors="pt")
    input_ids2 = inputs2.input_ids.to(model.device)
    attention_mask2 = inputs2.attention_mask.to(model.device)
    
    stopping_criteria2 = KeywordsStoppingCriteria(stop_words, tokenizer, input_ids2)
    
    # Generate reasoning (still need point clouds for context)
    with torch.inference_mode():
        output_ids2 = model.generate(
            input_ids=input_ids2,
            attention_mask=attention_mask2,
            point_clouds=point_clouds,  # Pass point clouds again for reasoning
            do_sample=do_sample,
            temperature=max(temperature, 1e-5) if do_sample else None,
            top_p=top_p if do_sample else None,
            top_k=top_k if do_sample else None,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=max(0, no_repeat_ngram_size),
            max_new_tokens=max_new_tokens,
            stopping_criteria=[stopping_criteria2],
            use_cache=True,
        )
    
    reasoning = tokenizer.decode(
        output_ids2[0, input_ids2.size(1):], skip_special_tokens=True
    ).strip()
    if reasoning.endswith(stop_str):
        reasoning = reasoning[: -len(stop_str)].strip()
    
    if debug:
        print(f"[DEBUG Multi-turn] Answer: {answer}")
        print(f"[DEBUG Multi-turn] Reasoning: {reasoning}")
    
    return answer, reasoning


def build_generation_prompt(
    conv_template,
    point_backbone_config: Dict,
    question: str,
    point_clouds: Optional[Sequence[Sequence[torch.Tensor]]],
) -> str:
    conv = conv_template.copy()
    if point_clouds is None:
        prompt_text = question.replace("<point>", "").strip()
    else:
        prompt_text = replace_point_tokens(
            question,
            point_backbone_config,
            num_point_clouds=count_point_clouds(point_clouds),
        )
    conv.append_message(conv.roles[0], prompt_text)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def score_continuation(
    model,
    tokenizer,
    prompt: str,
    continuation: str,
    point_clouds: Optional[Sequence[Sequence[torch.Tensor]]],
) -> float:
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    cont_ids = tokenizer(continuation, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
    if cont_ids.numel() == 0:
        return float("-inf")

    input_ids = torch.cat([prompt_ids, cont_ids], dim=1)
    attention_mask = torch.ones_like(input_ids, device=model.device)
    prompt_len = prompt_ids.size(1)

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            point_clouds=point_clouds,
            use_cache=False,
        )
    logits = (outputs.logits if hasattr(outputs, "logits") else outputs[0])[:, prompt_len - 1 : -1, :]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    token_log_probs = log_probs.gather(2, cont_ids.unsqueeze(-1)).squeeze(-1)
    return float(token_log_probs.mean().item())


def score_verify_answer(
    model,
    tokenizer,
    conv_template,
    point_backbone_config: Dict,
    question: str,
    point_clouds: Optional[Sequence[Sequence[torch.Tensor]]],
    reference: str,
) -> str:
    prompt = build_generation_prompt(conv_template, point_backbone_config, question, point_clouds)
    answer_prefix = "Answer: " if re.search(r"\bAnswer\s*:", reference or question, re.IGNORECASE) else ""
    candidates = {
        "Yes": f"{answer_prefix}Yes",
        "No": f"{answer_prefix}No",
    }
    scores = {
        label: score_continuation(model, tokenizer, prompt, text, point_clouds)
        for label, text in candidates.items()
    }
    label = max(scores, key=scores.get)
    return candidates[label]


def main() -> None:
    args = parse_args()
    if args.debug_feats:
        os.environ["POINTLLM_DEBUG_FEATS"] = "1"
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else os.path.join(args.model_path, "evaluation")
    )
    os.makedirs(output_dir, exist_ok=True)

    if args.output_name:
        output_name = args.output_name
    else:
        base = os.path.splitext(os.path.basename(args.anno_path))[0]
        output_name = f"{base}_multi3dllm_eval.json"
    output_path = os.path.join(output_dir, output_name)

    device = torch.device(args.device)
    (
        model,
        tokenizer,
        conv_template,
        stop_str,
        point_backbone_config,
        point_backbone_dtype,
        model_variant,
    ) = prepare_model(
        args.model_path, device, args.relation_mode, args.model_variant, args.debug, args.relation_gamma, args.disable_relation_inference
    )

    dataset = CVPRPatchEvalDataset(
        anno_path=args.anno_path,
        data_path=args.data_path,
        pointnum=args.pointnum,
        use_color=not args.no_color,
        debug=args.debug,
        select_one_mode=args.select_one_mode,
    )

    if args.limit is None or args.limit <= 0:
        total_samples = len(dataset)
    else:
        total_samples = min(args.limit, len(dataset))
    print(f"[INFO] Evaluating {total_samples} / {len(dataset)} samples.")

    results = []
    exact_matches = 0
    relaxed_matches = 0

    for idx in tqdm(range(total_samples), desc="Evaluating"):
        sample = dataset[idx]
        pcs = sample.point_clouds
        if args.reverse_order:
            pcs = list(reversed(pcs))

        if args.no_pc:
            point_clouds = None
        elif args.as_4d:
            # Pack as (1, M, N, C) to mimic eval_modelnet_multi
            stacked = torch.stack(
                [pc.to(device=device, dtype=point_backbone_dtype) for pc in pcs], dim=0
            ).unsqueeze(0)
            point_clouds = stacked
        else:
            # Keep list-of-tensors structure (training pipeline compatible)
            point_clouds = [[pc.to(device=device, dtype=point_backbone_dtype) for pc in pcs]]

        task_type = get_task_type(sample)
        # Check if this is a Shape Mating task
        is_sm_task = is_shape_mating_task(sample.question)
        
        # Generate prediction (with or without multi-turn reasoning)
        reasoning = None
        if args.score_verify_options and task_type == "verify":
            prediction = score_verify_answer(
                model=model,
                tokenizer=tokenizer,
                conv_template=conv_template,
                point_backbone_config=point_backbone_config,
                question=sample.question,
                point_clouds=point_clouds,
                reference=sample.answer,
            )
            if args.multi_turn:
                reasoning_prompt = args.verify_reasoning_prompt
                conv = conv_template.copy()
                if point_clouds is None:
                    prompt_text = sample.question.replace("<point>", "").strip()
                else:
                    prompt_text = replace_point_tokens(
                        sample.question,
                        point_backbone_config,
                        num_point_clouds=count_point_clouds(point_clouds),
                    )
                conv.append_message(conv.roles[0], prompt_text)
                conv.append_message(conv.roles[1], prediction)
                conv.append_message(conv.roles[0], reasoning_prompt)
                conv.append_message(conv.roles[1], None)
                prompt2 = conv.get_prompt()
                inputs2 = tokenizer([prompt2], return_tensors="pt")
                input_ids2 = inputs2.input_ids.to(model.device)
                attention_mask2 = inputs2.attention_mask.to(model.device)
                stop_words = [stop_str]
                if getattr(tokenizer, "eos_token", None):
                    stop_words.append(tokenizer.eos_token)
                stopping_criteria2 = KeywordsStoppingCriteria(stop_words, tokenizer, input_ids2)
                with torch.inference_mode():
                    output_ids2 = model.generate(
                        input_ids=input_ids2,
                        attention_mask=attention_mask2,
                        point_clouds=point_clouds,
                        do_sample=args.temperature > 0,
                        temperature=max(args.temperature, 1e-5) if args.temperature > 0 else None,
                        top_p=args.top_p if args.temperature > 0 else None,
                        top_k=args.top_k if args.temperature > 0 else None,
                        repetition_penalty=args.repetition_penalty,
                        no_repeat_ngram_size=max(0, args.no_repeat_ngram_size),
                        max_new_tokens=args.max_new_tokens,
                        stopping_criteria=[stopping_criteria2],
                        use_cache=True,
                    )
                reasoning = tokenizer.decode(
                    output_ids2[0, input_ids2.size(1) :], skip_special_tokens=True
                ).strip()
                if reasoning.endswith(stop_str):
                    reasoning = reasoning[: -len(stop_str)].strip()
        elif args.multi_turn and uses_multi_turn_inference(sample):
            reasoning_prompt = (
                args.verify_reasoning_prompt
                if task_type == "verify"
                else args.reasoning_prompt
            )
            prediction, reasoning = generate_answer_multi_turn(
                model=model,
                tokenizer=tokenizer,
                conv_template=conv_template,
                stop_str=stop_str,
                point_backbone_config=point_backbone_config,
                question=sample.question,
                point_clouds=point_clouds,
                reasoning_prompt=reasoning_prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                debug=args.debug and idx < 3,
            )
        else:
            prediction = generate_answer(
                model=model,
                tokenizer=tokenizer,
                conv_template=conv_template,
                stop_str=stop_str,
                point_backbone_config=point_backbone_config,
                question=sample.question,
                point_clouds=point_clouds,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                debug=args.debug and idx < 3,
            )

        if args.dedupe_delta_output and task_type == "delta_caption":
            prediction = dedupe_delta_caption_output(
                prediction,
                max_clauses=args.max_delta_output_clauses,
            )

        reference = sample.answer
        
        if is_sm_task:
            # For Shape Mating: extract and compare pair choices
            gt_choice = extract_sm_choice(reference)
            pred_choice = extract_sm_choice(prediction)
            is_exact = gt_choice is not None and gt_choice == pred_choice
            is_relaxed = is_exact  # For SM, relaxed = exact
        else:
            # For other tasks: use standard text matching
            norm_pred = normalize_text(prediction)
            norm_ref = normalize_text(reference)
            is_exact = norm_pred == norm_ref
            is_relaxed = bool(norm_ref) and (
                norm_ref in norm_pred or norm_pred in norm_ref
            )

        exact_matches += int(is_exact)
        relaxed_matches += int(is_relaxed)

        # Extract answer field if available in metadata for LLM evaluation
        answer_field = sample.metadata.get("answer") if sample.metadata else None
        
        result_entry = {
            "sample_index": sample.index,
            "object_ids": sample.object_ids,
            "question": sample.question,
            "ground_truth": reference,
            "answer": answer_field,  # Add answer field for LLM evaluation
            "prediction": prediction,
            "exact_match": bool(is_exact),
            "relaxed_match": bool(is_relaxed),
            "is_shape_mating": is_sm_task,
            "metadata": sample.metadata,
        }
        
        # Add reasoning if multi-turn mode was used
        if reasoning is not None:
            result_entry["reasoning"] = reasoning
        
        results.append(result_entry)

    num_samples = len(results)
    metrics = {
        "num_samples": num_samples,
        "exact_match": exact_matches / num_samples if num_samples else 0.0,
        "relaxed_match": relaxed_matches / num_samples if num_samples else 0.0,
    }

    output_payload = {
        "model_path": args.model_path,
        "anno_path": args.anno_path,
        "data_path": args.data_path,
        "pointnum": args.pointnum,
        "metrics": metrics,
        "generation_kwargs": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "dedupe_delta_output": args.dedupe_delta_output,
            "max_delta_output_clauses": args.max_delta_output_clauses,
        },
        "results": results,
    }

    with open(output_path, "w") as fp:
        json.dump(output_payload, fp, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved evaluation results to {output_path}")


if __name__ == "__main__":
    main()
