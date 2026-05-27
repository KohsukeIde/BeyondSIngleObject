# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# import os
# import torch
# import numpy as np

# # PyTorch 2.6 互換のためのsafe globals/torch.loadのパッチは、必要時のみ有効化します。
# # 環境変数 POINTLLM_DISABLE_SAFE_GLOBALS=1 が設定されていればスキップ。
# if os.getenv("POINTLLM_DISABLE_SAFE_GLOBALS", "0") != "1":
#     # Add comprehensive safe globals for PyTorch 2.6 compatibility
#     torch.serialization.add_safe_globals([
#         np._core.multiarray._reconstruct,
#         np.ndarray,
#         np.dtype,
#         np.core.multiarray._reconstruct,
#         np.random.RandomState,
#         np.random.Generator,
#         # Python built-in types
#         list, tuple, dict, set, frozenset,
#         int, float, str, bool, bytes, type(None),
#         # Common numpy types
#         np.int32, np.int64, np.float32, np.float64,
#         np.uint8, np.uint16, np.uint32, np.uint64,
#     ])

#     # Also set PyTorch's default load behavior for compatibility
#     import functools
#     _original_torch_load = torch.load
#     def _patched_torch_load(*args, **kwargs):
#         if 'weights_only' not in kwargs:
#             kwargs['weights_only'] = False
#         return _original_torch_load(*args, **kwargs)
#     torch.load = _patched_torch_load

# Need to call this before importing transformers.
from pointllm.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
replace_llama_attn_with_flash_attn()

# Ensure gradient checkpointing uses the non-reentrant implementation globally.
# PyTorch 2.7 removed set_checkpoint_implementation, so patch the default
# checkpoint() call used by transformers' gradient_checkpointing_enable().
try:
    import torch.utils.checkpoint as _cp
    if hasattr(_cp, "set_checkpoint_implementation"):
        _cp.set_checkpoint_implementation("no_reentrant")
    else:
        _orig_checkpoint = _cp.checkpoint

        def _checkpoint_no_reentrant(function, *args, **kwargs):
            if kwargs.get("use_reentrant") is None:
                kwargs["use_reentrant"] = False
            return _orig_checkpoint(function, *args, **kwargs)

        _cp.checkpoint = _checkpoint_no_reentrant
except Exception:
    pass

from pointllm.train.train import train

if __name__ == "__main__":
    train()
