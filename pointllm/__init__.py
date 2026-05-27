"""PointLLM package exports.

The heavyweight model stack imports torch, transformers, timm, and optional
training dependencies. Keep package import lightweight so data utilities and
release checks can run without importing the full model tree.
"""


def __getattr__(name):
    if name == "PointLLMLlamaForCausalLM":
        from .model import PointLLMLlamaForCausalLM

        return PointLLMLlamaForCausalLM
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["PointLLMLlamaForCausalLM"]
