import math
from typing import Optional

from transformers import TrainerCallback


def _unwrap_module(module):
    """Return the first non-wrapper module (handles FSDP / DDP wrappers)."""
    while hasattr(module, "module"):
        module = module.module
    return module


class RelationGammaSchedulerCallback(TrainerCallback):
    """Cosine warmup for relation gamma scaling."""

    def __init__(
        self,
        start: float,
        end: float,
        warmup_steps: Optional[int] = None,
        warmup_ratio: Optional[float] = None,
    ):
        self.start = float(start)
        self.end = float(end)
        self.warmup_steps = warmup_steps
        self.warmup_ratio = warmup_ratio
        self._total_steps: Optional[int] = None

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self._total_steps = state.max_steps
        if self.warmup_steps is None and self.warmup_ratio is not None and self._total_steps is not None:
            self.warmup_steps = max(1, int(self.warmup_ratio * self._total_steps))
        if self.warmup_steps is None:
            # Default to 0 warmup (keep start value) if neither steps nor ratio provided
            self.warmup_steps = 0

        core = _unwrap_module(model).get_model() if model is not None else None
        if core is not None and hasattr(core, "relation_gamma"):
            core.relation_gamma = float(self.start)
            
            # ★ デバッグ：Gamma スケジューラ設定を出力（rank 0のみ）
            if self.start != self.end:
                import torch.distributed as dist
                is_main_process = not dist.is_initialized() or dist.get_rank() == 0
                if is_main_process:
                    print("\n" + "="*60)
                    print("📈 [Relation Gamma Scheduler] Configuration")
                    print("="*60)
                    print(f"📊 Total steps: {self._total_steps}")
                    print(f"🎯 Gamma schedule: {self.start:.3f} → {self.end:.3f}")
                    print(f"⏱️  Warmup steps: {self.warmup_steps} ({self.warmup_steps/self._total_steps*100:.1f}%)")
                    print(f"🔧 Initial gamma: {self.start:.3f}")
                    print("="*60 + "\n")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        core = _unwrap_module(model).get_model() if model is not None else None
        if core is None or not hasattr(core, "relation_gamma"):
            return

        step = state.global_step
        warmup = max(0, int(self.warmup_steps or 0))
        if warmup == 0:
            gamma = self.end
        elif step >= warmup:
            gamma = self.end
        else:
            t = step / warmup
            gamma = self.start + 0.5 * (1.0 - math.cos(math.pi * t)) * (self.end - self.start)

        core.relation_gamma = float(gamma)

        if args.logging_steps and step % args.logging_steps == 0:
            state.log_history.append({"relation/gamma": float(gamma), "step": step})


class FreezeUnfreezeLLMCallback(TrainerCallback):
    """Freeze LLM initially, then unfreeze top-K layers at a given ratio."""

    def __init__(
        self,
        unfreeze_ratio: float,
        top_k_layers: Optional[int],
        target_lr: float,
    ):
        self.unfreeze_ratio = float(unfreeze_ratio)
        if top_k_layers is None:
            top_k_layers = 0
        self.unfreeze_all = int(top_k_layers) <= 0
        self.top_k_layers = max(0, int(top_k_layers))
        self.target_lr = float(target_lr)
        self._trigger_step: Optional[int] = None
        self._done = False

    def _collect_decoder_layers(self, model):
        core = _unwrap_module(model)
        try:
            layers = list(core.model.layers)
        except AttributeError:
            try:
                layers = list(core.model.model.layers)
            except AttributeError:
                # Fallback: scan modules
                from transformers.models.llama.modeling_llama import LlamaDecoderLayer

                layers = [m for m in core.modules() if isinstance(m, LlamaDecoderLayer)]
        return layers

    def on_train_begin(self, args, state, control, model=None, optimizer=None, **kwargs):
        import torch.distributed as dist
        is_main_process = not dist.is_initialized() or dist.get_rank() == 0
        
        total = state.max_steps or 0
        self._trigger_step = max(1, int(self.unfreeze_ratio * total))
        layers = self._collect_decoder_layers(model)
        
        # ★ デバッグ：訓練開始時の設定サマリーを出力（rank 0のみ）
        if is_main_process:
            print("\n" + "="*60)
            print("🔒 [LLM Stagewise] Initial Freeze Configuration")
            print("="*60)
            print(f"📊 Total training steps: {total}")
            print(f"📌 Unfreeze trigger step: {self._trigger_step} ({self.unfreeze_ratio*100:.1f}%)")
            print(f"🎯 Target layers to unfreeze: {'ALL' if self.unfreeze_all else f'top-{self.top_k_layers}'}")
            print(f"📚 Total decoder layers: {len(layers)}")
            print(f"📖 Target learning rate after unfreeze: {self.target_lr}")
        
        # LLM は train.py で既に freeze されているはずなので、ここでは確認のみ
        core = _unwrap_module(model)
        llm_frozen_count = 0
        if layers:
            for layer in layers:
                for param in layer.parameters():
                    if not param.requires_grad:
                        llm_frozen_count += 1
                    break  # 各層の最初のパラメータのみチェック
        
        if is_main_process:
            print(f"✅ LLM layers already frozen: {llm_frozen_count}/{len(layers)}")
            print("="*60 + "\n")
        
        # Optimizer の学習率を 0 に設定（FSDP環境では optimizer が None の可能性あり）
        if optimizer is not None:
            llm_group_found = False
            for group in optimizer.param_groups:
                if group.get("name") == "llm":
                    group["lr"] = 0.0
                    llm_group_found = True
                    if is_main_process:
                        print(f"✅ [LLM Stagewise] Set LLM learning rate to 0.0")
            if not llm_group_found and is_main_process:
                print(f"⚠️  [LLM Stagewise] WARNING: 'llm' param group not found in optimizer")
        else:
            if is_main_process:
                print(f"⚠️  [LLM Stagewise] WARNING: optimizer is None (FSDP environment?)")
        
        state.log_history.append({"llm/unfrozen": 0, "llm/lr": 0.0, "step": state.global_step})

    def on_step_end(self, args, state, control, model=None, optimizer=None, **kwargs):
        if self._done or self._trigger_step is None:
            return
        if state.global_step < self._trigger_step:
            return

        layers = self._collect_decoder_layers(model)
        if not layers:
            self._done = True
            return

        core = _unwrap_module(model)
        if self.unfreeze_all or self.top_k_layers == 0:
            # restore full trainability (match fix_llm=False)
            target_layers = layers
            if hasattr(core, "model") and hasattr(core.model, "embed_tokens"):
                for param in core.model.embed_tokens.parameters():
                    param.requires_grad = True
            if hasattr(core, "lm_head"):
                for param in core.lm_head.parameters():
                    param.requires_grad = True
        else:
            target_layers = layers[-self.top_k_layers:]
            # embed_tokens / lm_head are also unfrozen to mimic fix_llm=False behavior
            if hasattr(core, "model") and hasattr(core.model, "embed_tokens"):
                for param in core.model.embed_tokens.parameters():
                    param.requires_grad = True
            if hasattr(core, "lm_head"):
                for param in core.lm_head.parameters():
                    param.requires_grad = True

        for layer in target_layers:
            for param in layer.parameters():
                param.requires_grad = True

        if optimizer is not None:
            for group in optimizer.param_groups:
                if group.get("name") == "llm":
                    group["lr"] = self.target_lr

        # ★ デバッグ：unfreeze 後の統計を出力（rank 0のみ）
        import torch.distributed as dist
        is_main_process = not dist.is_initialized() or dist.get_rank() == 0
        
        if is_main_process:
            print("\n" + "="*60)
            print(f"🔓 [LLM Stagewise] Unfreezing at step {state.global_step}")
            print("="*60)
        
        # named_parameters() で requires_grad=True の統計を集計
        trainable_llm_params = 0
        trainable_llm_layers = set()
        frozen_llm_layers = set()
        
        for name, param in _unwrap_module(model).named_parameters():
            # LLM layers のみをカウント（point_proj, relation_module は除外）
            if "point_proj" in name or "relation_module" in name or "point_backbone" in name:
                continue
            
            # layer 番号を抽出
            if ".layers." in name:
                try:
                    layer_idx = int(name.split(".layers.")[1].split(".")[0])
                    if param.requires_grad:
                        trainable_llm_params += param.numel()
                        trainable_llm_layers.add(layer_idx)
                    else:
                        frozen_llm_layers.add(layer_idx)
                except (IndexError, ValueError):
                    pass
            elif param.requires_grad and ("embed_tokens" in name or "lm_head" in name):
                trainable_llm_params += param.numel()
        
        if is_main_process:
            print(f"✅ Trainable LLM layers: {sorted(trainable_llm_layers)} ({len(trainable_llm_layers)} layers)")
            print(f"❄️  Frozen LLM layers: {sorted(frozen_llm_layers)} ({len(frozen_llm_layers)} layers)")
            print(f"📊 Trainable LLM parameters: {trainable_llm_params:,}")
            print(f"📖 New learning rate: {self.target_lr}")
            print("="*60 + "\n")
        
        state.log_history.append({
            "llm/unfrozen": 1, 
            "llm/lr": self.target_lr, 
            "llm/trainable_layers": len(trainable_llm_layers),
            "llm/frozen_layers": len(frozen_llm_layers),
            "step": state.global_step
        })

        self._done = True


class RelationDeltaLoggerCallback(TrainerCallback):
    """Log relation delta statistics (if available)."""
    
    def __init__(self):
        self._first_log = True

    def on_step_end(self, args, state, control, model=None, **kwargs):
        core = _unwrap_module(model).get_model() if model is not None else None
        relation_module = getattr(core, "relation_module", None) if core is not None else None
        encoder = getattr(relation_module, "encoder", None) if relation_module is not None else None
        metrics = getattr(encoder, "_last_debug_metrics", None)
        
        # ★ デバッグ：最初のログ時に環境変数チェック（rank 0のみ）
        if self._first_log:
            import os
            import torch.distributed as dist
            is_main_process = not dist.is_initialized() or dist.get_rank() == 0
            debug_enabled = os.environ.get("POINTLLM_RELATION_DEBUG", "0") == "1"
            
            if is_main_process:
                if debug_enabled and metrics:
                    print("\n" + "="*60)
                    print("🔍 [Relation Delta Monitor] Debug Mode Enabled")
                    print("="*60)
                    print("✅ POINTLLM_RELATION_DEBUG=1 detected")
                    print("📊 Logging relation module delta statistics:")
                    print(f"   Available metrics: {list(metrics.keys())}")
                    print("="*60 + "\n")
                elif not debug_enabled:
                    print("\n" + "="*60)
                    print("⚠️  [Relation Delta Monitor] Debug Mode Disabled")
                    print("="*60)
                    print("❌ POINTLLM_RELATION_DEBUG is not set to 1")
                    print("💡 To enable delta monitoring, set: export POINTLLM_RELATION_DEBUG=1")
                    print("="*60 + "\n")
            self._first_log = False
        
        if not metrics:
            return
        log_record = {f"relation/{k}": float(v) for k, v in metrics.items()}
        log_record["step"] = state.global_step
        state.log_history.append(log_record)
