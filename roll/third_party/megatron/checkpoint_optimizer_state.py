"""Initialize pristine all-CPU AdamW checkpoints without taking an Adam step."""
import torch


def initialize_cpu_adamw_state(optimizer):
    """Ask AdamW to allocate its lazy state with at most one temporary gradient.

    Calling ``step`` with a zero learning rate advances the bias correction
    counter. Use AdamW's initialization routine so a checkpoint before the
    first update retains exactly the same subsequent training trajectory.
    """
    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError('initial CPU checkpoint state requires torch.optim.AdamW')
    for group in optimizer.param_groups:
        for parameter in group['params']:
            if parameter.device.type != 'cpu':
                raise ValueError('initial CPU checkpoint state requires CPU parameters')
    for group in optimizer.param_groups:
        for parameter in group['params']:
            if optimizer.state.get(parameter):
                continue
            previous_gradient = parameter.grad
            try:
                if previous_gradient is None:
                    parameter.grad = torch.zeros_like(parameter)
                # The native initializer owns dtype, fused/capturable step
                # placement and optional AMSGrad state. It never updates p.
                optimizer._init_group(dict(group, params=[parameter]), *[[] for _ in range(6)])
            except Exception:
                # A failed allocation may have installed only step/exp_avg.
                # Leave this previously empty entry retryable.
                optimizer.state.pop(parameter, None)
                raise
            finally:
                parameter.grad = previous_gradient


def prepare_cpu_adam_for_checkpoint(optimizer):
    """Synchronize native all-CPU Hybrid AdamW state before distributed save."""
    from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer

    if hasattr(optimizer, 'chained_optimizers'):
        for child in optimizer.chained_optimizers:
            prepare_cpu_adam_for_checkpoint(child)
    elif isinstance(optimizer, HybridDeviceOptimizer):
        # Mixed CPU/GPU optimizers keep their existing initialization contract.
        if optimizer.gpu_optimizer is not None or not optimizer.cpu_optimizers:
            return
        if not all(isinstance(child, torch.optim.AdamW) for child in optimizer.cpu_optimizers):
            return
        for child in optimizer.cpu_optimizers:
            initialize_cpu_adamw_state(child)
        optimizer._sync_sub_optimizers_state_to_hdo()
    elif getattr(optimizer, 'optimizer', None) is not None:
        prepare_cpu_adam_for_checkpoint(optimizer.optimizer)


# ---------------------------------------------------------------------------
# Opt-in Hybrid checkpoint serialization
# ---------------------------------------------------------------------------
# Two defects in Megatron's Hybrid checkpoint path, both only reachable when
# the optimizer is the all-CPU Hybrid that ROLL's CPU-offloaded training
# builds. Both are behind ``bounded_cpu_grad_staging`` because that is the flag
# that selects this optimizer shape in ROLL, and neither changes anything for
# the native Hybrid path.
#
# Replaces scripts/qwen38/patch_megatron_checkpoint_initialization.py and
# scripts/qwen38/patch_megatron_hybrid_checkpoint.py: installed at runtime
# instead of requiring every user to patch their own Megatron checkout.

_HYBRID_CHECKPOINT_INSTALLED = "_qwen4_hybrid_checkpoint_installed"
_ORIGINAL_DUMMY_STEP = "_qwen4_unpatched_dummy_step"
_ORIGINAL_LOAD_HOOKS = "_qwen4_unpatched_register_load_state_dict_hooks"
_ORIGINAL_SYNC_HDO_STATE = "_qwen4_unpatched_sync_hdo_state_to_sub_optimizers"
_ORIGINAL_GET_MAIN_PARAM = "_qwen4_unpatched_get_main_param_and_optimizer_states"


def _bounded_dummy_step(self):
    """Allocate CPU optimizer state from the reusable buffer, not a full image.

    Megatron's version calls ``self.step()`` (via the randn path) or rebuilds
    every CPU master. In bounded mode the gradients come from the staging
    buffer, so cold restore does not need a second complete CUDA gradient image
    nor a transient full CPU master copy.
    """
    if not getattr(self, "bounded_cpu_grad_staging", False):
        return getattr(type(self), _ORIGINAL_DUMMY_STEP)(self)
    if self.state:
        return None
    self._sync_hdo_param_groups_to_sub_optimizers()
    self._prepare_bounded_cpu_grad_staging()
    for optimizer in self.sub_optimizers:
        learning_rates = []
        gradients = []
        try:
            for group in optimizer.param_groups:
                learning_rates.append((group, group["lr"]))
                # Let the actual optimizer allocate its state without changing
                # the master or applying weight decay.
                group["lr"] = 0.0
                for param in group["params"]:
                    gradients.append((param, param.grad))
                    if optimizer is self.gpu_optimizer:
                        param.grad = torch.zeros_like(param)
                    else:
                        size = param.numel() * param.element_size()
                        staged = self._cpu_grad_staging_buffer[:size]
                        param.grad = staged.view(param.dtype).view(param.shape).zero_()
            optimizer.step()
        finally:
            for group, learning_rate in learning_rates:
                group["lr"] = learning_rate
            for param, gradient in gradients:
                param.grad = gradient
    torch.cuda.current_stream().wait_stream(self._h2d_stream)
    self._sync_sub_optimizers_state_to_hdo()
    return None


def _bounded_register_load_state_dict_hooks(self, *args, **kwargs):
    """Install the checkpoint hooks, with the post-hook's rebuild skipped.

    Reproduced rather than wrapped because the change sits inside the hook
    closures: Megatron's post-hook calls ``_init_sub_optimizers`` to regenerate
    the parameter/copy pairs after a load. In bounded mode the loaded state
    already aliases the live FP32 owners, so rebuilding would transiently clone
    a second complete CPU master image; the ownership is asserted unchanged
    instead. The pre-hook is Megatron's, verbatim.
    """
    original = getattr(type(self), _ORIGINAL_LOAD_HOOKS)
    if not getattr(self, "bounded_cpu_grad_staging", False):
        return original(self, *args, **kwargs)

    def pre_load_state_dict_hook(optimizer, state_dict):
        if not optimizer.param_update_in_fp32:
            return state_dict
        new_state = {}
        for param, value in optimizer.state.items():
            new_state[optimizer.param_to_fp32_param.get(param, param)] = value
        optimizer.state = new_state
        for group in optimizer.param_groups:
            for index, param in enumerate(group["params"]):
                group["params"][index] = optimizer.param_to_fp32_param.get(param, param)
        return state_dict

    def post_load_state_dict_hook(optimizer):
        if optimizer.param_update_in_fp32:
            new_state = {}
            for param, value in optimizer.state.items():
                new_state[optimizer.fp32_param_to_orig_param.get(param, param)] = value
            optimizer.state = new_state
            for group in optimizer.param_groups:
                for index, param in enumerate(group["params"]):
                    group["params"][index] = optimizer.fp32_param_to_orig_param.get(param, param)
        current_params = {p for group in optimizer.param_groups for p in group["params"]}
        if current_params != set(optimizer.param_to_inner_param):
            raise ValueError("bounded Hybrid restore requires unchanged parameter ownership")
        optimizer._sync_hdo_param_groups_to_sub_optimizers()
        optimizer._sync_hdo_state_to_sub_optimizers()

    self.register_load_state_dict_pre_hook(pre_load_state_dict_hook)
    self.register_load_state_dict_post_hook(post_load_state_dict_hook)


def _bounded_sync_hdo_state_to_sub_optimizers(self):
    """Megatron's method plus the canonical master owner in the state entry."""
    original = getattr(type(self), _ORIGINAL_SYNC_HDO_STATE)
    result = original(self)
    if getattr(self, "bounded_cpu_grad_staging", False) and self.param_update_in_fp32:
        # The loaded master has been copied into the actual optimizer
        # parameter. Keep only that canonical owner in the checkpoint state.
        for original_param, inner_param in self.param_to_inner_param.items():
            if original_param in self.state:
                self.state[original_param]["master_param"] = inner_param
    return result


def _bounded_get_main_param_and_optimizer_states(self, model_param):
    """Megatron's method, without serializing Hybrid's scalar Adam step.

    Adam's ``step`` is a scalar restored from common parameter-group metadata,
    so it has no model-shaped tensor shard to live in.
    """
    original = getattr(type(self), _ORIGINAL_GET_MAIN_PARAM)
    tensors = original(self, model_param)
    if (
        getattr(self.config, "use_precision_aware_optimizer_no_fp8_or_ds_fp8", False)
        and isinstance(getattr(self, "optimizer", None), _hybrid_class())
    ):
        tensors.pop("step", None)
    return tensors


def _hybrid_class():
    from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer

    return HybridDeviceOptimizer


def _verify_hybrid_checkpoint_contract():
    """Fail unless Megatron still carries the methods these replacements wrap."""
    import inspect

    hybrid = _hybrid_class()
    required = {
        "dummy_step": "the cold-restore state initializer",
        "_register_load_state_dict_hooks": "the checkpoint hook installer",
        "_sync_hdo_state_to_sub_optimizers": "the pre-save state sync",
    }
    missing = [name for name in required if not hasattr(hybrid, name)]
    if missing:
        raise RuntimeError(
            "Megatron's HybridDeviceOptimizer lacks "
            + ", ".join(f"{name} ({required[name]})" for name in missing)
            + "; re-derive the patch in roll/third_party/megatron/checkpoint_optimizer_state.py."
        )
    try:
        source = inspect.getsource(hybrid.dummy_step)
    except (OSError, TypeError) as exc:
        raise RuntimeError(
            "Cannot read Megatron's HybridDeviceOptimizer.dummy_step "
            f"({type(exc).__name__}: {exc}); the bounded replacement in "
            "roll/third_party/megatron/checkpoint_optimizer_state.py cannot be verified."
        ) from exc
    if "torch.randn_like" not in source:
        raise RuntimeError(
            "Megatron's HybridDeviceOptimizer.dummy_step no longer uses the randn "
            "initialization this replacement branches ahead of; re-derive it."
        )


def install_hybrid_checkpoint_patches() -> bool:
    """Install both Hybrid checkpoint replacements. Idempotent."""
    hybrid = _hybrid_class()
    if getattr(hybrid, _HYBRID_CHECKPOINT_INSTALLED, False):
        return True
    _verify_hybrid_checkpoint_contract()

    setattr(hybrid, _ORIGINAL_DUMMY_STEP, hybrid.dummy_step)
    setattr(hybrid, _ORIGINAL_LOAD_HOOKS, hybrid._register_load_state_dict_hooks)
    setattr(hybrid, _ORIGINAL_SYNC_HDO_STATE, hybrid._sync_hdo_state_to_sub_optimizers)
    hybrid.dummy_step = _bounded_dummy_step
    hybrid._register_load_state_dict_hooks = _bounded_register_load_state_dict_hooks
    hybrid._sync_hdo_state_to_sub_optimizers = _bounded_sync_hdo_state_to_sub_optimizers

    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    if not hasattr(DistributedOptimizer, _ORIGINAL_GET_MAIN_PARAM):
        setattr(DistributedOptimizer, _ORIGINAL_GET_MAIN_PARAM,
                DistributedOptimizer._get_main_param_and_optimizer_states)
        DistributedOptimizer._get_main_param_and_optimizer_states = (
            _bounded_get_main_param_and_optimizer_states
        )

    setattr(hybrid, _HYBRID_CHECKPOINT_INSTALLED, True)
    return True


def uninstall_hybrid_checkpoint_patches() -> bool:
    """Restore Megatron's own methods. For tests and scope assertions."""
    hybrid = _hybrid_class()
    if not getattr(hybrid, _HYBRID_CHECKPOINT_INSTALLED, False):
        return False
    for name, original_name in (("dummy_step", _ORIGINAL_DUMMY_STEP),
                                ("_register_load_state_dict_hooks", _ORIGINAL_LOAD_HOOKS),
                                ("_sync_hdo_state_to_sub_optimizers", _ORIGINAL_SYNC_HDO_STATE)):
        setattr(hybrid, name, getattr(hybrid, original_name))
        delattr(hybrid, original_name)
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    original = getattr(DistributedOptimizer, _ORIGINAL_GET_MAIN_PARAM, None)
    if original is not None:
        DistributedOptimizer._get_main_param_and_optimizer_states = original
        delattr(DistributedOptimizer, _ORIGINAL_GET_MAIN_PARAM)
    delattr(hybrid, _HYBRID_CHECKPOINT_INSTALLED)
    return True
