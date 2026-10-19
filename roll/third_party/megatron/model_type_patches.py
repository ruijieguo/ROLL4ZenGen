"""Megatron patches that must apply wherever the Qwen4 model is used.

Four patches in this batch, all of them reaching into code Megatron owns rather
than into a Qwen4 class, so none of them can be scoped by subclassing:

* ``gradient_clipping``  -- bound the TE tensor handles one scale call holds.
* ``moe_cache``          -- drop the completed router graph the dispatcher pins.
* ``checkpoint_padding`` -- never persist uninitialised allocator contents.
* ``ddp_memory``         -- remap weights before allocating gradient storage.

Each replaces a function (or a method) on the installed Megatron instead of the
caller editing their checkout, matching the runtime-install form the rest of
``roll/third_party/megatron`` uses. Everything is gated on the Qwen4 model type
by :func:`install_for_model_type`, so other models keep Megatron's own code.

A note on the reproduced bodies: two of these (``checkpoint_padding`` and
``ddp_memory``) are copies of upstream methods with the change inline, because
the change sits in the middle of the body and cannot be expressed as a call
around it. Each copy carries a contract check against the installed source --
see :func:`_verify_contracts` -- so a Megatron refactor fails loudly rather than
silently reverting to the unpatched behaviour.
"""

import inspect
import logging
from typing import Optional, Union

from roll.models.qwen4_exp_scope import is_flash_next

logger = logging.getLogger(__name__)

# Attribute names this module uses to record what it replaced, so the install is
# reversible and can tell a fresh dependency from one it already patched.
_GRADIENT_CLIPPING_ORIGINAL = "_qwen4_unpatched_clip_grad_by_total_norm_fp32"
_CHECKPOINT_PADDING_ORIGINAL = "_qwen4_unpatched_sharded_param_state_dp_reshardable"
_DDP_MEMORY_ORIGINAL = "_qwen4_unpatched_param_and_grad_buffer_init"
_MOE_CACHE_ORIGINAL = "_qwen4_unpatched_combine_postprocess"
_INSTALLED_FLAG = "_qwen4_model_type_patches_installed"

# The scale call is batched because transformer-engine's multi_tensor_scale
# creates one tensor handle per input; an expert-adapter parameter list of tens
# of thousands exhausts them. Every batch reuses the globally computed
# coefficient, so the arithmetic is identical to one call over all grads.
_SCALE_BATCH_SIZE = 2048


# ---------------------------------------------------------------------------
# gradient_clipping
# ---------------------------------------------------------------------------
def _clip_grad_by_total_norm_fp32(
    parameters,
    max_norm: Union[int, float],
    total_norm: float,
    use_decoupled_grad: bool = False,
):
    """Megatron's ``clip_grad_by_total_norm_fp32`` with the scale call batched.

    The norm and the clipping coefficient are computed exactly as upstream;
    only the in-place scaling is partitioned. Replaces
    scripts/qwen38/patch_megatron_gradient_clipping.py.
    """
    import torch
    from megatron.core.utils import to_local_if_dtensor

    multi_tensor_applier, multi_tensor_scale_impl = _multi_tensor_apply()

    params = []
    grads = []
    for param in parameters:
        if use_decoupled_grad:
            if hasattr(param, "decoupled_grad") and param.decoupled_grad is not None:
                assert param.decoupled_grad.dtype in [torch.float32, torch.bfloat16]
                params.append(param)
                grads.append(to_local_if_dtensor(param.decoupled_grad).detach())
        else:
            if param.grad is not None:
                assert param.grad.type() == "torch.cuda.FloatTensor"
                params.append(param)
                grads.append(to_local_if_dtensor(param.grad).detach())

    clip_coeff = max_norm / (total_norm + 1.0e-6)
    if clip_coeff < 1.0:
        dummy_overflow_buf = torch.zeros(1, dtype=torch.int, device="cuda")
        for offset in range(0, len(grads), _SCALE_BATCH_SIZE):
            grad_batch = grads[offset : offset + _SCALE_BATCH_SIZE]
            multi_tensor_applier(
                multi_tensor_scale_impl, dummy_overflow_buf, [grad_batch, grad_batch], clip_coeff
            )
    return None


def _multi_tensor_apply():
    """The (applier, scale impl) pair Megatron selected at import time."""
    from megatron.core.optimizer import clip_grads

    return clip_grads.multi_tensor_applier, clip_grads.multi_tensor_scale_impl


# ---------------------------------------------------------------------------
# moe_cache
# ---------------------------------------------------------------------------
def _combine_postprocess(self, permutated_local_input_tokens):
    """Megatron's ``MoEAlltoAllTokenDispatcher.combine_postprocess``, plus detach.

    The dispatcher keeps ``self.probs`` from its last dispatch; that tensor
    carries the completed router graph, so leaving it attached pins the
    checkpoint input and its gradient across microbatches even though backward
    already holds its own saved tensors. Detaching keeps the dtype metadata and
    releases the graph. Replaces scripts/qwen38/patch_megatron_moe_cache.py.
    """
    from megatron.core.transformer.moe.moe_utils import unpermute

    if self.shared_experts is not None:
        self.shared_experts.linear_fc2_forward(permutated_local_input_tokens)
        self.shared_experts.post_forward_comm()

    output = unpermute(
        permutated_local_input_tokens,
        self.reversed_local_input_permutation_mapping,
        restore_shape=self.hidden_shape_before_permute,
        routing_map=self.routing_map,
        fused=self.config.moe_permute_fusion,
        drop_and_pad=self.drop_and_pad,
    )

    output = output.view(self.hidden_shape)

    if self.shared_experts is not None:
        shared_expert_output = self.shared_experts.get_output()
        output += shared_expert_output
    self.probs = self.probs.detach()
    return output


# ---------------------------------------------------------------------------
# checkpoint_padding
# ---------------------------------------------------------------------------
def _sharded_param_state_dp_reshardable(self, model_sharded_state_dict, is_loading=False, metadata=None):
    """Megatron's ``DistributedOptimizer.sharded_param_state_dp_reshardable``.

    Upstream fills the intra-bucket alignment gaps with ``torch.empty``, so the
    bytes written into a fresh checkpoint are whatever the allocator last had
    there. Two otherwise identical cold continuations then produce different
    checkpoint bytes. The gaps hold no parameter or optimizer state, so zeroing
    them removes the nondeterminism without changing anything read back.
    Replaces scripts/qwen38/patch_megatron_checkpoint_padding.py.
    """
    import torch
    from megatron.core.dist_checkpointing.mapping import ShardedObject, ShardedTensor
    from megatron.core.dist_checkpointing.utils import LocalNonpersistentObject

    data_parallel_rank = self.data_parallel_group.rank()
    data_parallel_world_size = self.data_parallel_group.size()

    state = self.get_parameter_state_dp_reshardable()
    for per_bucket_key in ("per_bucket_numel", "per_bucket_numel_unpadded"):
        key = (
            f"optimizer.distributed.dp_group_idx_{self.data_parallel_group_idx}"
            f".{per_bucket_key}"
        )
        state[per_bucket_key] = ShardedObject(
            key,
            state[per_bucket_key],
            (1,),
            (0,),
            replica_id=(self.distributed_optimizer_instance_id, 0, data_parallel_rank),
        )

    for gbuf_idx, gbuf_range_maps in enumerate(self.gbuf_ranges):
        for dtype, gbuf_range_map_for_all_buckets in state[gbuf_idx].items():
            for bucket_idx, bucket_state in enumerate(gbuf_range_map_for_all_buckets):
                gbuf_world_numel_unpadded = self.buffers[gbuf_idx].buckets[bucket_idx].numel_unpadded
                gbuf_world_numel = self.buffers[gbuf_idx].buckets[bucket_idx].grad_data.numel()
                assert gbuf_world_numel_unpadded <= gbuf_world_numel
                assert gbuf_world_numel % data_parallel_world_size == 0
                gbuf_local_numel = gbuf_world_numel // data_parallel_world_size

                sharded_bucket_key = (
                    f"optimizer.distributed.dp_group_idx_{self.data_parallel_group_idx}"
                    f".gbuf_idx_{gbuf_idx}.dtype_{dtype}.bucket_idx_{bucket_idx}"
                )
                assert bucket_state, "empty bucket encountered"

                all_pad_tensors = {}
                for i in range(-1, len(bucket_state)):
                    if i == len(bucket_state) - 1:
                        next_param_start = gbuf_local_numel
                    else:
                        next_param_start = bucket_state[i + 1]["gbuf_local_start"]
                    if i == -1:
                        cur_param_end = 0
                    else:
                        cur_param_end = bucket_state[i]["gbuf_local_end"]
                    world_param_end = data_parallel_rank * gbuf_local_numel + cur_param_end
                    if (
                        next_param_start != cur_param_end
                        and world_param_end < gbuf_world_numel_unpadded
                    ):
                        # These persisted gaps are not parameter/Adam state;
                        # never serialize uninitialized allocator contents.
                        pad_tensors = {
                            k: torch.zeros(
                                next_param_start - cur_param_end, dtype=v.dtype, device=v.device
                            )
                            for k, v in bucket_state[i].items()
                            if isinstance(v, torch.Tensor)
                        }
                        all_pad_tensors[i + 1] = {
                            **pad_tensors,
                            "gbuf_local_start": cur_param_end,
                            "gbuf_local_end": next_param_start,
                            "padding": True,
                        }

                for index_to_insert in reversed(sorted(list(all_pad_tensors.keys()))):
                    bucket_state.insert(index_to_insert, all_pad_tensors[index_to_insert])

                for bucket_params_idx in range(len(bucket_state)):
                    tensors = bucket_state[bucket_params_idx]
                    gbuf_local_start = tensors.pop("gbuf_local_start")
                    gbuf_local_end = tensors.pop("gbuf_local_end")
                    if "padding" not in tensors:
                        tensors["padding"] = False

                    for key in tensors:
                        if key == "padding":
                            tensors[key] = LocalNonpersistentObject(tensors[key])
                            continue
                        assert tensors[key].shape == (gbuf_local_end - gbuf_local_start,), (
                            tensors[key].shape,
                            gbuf_local_start,
                            gbuf_local_end,
                        )

                        tensors[key] = ShardedTensor(
                            f"{sharded_bucket_key}.{key}",
                            tensors[key],
                            tensors[key].dtype,
                            tensors[key].shape,
                            (gbuf_world_numel_unpadded,),
                            (data_parallel_rank * gbuf_local_numel + gbuf_local_start,),
                            axis_fragmentations=None,
                            flattened_range=None,
                            allow_shape_mismatch=False,
                            replica_id=(self.distributed_optimizer_instance_id, 0, 0),
                        )
    return state


# ---------------------------------------------------------------------------
# ddp_memory
# ---------------------------------------------------------------------------
def _param_and_grad_buffer_init(
    self,
    ddp_config,
    param_dtype,
    grad_dtype,
    params,
    data_parallel_group,
    bucket_size,
    param_to_name,
    gradient_scaling_factor,
    param_indices,
    nccl_ub,
    pg_collection=None,
):
    """Megatron's ``_ParamAndGradBuffer.__init__`` with the allocation reordered.

    Upstream maps each parameter's ``data`` onto the new buffer and then
    allocates a separate gradient buffer, so both the original weights and the
    gradient storage are live at once. This remaps the weights first and only
    then allocates gradients, removing one parameter-sized transient. Values,
    bucket layout and the shared MXFP8 buffer are unchanged.
    Replaces scripts/qwen38/patch_megatron_ddp_memory.py.
    """
    import functools
    import math
    from contextlib import nullcontext

    import torch
    from megatron.core import parallel_state
    from megatron.core.distributed.param_and_grad_buffer import (
        BufferType,
        modify_underlying_storage,
    )
    from megatron.core.fp8_utils import is_float8tensor, is_mxfp8tensor
    from megatron.core.utils import log_on_each_pipeline_stage

    if pg_collection is None:
        self.dp_cp_group = parallel_state.get_data_and_context_parallel_group(
            with_context_parallel=True
        )
        self.tp_group = parallel_state.get_tensor_model_parallel_group()
    else:
        assert hasattr(pg_collection, "tp") and hasattr(pg_collection, "dp_cp")
        self.dp_cp_group = pg_collection.dp_cp
        self.tp_group = pg_collection.tp

    self.ddp_config = ddp_config
    self.params = params
    self.param_indices = param_indices

    unique_params = set()
    for param in params:
        assert param not in unique_params
        unique_params.add(param)
    del unique_params

    self.param_dtype = param_dtype
    self.grad_dtype = grad_dtype
    self.data_parallel_group = data_parallel_group
    self.data_parallel_world_size = self.data_parallel_group.size()
    self.gradient_scaling_factor = gradient_scaling_factor
    self.nccl_ub = nccl_ub

    self.buckets = []
    self.param_to_bucket = {}
    self.param_index_map = {}

    def _pad(number_to_be_padded, divisor):
        return int(math.ceil(number_to_be_padded / divisor) * divisor)

    def _pad_end_of_bucket_if_needed(bucket_end_index):
        if self.ddp_config.use_distributed_optimizer:
            if self.ddp_config.pad_buckets_for_high_nccl_busbw:
                bucket_size_divisor = math.lcm(self.data_parallel_world_size, 128, 2**16)
            else:
                bucket_size_divisor = math.lcm(self.data_parallel_world_size, 128)
            return _pad(bucket_end_index, bucket_size_divisor)
        return bucket_end_index

    def _pad_start_of_param_if_needed(param_start_index):
        if self.ddp_config.use_distributed_optimizer:
            return _pad(param_start_index, 64)
        return param_start_index

    param_start_index = 0
    bucket_start_index = param_start_index
    bucket_params = set()
    self.bucket_indices = []
    per_bucket_numel_unpadded = []
    bucket_id = 0

    def _update_bucket_metadata(param_end_index):
        nonlocal bucket_start_index, bucket_params, bucket_id
        per_bucket_numel_unpadded.append(param_end_index - bucket_start_index)
        bucket_end_index = _pad_end_of_bucket_if_needed(param_end_index)
        self.bucket_indices.append((bucket_start_index, bucket_end_index))
        bucket_start_index = bucket_end_index
        bucket_params = set()
        bucket_id += 1
        return bucket_end_index

    def _does_param_require_new_bucket(param):
        return getattr(param, "shared_embedding", False) and self.ddp_config.use_distributed_optimizer

    for param in params[::-1]:
        this_numel = param.data.nelement()
        param_start_index = _pad_start_of_param_if_needed(param_start_index)

        if _does_param_require_new_bucket(param) and len(bucket_params) > 0:
            param_start_index = _update_bucket_metadata(param_start_index)

        param_end_index = param_start_index + this_numel
        self.param_index_map[param] = (param_start_index, param_end_index, bucket_id)
        bucket_params.add(param)

        if (
            bucket_size is not None and (param_end_index - bucket_start_index) >= bucket_size
        ) or _does_param_require_new_bucket(param):
            bucket_end_index = _update_bucket_metadata(param_end_index)
            param_start_index = bucket_end_index
        else:
            param_start_index = param_end_index

    if len(bucket_params) > 0:
        bucket_end_index = _update_bucket_metadata(param_end_index)

    self.numel = bucket_end_index
    self.numel_unpadded = sum(per_bucket_numel_unpadded)
    assert self.numel_unpadded <= self.numel
    if self.ddp_config.use_distributed_optimizer:
        assert self.numel % self.data_parallel_world_size == 0
    else:
        assert self.numel == self.numel_unpadded

    self.param_data = None
    self.grad_data = None

    if self.nccl_ub:
        from megatron.core.distributed.param_and_grad_buffer import nccl_allocator

        nccl_allocator.init()
        pool = nccl_allocator.create_nccl_mem_pool(
            symmetric=not self.ddp_config.disable_symmetric_registration
        )
        mem_alloc_context = functools.partial(
            nccl_allocator.nccl_mem,
            pool,
            group=self.data_parallel_group,
            symmetric=not self.ddp_config.disable_symmetric_registration,
        )
        torch.distributed.barrier()
        tmp_warmup_tensor = torch.zeros([1], device="cuda")
        torch.distributed.all_reduce(tmp_warmup_tensor, group=self.data_parallel_group)
        torch.distributed.barrier()
    else:
        mem_alloc_context = nullcontext

    with mem_alloc_context():
        if self.ddp_config.use_distributed_optimizer and any(is_mxfp8tensor(p) for p in params):
            self.shared_buffer = torch.zeros(
                self.numel,
                dtype=self.grad_dtype,
                device=torch.cuda.current_device(),
                requires_grad=False,
            )
            if self.grad_dtype == torch.float32:
                self.param_data = self.shared_buffer[: math.ceil(self.numel / 2)].view(
                    torch.bfloat16
                )
            else:
                self.param_data = self.shared_buffer
            # The shared buffer is the gradient storage too, so the later
            # allocation below is skipped for this path (grad_data is not None).
            self.grad_data = self.shared_buffer
        else:
            if self.ddp_config.use_distributed_optimizer:
                self.param_data = torch.zeros(
                    self.numel,
                    dtype=self.param_dtype,
                    device=torch.cuda.current_device(),
                    requires_grad=False,
                )

    # Remap original weights before allocating separate gradient storage.
    for param in params[::-1]:
        param_start_index, _, _ = self.param_index_map[param]
        if not self.ddp_config.reuse_grad_buf_for_mxfp8_param_ag:
            if self.param_data is not None:
                new_param_data = self._get(
                    param.data.shape, param_start_index, buffer_type=BufferType.PARAM
                )
                if is_float8tensor(param):
                    modify_underlying_storage(param, new_param_data)
                else:
                    old_param_data = param.data
                    param.data = new_param_data
                    assert old_param_data._base is None
                    param.data.detach().copy_(old_param_data)
                    del old_param_data

    with mem_alloc_context():
        if self.grad_data is None:
            self.grad_data = torch.zeros(
                self.numel,
                dtype=self.grad_dtype,
                device=torch.cuda.current_device(),
                requires_grad=False,
            )

    bucket_params = []
    bucket_start_index = 0
    cur_bucket_id = 0
    for param in params[::-1]:
        param_start_index, param_end_index, bucket_id = self.param_index_map[param]
        param.main_grad = self._get(param.data.shape, param_start_index, buffer_type=BufferType.GRAD)
        if bucket_id != cur_bucket_id:
            bucket_end_index = _pad_end_of_bucket_if_needed(param_start_index)
            self.buckets.append(
                self._new_bucket(
                    bucket_params=bucket_params,
                    start_index=bucket_start_index,
                    end_index=bucket_end_index,
                    numel_unpadded=per_bucket_numel_unpadded[cur_bucket_id],
                    bucket_id=cur_bucket_id,
                )
            )
            bucket_start_index = bucket_end_index
            bucket_params = []
            assert cur_bucket_id + 1 == len(self.buckets)
            assert bucket_id == cur_bucket_id + 1
            cur_bucket_id = bucket_id
        bucket_params.append(param)

    if len(bucket_params) > 0:
        bucket_end_index = _pad_end_of_bucket_if_needed(param_end_index)
        self.buckets.append(
            self._new_bucket(
                bucket_params=bucket_params,
                start_index=bucket_start_index,
                end_index=bucket_end_index,
                numel_unpadded=per_bucket_numel_unpadded[cur_bucket_id],
                bucket_id=cur_bucket_id,
            )
        )

    log_strs = []
    log_strs.append(f"Number of buckets for gradient all-reduce / reduce-scatter: {len(self.buckets)}")
    for index, bucket in enumerate(self.buckets):
        numel = sum(param.data.nelement() for param in bucket.params)
        log_strs.append(
            f"Params for bucket {index + 1} ({numel} elements, "
            f"{bucket.grad_data.nelement()} padded size):"
        )
        for param in bucket.params:
            log_strs.append(f"\t{param_to_name[param]}")
    log_on_each_pipeline_stage(
        logger,
        logging.INFO,
        "\n".join(log_strs),
        tp_group=self.tp_group,
        dp_cp_group=self.dp_cp_group,
    )


# ---------------------------------------------------------------------------
# installation
# ---------------------------------------------------------------------------
def _verify_contracts():
    """Fail unless the installed Megatron still has the anchors these need.

    The two reproduced bodies are copies of upstream methods; if one is renamed
    or restructured the copy silently diverges. Checked by inspecting the
    installed source, so a version bump is caught here rather than by a
    checkpoint or an update that quietly does the wrong thing.
    """
    from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBuffer
    from megatron.core.optimizer import clip_grads
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
    from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher

    required = {
        "clip_grad_by_total_norm_fp32": (clip_grads, "clip_grad_by_total_norm_fp32"),
        "combine_postprocess": (MoEAlltoAllTokenDispatcher, "combine_postprocess"),
        "sharded_param_state_dp_reshardable": (
            DistributedOptimizer,
            "sharded_param_state_dp_reshardable",
        ),
    }
    missing = [n for n, (owner, attr) in required.items() if not hasattr(owner, attr)]
    if not hasattr(_ParamAndGradBuffer, "__init__"):
        missing.append("_ParamAndGradBuffer.__init__")
    if missing:
        raise RuntimeError(
            "Megatron no longer defines "
            + ", ".join(missing)
            + "; the Qwen4 model-type patches in roll/third_party/megatron/model_type_patches.py "
            "need re-deriving against the installed Megatron."
        )

    try:
        scale_source = inspect.getsource(clip_grads.clip_grad_by_total_norm_fp32)
        padding_source = inspect.getsource(DistributedOptimizer.sharded_param_state_dp_reshardable)
    except (OSError, TypeError) as exc:
        raise RuntimeError(
            "Cannot read the source of the Megatron functions these patches replace "
            f"({type(exc).__name__}: {exc}); review the patches against the installed Megatron."
        ) from exc

    # Anchors each reproduced body borrows. If upstream renames one, its copy is
    # no longer a faithful superset of the function being replaced.
    for label, source, marker in (
        ("clip_grad_by_total_norm_fp32", scale_source, "clip_coeff"),
        ("sharded_param_state_dp_reshardable", padding_source, "gbuf_local_start"),
    ):
        if marker not in source:
            raise RuntimeError(
                f"Megatron's {label} no longer contains {marker!r}; the reproduced copy in "
                "roll/third_party/megatron/model_type_patches.py is stale."
            )


def install_for_model_type(model_type: Optional[str]) -> bool:
    """Install the four patches for Flash-Next model types only.

    Returns whether the patches are installed after this call. Idempotent, and
    a no-op for every other model type, so unrelated models keep Megatron's own
    implementation of all four functions.
    """
    if not is_flash_next(model_type):
        return False

    from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBuffer
    from megatron.core.optimizer import clip_grads
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
    from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher

    if getattr(clip_grads, _INSTALLED_FLAG, False):
        return True

    _verify_contracts()

    setattr(clip_grads, _GRADIENT_CLIPPING_ORIGINAL, clip_grads.clip_grad_by_total_norm_fp32)
    clip_grads.clip_grad_by_total_norm_fp32 = _clip_grad_by_total_norm_fp32

    setattr(
        MoEAlltoAllTokenDispatcher, _MOE_CACHE_ORIGINAL, MoEAlltoAllTokenDispatcher.combine_postprocess
    )
    MoEAlltoAllTokenDispatcher.combine_postprocess = _combine_postprocess

    setattr(
        DistributedOptimizer,
        _CHECKPOINT_PADDING_ORIGINAL,
        DistributedOptimizer.sharded_param_state_dp_reshardable,
    )
    DistributedOptimizer.sharded_param_state_dp_reshardable = _sharded_param_state_dp_reshardable

    setattr(_ParamAndGradBuffer, _DDP_MEMORY_ORIGINAL, _ParamAndGradBuffer.__init__)
    _ParamAndGradBuffer.__init__ = _param_and_grad_buffer_init

    setattr(clip_grads, _INSTALLED_FLAG, True)
    logger.info("Installed the Qwen4 model-type Megatron patches for model_type=%s", model_type)
    return True


def uninstall() -> bool:
    """Restore Megatron's implementations. For tests and scope assertions."""
    from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBuffer
    from megatron.core.optimizer import clip_grads
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
    from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher

    if not getattr(clip_grads, _INSTALLED_FLAG, False):
        return False

    clip_grads.clip_grad_by_total_norm_fp32 = getattr(clip_grads, _GRADIENT_CLIPPING_ORIGINAL)
    delattr(clip_grads, _GRADIENT_CLIPPING_ORIGINAL)

    MoEAlltoAllTokenDispatcher.combine_postprocess = getattr(
        MoEAlltoAllTokenDispatcher, _MOE_CACHE_ORIGINAL
    )
    delattr(MoEAlltoAllTokenDispatcher, _MOE_CACHE_ORIGINAL)

    DistributedOptimizer.sharded_param_state_dp_reshardable = getattr(
        DistributedOptimizer, _CHECKPOINT_PADDING_ORIGINAL
    )
    delattr(DistributedOptimizer, _CHECKPOINT_PADDING_ORIGINAL)

    _ParamAndGradBuffer.__init__ = getattr(_ParamAndGradBuffer, _DDP_MEMORY_ORIGINAL)
    delattr(_ParamAndGradBuffer, _DDP_MEMORY_ORIGINAL)

    delattr(clip_grads, _INSTALLED_FLAG)
    return True
