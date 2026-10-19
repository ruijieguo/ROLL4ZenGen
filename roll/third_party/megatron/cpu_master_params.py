"""Rebuild disposable model shards from authoritative CPU optimizer masters."""

import torch


def cpu_master_shard_pairs(optimizer):
    """Validate every shard before any model buffer is discarded.

    Hybrid owns one FP32 master per local DP shard. This deliberately rejects
    partial CPU offload and ordinary FP32 model parameters, whose originals
    could themselves be authoritative optimizer storage.
    """
    if any(optimizer.shard_fp32_groups):
        raise ValueError("CPU master model offload does not support FP32 model parameters")
    parameters = [p for group in optimizer.shard_float16_groups for p in group]
    hybrid = optimizer.optimizer
    originals = [p for group in hybrid.param_groups for p in group["params"]]
    if len(originals) != len(parameters) or {id(p) for p in originals} != {id(p) for p in parameters}:
        raise ValueError("CPU master model offload requires complete Hybrid shard ownership")
    pairs = []
    for parameter in parameters:
        master = hybrid.param_to_fp32_param.get(parameter)
        if (
            not isinstance(master, torch.Tensor)
            or master.device.type != "cpu"
            or master.dtype != torch.float32
            or master.shape != parameter.shape
            or not master.is_contiguous()
            or parameter.dtype not in (torch.bfloat16, torch.float16)
            or hybrid.gpu_params_map_cpu_copy.get(parameter) is not master
        ):
            raise ValueError("CPU master model offload requires an authoritative CPU FP32 master for every shard")
        pairs.append((parameter, master))
    return pairs


@torch.no_grad()
def restore_cpu_master_shards(pairs, chunk_bytes=64 * 1024 * 1024):
    """Cast directly into current model shards, with bounded transfer staging."""
    if chunk_bytes < 4:
        raise ValueError("chunk_bytes must fit at least one FP32 element")
    chunk_elements = chunk_bytes // 4
    for parameter, master in pairs:
        destination, source = parameter.view(-1), master.view(-1)
        for start in range(0, master.numel(), chunk_elements):
            # Blocking transfers make ownership explicit before DP all-gather.
            destination[start:start + chunk_elements].copy_(source[start:start + chunk_elements])


def empty_cpu_parameter_buffer(numel, dtype):
    """Retain shape metadata only; values are unreadable until model reload."""
    return torch.zeros(1, dtype=dtype, device="cpu").expand(numel)


class CpuMasterWeightProvider:
    """Reconstruct one model weight without reloading its complete DDP buffer.

    The model may contain zero-storage CPU placeholders. Only the authoritative
    Hybrid FP32 masters supply parameter values. Disjoint DP slices are summed
    on the communication device before the ordinary TP/EP converter runs.
    No reconstructed tensor is cached; memory is bounded by the caller's bucket
    and its largest individual parameter. Construct anew for each update because
    normal offload/reload rebinds the optimizer's shard objects.
    """

    def __init__(self, models, optimizer, device="cuda"):
        self.device = torch.device(device)
        error = None
        try:
            partitions = self._local_partitions(models, optimizer)
        except Exception as exc:
            error = exc
        self._agree_validation(error)
        # Finish every child group's collectives before rejecting coverage.
        # Expert-DP subgroups eventually join the common TP/EP converter.
        error = None
        for group, parameters, intervals in partitions:
            world = torch.distributed.get_world_size(group) if group is not None else 1
            all_intervals = [intervals]
            if world > 1 and intervals:
                local = torch.tensor(intervals, dtype=torch.int64, device=self.device)
                gathered = [torch.empty_like(local) for _ in range(world)]
                torch.distributed.all_gather(gathered, local, group=group)
                all_intervals = [value.cpu().tolist() for value in gathered]
            try:
                self._validate_coverage(parameters, all_intervals)
            except ValueError as exc:
                error = exc
        self._agree_validation(error)

    def _agree_validation(self, error):
        """All actor ranks reject before entering the next collective phase."""
        failed = torch.tensor(int(error is not None), dtype=torch.int32, device=self.device)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(failed, op=torch.distributed.ReduceOp.MAX)
        if failed.item():
            if error is not None:
                raise ValueError(str(error)) from error
            raise ValueError("CPU master weight streaming validation failed on another rank")

    def _local_partitions(self, models, optimizer):
        if len(models) != 1 or models[0].config.pipeline_model_parallel_size != 1:
            raise ValueError("CPU master weight streaming currently requires PP1 with one model")
        self.parameters = dict(models[0].named_parameters())
        self.buffers = dict(models[0].named_buffers())
        self.entries = {}
        names = {id(p): name for name, p in self.parameters.items()}
        leaves = getattr(optimizer, "chained_optimizers", [optimizer])
        partitions = []
        for leaf in leaves:
            if not getattr(leaf.config, "offload_model_from_cpu_master", False):
                raise ValueError("CPU master weight streaming requires complete CPU master offload")
            masters = {id(p): master for p, master in cpu_master_shard_pairs(leaf)}
            owned = {}
            for index, group in enumerate(leaf.opt_group_ranges):
                shards = leaf.shard_float16_groups[index]
                if len(shards) != len(group["params"]):
                    raise ValueError("CPU master shard count does not match optimizer ranges")
                for parameter, shard in zip(group["params"], shards):
                    buffer, dtype, bucket = leaf.model_param_gbuf_map[parameter]
                    interval = leaf.gbuf_ranges[buffer][dtype][bucket]["param_map"][parameter]["param"]
                    master = masters[id(shard)]
                    if not (0 <= interval.start < interval.end <= parameter.numel()
                            and interval.end - interval.start == master.numel()):
                        raise ValueError("CPU master shard range does not match its parameter")
                    owned[id(parameter)] = (interval.start, interval.end, master)
            parameters = [p for b in leaf.buffers for p in b.params]
            if set(owned) - {id(p) for p in parameters}:
                raise ValueError("CPU master shard has no matching model parameter buffer")
            intervals = []
            for parameter in parameters:
                name = names.get(id(parameter))
                if name is None or name in self.entries:
                    raise ValueError("CPU master parameter ownership is missing or ambiguous")
                segment = owned.get(id(parameter))
                intervals.append(segment[:2] if segment is not None else (-1, -1))
                self.entries[name] = (leaf.data_parallel_group, segment)
            partitions.append((leaf.data_parallel_group, parameters, intervals))
        if self.parameters.keys() != self.entries.keys():
            raise ValueError("CPU master weight streaming requires every model parameter to be owned")
        return partitions

    @staticmethod
    def _validate_coverage(parameters, all_intervals):
        # Megatron buffers use the same parameter order within each DP group.
        for index, parameter in enumerate(parameters):
            ranges = sorted(row[index] for row in all_intervals if row[index][0] >= 0)
            end = 0
            for start, stop in ranges:
                if start != end or stop <= start:
                    raise ValueError("CPU master DP shards overlap or leave a gap")
                end = stop
            if end != parameter.numel():
                raise ValueError("CPU master DP shards do not cover the complete parameter")

    @torch.no_grad()
    def __call__(self, name, metadata):
        parameter = self.parameters.get(name)
        if parameter is None:
            buffer = self.buffers.get(name)
            if buffer is None or buffer.shape != metadata.shape or buffer.dtype != metadata.dtype:
                raise ValueError(f"Unknown or mismatched CPU master export key: {name}")
            return buffer.to(device=self.device, copy=True)
        if parameter.shape != metadata.shape or parameter.dtype != metadata.dtype:
            raise ValueError(f"CPU master export metadata mismatch: {name}")
        group, segment = self.entries[name]
        result = torch.zeros(parameter.shape, dtype=parameter.dtype, device=self.device)
        if segment is not None:
            start, stop, master = segment
            restore_cpu_master_shards([(result.view(-1)[start:stop], master)])
        if group is not None and torch.distributed.get_world_size(group) > 1:
            torch.distributed.all_reduce(result, group=group)
        return result


# ---------------------------------------------------------------------------
# Opt-in bounded CPU gradient staging
# ---------------------------------------------------------------------------
# Megatron's HybridDeviceOptimizer stages every CPU parameter's gradient in a
# per-parameter cache (``cpu_copy_map_grad``) that is never released, so the
# cache grows with the parameter count and a full-model CPU optimizer holds a
# second complete gradient image. This replaces the staging path with one
# reusable buffer sized to the largest CPU parameter, allocating gradients
# immediately before each sub-optimizer's update.
#
# Replaces scripts/qwen38/patch_megatron_cpu_grad_staging.py: it is installed
# at runtime for the Flash-Next model type instead of requiring every user of
# the model to patch their own Megatron checkout. Behaviour is unchanged --
# every change is behind the ``bounded_cpu_grad_staging`` flag, which defaults
# off, and the unbounded path delegates to Megatron's own implementation.

_BOUNDED_INSTALLED = "_qwen4_bounded_cpu_grad_staging_installed"
_BOUNDED_SAVED = "_qwen4_bounded_cpu_grad_staging_saved"
_ORIGINAL_SET_MAIN_PARAM = "_qwen4_unpatched_set_main_param_and_optimizer_states"

# The methods this module replaces or adds, by the name they live under on
# HybridDeviceOptimizer. Installing records what was there under each name so
# uninstall can put it back: a Megatron carrying the legacy source patches
# already defines some of these, and deleting them would be wrong.
_REPLACED = ("__init__", "step", "_set_sub_optimizer_grads")
_ADDED = ("_prepare_bounded_cpu_grad_staging", "_stage_bounded_cpu_optimizer_grad")


def _param_generator(optimizer):
    for group in optimizer.param_groups:
        for param in group["params"]:
            yield param


def _hybrid_optimizer_class():
    from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer

    return HybridDeviceOptimizer


def _original(name):
    """The method this module replaced, captured at install time.

    Held in one record rather than a fixed attribute per method so uninstall
    can put back exactly what was there, including methods a patched dependency
    already provided.
    """
    saved = getattr(_hybrid_optimizer_class(), _BOUNDED_SAVED, None)
    if saved is None or saved.get(name) is None:
        raise RuntimeError(
            f"bounded_cpu_grad_staging is not installed (no saved original for {name!r}); "
            "install_bounded_cpu_grad_staging must run before any update step"
        )
    return saved[name]


def _prepare_bounded_cpu_grad_staging(self):
    """Reserve one reusable buffer, never a per-parameter gradient cache."""
    if not self.overlap_cpu_optimizer_d2h_h2d:
        raise ValueError('bounded_cpu_grad_staging requires overlap_cpu_optimizer_d2h_h2d=True')
    largest = 0
    for optimizer in self.cpu_optimizers:
        params = list(_param_generator(optimizer))
        if len(params) != 1:
            raise ValueError('bounded_cpu_grad_staging requires one parameter per CPU optimizer')
        param = params[0]
        param.grad = None
        largest = max(largest, param.numel() * param.element_size())
    self.cpu_copy_map_grad.clear()
    # Held on the instance rather than initialized in _init_sub_optimizers: the
    # checkpoint path deliberately skips re-initialization in bounded mode, and
    # a buffer reset there would be immediate garbage.
    buffer = getattr(self, '_cpu_grad_staging_buffer', None)
    if largest and buffer is None:
        # A byte buffer also supports mixed parameter dtypes without separate pools.
        buffer = torch.empty(largest, dtype=torch.uint8, device='cpu', pin_memory=self.pin_cpu_grads)
        self._cpu_grad_staging_buffer = buffer
    if largest and buffer.numel() < largest:
        raise RuntimeError('CPU parameter size changed after staging buffer allocation')


def _stage_bounded_cpu_optimizer_grad(self, optimizer):
    param = next(_param_generator(optimizer))
    gpu_param = self.cpu_copys_map_gpu_param[param]
    grad = getattr(gpu_param, 'decoupled_grad', gpu_param.grad)
    param.requires_grad = False
    if grad is None:
        param.grad = None
        return
    size = param.numel() * param.element_size()
    staged = self._cpu_grad_staging_buffer[:size].view(param.dtype).view(param.shape)
    staged.copy_(grad, non_blocking=True)
    param.grad = staged
    self.cpu_copy_map_grad[param] = staged
    self._cpu_optimizer_map_data_event[optimizer] = self._d2h_stream.record_event()


def _bounded_step(self, closure=None):
    """Megatron's step with each CPU gradient staged just before its update.

    Mirrors the ordering of the unbounded step; only the staging points differ,
    which is what bounds the retained memory. Kept as one body rather than
    delegating because the staging has to happen inside the per-optimizer loop.
    """
    self._sync_hdo_param_groups_to_sub_optimizers()
    self._prepare_bounded_cpu_grad_staging()
    self._d2h_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(self._d2h_stream):
        self._set_sub_optimizer_grads()

    if self.gpu_optimizer:
        torch.cuda.current_stream().wait_stream(self._d2h_stream)
        self.gpu_optimizer.step(closure)

    for cpu_optimizer in self.cpu_optimizers:
        with torch.cuda.stream(self._d2h_stream):
            self._stage_bounded_cpu_optimizer_grad(cpu_optimizer)
        d2h_event = self._cpu_optimizer_map_data_event.pop(cpu_optimizer, None)
        if d2h_event is not None:
            d2h_event.synchronize()
        try:
            cpu_optimizer.step(closure)
        finally:
            # CPU Adam consumes the gradient synchronously. H2D uses the master
            # parameter, so the gradient buffer is now reusable.
            for param in _param_generator(cpu_optimizer):
                param.grad = None
                self.cpu_copy_map_grad.pop(param, None)

    # Subsequent GPU work must observe all asynchronous master copies.
    torch.cuda.current_stream().wait_stream(self._h2d_stream)
    self._sync_sub_optimizers_state_to_hdo()


def _bounded_step_dispatch(self, closure=None):
    if not getattr(self, "bounded_cpu_grad_staging", False):
        return _original("step")(self, closure)
    return _bounded_step(self, closure)


def _bounded_init(self, *args, bounded_cpu_grad_staging=False, **kwargs):
    """Megatron's constructor plus the opt-in flag, consumed not forwarded.

    ``bounded_cpu_grad_staging`` is deliberately not passed into ``defaults``:
    it is a ROLL option, not an optimizer hyperparameter, and torch rejects
    unknown keys when param groups are added.
    """
    _original("__init__")(self, *args, **kwargs)
    self.bounded_cpu_grad_staging = bounded_cpu_grad_staging
    if bounded_cpu_grad_staging and not self.overlap_cpu_optimizer_d2h_h2d:
        raise ValueError('bounded_cpu_grad_staging requires overlap_cpu_optimizer_d2h_h2d=True')


def _bounded_set_sub_optimizer_grads(self):
    """Megatron's method with the per-parameter gradient cache left alone.

    The unbounded body ends by copying every CPU gradient into
    ``cpu_copy_map_grad``; bounded mode stages each one later instead. The
    original is called with no CPU optimizers visible so that second loop does
    nothing, which keeps the FP32-alias half Megatron's own.
    """
    original = _original("_set_sub_optimizer_grads")
    if not getattr(self, "bounded_cpu_grad_staging", False):
        return original(self)
    cpu_optimizers = self.cpu_optimizers
    try:
        self.cpu_optimizers = []
        original(self)
    finally:
        self.cpu_optimizers = cpu_optimizers


def _bounded_set_main_param_and_optimizer_states(self, model_param, tensors):
    """Megatron's setter, for the two defects in this path.

    ``padding`` entries describe gaps in the checkpoint bucket, not optimizer
    state (they come from a separate ``hybrid_checkpoint`` concern but land in
    the same method), so they are dropped for every caller. In bounded mode the
    master is additionally restored into the tensor its sub-optimizer actually
    reads: ``load_parameter_state_from_fs_model_space`` runs after
    ``Hybrid.load_state_dict`` re-points state at the live FP32 owners, and a
    replacement state entry alone would leave CPU Adam consuming a stale
    tensor.
    """
    original = getattr(type(self), _ORIGINAL_SET_MAIN_PARAM)
    config = getattr(self, "config", None)
    hybrid = _hybrid_optimizer_class()
    tensors = {key: value for key, value in tensors.items() if key != "padding"}
    if (
        isinstance(getattr(self, "optimizer", None), hybrid)
        and getattr(self.optimizer, "bounded_cpu_grad_staging", False)
        and getattr(config, "use_precision_aware_optimizer_no_fp8_or_ds_fp8", False)
    ):
        group_index, group_order = self.model_param_group_index_map[model_param]
        sharded = self.optimizer.param_groups[group_index]["params"][group_order]
        inner_param = self.optimizer.param_to_inner_param[sharded]
        for key, value in tensors.items():
            if key == "param":
                inner_param.data.copy_(value)
                self.optimizer.state[sharded]["master_param"] = inner_param
            else:
                self.optimizer.state[sharded][key] = value.to(inner_param.device)
        return None
    return original(self, model_param, tensors)


def _verify_bounded_contract():
    """Fail unless Megatron still has the methods this replacement wraps.

    The bounded path mirrors Megatron's update ordering, so the anchors it
    borrows must still exist. Checked by source inspection so a refactor that
    keeps them still passes; raising here is the point, since a renamed hook
    would otherwise only show up as a wrong update or a lost checkpoint.
    """
    import inspect

    hybrid = _hybrid_optimizer_class()
    required = {
        "_set_sub_optimizer_grads": "the gradient hand-off this stages around",
        "_sync_hdo_param_groups_to_sub_optimizers": "the pre-step parameter-group sync",
        "_sync_sub_optimizers_state_to_hdo": "the post-step state sync",
    }
    missing = [name for name in required if not hasattr(hybrid, name)]
    if missing:
        raise RuntimeError(
            "Megatron's HybridDeviceOptimizer lacks "
            + ", ".join(f"{name} ({required[name]})" for name in missing)
            + ", so bounded_cpu_grad_staging cannot reproduce its update order. "
            "Re-derive the patch in roll/third_party/megatron/cpu_master_params.py."
        )
    try:
        source = inspect.getsource(hybrid.step)
    except (OSError, TypeError) as exc:
        raise RuntimeError(
            "Cannot read Megatron's HybridDeviceOptimizer.step "
            f"({type(exc).__name__}: {exc}); bounded_cpu_grad_staging mirrors it and cannot "
            "be verified. Review that patch against the installed Megatron."
        ) from exc
    for marker in ("_sync_hdo_param_groups_to_sub_optimizers", "self._d2h_stream",
                   "for cpu_optimizer in self.cpu_optimizers", "self._sync_sub_optimizers_state_to_hdo"):
        if marker not in source:
            raise RuntimeError(
                f"Megatron's HybridDeviceOptimizer.step no longer contains {marker!r}; the "
                "bounded copy in roll/third_party/megatron/cpu_master_params.py is stale."
            )


def install_bounded_cpu_grad_staging() -> bool:
    """Install the bounded-staging replacement on the installed Megatron.

    Idempotent. Called only when the ROLL config asks for bounded staging, so a
    run without the flag never touches these classes. Every attribute the
    install touches is recorded beforehand, so uninstall restores exactly what
    was there -- a dependency that already carries some of these methods (an
    older Megatron with the legacy source patches) is put back, not stripped.
    """
    hybrid = _hybrid_optimizer_class()
    if getattr(hybrid, _BOUNDED_INSTALLED, False):
        return True
    _verify_bounded_contract()

    saved = {name: hybrid.__dict__.get(name) for name in _REPLACED + _ADDED}
    hybrid.__init__ = _bounded_init
    hybrid.step = _bounded_step_dispatch
    hybrid._set_sub_optimizer_grads = _bounded_set_sub_optimizer_grads
    hybrid._prepare_bounded_cpu_grad_staging = _prepare_bounded_cpu_grad_staging
    hybrid._stage_bounded_cpu_optimizer_grad = _stage_bounded_cpu_optimizer_grad
    setattr(hybrid, _BOUNDED_SAVED, saved)

    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    if not hasattr(DistributedOptimizer, _ORIGINAL_SET_MAIN_PARAM):
        setattr(DistributedOptimizer, _ORIGINAL_SET_MAIN_PARAM,
                DistributedOptimizer._set_main_param_and_optimizer_states)
        DistributedOptimizer._set_main_param_and_optimizer_states = _bounded_set_main_param_and_optimizer_states

    setattr(hybrid, _BOUNDED_INSTALLED, True)
    return True


def uninstall_bounded_cpu_grad_staging() -> bool:
    """Restore exactly what was present before install. For scope tests."""
    hybrid = _hybrid_optimizer_class()
    if not getattr(hybrid, _BOUNDED_INSTALLED, False):
        return False
    for name, previous in getattr(hybrid, _BOUNDED_SAVED, {}).items():
        if previous is None:
            # Absent before the install; drop the attribute rather than keep
            # a method this module added.
            delattr(hybrid, name)
        else:
            setattr(hybrid, name, previous)
    delattr(hybrid, _BOUNDED_SAVED)
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    original = getattr(DistributedOptimizer, _ORIGINAL_SET_MAIN_PARAM, None)
    if original is not None:
        DistributedOptimizer._set_main_param_and_optimizer_states = original
        delattr(DistributedOptimizer, _ORIGINAL_SET_MAIN_PARAM)
    delattr(hybrid, _BOUNDED_INSTALLED)
    return True
