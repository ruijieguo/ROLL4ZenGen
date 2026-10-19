"""Qwen4 GDN keeps convolution activation separate from the output gate."""
from __future__ import annotations

import inspect
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from megatron.core.ssm.gated_delta_net import GatedDeltaNet

# Megatron's GatedDeltaNet.forward, in the two layouts this model supports. The
# overrides below only act if forward still reaches these points; anything else
# means the override is dead code and the numerical fix it carries is absent.
#
#   hook layout   megatron-core >= 0.18: forward calls self._compute_g_and_beta,
#                 which this subclass replaces outright.
#   inline layout megatron-core 0.16 with scripts/qwen38/patch_megatron_gdn_
#                 {decay,gate_precision}.py applied: the FP32 forms sit in
#                 forward itself, so the decay and write-gate override is
#                 redundant rather than wrong. The convolution rebind still
#                 applies, because the source patch's method calls the same
#                 module-level name.
HOOK_LAYOUT_CALL = "g, beta = self._compute_g_and_beta("
INLINE_LAYOUT_DECAY = "A_log_local_cp.float().exp()"
INLINE_LAYOUT_BETA = "beta.float().sigmoid()"
MODULE_LEVEL_CONVOLUTION = "qkv, _ = causal_conv1d("
PATCHED_CONVOLUTION = "qkv = self._apply_causal_conv1d("


def verify_upstream_route() -> str:
    """Fail unless the overrides below can take effect; report which layout.

    ``_compute_g_and_beta`` and the causal-convolution rebind change nothing
    merely by existing: the first runs only if Megatron's ``forward`` calls it,
    the second only if ``forward`` reaches the convolution through the
    module-level name it rebinds. Megatron 0.16 does neither -- it computes the
    decay, the write gate and the convolution inline -- so on an unpatched 0.16
    tree these overrides are dead code and the FP32 fixes vanish with no
    symptom. That is the one failure this model cannot detect on its own, so it
    is checked here, before training, by inspecting the source rather than by
    comparing version numbers, so a backport or a refactor that keeps the
    routing still passes.
    """
    from megatron.core.ssm.gated_delta_net import GatedDeltaNet

    try:
        source = inspect.getsource(GatedDeltaNet.forward)
    except (OSError, TypeError) as exc:
        raise RuntimeError(
            "Cannot read the source of Megatron's GatedDeltaNet.forward "
            f"({type(exc).__name__}: {exc}), so the Qwen4ExpGatedDeltaNet overrides "
            "cannot be verified as reachable. Review this model against the installed "
            "Megatron."
        ) from exc

    if hasattr(GatedDeltaNet, "_compute_g_and_beta"):
        if HOOK_LAYOUT_CALL not in source:
            raise RuntimeError(
                "Megatron defines GatedDeltaNet._compute_g_and_beta but its forward never "
                "calls it, so the Qwen4ExpGatedDeltaNet FP32 decay and write-gate override "
                "would be silently skipped. Re-derive that override against the installed "
                "Megatron."
            )
        layout = "hook"
    else:
        missing = [
            name
            for name, text in (
                ("the FP32 decay exponential", INLINE_LAYOUT_DECAY),
                ("the FP32 write gate", INLINE_LAYOUT_BETA),
            )
            if text not in source
        ]
        if missing:
            raise RuntimeError(
                "Megatron's GatedDeltaNet.forward does not define _compute_g_and_beta and "
                "does not carry " + " or ".join(missing) + " inline, so this model would "
                "train with BF16 decay and write-gate rounding and no fix in either the "
                "override or the source. Install a Megatron that defines "
                "_compute_g_and_beta, or apply scripts/qwen38/patch_megatron_gdn_decay.py "
                "and patch_megatron_gdn_gate_precision.py to the dependency."
            )
        layout = "inline"

    if MODULE_LEVEL_CONVOLUTION not in source and PATCHED_CONVOLUTION not in source:
        raise RuntimeError(
            "Megatron's GatedDeltaNet.forward does not reach a causal convolution this "
            "model can intercept, so the native-rounding override would be silently "
            "skipped. Re-derive it against the installed Megatron."
        )
    return layout


class Qwen4ExpGatedDeltaNet(GatedDeltaNet):
    def __init__(self, *args, **kwargs):
        # The overrides below act inside forward, so nothing at import time can
        # tell whether they will be reached. Checked here, before any training,
        # rather than silently running with unfixed numerics.
        verify_upstream_route()
        super().__init__(*args, **kwargs)

    def _compute_g_and_beta(self, A_log_local_cp, dt_bias_local_cp, alpha, beta):
        """Keep the decay exponential and the write gate in FP32.

        Two distinct quantities, both rounded too early upstream.

        ``g`` is the log-domain decay. Megatron already evaluates the softplus
        term in FP32 but exponentiates ``A_log`` in the parameter dtype, so a
        BF16 checkpoint rounds the decay base before the multiply. The
        recurrence applies ``g`` once per step, so that error compounds along
        the sequence instead of staying in the last bit.

        ``beta`` is the write strength handed to the delta rule. Upstream calls
        ``sigmoid`` in the input dtype; promoting first stops the saturating
        ends of the curve from collapsing onto one BF16 value.

        Replaces scripts/qwen38/patch_megatron_gdn_{decay,gate_precision}.py.
        The parent carries ``@jit_fuser``; overriding gives up that fusion for
        these few elementwise ops, which is the cost of the added precision.
        """
        g = -A_log_local_cp.float().exp() * F.softplus(alpha.float() + dt_bias_local_cp)
        return g, beta.float().sigmoid()

    @staticmethod
    def _split_input_projection(module, hidden_states, qkvz_rows):
        """Run the native Qwen3.8 two-GEMM input projection.

        The checkpoint intentionally keeps one Megatron ``in_proj`` parameter
        (q, k, v, gate, beta, alpha) for DCP and LoRA compatibility.  Native
        Qwen3.8 computes ``qkvz`` and ``ba`` with separate linear layers.  A
        single BF16 GEMM over the concatenated rows changes the accumulation
        rounding of beta/alpha, and the small difference is amplified by the
        recurrent GDN update.  Splitting the existing local TP weight at
        runtime preserves the checkpoint layout while matching the native
        arithmetic.

        ``module`` may be a plain Transformer Engine column linear or the
        adapter's ``LoraColumnParallelLinear`` wrapper.  The base projection is
        computed from the local TP weight. Preserve column-parallel input
        communication here; the parent GDN then performs CP movement.
        """
        if getattr(module, "disable_adapters", False) and getattr(module, "merged", False):
            module.unmerge()
        base = module.get_base_layer() if hasattr(module, "get_base_layer") else module
        weight = getattr(base, "weight", None)
        if weight is None or weight.ndim != 2:
            raise TypeError(
                "Qwen4 native GDN projection requires a 2-D local weight; "
                f"got {type(base).__name__}"
            )
        if not 0 < qkvz_rows < weight.shape[0]:
            raise ValueError(
                f"invalid Qwen4 GDN projection split {qkvz_rows} for {tuple(weight.shape)}"
            )
        bias = getattr(base, "bias", None)
        # Transformer Engine represents ``bias=False`` as an empty parameter
        # on some releases.  Passing that zero-length tensor to F.linear is
        # interpreted as a real bias and fails only when the first GDN layer
        # executes, so normalize it to the native no-bias path here.
        if bias is not None and bias.numel() == 0:
            bias = None
        qkvz_bias = None if bias is None else bias[:qkvz_rows]
        ba_bias = None if bias is None else bias[qkvz_rows:]
        projection_input = hidden_states
        tp_group = getattr(base, "tp_group", None)
        if tp_group is not None and tp_group.size() > 1:
            from megatron.core.tensor_parallel.mappings import (
                copy_to_tensor_model_parallel_region,
                gather_from_sequence_parallel_region,
            )

            if getattr(base, "sequence_parallel", False):
                projection_input = gather_from_sequence_parallel_region(
                    hidden_states, tensor_parallel_output_grad=True, group=tp_group,
                )
            else:
                projection_input = copy_to_tensor_model_parallel_region(hidden_states, group=tp_group)
        qkvz = F.linear(projection_input, weight[:qkvz_rows], qkvz_bias)
        ba = F.linear(projection_input, weight[qkvz_rows:], ba_bias)
        result = torch.cat((qkvz, ba), dim=-1)

        # Preserve the existing adapter semantics.  LoRA-B has the same fused
        # output layout as the base weight, so splitting its result at the same
        # boundary is exact while leaving adapter state/checkpoint keys intact.
        if (
            hasattr(module, "active_adapters")
            and not getattr(module, "disable_adapters", False)
            and not getattr(module, "merged", False)
        ):
            if getattr(module, "sequence_parallel", False) and getattr(base, "parallel_mode", None) == "column":
                from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

                # LoRA-B is a TE column linear and already sums its input
                # gradient over TP. Scatter that replicated gradient back to
                # sequence shards without a second sum.
                lora_input = gather_from_sequence_parallel_region(
                    hidden_states, tensor_parallel_output_grad=False, group=tp_group,
                )
            else:
                lora_input = hidden_states
            for adapter_name in module.active_adapters:
                if adapter_name not in module.lora_A:
                    continue
                lora_a = module.lora_A[adapter_name]
                lora_b = module.lora_B[adapter_name]
                dropout = module.lora_dropout[adapter_name]
                delta = lora_a(dropout(lora_input.to(lora_a.weight.dtype)))
                if isinstance(delta, tuple):
                    delta = delta[0]
                delta = lora_b(delta)
                if isinstance(delta, tuple):
                    delta = delta[0]
                scaling = module.scaling[adapter_name]
                if scaling != 1.0:
                    delta = delta * scaling
                result = result + delta
        # Match the shared LoRA wrapper's promoted accumulation, including
        # FP32 adapters on a BF16 base and multiple active adapters.
        return result.to(hidden_states.dtype), None

    @contextmanager
    def _native_input_projection(self):
        """Temporarily replace only ``in_proj.forward`` for parent GDN code."""
        module = self.in_proj
        # Qwen4's GR layer spec removes the fused input layernorm.  If a caller
        # constructs a non-GR layer with the norm-bearing TE module, retain the
        # original path rather than silently dropping that norm.
        if hasattr(module, "layer_norm_weight"):
            yield
            return
        original = module.forward
        # ``in_proj`` is column-parallel, so the runtime weight contains only
        # the local TP rows.  The native boundary is split after q/k/v/z on
        # every rank, before the local beta/alpha rows.
        qkvz_rows = (2 * self.qk_dim + 2 * self.v_dim) // self.tp_size

        def split_forward(_module, hidden_states, *args, **kwargs):
            if args or kwargs:
                raise TypeError("Qwen4 native GDN projection does not accept extra linear arguments")
            return self._split_input_projection(_module, hidden_states, qkvz_rows)

        module.forward = split_forward.__get__(module, type(module))
        try:
            yield
        finally:
            module.forward = original

    def forward(self, *args, **kwargs):
        # Keep Megatron's tested CP and recurrence implementation; retain
        # projection precision across the TP communication boundary and the
        # native rounding of the causal convolution.
        with (
            self._native_input_projection(),
            self._fp32_output_projection(),
            self._native_causal_convolution(),
        ):
            return super().forward(*args, **kwargs)

    @contextmanager
    def _fp32_output_projection(self):
        """Cast GDN row-projection results only after the FP32 TP sum.

        Rounding each rank's GEMM before reduction changes the forward result
        with TP size. PLE's signed square-root gate amplifies this difference
        in backward. Keep the checkpoint parameters and any LoRA wrapper;
        replace only the base row linear for the duration of this forward.
        """
        from .precision_linear import linear_with_fp32_output

        module = self.out_proj
        base = module.get_base_layer() if hasattr(module, "get_base_layer") else module
        original = base.forward

        def forward(_module, hidden_states, *args, **kwargs):
            if args or kwargs:
                raise TypeError("Qwen4 GDN row projection does not accept extra linear arguments")
            result = linear_with_fp32_output(hidden_states, _module.weight)
            group = getattr(_module, "tp_group", None)
            if group is not None and group.size() > 1:
                from megatron.core.tensor_parallel.mappings import (
                    reduce_from_tensor_model_parallel_region,
                    reduce_scatter_to_sequence_parallel_region,
                )

                if getattr(_module, "sequence_parallel", False):
                    result = reduce_scatter_to_sequence_parallel_region(result, group=group)
                else:
                    result = reduce_from_tensor_model_parallel_region(result, group=group)
            result = result.to(hidden_states.dtype)
            bias = getattr(_module, "bias", None)
            if bias is not None and bias.numel() == 0:
                bias = None
            if bias is not None and not getattr(_module, "skip_bias_add", False):
                result = result + bias
                bias = None
            return result, bias

        base.forward = forward.__get__(base, type(base))
        try:
            yield
        finally:
            base.forward = original

    @contextmanager
    def _native_causal_convolution(self):
        """Swap FLA's ``causal_conv1d`` for the native-rounding one.

        Megatron calls the FLA kernel through its own module namespace, so
        rebinding that attribute for the duration of this forward replaces the
        implementation without a source patch. Replaces
        scripts/qwen38/patch_megatron_gdn_convolution.py, which inserted an
        ``_apply_causal_conv1d`` hook into the dependency instead.

        Only the non-deterministic branch routes here. ``deterministic_mode``
        uses ``F.conv1d`` directly, and that path is left alone.
        """
        from megatron.core.ssm import gated_delta_net as megatron_gdn

        from .causal_convolution import causal_conv1d as native_causal_conv1d

        original = getattr(megatron_gdn, "causal_conv1d", None)
        if original is None:
            # FLA missing entirely: upstream asserts on this before reaching the
            # call, so do not mask it with a silent substitution.
            yield
            return

        def shim(
            x,
            weight,
            bias=None,
            activation=None,
            initial_state=None,
            output_final_state=False,
            cu_seqlens=None,
            **kwargs,
        ):
            if initial_state is not None or output_final_state:
                raise NotImplementedError(
                    "Qwen4 native causal convolution does not carry conv state; "
                    "Megatron requested recurrent-cache behaviour"
                )
            if activation not in ("silu", "swish"):
                raise ValueError(f"Qwen4 native causal convolution expects silu/swish, got {activation!r}")
            if cu_seqlens is not None:
                # Variable-length packing would make the causal window span
                # sequence boundaries. Flash-Next Megatron roles already reject
                # packed sequences (use_sequence_packing=false), so this is
                # unreachable rather than merely unsupported.
                raise NotImplementedError("Qwen4 native causal convolution does not support packed cu_seqlens")
            return native_causal_conv1d(x, weight, bias), None

        megatron_gdn.causal_conv1d = shim
        try:
            yield
        finally:
            megatron_gdn.causal_conv1d = original

    def _apply_gated_norm(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """Round after normalization, affine scaling and the output gate.

        Keep the existing norm parameter for checkpoint and optimizer compatibility.
        Calling the norm module first would round its BF16 output before gating.
        """
        x = x.reshape(-1, x.shape[-1])
        gate = gate.reshape(-1, gate.shape[-1])
        weight = self.out_norm.weight.float()
        if self.out_norm.zero_centered_gamma:
            weight = weight + 1
        activation = self.config.gdn_output_gate_type
        if x.is_cuda:
            if activation == "sigmoid":
                from .gated_norm import sigmoid_gated_rms_norm

                return sigmoid_gated_rms_norm(x, gate, weight, self.out_norm.eps)

            from fla.modules.fused_norm_gate import rms_norm_gated

            return rms_norm_gated(x, gate, weight, None, activation=activation, eps=self.out_norm.eps)

        # CPU reference avoids allocating CUDA state for metadata/tests.
        xf = x.float()
        normalized = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + self.out_norm.eps)
        output_gate = torch.sigmoid(gate.float()) if activation == "sigmoid" else self.act_fn(gate.float())
        return (normalized * weight * output_gate).to(x.dtype)
