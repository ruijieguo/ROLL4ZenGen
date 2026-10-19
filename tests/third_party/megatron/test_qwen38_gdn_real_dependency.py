"""Numeric checks for the GDN overrides, on real torch and real megatron-core.

``test_qwen38_gdn_route_contract.py`` pins *which* dependency layout the guard
accepts; this file pins *what the fix does to the numbers*, so that a change
that keeps the routing but loses the arithmetic is caught. Both import the
installed megatron-core rather than a stub.

The two quantities here are the ones the earlier source patches changed, so
these assertions are the executable form of those patches' intent.
"""

import pytest

torch = pytest.importorskip("torch")
megatron_gdn = pytest.importorskip("megatron.core.ssm.gated_delta_net")

from mcore_adapter.models.qwen4_exp.gated_delta_net import (  # noqa: E402
    Qwen4ExpGatedDeltaNet,
)

# Neither method reads ``self``, so a bare object drives them without building
# a real GatedDeltaNet (which needs a TransformerConfig and TE layers).
_LAYER = object()


# --------------------------------------------------------------------------
# decay and write gate
# --------------------------------------------------------------------------
def test_decay_matches_an_fp32_reference_exactly():
    """``g`` must be ``-(A_log.float().exp()) * softplus(alpha.float() + dt_bias)``.

    Upstream exponentiates ``A_log`` in the parameter dtype, so a BF16
    checkpoint rounds the decay base before the multiply and the recurrence
    compounds that along the sequence. Bit-exact against the FP32 reference.
    """
    torch.manual_seed(7)
    a_log = torch.randn(64, dtype=torch.bfloat16) * 2
    dt_bias = torch.randn(64, dtype=torch.bfloat16)
    alpha = torch.randn(4, 8, 64, dtype=torch.bfloat16)
    beta = torch.randn(4, 8, 64, dtype=torch.bfloat16)

    g, _ = Qwen4ExpGatedDeltaNet._compute_g_and_beta(_LAYER, a_log, dt_bias, alpha, beta)
    reference = -(a_log.float().exp()) * torch.nn.functional.softplus(alpha.float() + dt_bias.float())
    assert g.dtype == torch.float32
    torch.testing.assert_close(g, reference, rtol=0, atol=0)


def test_decay_differs_from_the_unpromoted_upstream_form():
    """The chosen inputs must actually expose the difference.

    A test that passes on both the fixed and the unfixed arithmetic would
    certify nothing; require the parameter-dtype path to disagree.
    """
    torch.manual_seed(7)
    a_log = torch.randn(64, dtype=torch.bfloat16) * 2
    dt_bias = torch.randn(64, dtype=torch.bfloat16)
    alpha = torch.randn(4, 8, 64, dtype=torch.bfloat16)
    beta = torch.randn(4, 8, 64, dtype=torch.bfloat16)

    g, _ = Qwen4ExpGatedDeltaNet._compute_g_and_beta(_LAYER, a_log, dt_bias, alpha, beta)
    dtypes_rounded = (-(a_log.exp()) * torch.nn.functional.softplus(alpha.float() + dt_bias)).float()
    assert not torch.equal(g, dtypes_rounded)


def test_write_gate_sigmoid_is_promoted_before_saturation():
    """``beta.float().sigmoid()`` vs ``beta.sigmoid()`` at the saturating ends.

    BF16 near +/-1 has too little resolution to keep distinct pre-activations
    distinct, so the gate collapses. 12.5 and 20 both saturate to 1.0 in BF16
    but not in FP32.
    """
    beta = torch.tensor([20.0, -20.0, 12.5, -12.5], dtype=torch.bfloat16)
    zeros = torch.zeros(4, dtype=torch.bfloat16)
    _, gate = Qwen4ExpGatedDeltaNet._compute_g_and_beta(_LAYER, torch.zeros(4), torch.zeros(4), zeros, beta)
    assert gate.dtype == torch.float32
    torch.testing.assert_close(gate, beta.float().sigmoid(), rtol=0, atol=0)
    assert beta.sigmoid()[0] == beta.sigmoid()[2], "BF16 is expected to saturate these to the same value"
    assert gate[0] != gate[2]


# --------------------------------------------------------------------------
# the causal convolution the shim hands back
# --------------------------------------------------------------------------
def test_shim_returns_the_native_cpu_convolution_result():
    """The rebind must return the Qwen4 native convolution's own output."""
    from mcore_adapter.models.qwen4_exp.causal_convolution import (
        causal_conv1d as native,
    )

    torch.manual_seed(11)
    x = torch.randn(2, 6, 8, dtype=torch.float32)
    weight = torch.randn(8, 3, dtype=torch.float32)
    bias = torch.randn(8, dtype=torch.float32)
    original = megatron_gdn.causal_conv1d
    with Qwen4ExpGatedDeltaNet._native_causal_convolution(_LAYER):
        result, state = megatron_gdn.causal_conv1d(
            x=x,
            weight=weight,
            bias=bias,
            activation="silu",
            initial_state=None,
            output_final_state=False,
        )
    assert megatron_gdn.causal_conv1d is original
    assert state is None
    torch.testing.assert_close(result, native(x, weight, bias), rtol=0, atol=0)


def test_native_convolution_differs_from_a_plain_depthwise_conv():
    """Rounding each product before accumulating is the point of the fix.

    If the native path equalled a fused ``conv1d`` + SiLU, the override would
    be carrying no arithmetic change and this file would not be testing a fix.
    The reference is depthwise over time (``groups=dim``), matching the
    ``weight[:, tap]`` indexing the implementation uses.
    """
    from torch.nn import functional as F

    from mcore_adapter.models.qwen4_exp.causal_convolution import (
        causal_conv1d as native,
    )

    torch.manual_seed(12)
    x = torch.randn(2, 6, 8, dtype=torch.bfloat16)
    weight = torch.randn(8, 3, dtype=torch.bfloat16)
    dim, width = x.shape[-1], weight.shape[-1]
    # conv1d requires matching dtypes, so the fused reference runs in BF16 too;
    # that is the whole contrast -- the native path rounds each product and
    # accumulates in FP32, this one accumulates in BF16.
    padded = F.pad(x, (0, 0, width - 1, 0)).transpose(1, 2)
    fused = F.conv1d(padded, weight.unsqueeze(1), groups=dim).transpose(1, 2)
    plain = F.silu(fused).to(x.dtype)
    assert not torch.equal(native(x, weight, None), plain)
