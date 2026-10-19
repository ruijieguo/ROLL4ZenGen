"""The GDN subclass overrides must be reached, scoped, and reversible.

``Qwen4ExpGatedDeltaNet`` replaces Megatron's decay and write-gate computation
(``_compute_g_and_beta``) and its causal convolution (a module-level rebind).
Neither does anything by merely existing: the first runs only if Megatron's
``forward`` calls it, the second only if ``forward`` reaches the convolution
through the name it rebinds. Megatron 0.16 computes both inline, so on an
unpatched 0.16 tree these overrides are dead code and the model trains with a
BF16 decay and a non-native convolution with no symptom at all -- which is why
``verify_upstream_route`` refuses to construct the layer there.

Everything here runs against the **installed** megatron-core: the guard reads
the real ``GatedDeltaNet.forward``, and the layouts it must accept and reject
are produced by substituting that method (and the hook attribute) on the real
class. The repository has the two layouts that matter -- megatron-core 0.18's
hook, and the .181 image's 0.16 tree carrying the scripts/qwen38 source patches
-- and only one of them is installed at a time, so the other is synthesized by
substituting exactly the strings the patches themselves substitute. No torch
and no GPU: the class attributes under test are pure Python.
"""

import linecache
from pathlib import Path

import pytest

megatron_gdn = pytest.importorskip("megatron.core.ssm.gated_delta_net")

from mcore_adapter.models.qwen4_exp.gated_delta_net import (  # noqa: E402
    Qwen4ExpGatedDeltaNet,
    verify_upstream_route,
)

_ROLL = Path(__file__).parents[3]
_UPSTREAM = megatron_gdn.GatedDeltaNet
# Neither method under test reads ``self``, so a bare object drives them.
_LAYER = object()


# --------------------------------------------------------------------------
# building a forward() with the routing a given dependency layout has
# --------------------------------------------------------------------------
def _readable_forward(body_lines, name=None):
    """Compile a ``forward`` whose source ``inspect.getsource`` can read.

    ``verify_upstream_route`` inspects the source text, so a function built
    with ``exec`` is unreadable unless its text is registered with linecache.
    """
    source = "def forward(self, hidden_states):\n" + "".join(f"    {line}\n" for line in body_lines)
    namespace = {}
    label = name or f"<synthetic forward {abs(hash(source)) % 100000}>"
    exec(compile(source, label, "exec"), namespace)
    linecache.cache[label] = (len(source), None, source.splitlines(True), label)
    return namespace["forward"]


_HOOK_CALL = "g, beta = self._compute_g_and_beta(" "A_log_local_cp, dt_bias_local_cp, alpha, beta)"
_INLINE_DECAY = "g = -A_log_local_cp.float().exp() * softplus(alpha.float() + dt_bias_local_cp)"
_INLINE_BETA = "beta = beta.float().sigmoid()"
_MODULE_CONV = "qkv, _ = causal_conv1d(x=hidden_states, weight=None)"
_PATCHED_CONV = "qkv = self._apply_causal_conv1d(hidden_states, None, None)"


def install_layout(monkeypatch, *, hook, hook_call, inline_fp32, convolution="module"):
    """Give the *real* GatedDeltaNet the routing a dependency layout has.

    ``hook`` toggles the ``_compute_g_and_beta`` attribute, ``hook_call``
    whether forward calls it, ``inline_fp32`` whether the FP32 forms sit in
    forward (the source-patch shape), ``convolution`` one of "module",
    "patched" or "none".
    """
    if hook:
        monkeypatch.setattr(_UPSTREAM, "_compute_g_and_beta", lambda self, *a, **k: a, raising=False)
    else:
        monkeypatch.delattr(_UPSTREAM, "_compute_g_and_beta", raising=False)
    if convolution == "patched":
        monkeypatch.setattr(_UPSTREAM, "_apply_causal_conv1d", lambda self, x, w, b: x, raising=False)

    body = ["A_log_local_cp = dt_bias_local_cp = alpha = beta = hidden_states"]
    if convolution == "module":
        body.append(_MODULE_CONV)
    elif convolution == "patched":
        body.append(_PATCHED_CONV)
    if inline_fp32:
        body += [_INLINE_DECAY, _INLINE_BETA]
    if hook and hook_call:
        body.append(_HOOK_CALL)
    body.append("return beta")
    monkeypatch.setattr(_UPSTREAM, "forward", _readable_forward(body))


# --------------------------------------------------------------------------
# routing contract: are the overrides reachable at all?
# --------------------------------------------------------------------------
def test_hook_layout_is_accepted(monkeypatch):
    """Megatron >= 0.18: forward calls the hook this subclass overrides."""
    install_layout(monkeypatch, hook=True, hook_call=True, inline_fp32=False)
    assert verify_upstream_route() == "hook"


def test_inline_layout_with_source_patches_is_accepted(monkeypatch):
    """0.16 + the decay/gate source patches: the FP32 forms sit in forward.

    The subclass override is redundant here, not wrong, so this must pass --
    it is the layout of the recorded .181 validation image.
    """
    install_layout(monkeypatch, hook=False, hook_call=False, inline_fp32=True)
    assert verify_upstream_route() == "inline"


def test_patched_convolution_layout_is_accepted(monkeypatch):
    """0.16 + the convolution source patch routes through its own method.

    That method still calls the module-level name, so the rebind applies.
    """
    install_layout(monkeypatch, hook=False, hook_call=False, inline_fp32=True, convolution="patched")
    assert verify_upstream_route() == "inline"


def test_hook_defined_but_never_called_is_rejected(monkeypatch):
    """The exact silent failure: the method exists, forward ignores it."""
    install_layout(monkeypatch, hook=True, hook_call=False, inline_fp32=False)
    with pytest.raises(RuntimeError, match="never calls it"):
        verify_upstream_route()


def test_unpatched_inline_layout_is_rejected(monkeypatch):
    """0.16 with neither the hook nor the FP32 source patches: no fix at all."""
    install_layout(monkeypatch, hook=False, hook_call=False, inline_fp32=False)
    with pytest.raises(RuntimeError, match="does not carry"):
        verify_upstream_route()


def test_missing_convolution_route_is_rejected(monkeypatch):
    """A forward that reaches no interception point must not pass silently."""
    install_layout(monkeypatch, hook=True, hook_call=True, inline_fp32=False, convolution="none")
    with pytest.raises(RuntimeError, match="causal convolution"):
        verify_upstream_route()


def test_unreadable_forward_is_refused(monkeypatch):
    """A compiled or externally wrapped forward cannot be verified."""
    monkeypatch.setattr(_UPSTREAM, "forward", len)
    with pytest.raises(RuntimeError, match="Cannot read the source"):
        verify_upstream_route()


def test_constructing_the_layer_runs_the_route_check(monkeypatch):
    """The check must happen before training, not only when called by hand.

    ``__init__`` calls the guard before ``super().__init__``, so on a tree with
    no fix at all the constructor raises before any Megatron state is built.
    """
    install_layout(monkeypatch, hook=False, hook_call=False, inline_fp32=False)
    with pytest.raises(RuntimeError, match="does not carry"):
        Qwen4ExpGatedDeltaNet.__init__(_LAYER)


# --------------------------------------------------------------------------
# scope: this module must not disturb the dependency
# --------------------------------------------------------------------------
def test_the_base_class_keeps_its_own_hook():
    """The override lives on the subclass; other GDN models keep Megatron's."""
    assert "_compute_g_and_beta" in Qwen4ExpGatedDeltaNet.__dict__


def test_the_override_keeps_both_operands_in_fp32():
    """Pin the two precision changes so a refactor cannot quietly drop one.

    The decay ``g`` must be exponentiated in FP32 (checked by the source
    patches' own before/after hashes too) and the write gate ``beta`` promoted
    before ``sigmoid``.
    """
    source = (_ROLL / "mcore_adapter/src/mcore_adapter/models/qwen4_exp/gated_delta_net.py").read_text(
        encoding="utf-8"
    )
    body = source.split("def _compute_g_and_beta", 1)[1].split("@staticmethod", 1)[0]
    assert "A_log_local_cp.float().exp()" in body
    assert "beta.float().sigmoid()" in body
    assert "A_log_local_cp.exp()" not in body.replace("A_log_local_cp.float().exp()", "")


def test_the_inline_markers_match_the_source_patches_they_recognize():
    """The guard's inline strings must be the patches' own NEW text.

    ``verify_upstream_route`` accepts a 0.16 tree only when it carries the
    inline FP32 forms; those forms are produced by
    scripts/qwen38/patch_megatron_gdn_{decay,gate_precision}.py. If either
    patch's NEW is reworded without updating the guard, the guard would reject
    the very tree the patch creates.
    """
    import mcore_adapter.models.qwen4_exp.gated_delta_net as under_test

    for filename, marker in [
        ("patch_megatron_gdn_decay.py", under_test.INLINE_LAYOUT_DECAY),
        ("patch_megatron_gdn_gate_precision.py", under_test.INLINE_LAYOUT_BETA),
    ]:
        text = (_ROLL / "scripts/qwen38" / filename).read_text(encoding="utf-8")
        assert marker in text, (
            f"{filename} no longer contains {marker!r}, which verify_upstream_route "
            "accepts as the inline layout; one of the two drifted"
        )


# --------------------------------------------------------------------------
# the convolution rebind: lifetime and inputs, on the real module
# --------------------------------------------------------------------------
@pytest.fixture
def fla_present(monkeypatch):
    """A recognizable stand-in for the FLA convolution the shim wraps."""
    monkeypatch.setattr(megatron_gdn, "causal_conv1d", lambda *a, **k: ("fla", a, k))


def test_rebind_is_scoped_and_restored(fla_present):
    native = megatron_gdn.causal_conv1d
    with Qwen4ExpGatedDeltaNet._native_causal_convolution(_LAYER):
        assert megatron_gdn.causal_conv1d is not native
        assert megatron_gdn.causal_conv1d.__name__ == "shim"
    assert megatron_gdn.causal_conv1d is native


def test_rebind_is_restored_when_the_body_raises(fla_present):
    native = megatron_gdn.causal_conv1d
    with pytest.raises(ValueError, match="boom"):
        with Qwen4ExpGatedDeltaNet._native_causal_convolution(_LAYER):
            raise ValueError("boom")
    assert megatron_gdn.causal_conv1d is native


def test_rebind_restores_after_nesting(fla_present):
    native = megatron_gdn.causal_conv1d
    with Qwen4ExpGatedDeltaNet._native_causal_convolution(_LAYER):
        inner = megatron_gdn.causal_conv1d
        with Qwen4ExpGatedDeltaNet._native_causal_convolution(_LAYER):
            assert megatron_gdn.causal_conv1d is not inner
        assert megatron_gdn.causal_conv1d is inner
    assert megatron_gdn.causal_conv1d is native


def test_missing_fla_convolution_is_left_alone(monkeypatch):
    """No FLA: upstream asserts before the call, so do not mask that."""
    monkeypatch.setattr(megatron_gdn, "causal_conv1d", None)
    with Qwen4ExpGatedDeltaNet._native_causal_convolution(_LAYER):
        assert megatron_gdn.causal_conv1d is None


def test_shim_forwards_to_the_native_convolution(fla_present):
    from mcore_adapter.models.qwen4_exp import causal_convolution

    calls = []
    native = causal_convolution.causal_conv1d
    try:
        causal_convolution.causal_conv1d = lambda x, w, b: calls.append((x, w, b)) or "native"
        with Qwen4ExpGatedDeltaNet._native_causal_convolution(_LAYER):
            result = megatron_gdn.causal_conv1d("x", "w", bias="b", activation="silu")
    finally:
        causal_convolution.causal_conv1d = native
    assert result == ("native", None)
    assert calls == [("x", "w", "b")]


@pytest.mark.parametrize("activation", ["silu", "swish"])
def test_shim_accepts_the_two_native_activations(fla_present, activation):
    from mcore_adapter.models.qwen4_exp import causal_convolution

    native = causal_convolution.causal_conv1d
    try:
        causal_convolution.causal_conv1d = lambda x, w, b: "native"
        with Qwen4ExpGatedDeltaNet._native_causal_convolution(_LAYER):
            assert megatron_gdn.causal_conv1d("x", "w", activation=activation) == ("native", None)
    finally:
        causal_convolution.causal_conv1d = native


def test_shim_rejects_recurrent_conv_state(fla_present):
    """The native path carries no conv cache; a request for one must be loud."""
    with Qwen4ExpGatedDeltaNet._native_causal_convolution(_LAYER):
        with pytest.raises(NotImplementedError, match="conv state"):
            megatron_gdn.causal_conv1d("x", "w", activation="silu", initial_state=object())
        with pytest.raises(NotImplementedError, match="conv state"):
            megatron_gdn.causal_conv1d("x", "w", activation="silu", output_final_state=True)


def test_shim_rejects_a_foreign_activation(fla_present):
    with Qwen4ExpGatedDeltaNet._native_causal_convolution(_LAYER):
        with pytest.raises(ValueError, match="silu/swish"):
            megatron_gdn.causal_conv1d("x", "w", activation="gelu")


def test_shim_rejects_packed_sequences(fla_present):
    """A causal window must not span packed-sequence boundaries."""
    with Qwen4ExpGatedDeltaNet._native_causal_convolution(_LAYER):
        with pytest.raises(NotImplementedError, match="packed cu_seqlens"):
            megatron_gdn.causal_conv1d("x", "w", activation="silu", cu_seqlens=object())
