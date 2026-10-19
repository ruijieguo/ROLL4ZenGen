"""The four model-type Megatron patches: scope, idempotence, and reversibility.

These replace functions on the installed Megatron instead of asking each user to
patch their own checkout, so the install must be exact: nothing touched for
another model type, nothing left unreplaced for a Flash-Next one, and put back
byte-for-byte on uninstall. Two of the four reproduce upstream bodies, so a
renamed anchor has to fail the install rather than silently diverge.

Runs against the installed megatron-core (skipped when absent). The attribute
assertions create no CUDA context.
"""

import inspect

import pytest

pytest.importorskip("torch")
pytest.importorskip("megatron.core.optimizer.distrib_optimizer")

from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBuffer  # noqa: E402
from megatron.core.optimizer import clip_grads  # noqa: E402
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer  # noqa: E402
from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher  # noqa: E402

from roll.models.qwen4_exp_scope import FLASH_NEXT_MODEL_TYPES  # noqa: E402
from roll.third_party.megatron import model_type_patches  # noqa: E402

_ORIGINALS = (
    "_qwen4_unpatched_clip_grad_by_total_norm_fp32",
    "_qwen4_unpatched_sharded_param_state_dp_reshardable",
    "_qwen4_unpatched_param_and_grad_buffer_init",
    "_qwen4_unpatched_combine_postprocess",
)
_FLAG = "_qwen4_model_type_patches_installed"


@pytest.fixture
def clean():
    """Patches uninstalled before and after, in case an earlier test left them."""
    model_type_patches.uninstall()
    yield
    model_type_patches.uninstall()


def _originals_absent():
    return (
        not hasattr(clip_grads, _ORIGINALS[0])
        and not hasattr(DistributedOptimizer, _ORIGINALS[1])
        and not hasattr(_ParamAndGradBuffer, _ORIGINALS[2])
        and not hasattr(MoEAlltoAllTokenDispatcher, _ORIGINALS[3])
    )


def test_import_does_not_install(clean):
    assert _originals_absent()
    assert not hasattr(clip_grads, _FLAG)


@pytest.mark.parametrize("model_type", ["qwen3", "llama", "mamba", None, ""])
def test_other_model_types_are_untouched(clean, model_type):
    """A non-Flash-Next model must keep Megatron's own four functions."""
    before = (
        clip_grads.clip_grad_by_total_norm_fp32,
        DistributedOptimizer.sharded_param_state_dp_reshardable,
        _ParamAndGradBuffer.__init__,
        MoEAlltoAllTokenDispatcher.combine_postprocess,
    )
    assert model_type_patches.install_for_model_type(model_type) is False
    after = (
        clip_grads.clip_grad_by_total_norm_fp32,
        DistributedOptimizer.sharded_param_state_dp_reshardable,
        _ParamAndGradBuffer.__init__,
        MoEAlltoAllTokenDispatcher.combine_postprocess,
    )
    assert before == after
    assert _originals_absent()


@pytest.mark.parametrize("model_type", sorted(FLASH_NEXT_MODEL_TYPES))
def test_every_flash_next_type_installs_all_four(clean, model_type):
    before = (
        clip_grads.clip_grad_by_total_norm_fp32,
        DistributedOptimizer.sharded_param_state_dp_reshardable,
        _ParamAndGradBuffer.__init__,
        MoEAlltoAllTokenDispatcher.combine_postprocess,
    )
    assert model_type_patches.install_for_model_type(model_type) is True
    after = (
        clip_grads.clip_grad_by_total_norm_fp32,
        DistributedOptimizer.sharded_param_state_dp_reshardable,
        _ParamAndGradBuffer.__init__,
        MoEAlltoAllTokenDispatcher.combine_postprocess,
    )
    for original, replacement in zip(before, after):
        assert replacement is not original


def test_install_is_idempotent(clean):
    assert model_type_patches.install_for_model_type("qwen4_exp") is True
    installed = clip_grads.clip_grad_by_total_norm_fp32
    assert model_type_patches.install_for_model_type("qwen4_exp") is True
    assert clip_grads.clip_grad_by_total_norm_fp32 is installed


def test_uninstall_restores_exactly(clean):
    before = (
        clip_grads.clip_grad_by_total_norm_fp32,
        DistributedOptimizer.sharded_param_state_dp_reshardable,
        _ParamAndGradBuffer.__init__,
        MoEAlltoAllTokenDispatcher.combine_postprocess,
    )
    model_type_patches.install_for_model_type("qwen4_exp")
    assert model_type_patches.uninstall() is True
    after = (
        clip_grads.clip_grad_by_total_norm_fp32,
        DistributedOptimizer.sharded_param_state_dp_reshardable,
        _ParamAndGradBuffer.__init__,
        MoEAlltoAllTokenDispatcher.combine_postprocess,
    )
    assert before == after
    assert _originals_absent()


def test_uninstall_without_install_is_a_noop(clean):
    assert model_type_patches.uninstall() is False


def test_reinstall_after_uninstall_starts_clean(clean):
    model_type_patches.install_for_model_type("qwen4_exp")
    model_type_patches.uninstall()
    before = clip_grads.clip_grad_by_total_norm_fp32
    model_type_patches.install_for_model_type("qwen4_exp")
    model_type_patches.uninstall()
    assert clip_grads.clip_grad_by_total_norm_fp32 is before


def test_installer_verifies_a_renamed_anchor(clean, monkeypatch):
    """A reproduced body whose anchor vanished must fail the install."""

    def broken(self, *args, **kwargs):
        return {}

    monkeypatch.setattr(DistributedOptimizer, "sharded_param_state_dp_reshardable", broken)
    monkeypatch.setattr(DistributedOptimizer, "_qwen4_unpatched_sharded_param_state_dp_reshardable",
                        broken, raising=False)
    with pytest.raises(RuntimeError, match="no longer contains|need re-deriving"):
        model_type_patches.install_for_model_type("qwen4_exp")


def test_the_reproduced_clipping_keeps_the_native_coefficient():
    """Pin that only the scale call was batched, not the coefficient math."""
    source = inspect.getsource(model_type_patches._clip_grad_by_total_norm_fp32)
    assert "max_norm / (total_norm + 1.0e-6)" in source
    assert "clip_coeff < 1.0" in source
    assert "_SCALE_BATCH_SIZE" in source


def test_the_reproduced_padding_zeroes_the_gaps():
    """The gaps must be zeros, not torch.empty."""
    source = inspect.getsource(model_type_patches._sharded_param_state_dp_reshardable)
    assert "torch.zeros(" in source
    assert "torch.empty(" not in source


def test_the_reproduced_dispatcher_detaches_probs():
    source = inspect.getsource(model_type_patches._combine_postprocess)
    assert "self.probs.detach()" in source


def test_the_reproduced_buffer_defers_gradient_allocation():
    """The allocation must come after the weight remap, not before it."""
    source = inspect.getsource(model_type_patches._param_and_grad_buffer_init)
    remap = source.index("Remap original weights")
    allocation = source.index("self.grad_data = torch.zeros(")
    assert remap < allocation
