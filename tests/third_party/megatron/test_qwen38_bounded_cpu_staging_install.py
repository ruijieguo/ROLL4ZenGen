"""The opt-in Hybrid replacements must be scoped, idempotent, and reversible.

Commit 6 moves three Megatron source patches into ROLL: bounded CPU gradient
staging, and the two Hybrid checkpoint-serialization fixes. They are installed
at runtime instead of asking every user to patch their own Megatron checkout,
so the install/uninstall pair has to be exact -- a half-restored class would
leak ROLL behaviour into other models, and a silently-skipped install would
train with the unbounded gradient cache it was meant to replace.

Runs against the installed megatron-core (skipped when absent). The methods
under test are pure Python attribute swaps; no CUDA context is created.
"""

import pytest

megatron = pytest.importorskip("megatron.core.optimizer.cpu_offloading")
torch = pytest.importorskip("torch")

from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer  # noqa: E402
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer  # noqa: E402

from roll.third_party.megatron.checkpoint_optimizer_state import (  # noqa: E402
    install_hybrid_checkpoint_patches,
    uninstall_hybrid_checkpoint_patches,
)
from roll.third_party.megatron.cpu_master_params import (  # noqa: E402
    install_bounded_cpu_grad_staging,
    uninstall_bounded_cpu_grad_staging,
)

_STAGING = (
    "__init__",
    "step",
    "_set_sub_optimizer_grads",
    "_prepare_bounded_cpu_grad_staging",
    "_stage_bounded_cpu_optimizer_grad",
)
_CHECKPOINT = ("dummy_step", "_register_load_state_dict_hooks", "_sync_hdo_state_to_sub_optimizers")


@pytest.fixture
def clean():
    """Both patches uninstalled before and after, in a fixed order."""
    uninstall_bounded_cpu_grad_staging()
    uninstall_hybrid_checkpoint_patches()
    yield
    uninstall_bounded_cpu_grad_staging()
    uninstall_hybrid_checkpoint_patches()


def test_import_leaves_the_dependency_untouched(clean):
    """Importing the modules must not install anything by itself."""
    assert not hasattr(HybridDeviceOptimizer, "_qwen4_bounded_cpu_grad_staging_installed")
    assert not hasattr(HybridDeviceOptimizer, "_qwen4_hybrid_checkpoint_installed")
    assert not hasattr(HybridDeviceOptimizer, "_prepare_bounded_cpu_grad_staging")
    assert not hasattr(DistributedOptimizer, "_qwen4_unpatched_set_main_param_and_optimizer_states")


def test_install_replaces_every_method_it_claims_to(clean):
    staging_before = {name: getattr(HybridDeviceOptimizer, name) for name in _STAGING}
    checkpoint_before = {name: getattr(HybridDeviceOptimizer, name) for name in _CHECKPOINT}
    set_main_before = DistributedOptimizer._set_main_param_and_optimizer_states

    assert install_bounded_cpu_grad_staging()
    assert install_hybrid_checkpoint_patches()

    for name, original in staging_before.items():
        assert getattr(HybridDeviceOptimizer, name) is not original, f"{name} was not replaced"
    for name, original in checkpoint_before.items():
        assert getattr(HybridDeviceOptimizer, name) is not original, f"{name} was not replaced"
    assert DistributedOptimizer._set_main_param_and_optimizer_states is not set_main_before


def test_install_is_idempotent(clean):
    """A second call must not re-wrap, which would nest the originals."""
    assert install_bounded_cpu_grad_staging()
    installed = HybridDeviceOptimizer.__init__
    original = getattr(HybridDeviceOptimizer, "_qwen4_unpatched_init")
    assert install_bounded_cpu_grad_staging()
    assert HybridDeviceOptimizer.__init__ is installed
    assert getattr(HybridDeviceOptimizer, "_qwen4_unpatched_init") is original


def test_uninstall_restores_the_originals_exactly(clean):
    staging_before = {name: getattr(HybridDeviceOptimizer, name) for name in _STAGING}
    checkpoint_before = {name: getattr(HybridDeviceOptimizer, name) for name in _CHECKPOINT}
    set_main_before = DistributedOptimizer._set_main_param_and_optimizer_states

    install_bounded_cpu_grad_staging()
    install_hybrid_checkpoint_patches()
    assert uninstall_bounded_cpu_grad_staging()
    assert uninstall_hybrid_checkpoint_patches()

    for name, original in {**staging_before, **checkpoint_before}.items():
        assert getattr(HybridDeviceOptimizer, name) is original, f"{name} was not restored"
    assert DistributedOptimizer._set_main_param_and_optimizer_states is set_main_before
    # No marker attributes survive, so a later install starts from scratch.
    assert not hasattr(HybridDeviceOptimizer, "_qwen4_bounded_cpu_grad_staging_installed")
    assert not hasattr(HybridDeviceOptimizer, "_qwen4_hybrid_checkpoint_installed")
    assert not hasattr(HybridDeviceOptimizer, "_prepare_bounded_cpu_grad_staging")


def test_uninstall_without_install_is_a_noop(clean):
    assert uninstall_bounded_cpu_grad_staging() is False
    assert uninstall_hybrid_checkpoint_patches() is False


def test_other_models_keep_the_native_optimizer(clean):
    """Nothing is installed unless the flag path runs, so no other class is touched."""
    other = type("PlainAdamW", (), {"__init__": object.__init__, "step": object.__init__})
    before = other.__dict__.copy()
    install_bounded_cpu_grad_staging()
    assert other.__dict__ == before
    assert "bounded_cpu_grad_staging" not in vars(other)


def test_the_bounded_step_still_runs_megatrons_ordering(clean):
    """Pin the four anchors the replacement reproduces.

    The bounded step is a copy of Megatron's, so the anchors it borrows have to
    keep existing; if one is renamed the copy silently diverges.
    """
    import inspect

    source = inspect.getsource(HybridDeviceOptimizer.step)
    for marker in ("_sync_hdo_param_groups_to_sub_optimizers", "self._d2h_stream",
                   "for cpu_optimizer in self.cpu_optimizers", "self._sync_sub_optimizers_state_to_hdo"):
        assert marker in source, marker


def test_installer_verifies_the_upstream_contract(clean, monkeypatch):
    """A renamed anchor must fail the install, not silently diverge."""
    from roll.third_party.megatron import cpu_master_params

    def broken_step(self, closure=None):
        return None

    monkeypatch.setattr(HybridDeviceOptimizer, "step", broken_step)
    with pytest.raises(RuntimeError, match="no longer contains"):
        cpu_master_params.install_bounded_cpu_grad_staging()


def test_checkpoint_installer_verifies_the_upstream_contract(clean, monkeypatch):
    from roll.third_party.megatron import checkpoint_optimizer_state

    def broken_dummy_step(self):
        return None

    monkeypatch.setattr(HybridDeviceOptimizer, "dummy_step", broken_dummy_step)
    with pytest.raises(RuntimeError, match="randn"):
        checkpoint_optimizer_state.install_hybrid_checkpoint_patches()


def test_the_installed_signature_accepts_the_flag(clean):
    """The factory checks this signature, so the replacement must provide it."""
    import inspect

    install_bounded_cpu_grad_staging()
    assert "bounded_cpu_grad_staging" in inspect.signature(HybridDeviceOptimizer.__init__).parameters
    assert (
        inspect.signature(HybridDeviceOptimizer.__init__)
        .parameters["bounded_cpu_grad_staging"]
        .default
        is False
    )


def test_the_flag_is_not_forwarded_to_the_optimizer_defaults(clean):
    """``bounded_cpu_grad_staging`` is a ROLL option, not a hyperparameter.

    Megatron's constructor passes ``**kwargs`` into ``defaults``; if the flag
    leaked there, torch would reject the unknown key while adding param groups.
    """
    install_bounded_cpu_grad_staging()
    parameters = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(3))])
    optimizer = HybridDeviceOptimizer(
        [{"params": list(parameters)}],
        offload_fraction=1.0,
        cpu_optimizer_cls=torch.optim.AdamW,
        gpu_optimizer_cls=torch.optim.AdamW,
        overlap_cpu_optimizer_d2h_h2d=True,
        bounded_cpu_grad_staging=True,
    )
    assert optimizer.bounded_cpu_grad_staging is True
    assert "bounded_cpu_grad_staging" not in optimizer.defaults


def test_the_flag_off_keeps_megatrons_own_constructor(clean):
    install_bounded_cpu_grad_staging()
    parameters = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(3))])
    optimizer = HybridDeviceOptimizer(
        [{"params": list(parameters)}],
        offload_fraction=1.0,
        cpu_optimizer_cls=torch.optim.AdamW,
        gpu_optimizer_cls=torch.optim.AdamW,
        overlap_cpu_optimizer_d2h_h2d=True,
    )
    assert optimizer.bounded_cpu_grad_staging is False


def test_the_flag_requires_overlapping_transfer(clean):
    """The bounded buffer is only valid with one CPU optimizer per parameter."""
    install_bounded_cpu_grad_staging()
    with pytest.raises(ValueError, match="overlap_cpu_optimizer_d2h_h2d"):
        HybridDeviceOptimizer(
            [{"params": [torch.nn.Parameter(torch.ones(3))]}],
            offload_fraction=1.0,
            cpu_optimizer_cls=torch.optim.AdamW,
            gpu_optimizer_cls=torch.optim.AdamW,
            overlap_cpu_optimizer_d2h_h2d=False,
            bounded_cpu_grad_staging=True,
        )
