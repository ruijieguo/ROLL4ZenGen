"""The opt-in Hybrid replacements must be scoped, idempotent, and reversible.

Commit 6 moves three Megatron source patches into ROLL: bounded CPU gradient
staging, and the two Hybrid checkpoint-serialization fixes. They are installed
at runtime instead of asking every user to patch their own Megatron checkout,
so the install/uninstall pair has to be exact -- a half-restored class would
leak ROLL behaviour into other models, and a silently-skipped install would
train with the unbounded gradient cache it was meant to replace.

The assertions are deliberately *relative*: the validation image's Megatron
already carries the legacy source patches, so several of these methods exist
before install. What matters is that install swaps them and uninstall puts back
exactly what was there -- not that any given attribute is absent.

Runs against the installed megatron-core (skipped when absent). The attribute
tests create no CUDA context; the constructor tests are skipped without a GPU.
"""

import inspect

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

# Every method either installer replaces or adds.
TOUCHED = (
    "__init__",
    "step",
    "_set_sub_optimizer_grads",
    "_prepare_bounded_cpu_grad_staging",
    "_stage_bounded_cpu_optimizer_grad",
    "dummy_step",
    "_register_load_state_dict_hooks",
    "_sync_hdo_state_to_sub_optimizers",
)
_MARKERS = ("_qwen4_bounded_cpu_grad_staging_installed", "_qwen4_bounded_cpu_grad_staging_saved",
            "_qwen4_hybrid_checkpoint_installed")
_SET_MAIN = "_qwen4_unpatched_set_main_param_and_optimizer_states"


def _snapshot():
    """What the class looks like right now, for each method under test.

    Records the class dict entry rather than the attribute, so an inherited
    method (absent from the dict) is distinguished from one the class owns.
    """
    return {name: HybridDeviceOptimizer.__dict__.get(name) for name in TOUCHED}


def _restore(snapshot):
    for name, value in snapshot.items():
        if value is None:
            if name in HybridDeviceOptimizer.__dict__:
                delattr(HybridDeviceOptimizer, name)
        else:
            setattr(HybridDeviceOptimizer, name, value)


@pytest.fixture
def clean():
    """Both patches uninstalled before and after, exactly as found."""
    uninstall_bounded_cpu_grad_staging()
    uninstall_hybrid_checkpoint_patches()
    snapshot = _snapshot()
    set_main = getattr(DistributedOptimizer, _SET_MAIN, None)
    yield
    uninstall_bounded_cpu_grad_staging()
    uninstall_hybrid_checkpoint_patches()
    _restore(snapshot)
    if set_main is not None:
        setattr(DistributedOptimizer, _SET_MAIN, set_main)


def test_import_does_not_install_by_itself(clean):
    """Importing the modules must not change the dependency."""
    for marker in _MARKERS:
        assert not hasattr(HybridDeviceOptimizer, marker)


def test_install_replaces_every_method_it_claims_to(clean):
    """Each named method must actually be swapped, not silently skipped."""
    before = {}
    for name in TOUCHED:
        entry = HybridDeviceOptimizer.__dict__.get(name)
        before[name] = entry
    set_main_before = DistributedOptimizer._set_main_param_and_optimizer_states

    assert install_bounded_cpu_grad_staging()
    assert install_hybrid_checkpoint_patches()

    for name, previous in before.items():
        assert HybridDeviceOptimizer.__dict__.get(name) is not previous, f"{name} was not replaced"
    assert DistributedOptimizer._set_main_param_and_optimizer_states is not set_main_before


def test_install_is_idempotent(clean):
    """A second call must not re-wrap, which would nest the previous install."""
    assert install_bounded_cpu_grad_staging()
    first = HybridDeviceOptimizer.__dict__["__init__"]
    assert install_bounded_cpu_grad_staging()
    assert HybridDeviceOptimizer.__dict__["__init__"] is first


def test_uninstall_restores_the_class_exactly(clean):
    """Uninstall must put back the original entries, removals included.

    The distinction that matters: a method the install *added* is removed
    again, while one it merely *replaced* -- possibly because the dependency
    already had it -- comes back as the same object.
    """
    before = _snapshot()
    set_main_before = DistributedOptimizer._set_main_param_and_optimizer_states

    install_bounded_cpu_grad_staging()
    install_hybrid_checkpoint_patches()
    assert uninstall_bounded_cpu_grad_staging()
    assert uninstall_hybrid_checkpoint_patches()

    after = _snapshot()
    assert after == before
    assert DistributedOptimizer._set_main_param_and_optimizer_states is set_main_before
    for marker in _MARKERS:
        assert not hasattr(HybridDeviceOptimizer, marker)
    # The saved-originals record must not survive, or a later install would
    # restore this run's originals instead of its own.
    assert "_qwen4_bounded_cpu_grad_staging_saved" not in vars(HybridDeviceOptimizer)


def test_uninstall_without_install_is_a_noop(clean):
    assert uninstall_bounded_cpu_grad_staging() is False
    assert uninstall_hybrid_checkpoint_patches() is False


def test_reinstall_after_uninstall_starts_clean(clean):
    """The install must not depend on state left by a previous cycle."""
    install_bounded_cpu_grad_staging()
    install_hybrid_checkpoint_patches()
    uninstall_bounded_cpu_grad_staging()
    uninstall_hybrid_checkpoint_patches()
    before = _snapshot()
    install_bounded_cpu_grad_staging()
    install_hybrid_checkpoint_patches()
    uninstall_bounded_cpu_grad_staging()
    uninstall_hybrid_checkpoint_patches()
    assert _snapshot() == before


def test_no_other_class_is_touched(clean):
    """Nothing is installed unless the flag path runs."""
    other = type("PlainAdamW", (), {"__init__": object.__init__, "step": object.__init__})
    before = dict(other.__dict__)
    install_bounded_cpu_grad_staging()
    install_hybrid_checkpoint_patches()
    assert other.__dict__ == before
    assert not hasattr(other, "bounded_cpu_grad_staging")


def test_the_bounded_step_reproduces_megatrons_ordering(clean):
    """Pin the four anchors the replacement borrows.

    The bounded step is a copy of Megatron's, so if one of these is renamed the
    copy diverges without any error.
    """
    source = inspect.getsource(HybridDeviceOptimizer.step)
    for marker in ("_sync_hdo_param_groups_to_sub_optimizers", "self._d2h_stream",
                   "for cpu_optimizer in self.cpu_optimizers", "self._sync_sub_optimizers_state_to_hdo"):
        assert marker in source, marker


def test_staging_installer_verifies_the_upstream_contract(clean, monkeypatch):
    """A renamed anchor must fail the install, not silently diverge."""
    from roll.third_party.megatron import cpu_master_params

    monkeypatch.setattr(HybridDeviceOptimizer, "step", lambda self, closure=None: None)
    with pytest.raises(RuntimeError, match="no longer contains"):
        cpu_master_params.install_bounded_cpu_grad_staging()


def test_checkpoint_installer_verifies_the_upstream_contract(clean, monkeypatch):
    from roll.third_party.megatron import checkpoint_optimizer_state

    monkeypatch.setattr(HybridDeviceOptimizer, "dummy_step", lambda self: None)
    with pytest.raises(RuntimeError, match="randn"):
        checkpoint_optimizer_state.install_hybrid_checkpoint_patches()


def test_installers_are_independent(clean):
    """Uninstalling one must not disturb the other's methods."""
    install_bounded_cpu_grad_staging()
    staged = HybridDeviceOptimizer.__dict__["step"]
    original_dummy = HybridDeviceOptimizer.__dict__.get("dummy_step")

    install_hybrid_checkpoint_patches()
    assert HybridDeviceOptimizer.__dict__["step"] is staged
    assert uninstall_hybrid_checkpoint_patches()
    # The staging install is still in place and functional.
    assert HybridDeviceOptimizer.__dict__["step"] is staged
    assert HybridDeviceOptimizer.__dict__.get("dummy_step") is original_dummy
    assert uninstall_bounded_cpu_grad_staging()


def test_the_installed_signature_accepts_the_flag(clean):
    """The optimizer factory reads this signature, so the replacement must provide it."""
    install_bounded_cpu_grad_staging()
    parameters = inspect.signature(HybridDeviceOptimizer.__init__).parameters
    assert "bounded_cpu_grad_staging" in parameters
    assert parameters["bounded_cpu_grad_staging"].default is False
    # Keyword-only, so a positional argument cannot collide with it.
    assert parameters["bounded_cpu_grad_staging"].kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.skipif(not torch.cuda.is_available(), reason="constructing the Hybrid optimizer needs CUDA")
class TestFlagPlumbing:
    """The flag must reach the instance without becoming a hyperparameter.

    Separate from the attribute tests because Megatron's constructor opens CUDA
    streams; this is the only part that needs a device.
    """

    def _build(self, **overrides):
        parameters = [torch.nn.Parameter(torch.ones(3, device="cuda"))]
        kwargs = dict(
            offload_fraction=1.0,
            cpu_optimizer_cls=torch.optim.AdamW,
            gpu_optimizer_cls=torch.optim.AdamW,
            overlap_cpu_optimizer_d2h_h2d=True,
        )
        kwargs.update(overrides)
        return HybridDeviceOptimizer([{"params": parameters}], **kwargs)

    def test_the_flag_reaches_the_instance(self, clean):
        install_bounded_cpu_grad_staging()
        assert self._build(bounded_cpu_grad_staging=True).bounded_cpu_grad_staging is True

    def test_the_flag_defaults_off(self, clean):
        install_bounded_cpu_grad_staging()
        assert self._build().bounded_cpu_grad_staging is False

    def test_the_flag_is_not_forwarded_to_the_optimizer_defaults(self, clean):
        """Megatron passes **kwargs into ``defaults``; the flag must not leak.

        torch rejects unknown keys when param groups are added, so a leak would
        surface as a construction failure rather than a wrong value.
        """
        install_bounded_cpu_grad_staging()
        optimizer = self._build(bounded_cpu_grad_staging=True)
        assert "bounded_cpu_grad_staging" not in optimizer.defaults

    def test_the_flag_requires_overlapping_transfer(self, clean):
        """The single shared buffer is only valid with one CPU optimizer per parameter."""
        install_bounded_cpu_grad_staging()
        with pytest.raises(ValueError, match="overlap_cpu_optimizer_d2h_h2d"):
            self._build(overlap_cpu_optimizer_d2h_h2d=False, bounded_cpu_grad_staging=True)
