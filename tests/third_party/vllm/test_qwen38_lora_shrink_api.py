"""The LoRA-shrink patch point must still exist in the installed vLLM.

``roll/third_party/vllm/lora_shrink.py`` wraps
``vllm.lora.ops.triton_ops.utils.get_lora_op_configs``. vLLM 0.23.0 moved that
selector there from ``lora_shrink_op.py``, and the move was found by hand --
the previous patch point imported fine and only the attribute lookup failed, so
nothing in the CPU suite noticed until someone read the wheel.

These assertions run against the installed vLLM (skipped when it is absent) so
the next relocation fails a test instead of surfacing as a runtime error in a
training run. They read the module the patch targets and check the contract the
patch relies on: a callable selector, and a "shrink" config carrying "split_k".
"""

import importlib

import pytest

vllm = pytest.importorskip("vllm")

from roll.third_party.vllm.lora_shrink import _SHRINK_MODULE  # noqa: E402


def _selector_module(monkeypatch):
    module = importlib.import_module(_SHRINK_MODULE)
    # The patch replaces a process-global attribute; undo it for this process.
    monkeypatch.setattr(module, "get_lora_op_configs", module.get_lora_op_configs)
    return module


def test_the_patched_module_exists_in_the_installed_vllm(monkeypatch):
    """A relocation changes the import path, not just the attribute."""
    module = _selector_module(monkeypatch)
    assert module is not None


def test_the_patched_module_exposes_a_callable_selector(monkeypatch):
    """The patch reads this attribute and raises if it is not callable.

    This is what broke silently: ``lora_shrink_op`` still imported after the
    move, but no longer carried the attribute.
    """
    module = _selector_module(monkeypatch)
    assert callable(
        getattr(module, "get_lora_op_configs", None)
    ), f"{_SHRINK_MODULE} has no callable get_lora_op_configs; the patch point moved again"


def test_the_selector_returns_a_shrink_config_with_split_k(monkeypatch):
    """Pin the key the patch rewrites, at the values vLLM documents.

    ``select`` raises when "split_k" is missing, so a config-shape change must
    fail here rather than at the first rollout.
    """
    module = _selector_module(monkeypatch)
    config = module.get_lora_op_configs("shrink", 1, 210, 2560, 64, 1)
    assert isinstance(config, dict)
    assert "split_k" in config


def test_the_selector_signature_has_the_arguments_the_patch_forwards():
    """The patch wraps with ``select(op_type, *args, **kwargs)``.

    A signature narrowed to keyword-only or reordered positional arguments
    would still pass the calls above but change what the wrapper forwards; the
    first five parameters must be positional in this order.
    """
    import inspect

    module = importlib.import_module(_SHRINK_MODULE)
    params = list(inspect.signature(module.get_lora_op_configs).parameters.values())
    positional = [p.name for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    assert positional[:6] == ["op_type", "max_loras", "batch", "hidden_size", "rank", "num_slices"]
