"""The decoder-class rebinding must be narrow, restored, and self-checking.

``Qwen4ExpModel`` substitutes its own decoder by rebinding the
``TransformerBlock`` name inside Megatron's ``gpt_model`` module for the
duration of its constructor. These assertions cover the rebinding's scope and
lifetime; no model is built and no GPU is used.
"""

import pytest
from megatron.core.models.gpt import gpt_model as megatron_gpt_model

from mcore_adapter.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpModel
from mcore_adapter.models.qwen4_exp.transformer_block import Qwen4ExpTransformerBlock


@pytest.fixture
def native_block():
    """Megatron's own decoder class, restored afterwards no matter what."""
    original = megatron_gpt_model.TransformerBlock
    yield original
    megatron_gpt_model.TransformerBlock = original


def test_substitutes_the_decoder_class_inside_the_block(native_block):
    with Qwen4ExpModel._qwen4_decoder_class():
        assert megatron_gpt_model.TransformerBlock is Qwen4ExpTransformerBlock


def test_restores_the_native_class_on_exit(native_block):
    with Qwen4ExpModel._qwen4_decoder_class():
        pass

    assert megatron_gpt_model.TransformerBlock is native_block


def test_restores_the_native_class_when_construction_raises(native_block):
    """A failed build must not leave Megatron pointing at this model's decoder."""
    with pytest.raises(ValueError, match="construction failed"):
        with Qwen4ExpModel._qwen4_decoder_class():
            raise ValueError("construction failed")

    assert megatron_gpt_model.TransformerBlock is native_block


def test_leaves_the_native_class_behind_after_nesting(native_block):
    """Models built in sequence or nested must each restore what they found."""
    with Qwen4ExpModel._qwen4_decoder_class():
        with Qwen4ExpModel._qwen4_decoder_class():
            assert megatron_gpt_model.TransformerBlock is Qwen4ExpTransformerBlock

    assert megatron_gpt_model.TransformerBlock is native_block


def test_the_substitute_is_a_megatron_transformer_block(native_block):
    """Megatron's GPTModel calls this class with its own keyword arguments."""
    assert issubclass(Qwen4ExpTransformerBlock, native_block)


def test_megatron_still_builds_its_decoder_from_the_rebound_name():
    """Pin the assumption the rebinding depends on.

    If a future Megatron constructs the decoder some other way, the rebinding
    silently stops applying and every GDN layer loses its PLE token ids. The
    constructor raises in that case; this test says why.
    """
    import inspect

    source = inspect.getsource(megatron_gpt_model.GPTModel.__init__)

    assert "self.decoder = TransformerBlock(" in source, (
        "Megatron's GPTModel no longer builds its decoder by calling the "
        "module-level TransformerBlock name, so Qwen4ExpModel._qwen4_decoder_class "
        "cannot substitute the Qwen4 decoder. Revisit that context manager."
    )
