"""The GDN checkpoint patch must be scoped, reversible, and fail loudly.

No GPU and no real Megatron: a stub module stands in for
``megatron.core.ssm.gated_delta_net`` so these assertions are about the
patch's scope and contract, not about shard arithmetic.
"""

import linecache
import sys
from types import ModuleType

import pytest

PATCH_MODULE = "roll.third_party.megatron.gdn_lora_checkpoint"
GDN_MODULE = "megatron.core.ssm.gated_delta_net"
UTILS_MODULE = "megatron.core.transformer.utils"

# The markers verify_upstream_contract() looks for, as they appear upstream.
UPSTREAM_BODY = """
    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None, tp_group=None):
        metadata = ensure_metadata_has_dp_cp_group(metadata)
        sharded_state_dict = make_sharded_tensors_for_checkpoint(
            sharded_state_dict, prefix,
            tensor_parallel_layers_axis_map={"A_log": 0, "dt_bias": 0},
        )
        assert sharded_state_dict[f"{prefix}in_proj.weight"].data.size(0) == in_proj_dim_local_tp
        sharded_state_dict[f"{prefix}in_proj.weight"] = _split_tensor_factory(
            sharded_state_dict[f"{prefix}in_proj.weight"],
            sections,
            ["query", "key", "value", "z", "beta", "alpha"],
            0,
        )
        conv_layer_name_list = ["conv1d.weight"]
        return sharded_state_dict
"""


def _compile_method(body, name):
    """Compile a stub method whose source ``inspect.getsource`` can read back.

    The patch checks upstream's source text, so a stub built with a bare
    ``exec`` would make ``getsource`` raise ``OSError`` and exercise the
    unreadable-source path instead of the contract check. Registering the
    source with ``linecache`` is what makes it readable.
    """
    source = "class _Holder:" + body
    namespace = {}
    exec(compile(source, name, "exec"), namespace)
    linecache.cache[name] = (len(source), None, source.splitlines(True), name)
    return namespace["_Holder"].sharded_state_dict


def _install_stub_megatron(monkeypatch, *, upstream_fixed=False):
    """Provide just enough of Megatron for the patch to install and run."""
    gdn = ModuleType(GDN_MODULE)
    split_calls = []

    def split_tensor_factory(shard, sections, names, dim):
        split_calls.append((shard.tag, tuple(sections), tuple(names), dim))
        return f"split({shard.tag})"

    body = UPSTREAM_BODY
    if upstream_fixed:
        body = body.replace('in_proj.weight"', 'in_proj.base_layer.weight"')

    class GatedDeltaNet:
        pass

    GatedDeltaNet.sharded_state_dict = _compile_method(body, "stub_gdn_upstream")

    gdn.GatedDeltaNet = GatedDeltaNet
    gdn._split_tensor_factory = split_tensor_factory
    monkeypatch.setitem(sys.modules, GDN_MODULE, gdn)

    utils = ModuleType(UTILS_MODULE)
    utils.ensure_metadata_has_dp_cp_group = lambda m: dict(m or {}, dp_cp_group=None)
    utils.make_sharded_tensors_for_checkpoint = lambda sd, prefix, **kw: dict(sd)
    utils.sharded_state_dict_default = lambda m, prefix, off, md, **kw: {}
    monkeypatch.setitem(sys.modules, UTILS_MODULE, utils)
    return gdn, split_calls


@pytest.fixture
def patch_module(monkeypatch):
    sys.modules.pop(PATCH_MODULE, None)
    import importlib

    module = importlib.import_module(PATCH_MODULE)
    yield module
    if GDN_MODULE in sys.modules:
        # Tests that never stubbed Megatron have nothing to restore, and
        # uninstall() would fail on the missing import rather than report it.
        module.uninstall()


@pytest.mark.parametrize(
    "model_type",
    ["qwen4_exp", "qwen4_exp_text", "qwen3_8_flash_next", "qwen3_8_flash_next_text"],
)
def test_installs_for_every_flash_next_model_type(monkeypatch, patch_module, model_type):
    gdn, _ = _install_stub_megatron(monkeypatch)
    native = gdn.GatedDeltaNet.sharded_state_dict

    assert patch_module.install_for_model_type(model_type) is True
    assert gdn.GatedDeltaNet.sharded_state_dict is not native


@pytest.mark.parametrize("model_type", ["qwen3_next", "mamba", "llama", None, ""])
def test_leaves_other_models_on_the_native_implementation(monkeypatch, patch_module, model_type):
    gdn, _ = _install_stub_megatron(monkeypatch)
    native = gdn.GatedDeltaNet.sharded_state_dict

    assert patch_module.install_for_model_type(model_type) is False
    assert gdn.GatedDeltaNet.sharded_state_dict is native


def test_installation_is_idempotent(monkeypatch, patch_module):
    gdn, _ = _install_stub_megatron(monkeypatch)

    assert patch_module.install_for_model_type("qwen4_exp") is True
    first = gdn.GatedDeltaNet.sharded_state_dict
    assert patch_module.install_for_model_type("qwen4_exp") is True
    assert gdn.GatedDeltaNet.sharded_state_dict is first


def test_uninstall_restores_the_native_implementation(monkeypatch, patch_module):
    gdn, _ = _install_stub_megatron(monkeypatch)
    native = gdn.GatedDeltaNet.sharded_state_dict

    patch_module.install_for_model_type("qwen4_exp")
    assert patch_module.uninstall() is True
    assert gdn.GatedDeltaNet.sharded_state_dict is native
    # A second uninstall reports that there was nothing to undo.
    assert patch_module.uninstall() is False


def test_changed_upstream_contract_is_rejected(monkeypatch, patch_module):
    gdn, _ = _install_stub_megatron(monkeypatch)
    # Drop one marker the reproduced copy depends on.
    changed = UPSTREAM_BODY.replace('"dt_bias": 0', '"dt_bias_renamed": 0')
    gdn.GatedDeltaNet.sharded_state_dict = _compile_method(changed, "stub_gdn_changed")

    with pytest.raises(RuntimeError, match="no longer matches"):
        patch_module.install_for_model_type("qwen4_exp")
    # A rejected install must leave the native implementation in place.
    assert not hasattr(gdn.GatedDeltaNet, "_qwen4_unpatched_sharded_state_dict")


def test_unreadable_upstream_source_is_refused(monkeypatch, patch_module):
    """A compiled or externally wrapped method cannot be verified, so refuse."""
    gdn, _ = _install_stub_megatron(monkeypatch)
    # A builtin has no retrievable source, which is what getsource chokes on.
    gdn.GatedDeltaNet.sharded_state_dict = len

    with pytest.raises(RuntimeError, match="Cannot read the source"):
        patch_module.install_for_model_type("qwen4_exp")


def test_skips_itself_once_upstream_handles_the_lora_key(monkeypatch, patch_module):
    gdn, _ = _install_stub_megatron(monkeypatch, upstream_fixed=True)
    native = gdn.GatedDeltaNet.sharded_state_dict

    assert patch_module.install_for_model_type("qwen4_exp") is False
    assert gdn.GatedDeltaNet.sharded_state_dict is native


def test_stacked_keys_cover_base_and_lora_b_but_not_lora_a(patch_module):
    state = {
        "l.in_proj.base_layer.weight": object(),
        "l.in_proj.lora_A.default.weight": object(),
        "l.in_proj.lora_B.default.weight": object(),
        "l.in_proj.lora_B.other.weight": object(),
        "l.in_proj.lora_B.default.bias": object(),
        "l.conv1d.weight": object(),
    }
    keys = patch_module.stacked_output_keys(state, "l.in_proj.")

    assert keys == [
        "l.in_proj.base_layer.weight",
        "l.in_proj.lora_B.default.weight",
        "l.in_proj.lora_B.other.weight",
    ]


def test_plain_base_weight_is_recognized_without_lora(patch_module):
    state = {"l.in_proj.weight": object(), "l.conv1d.weight": object()}

    assert patch_module.stacked_output_keys(state, "l.in_proj.") == ["l.in_proj.weight"]
