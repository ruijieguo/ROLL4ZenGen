"""Let Megatron's GDN write a distributed checkpoint when LoRA wraps in_proj.

Megatron's ``GatedDeltaNet.sharded_state_dict`` splits the stacked input
projection by looking up one fixed key::

    assert sharded_state_dict[f"{prefix}in_proj.weight"].data.size(0) == ...

Once MCA's LoRA wraps that projection the base weight is reported as
``in_proj.base_layer.weight``, so the lookup raises ``KeyError`` and no
checkpoint can be written at all. Upstream ROLL issue #411 reports this for
Qwen3.5-35B-A3B; the maintainers' answer there is ``--save_only_model``, which
omits optimizer state and so gives up cold resume. Flash-Next needs resume, so
the lookup is widened here instead.

The replacement also splits each adapter's LoRA-B, which shares the base
layer's stacked output layout. LoRA-A maps into the adapter's low-rank space,
not the stacked output space, so it is left alone.

The assert sits between steps that must still run (parameter sharding,
submodule recursion, the conv1d split), so the method is replaced as a whole
and those steps are reproduced here. ``verify_upstream_contract`` pins the
assumptions that copy relies on, and is called at install time: if Megatron
changes its own implementation, installation fails loudly instead of silently
writing a checkpoint under stale rules.

Installed only for Flash-Next model types. Delete once Megatron handles
LoRA-wrapped projections itself.
"""

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

SPLIT_NAMES = ["query", "key", "value", "z", "beta", "alpha"]
CONV_SPLIT_NAMES = ["query", "key", "value"]
_ORIGINAL = "_qwen4_unpatched_sharded_state_dict"


def stacked_output_keys(sharded_state_dict: Dict, input_prefix: str) -> List[str]:
    """Projection keys carrying the stacked q/k/v/z/beta/alpha rows.

    The base weight does, under either its plain or its LoRA-wrapped name, and
    so does every adapter's LoRA-B.
    """
    keys = [key for key in (input_prefix + "weight", input_prefix + "base_layer.weight") if key in sharded_state_dict]
    keys += sorted(
        key for key in sharded_state_dict if key.startswith(input_prefix + "lora_B.") and key.endswith(".weight")
    )
    return keys


def _sharded_state_dict(
    self,
    prefix: str = "",
    sharded_offsets: Tuple = (),
    metadata: Optional[dict] = None,
    tp_group=None,
):
    """Megatron's own method with a LoRA-aware input-projection split.

    Everything before the projection split is Megatron's logic, reproduced so
    the split can cover renamed and adapter tensors.
    """
    from megatron.core.ssm.gated_delta_net import _split_tensor_factory
    from megatron.core.transformer.utils import (
        ensure_metadata_has_dp_cp_group,
        make_sharded_tensors_for_checkpoint,
        sharded_state_dict_default,
    )

    metadata = ensure_metadata_has_dp_cp_group(metadata)

    sharded_state_dict = {}
    self._save_to_state_dict(sharded_state_dict, "", keep_vars=True)
    sharded_state_dict = make_sharded_tensors_for_checkpoint(
        sharded_state_dict,
        prefix,
        tensor_parallel_layers_axis_map={"A_log": 0, "dt_bias": 0},
        sharded_offsets=sharded_offsets,
        tp_group=(tp_group if tp_group is not None else self.pg_collection.tp),
        dp_cp_group=metadata["dp_cp_group"],
    )

    tp_group = tp_group if tp_group is not None else self.pg_collection.tp
    for name, module in self.named_children():
        if name == "conv1d":
            module_sd = module.state_dict(prefix="", keep_vars=True)
            tp_sharding_map = {"weight": 0}
            if self.conv_bias:
                tp_sharding_map["bias"] = 0
            module_sharded_sd = make_sharded_tensors_for_checkpoint(
                module_sd,
                f"{prefix}{name}.",
                tp_sharding_map,
                sharded_offsets,
                tp_group=tp_group,
                dp_cp_group=metadata["dp_cp_group"],
            )
        else:
            module_sharded_sd = sharded_state_dict_default(
                module, f"{prefix}{name}.", sharded_offsets, metadata, tp_group=tp_group
            )
        sharded_state_dict.update(module_sharded_sd)

    # --- the one behavioural change -------------------------------------
    input_prefix = f"{prefix}in_proj."
    in_proj_dim_local_tp = self.in_proj_dim // self.tp_size
    projection_sections = [
        self.qk_dim_local_tp,
        self.qk_dim_local_tp,
        self.v_dim_local_tp,
        self.v_dim_local_tp,
        self.num_value_heads // self.tp_size,
        self.num_value_heads // self.tp_size,
    ]
    projection_keys = stacked_output_keys(sharded_state_dict, input_prefix)
    if not projection_keys:
        raise RuntimeError(
            f"GDN reported no stacked input-projection weight under {input_prefix}; "
            "the checkpoint contract changed and this patch must be revisited"
        )
    for key in projection_keys:
        tensor = sharded_state_dict[key]
        if tensor.data.size(0) != in_proj_dim_local_tp:
            raise RuntimeError(
                f"GDN stacked output shape mismatch for {key}: got "
                f"{tuple(tensor.data.shape)}, expected first dim {in_proj_dim_local_tp}"
            )
        sharded_state_dict[key] = _split_tensor_factory(tensor, projection_sections, SPLIT_NAMES, 0)
    # --------------------------------------------------------------------

    conv_layer_names = ["conv1d.weight"]
    if self.conv_bias:
        conv_layer_names.append("conv1d.bias")
    for conv_layer_name in conv_layer_names:
        key = f"{prefix}{conv_layer_name}"
        if sharded_state_dict[key].data.size(0) != self.conv_dim_local_tp:
            raise RuntimeError(
                f"GDN conv shape mismatch for {key}: got "
                f"{tuple(sharded_state_dict[key].data.shape)}, expected first dim "
                f"{self.conv_dim_local_tp}"
            )
        sharded_state_dict[key] = _split_tensor_factory(
            sharded_state_dict[key],
            [self.qk_dim_local_tp, self.qk_dim_local_tp, self.v_dim_local_tp],
            CONV_SPLIT_NAMES,
            0,
        )

    return sharded_state_dict


def verify_upstream_contract() -> None:
    """Fail if Megatron's own implementation no longer matches this copy.

    Checked by source inspection rather than by version number: the method is
    reproduced above, so a changed upstream body means the copy may be stale.
    Raising here is the point. A silent mismatch would write checkpoints under
    rules nobody reviewed.
    """
    import inspect

    from megatron.core.ssm.gated_delta_net import GatedDeltaNet

    target = getattr(GatedDeltaNet, _ORIGINAL, None) or GatedDeltaNet.sharded_state_dict
    try:
        source = inspect.getsource(target)
    except (OSError, TypeError) as exc:
        # No readable source: a compiled build, or someone else wrapped the
        # method. Either way the copy below cannot be checked, so refuse rather
        # than install against an unverified implementation.
        raise RuntimeError(
            "Cannot read the source of Megatron's GatedDeltaNet.sharded_state_dict "
            f"({type(exc).__name__}: {exc}), so the reproduced implementation in "
            "roll/third_party/megatron/gdn_lora_checkpoint.py cannot be verified "
            "against it. Review that patch against the installed Megatron."
        ) from exc

    required = {
        'sharded_state_dict[f"{prefix}in_proj.weight"]': "the single-key projection lookup",
        '"A_log": 0': "the A_log tensor-parallel axis map",
        '"dt_bias": 0': "the dt_bias tensor-parallel axis map",
        '["query", "key", "value", "z", "beta", "alpha"]': "the projection split names",
        'conv_layer_name_list = ["conv1d.weight"]': "the conv1d split list",
        "_split_tensor_factory": "the shard-splitting factory",
    }
    missing = [why for marker, why in required.items() if marker not in source]
    if missing:
        raise RuntimeError(
            "Megatron's GatedDeltaNet.sharded_state_dict no longer matches the "
            "implementation reproduced in roll/third_party/megatron/gdn_lora_checkpoint.py; "
            "missing " + "; ".join(missing) + ". Re-derive the patch against the installed "
            "Megatron before training with LoRA."
        )


def is_already_fixed() -> bool:
    """Has Megatron started handling the LoRA-wrapped key itself?

    Unreadable source answers "no" here, which hands the decision to
    ``verify_upstream_contract``; that function treats the same condition as a
    hard error. Returning True would silently skip the patch and let the
    KeyError resurface at the first checkpoint.
    """
    import inspect

    from megatron.core.ssm.gated_delta_net import GatedDeltaNet

    if hasattr(GatedDeltaNet, _ORIGINAL):
        return False
    try:
        source = inspect.getsource(GatedDeltaNet.sharded_state_dict)
    except (OSError, TypeError):
        return False
    return "base_layer.weight" in source


def install_for_model_type(model_type: Optional[str]) -> bool:
    """Install the replacement for Flash-Next model types only.

    Returns whether the patch is installed after this call. Idempotent, and a
    no-op for every other model type so unrelated GDN models keep Megatron's
    own implementation.
    """
    from roll.models.qwen4_exp_scope import is_flash_next

    if not is_flash_next(model_type):
        return False

    from megatron.core.ssm.gated_delta_net import GatedDeltaNet

    if hasattr(GatedDeltaNet, _ORIGINAL):
        return True

    if is_already_fixed():
        logger.info(
            "Megatron handles LoRA-wrapped GDN input projections itself; " "skipping the Qwen4 checkpoint patch"
        )
        return False

    verify_upstream_contract()
    setattr(GatedDeltaNet, _ORIGINAL, GatedDeltaNet.sharded_state_dict)
    GatedDeltaNet.sharded_state_dict = _sharded_state_dict
    logger.info("Installed the Qwen4 GDN LoRA checkpoint patch for model_type=%s", model_type)
    return True


def uninstall() -> bool:
    """Restore Megatron's implementation. For tests and scope assertions."""
    from megatron.core.ssm.gated_delta_net import GatedDeltaNet

    original = getattr(GatedDeltaNet, _ORIGINAL, None)
    if original is None:
        return False
    GatedDeltaNet.sharded_state_dict = original
    delattr(GatedDeltaNet, _ORIGINAL)
    return True
