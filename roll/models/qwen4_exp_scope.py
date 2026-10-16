"""Which HF model types the Flash-Next patches apply to.

Both the training-side Megatron patches and the inference-side vLLM LoRA
shrink fix must agree on this set: a patch that is installed for one but not
the other would make training and rollout disagree numerically. Keeping the
list here, rather than in either of those packages, avoids a dependency
between them and gives the scope one owner.

``qwen4_exp`` is the HF ``model_type`` registered by
``mcore_adapter.models.qwen4_exp``; the ``_text`` variants are the text
towers reported when the config is nested under a multimodal wrapper.
"""

FLASH_NEXT_MODEL_TYPES = frozenset(
    {
        "qwen4_exp",
        "qwen4_exp_text",
        "qwen3_8_flash_next",
        "qwen3_8_flash_next_text",
    }
)


def is_flash_next(model_type) -> bool:
    """Is ``model_type`` a Flash-Next variant these patches are scoped to?"""
    return model_type in FLASH_NEXT_MODEL_TYPES
