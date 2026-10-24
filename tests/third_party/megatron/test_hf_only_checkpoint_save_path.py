"""HF-only checkpoint save path for full-parameter qwen4_exp MoE training.

Covers the four fixes that let multi-node GRPO write a usable HF checkpoint:

1. ``gather_tensor_parallel`` accepts ``is_expert_parallel`` and gathers over the
   expert tensor-parallel group, skipping the collective for a single-rank group.
2. ``_needs_moe_allgather`` includes ``qwen4_exp`` so expert weights are gathered
   across EP before the fused-expert template sees them.
3. ``VirtualModels.save_external_assets`` unwraps DDP before looking for the
   N-gram asset hooks, so ``mca_external_assets.json`` is written.
4. ``MegatronTrainStrategy.save_checkpoint`` returns before the torch_dist path when
   ``save_hf_model`` and ``save_only_model`` are both set.

These run without GPUs or Megatron by exercising the functions with stubs or by
inspecting the source.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
CONVERT_UTILS = ROOT / "mcore_adapter/src/mcore_adapter/models/converter/convert_utils.py"
MODEL_CONVERTER = ROOT / "mcore_adapter/src/mcore_adapter/models/converter/model_converter.py"
MODEL_FACTORY = ROOT / "mcore_adapter/src/mcore_adapter/models/model_factory.py"
MEGATRON_STRATEGY = ROOT / "roll/distributed/strategy/megatron_strategy.py"


def _function_source(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(path.read_text(encoding="utf-8"), node)
    raise AssertionError(f"{name} not found in {path}")


def _load_gather_tensor_parallel(mpu: SimpleNamespace, dist: SimpleNamespace):
    """Compile gather_tensor_parallel alone against stub mpu / dist modules."""
    src = _function_source(CONVERT_UTILS, "gather_tensor_parallel")
    namespace = {"torch": torch, "mpu": mpu, "dist": dist}
    exec(compile(src, str(CONVERT_UTILS), "exec"), namespace)
    return namespace["gather_tensor_parallel"]


class _RecordingDist:
    def __init__(self, world_size: int, ranks: list[int]) -> None:
        self._world_size = world_size
        self._ranks = ranks
        self.gather_calls: list[dict] = []

    def get_world_size(self, group) -> int:
        return self._world_size

    def get_process_group_ranks(self, group) -> list[int]:
        return self._ranks

    def gather(self, tensor, gather_list, dst, group, async_op):
        self.gather_calls.append({"group": group, "dst": dst, "has_list": gather_list is not None})
        if gather_list is not None:
            for i in range(len(gather_list)):
                gather_list[i].copy_(tensor + i)
        return None


def _mpu(etp_group, tp_rank: int = 0, etp_rank: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        get_tensor_model_parallel_group=lambda: "tp-group",
        get_tensor_model_parallel_rank=lambda: tp_rank,
        get_expert_tensor_parallel_group=lambda check_initialized=True: etp_group,
        get_expert_tensor_parallel_rank=lambda: etp_rank,
    )


def test_expert_gather_uses_etp_group_not_tp_group():
    dist = _RecordingDist(world_size=2, ranks=[10, 11])
    gather = _load_gather_tensor_parallel(_mpu("etp-group"), dist)

    out = gather(torch.zeros(3), is_expert_parallel=True)

    assert [c["group"] for c in dist.gather_calls] == ["etp-group"]
    assert dist.gather_calls[0]["dst"] == 10
    assert len(out) == 2


def test_single_rank_expert_group_skips_the_collective():
    dist = _RecordingDist(world_size=1, ranks=[10])
    gather = _load_gather_tensor_parallel(_mpu("etp-group"), dist)
    t = torch.ones(4)

    out = gather(t, is_expert_parallel=True)

    assert dist.gather_calls == []
    assert len(out) == 1 and out[0] is t


def test_missing_etp_group_falls_back_to_tp_group():
    dist = _RecordingDist(world_size=4, ranks=[0, 1, 2, 3])
    gather = _load_gather_tensor_parallel(_mpu(None), dist)

    gather(torch.zeros(2), is_expert_parallel=True)

    assert [c["group"] for c in dist.gather_calls] == ["tp-group"]


def test_non_root_rank_still_enters_the_gather():
    # A rank that is not the gather root must still call gather, otherwise the group
    # deadlocks. It passes no output list.
    dist = _RecordingDist(world_size=2, ranks=[10, 11])
    gather = _load_gather_tensor_parallel(_mpu("etp-group", etp_rank=1), dist)

    gather(torch.zeros(3), is_expert_parallel=True)

    assert dist.gather_calls == [{"group": "etp-group", "dst": 10, "has_list": False}]


def test_dense_gather_keeps_the_tp_group():
    dist = _RecordingDist(world_size=2, ranks=[0, 1])
    gather = _load_gather_tensor_parallel(_mpu("etp-group"), dist)

    gather(torch.zeros(3))

    assert [c["group"] for c in dist.gather_calls] == ["tp-group"]


def _needs_moe_allgather(hf_model_type: str, ep_world_size: int, is_expert: bool) -> bool:
    src = _function_source(MODEL_CONVERTER, "_needs_moe_allgather")
    namespace: dict = {}
    exec(compile(src, str(MODEL_CONVERTER), "exec"), namespace)
    owner = SimpleNamespace(
        mca_config=SimpleNamespace(hf_model_type=hf_model_type),
        dist_converter=SimpleNamespace(is_expert_parallel_weight=lambda name: is_expert),
    )
    return namespace["_needs_moe_allgather"](owner, "decoder.layers.0.mlp.experts.weight1", ep_world_size)


@pytest.mark.parametrize("hf_model_type", ["qwen3_vl_moe", "qwen3_5_moe", "qwen4_exp"])
def test_fused_expert_models_gather_across_ep(hf_model_type):
    assert _needs_moe_allgather(hf_model_type, ep_world_size=8, is_expert=True)


def test_allgather_is_skipped_without_ep_or_for_dense_weights():
    assert not _needs_moe_allgather("qwen4_exp", ep_world_size=1, is_expert=True)
    assert not _needs_moe_allgather("qwen4_exp", ep_world_size=8, is_expert=False)
    assert not _needs_moe_allgather("qwen2_moe", ep_world_size=8, is_expert=True)


def _load_save_external_assets():
    src = _function_source(MODEL_FACTORY, "save_external_assets")
    calls: list[tuple] = []
    pkg = types.ModuleType("_hf_only_pkg")
    pkg.__path__ = []
    lifecycle = types.ModuleType("_hf_only_pkg.qwen4_exp.asset_lifecycle")

    def persist_ngram_asset_models(models, save_directory):
        calls.append((tuple(models), save_directory))
        return {"manifest": save_directory}

    lifecycle.persist_ngram_asset_models = persist_ngram_asset_models
    sub = types.ModuleType("_hf_only_pkg.qwen4_exp")
    sys.modules.update({"_hf_only_pkg": pkg, "_hf_only_pkg.qwen4_exp": sub, lifecycle.__name__: lifecycle})
    namespace = {"torch": torch, "__name__": "_hf_only_pkg.model_factory", "__package__": "_hf_only_pkg"}
    exec(compile(src, str(MODEL_FACTORY), "exec"), namespace)
    return namespace["save_external_assets"], calls


class _NgramModel:
    def attach_ngram_assets(self):  # pragma: no cover - only its presence matters
        pass


class _MegatronStyleDDP:
    """Mimics Megatron DDP: holds the module and forwards no attributes."""

    def __init__(self, module) -> None:
        self.module = module


def test_ddp_wrapped_ngram_model_still_writes_the_manifest():
    save_external_assets, calls = _load_save_external_assets()
    inner = _NgramModel()
    owner = SimpleNamespace(models=[_MegatronStyleDDP(inner)])

    saved = save_external_assets(owner, "/ckpt")

    assert calls == [((inner,), "/ckpt")]
    assert saved == [{"manifest": "/ckpt"}]


def test_plain_hooks_also_see_unwrapped_models(monkeypatch):
    save_external_assets, calls = _load_save_external_assets()
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    seen = []
    inner = SimpleNamespace(save_external_assets=lambda d: seen.append(d) or "ok")
    owner = SimpleNamespace(models=[_MegatronStyleDDP(inner)])

    assert save_external_assets(owner, "/ckpt") == ["ok"]
    assert seen == ["/ckpt"] and calls == []


def test_hf_only_save_returns_before_the_torch_dist_path():
    src = _function_source(MEGATRON_STRATEGY, "save_checkpoint")
    guard = "if self.megatron_train_args.save_hf_model and self.megatron_train_args.save_only_model:"
    assert guard in src, "HF-only early return is missing"
    head, tail = src.split(guard, 1)
    assert "ckpt_format = self.megatron_train_args.ckpt_format" in head
    early_block = tail.split("# save model and tokenizer", 1)[0]
    assert "return {" in early_block
    assert "dist.barrier()" in early_block
    assert "checkpoint_manager.upload" in early_block
    # The torch_dist writer must come after the early return.
    assert "dist_checkpointing" not in early_block
