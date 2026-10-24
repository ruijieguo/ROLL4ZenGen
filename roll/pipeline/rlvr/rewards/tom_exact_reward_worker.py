"""Binary result-only reward for single, multi, and grouped boxed answers."""

from __future__ import annotations

import re
from typing import Any

try:
    import torch
    from roll.distributed.executor.worker import Worker
    from roll.distributed.scheduler.decorator import Dispatch, register
    from roll.distributed.scheduler.protocol import DataProto
    from roll.models.model_providers import default_tokenizer_provider
    from roll.utils.logging import get_logger

    _ROLL_AVAILABLE = True
    logger = get_logger()
except ImportError:
    torch = None
    Worker = object
    DataProto = object
    default_tokenizer_provider = None
    _ROLL_AVAILABLE = False

    class _Dispatch:
        ONE_TO_ALL = "one_to_all"
        DP_MP_COMPUTE = "dp_mp_compute"

    Dispatch = _Dispatch()

    def register(*args, **kwargs):
        def decorator(function):
            return function
        return decorator


BOX_RE = re.compile(r"\\box(?:ed)?\s*\{([^{}]*)\}", re.IGNORECASE)
LETTER_RE = re.compile(r"[A-Z]")


def normalize_letters(value: Any) -> list[str]:
    return sorted(set(LETTER_RE.findall(str(value or "").upper())))


def parse_ground_truth(value: str) -> tuple[str, list[str]]:
    if ":" not in str(value):
        raise ValueError(f"Missing answer mode in ground truth: {value!r}")
    mode, payload = str(value).split(":", 1)
    if mode == "grouped":
        expected = [part.strip().upper() for part in payload.split("|") if part.strip()]
    elif mode in {"single", "multi"}:
        expected = normalize_letters(payload)
    else:
        raise ValueError(f"Unsupported answer mode: {mode}")
    if not expected:
        raise ValueError(f"Empty ground truth: {value!r}")
    return mode, expected


def exact_result_reward(response: str, ground_truth: str) -> tuple[float, bool, Any]:
    """Return (0/1 reward, extraction_failed, normalized_prediction)."""
    boxes = BOX_RE.findall(response or "")
    mode, expected = parse_ground_truth(ground_truth)
    if mode == "grouped":
        if len(boxes) < len(expected):
            return 0.0, True, []
        values = [normalize_letters(box) for box in boxes[-len(expected):]]
        if any(len(value) != 1 for value in values):
            return 0.0, any(not value for value in values), values
        predicted = [value[0] for value in values]
        return float(predicted == expected), False, predicted
    if not boxes:
        return 0.0, True, []
    predicted = normalize_letters(boxes[-1])
    if not predicted:
        return 0.0, True, []
    if mode == "single" and len(predicted) != 1:
        return 0.0, False, predicted
    return float(predicted == expected), False, predicted


class TomExactRewardWorker(Worker):
    """ROLL reward worker with no format, length, or process shaping."""

    def __init__(self, worker_config):
        if not _ROLL_AVAILABLE:
            raise RuntimeError("TomExactRewardWorker requires ROLL")
        super().__init__(worker_config=worker_config)
        self.rank_info.dp_rank = self.rank_info.rank
        self.rank_info.dp_size = self.rank_info.world_size
        self.tokenizer = default_tokenizer_provider(model_args=self.worker_config.model_args)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        pass

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, clear_cache=False)
    def compute_rewards(self, data):
        responses = self.tokenizer.batch_decode(data.batch["responses"], skip_special_tokens=True)
        ground_truths = data.non_tensor_batch["ground_truth"]
        scores = []
        extraction_failures = []
        for response, ground_truth in zip(responses, ground_truths):
            reward, extraction_failed, prediction = exact_result_reward(response, str(ground_truth))
            scores.append(reward)
            extraction_failures.append(float(extraction_failed))
            logger.debug(
                "[tom_exact_reward] reward=%s extraction_failed=%s prediction=%s gold=%s",
                reward,
                extraction_failed,
                prediction,
                ground_truth,
            )

        scores_tensor = torch.tensor(scores, dtype=torch.float16)
        token_level_rewards = torch.zeros_like(data.batch["responses"], dtype=torch.float16)
        n = max(1, len(scores))
        output = DataProto.from_dict(
            tensors={
                "token_level_rewards": token_level_rewards,
                "response_level_rewards": scores_tensor,
                "scores": scores_tensor,
            }
        )
        output.meta_info = {
            "metrics": {
                "reward/exact_match_mean": sum(scores) / n,
                "reward/extraction_failure_rate": sum(extraction_failures) / n,
            }
        }
        return output
