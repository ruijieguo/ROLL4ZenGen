"""Result-only reward for boxed single / multi / grouped multiple-choice answers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# Load the module file directly: importing it through the ``roll`` package pulls in ray and
# every other reward worker, while the parsing logic under test has no such dependency.
_PATH = Path(__file__).resolve().parents[2] / "roll/pipeline/rlvr/rewards/tom_exact_reward_worker.py"
_spec = importlib.util.spec_from_file_location("_tom_exact_reward_worker_under_test", _PATH)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
exact_result_reward = _module.exact_result_reward
parse_ground_truth = _module.parse_ground_truth


@pytest.mark.parametrize(
    "ground_truth, expected",
    [
        ("single:B", ("single", ["B"])),
        ("multi:C,A", ("multi", ["A", "C"])),
        ("grouped:A|c", ("grouped", ["A", "C"])),
    ],
)
def test_parse_ground_truth_reads_the_answer_mode(ground_truth, expected):
    assert parse_ground_truth(ground_truth) == expected


@pytest.mark.parametrize("ground_truth", ["B", "unknown:A", "single:", "grouped:|"])
def test_parse_ground_truth_rejects_malformed_labels(ground_truth):
    with pytest.raises(ValueError):
        parse_ground_truth(ground_truth)


def test_single_answer_uses_the_mode_prefix_not_its_first_character():
    # The bundled multiple-choice worker read ground_truth[0] and compared against "s".
    assert exact_result_reward("so the answer is \\boxed{B}", "single:B")[0] == 1.0
    assert exact_result_reward("\\boxed{S}", "single:B")[0] == 0.0


def test_single_answer_takes_the_last_box_and_rejects_multiple_letters():
    assert exact_result_reward("\\boxed{A} wait, \\boxed{C}", "single:C")[:2] == (1.0, False)
    assert exact_result_reward("\\boxed{A,C}", "single:C") == (0.0, False, ["A", "C"])


def test_multi_answer_is_order_insensitive_and_exact():
    assert exact_result_reward("\\boxed{C, A}", "multi:A,C")[0] == 1.0
    assert exact_result_reward("\\boxed{A}", "multi:A,C")[0] == 0.0


def test_grouped_answer_matches_the_trailing_boxes_in_order():
    response = "part one \\boxed{A} part two \\boxed{C}"
    assert exact_result_reward(response, "grouped:A|C") == (1.0, False, ["A", "C"])
    assert exact_result_reward(response, "grouped:C|A")[0] == 0.0


def test_missing_box_is_an_extraction_failure():
    assert exact_result_reward("the answer is B", "single:B") == (0.0, True, [])
    assert exact_result_reward("\\boxed{A}", "grouped:A|C") == (0.0, True, [])


def test_box_alias_is_accepted():
    assert exact_result_reward("\\box{D}", "single:D")[0] == 1.0
