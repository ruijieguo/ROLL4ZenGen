"""Thinking-trace SFT targets must land in reasoning_content, not inside content.

With a thinking-enabled template (``qwen3``) the assistant turn renders as
``<think>\\n{reasoning_content}\\n</think>\\n\\n{content}``. Passing a response that already
contains ``<think>...</think>`` as plain ``content`` used to supervise an *empty* think block
followed by the trace inside the answer. ``_assistant_turn`` splits such responses.

The pure-Python split is tested without any tokenizer. The end-to-end check needs the real
Flash-Next tokenizer; set ``QWEN38_TOKENIZER_PATH`` to run it.
"""
import os

import pytest

from roll.pipeline.sft.sft_pipeline import _assistant_turn


def test_assistant_turn_splits_explicit_think_block():
    turn = _assistant_turn("<think>\nreason A\n</think>\n\nanswer B")
    assert turn == {"role": "assistant", "reasoning_content": "reason A", "content": "answer B"}


def test_assistant_turn_accepts_trace_without_opening_tag():
    # Models prompted with a generation prefix ending in ``<think>\n`` emit only the closing tag.
    turn = _assistant_turn("reason A\n</think>\n\nanswer B")
    assert turn == {"role": "assistant", "reasoning_content": "reason A", "content": "answer B"}


def test_assistant_turn_leaves_plain_answers_alone():
    assert _assistant_turn("just an answer") == {"role": "assistant", "content": "just an answer"}


def test_assistant_turn_splits_on_first_close_tag_only():
    turn = _assistant_turn("<think>r</think>a mentions </think> literally")
    assert turn["reasoning_content"] == "r"
    assert turn["content"] == "a mentions </think> literally"


@pytest.fixture(scope="module")
def tokenizer():
    model = os.environ.get("QWEN38_TOKENIZER_PATH")
    if not model:
        pytest.skip("set QWEN38_TOKENIZER_PATH to the real Flash-Next tokenizer")
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model, trust_remote_code=True)


def test_qwen3_supervised_segment_has_single_think_block(tokenizer):
    from roll.datasets.chat_template import get_chat_template
    from roll.pipeline.sft.sft_pipeline import get_encode_function

    template = get_chat_template("qwen3", tokenizer)
    encode = get_encode_function("qwen3", tokenizer, "prompt", None, "response", "system")
    base = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    expected = template(base + [{"role": "assistant", "reasoning_content": "reason A", "content": "answer B"}],
                        add_generation_prompt=False).rstrip("\n")
    prefix_len = len(tokenizer(template(base, add_generation_prompt=True))["input_ids"])
    for response in ("<think>\nreason A\n</think>\n\nanswer B", "reason A\n</think>\n\nanswer B"):
        out = encode({"system": ["S"], "prompt": ["U"], "response": [response]})
        ids, labels = out["input_ids"][0], out["labels"][0]
        full = tokenizer.decode(ids)
        assert full.count("<think>") == 1
        assert full == expected
        supervised = tokenizer.decode([t for t, l in zip(ids, labels) if l != -100])
        assert supervised.startswith("reason A"), supervised
        assert labels[:prefix_len] == [-100] * prefix_len
