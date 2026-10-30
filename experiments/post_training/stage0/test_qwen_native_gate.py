# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.post_training.stage0.protocol_gate import Decode
from experiments.post_training.stage0.qwen_native_gate import (
    Thinking,
    gate_summary,
    math_correct,
    native_facts,
    native_request,
)


def test_qwen_native_requests_keep_full_cap_and_model_specific_prefills():
    messages = [{"role": "user", "content": "Two plus two?"}]
    qwen3 = native_request("qwen3_moe", messages, Thinking.ENABLED, Decode(), [151645, 151643])
    assert qwen3["messages"][-1] == {"role": "assistant", "content": "<think>\n"}
    assert qwen3["continue_final_message"] and not qwen3["add_generation_prompt"]
    assert qwen3["max_tokens"] == 32768 and qwen3["stop_token_ids"] == [151645, 151643]
    plain = native_request("qwen3_moe", messages, Thinking.DISABLED, Decode(), [151645, 151643])
    assert plain["messages"] == messages and "continue_final_message" not in plain
    qwen35 = native_request("qwen3_5_moe", messages, Thinking.ENABLED, Decode(), [248046, 248044])
    assert qwen35["messages"] == messages and qwen35["chat_template_kwargs"] == {"enable_thinking": True}
    plain35 = native_request("qwen3_5_moe", messages, Thinking.DISABLED, Decode(), [248046, 248044])
    assert plain35["chat_template_kwargs"] == {"enable_thinking": False}


def test_native_gate_separates_eos_empty_answers_and_unclosed_reasoning():
    ids = [151645, 151643]
    closed = native_facts("work</think>4", "stop", ids[0], "<think>\n", ids)
    empty = native_facts("work</think><|endoftext|>", "stop", ids[1], "<think>\n", ids)
    capped = native_facts("The answer might be 4", "length", None, "<think>\n", ids)
    assert closed["think_closed_with_answer"] and closed["answer_text"] == "4"
    assert empty["think_closed"] and not empty["think_closed_with_answer"]
    assert capped["answer_text"] == "" and not capped["think_closed"]
    records = [
        {"probe_id": str(index), "facts": fact, "text": "sample", "correct": correct, "finish_reason": finish}
        for index, (fact, correct, finish) in enumerate(
            [(closed, True, "stop"), (empty, False, "stop"), (capped, False, "length")]
        )
    ]
    report = gate_summary(records)
    assert report["end_tokens"] == {"<|im_end|>": 1, "<|endoftext|>": 1, "none": 1}
    assert report["think_closed_with_answer_share"] == 1 / 2
    assert report["accuracy"] == 1 / 3
    assert report["turn_end_share"] == 1 / 2 and report["length_limited"] == 1
    assert report["length_capped"] == 1
    plain = native_facts("4", "stop", ids[0], "", ids)
    assert gate_summary([*records, {**records[0], "facts": plain}])["replies_opening_think"] == 3


def test_math_accuracy_does_not_require_think_closure():
    text = r"The answer is \boxed{\frac{1}{2}}.<|endoftext|>"
    facts = native_facts(text, "stop", 151643, "<think>\n", [151645, 151643])
    assert not facts["think_closed_with_answer"]
    assert math_correct("0.5", text)
    assert not math_correct("0.25", text)
    assert math_correct("0.5", "work</think>" + text)
