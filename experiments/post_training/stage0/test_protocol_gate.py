# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.post_training.chat_template import QWEN_TOKENS
from experiments.post_training.stage0.protocol_gate import (
    END_OF_TURN_ID,
    Completion,
    Delimiters,
    completion_protocol,
    is_agent_turn,
    protocol_report,
)

CLOSED = "<|start_think|>r<|end_think|>answer"


def _completion(family: str, text: str, *, ended: bool = True) -> Completion:
    return Completion(family, "p", "stop" if ended else "length", END_OF_TURN_ID if ended else None, text)


def test_report_counts_turn_ends_and_closed_think_blocks_per_family():
    completions = [_completion("P_think", CLOSED) for _ in range(98)]
    completions += [_completion("P_think", "<|end_think|>a<|start_think|>b")]  # reopened, never closed
    completions += [_completion("P_chat", "plain answer") for _ in range(100)]
    report = protocol_report(completions)
    think = report["families"]["P_think"]
    assert think["turn_end_share"] == 1.0 and think["think_close_share"] == 98 / 99 and think["passed"]
    assert report["families"]["P_chat"]["think_close_share"] is None and report["passed"]


def test_a_reply_capped_mid_think_is_length_limited_and_excluded_from_both_shares():
    completions = [_completion("P_think", CLOSED) for _ in range(97)]
    completions.append(_completion("P_think", "<|start_think|>still thinking" + "x" * 500, ended=False))
    think = protocol_report(completions)["families"]["P_think"]
    assert think["length_limited"] == 1 and think["scored"] == 97
    assert think["turn_end_share"] == 1.0 and think["think_close_share"] == 1.0 and think["passed"]


def test_a_reply_capped_after_its_think_block_still_fails_turn_end():
    capped = CLOSED + "word " * 200
    completions = [_completion("P_chat", CLOSED) for _ in range(9)] + [_completion("P_chat", capped, ended=False)]
    chat = protocol_report(completions)["families"]["P_chat"]
    assert chat["length_limited"] == 0 and chat["scored"] == 10
    assert chat["turn_end_share"] == 0.9 and not chat["passed"]


def test_a_completion_stopped_by_another_token_does_not_end_its_turn():
    stopped_elsewhere = Completion("P_chat", "p", "stop", None, "answer")
    report = protocol_report([stopped_elsewhere] + [_completion("P_chat", "a") for _ in range(9)])
    assert report["families"]["P_chat"]["turn_end_share"] == 0.9 and not report["passed"]


def test_length_limited_replies_report_how_many_end_in_a_repetition_loop():
    looping = "<|start_think|>hmm " + "abc " * 2000
    progressing = "<|start_think|>working: " + " ".join(f"step{index}" for index in range(500))
    completions = [_completion("P_think", looping, ended=False), _completion("P_think", progressing, ended=False)]
    think = protocol_report(completions)["families"]["P_think"]
    assert think["length_limited"] == 2 and think["length_limited_repetition_loop"] == 1
    # Every reply was length-limited, so there is no share to gate on.
    assert think["turn_end_share"] is None and not think["passed"]


def test_length_limited_loop_count_keeps_nonloop_caps_separate():
    looping = _completion(
        "P_think", "<|start_think|>" + "Let us reconsider this calculation carefully. " * 20, ended=False
    )
    ordinary = _completion("P_think", "<|start_think|>a calculation unfinished at the cap", ended=False)
    report = protocol_report([looping, ordinary])["families"]["P_think"]
    assert report["length_limited"] == 2
    assert report["length_limited_repetition_loop"] == 1


def test_cap_exclusion_requires_an_open_thinking_block_and_evaluable_replies():
    open_cap = _completion("P_think", "<|start_think|>unfinished", ended=False)
    plain_cap = _completion("P_think", "unfinished answer", ended=False)
    closed_cap = _completion("P_think", "<|start_think|>work<|end_think|>answer", ended=False)
    report = protocol_report([open_cap, plain_cap, closed_cap])["families"]["P_think"]
    assert report["length_limited"] == 1
    assert report["scored"] == 2
    assert report["turn_end_share"] == 0 and report["think_close_share"] == 1
    assert not report["passed"]
    empty = protocol_report([open_cap])["families"]["P_think"]
    assert empty["turn_end_share"] is None and not empty["passed"]


def test_agent_format_counts_replies_whose_answer_is_a_terminus_json_turn():
    agent = '<|start_think|>r<|end_think|>{"analysis": "a", "plan": "p", "commands": []}'
    assert is_agent_turn(agent)
    assert not is_agent_turn('<|start_think|>{"analysis"<|end_think|>The answer is 4.')
    report = protocol_report([_completion("P_chat", agent), _completion("P_chat", "plain")])
    assert report["families"]["P_chat"]["agent_format_share"] == 0.5


def test_prefilled_thinking_closes_but_native_eos_is_not_an_end_of_turn():
    delimiters = Delimiters(QWEN_TOKENS, 151645)
    ended = completion_protocol("reason</think>answer", "stop", 151645, delimiters, prefill="<think>\n")
    eos = completion_protocol("reason</think>answer", "stop", 151643, delimiters, prefill="<think>\n")
    unclosed = completion_protocol("reason", "length", None, delimiters, prefill="<think>\n")
    assert ended["think_blocks"] == 1 and ended["think_closed"] and ended["turn_ended"]
    assert eos["think_closed"] and not eos["turn_ended"]
    assert not unclosed["think_closed"] and not unclosed["turn_ended"]
