# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.post_training.stage0.protocol_gate import END_OF_TURN_ID, Completion, protocol_report


def _completion(family: str, text: str, *, ended: bool = True) -> Completion:
    return Completion(family, "p", "stop" if ended else "length", END_OF_TURN_ID if ended else None, text)


def test_report_counts_turn_ends_and_closed_think_blocks_per_family():
    closed = "<|start_think|>r<|end_think|>answer"
    completions = [_completion("P_think", closed) for _ in range(98)]
    completions += [_completion("P_think", "<|start_think|>never closed", ended=False)]
    completions += [_completion("P_think", "<|end_think|>a<|start_think|>b")]  # reopened, never closed
    completions += [_completion("P_chat", "plain answer") for _ in range(100)]
    report = protocol_report(completions)
    think = report["families"]["P_think"]
    assert think["turn_end_share"] == 0.99 and think["think_close_share"] == 0.98 and think["passed"]
    assert report["families"]["P_chat"]["think_close_share"] is None and report["passed"]


def test_a_completion_stopped_by_another_token_does_not_end_its_turn():
    stopped_elsewhere = Completion("P_chat", "p", "stop", None, "answer")
    report = protocol_report([stopped_elsewhere] + [_completion("P_chat", "a") for _ in range(9)])
    assert report["families"]["P_chat"]["turn_end_share"] == 0.9 and not report["passed"]
