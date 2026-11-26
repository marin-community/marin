# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

# The converter tests whose grader construction needs the atlas runtime stack (the
# fresh-machine ScriptGrader contract) run on the atlas lanes; this catalog branch
# keeps the modules as inventory records and tests what its stack can build.
from experiments.post_training.task_curation.datasets.qa_formatting_grade import format_verification, score_abstention

ROWS = json.loads((Path(__file__).parent / "fixtures/instruction_sources/qa-formatting.json").read_text())


def test_normalized_boxed_abstention_bypasses_judge_but_never_receives_correct_reward():
    def unexpected_judge(prompt):
        raise AssertionError("Explicit abstention must not invoke the judge")

    reward, detail = score_abstention("Question", "Target", r"<think>reasoning</think>\boxed{ [IDK] }", unexpected_judge)
    assert reward == 0.5 and detail["verdict"] == "abstain" and detail["judge_output"] is None


@pytest.mark.parametrize("reply,reward", [("A", 1.0), ("B", 0.0), ("C", 0.5), ("?", 0.0)])
def test_semantic_judge_uses_native_prompt_and_three_way_reward(reply, reward):
    def judge(prompt):
        assert "Gold target: CQ Press" in prompt and "Predicted answer: Congressional Quarterly" in prompt
        return reply

    actual, detail = score_abstention("Publisher?", "CQ Press", r"\boxed{Congressional Quarterly}", judge)
    assert actual == reward and detail["extracted_answer"] == "Congressional Quarterly"


def test_formatter_counts_lines_once_and_rejects_spurious_citations():
    verifier = {"type": "regex", "verify_regex": [r"^\d+\.", r"^1\."], "verify_min_matches": 2}
    assert format_verification.grade_format("1. one line", verifier)[0] == 0
    assert format_verification.grade_format("1. one line\n2. another", verifier)[0] == 1
    verifier = {"type": "string_match", "patterns": [r"\[ref:\d+\]"], "expected_markers": ["[ref:2]"]}
    assert format_verification.grade_format("[ref:2]", verifier)[0] == 1
    assert format_verification.grade_format("[ref:2] [ref:3]", verifier)[0] == 0
