# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from experiments.sft_new_mix.exp3_filters import (
    BOS,
    END_HEADER,
    EOS,
    EOT,
    HEADER,
    Decision,
    Document,
    FilterConfig,
    Stage,
    apply_stage,
    turns,
    unquote_opencode_prompt,
)
from experiments.sft_new_mix.opencode_questions import repair_text


def chat(*messages: tuple[str, str]) -> str:
    return BOS + "".join(f"{HEADER}{role}{END_HEADER}\n{text}{EOT}\n" for role, text in messages) + EOS


def apply(source: str, text: str, stage: Stage, *, passed: bool = False) -> Decision:
    config = FilterConfig(frozenset(Stage), {}, {})
    return apply_stage(Document(source, "fixture", text, passed), stage, config, len)


def test_outcome_requires_recorded_pass_even_for_confident_answer():
    text = chat(("user", "Fix this"), ("assistant", "All tests passed. Done."))
    assert not apply("agenttrove", text, Stage.OUTCOME).keep
    assert apply("agenttrove", text, Stage.OUTCOME, passed=True).keep


def test_system_cleanup_preserves_task_and_relabels_only_all_spanless_think():
    text = chat(
        ("system", "Reasoning: /think"), ("system", ""), ("system", "Be precise."), ("user", "2+2?"), ("assistant", "4")
    )
    result = apply("chat", text, Stage.SYSTEM_TURNS)
    assert [(turn.role, turn.content) for turn in turns(result.document.text)] == [
        ("system", "Reasoning: /think\n\nBe precise."),
        ("user", "2+2?"),
        ("assistant", "4"),
    ]
    result = apply("chat", result.document.text, Stage.THINK_MODE)
    assert turns(result.document.text)[0].content == "Reasoning: /nothink\n\nBe precise."


def test_mixed_think_mode_is_dropped_without_removing_reasoning():
    text = chat(
        ("system", "Reasoning: /think"),
        ("assistant", "<|start_think|>Compute.<|end_think|>4"),
        ("user", "Again"),
        ("assistant", "4"),
    )
    result = apply("chat", text, Stage.THINK_MODE)
    assert not result.keep
    assert result.document.text == text


def test_think_mode_cleanup_does_not_rewrite_quoted_user_instructions():
    text = chat(
        ("system", "Reasoning: /nothink"),
        ("user", "Explain the line Reasoning: /think"),
        ("assistant", "It enables reasoning."),
    )
    assert apply("chat", text, Stage.THINK_MODE).document.text == text


def test_lean_prompt_helpers_survive_but_duplicate_answer_declarations_do_not():
    source = "nemotron_sft_v3/math_proofs_v1/lean"
    task = "def helper := 2\nProve the statement."
    good = chat(("user", task), ("assistant", "def helper := 2\ntheorem answer : helper = 2 := rfl"))
    bad = chat(("user", task), ("assistant", "def helper := 2\ndef helper := 2"))
    assert apply(source, good, Stage.LEAN).keep
    assert not apply(source, bad, Stage.LEAN).keep


def test_native_miniswe_json_survives_without_unescaping_output():
    text = chat(
        ("user", "Fix"), ("assistant", "Run"), ("tool", '{"returncode": 0, "output": "a\\nb"}'), ("assistant", "Done")
    )
    result = apply("open_swe_traces/v1_1/minisweagent/qwen36_27b/scale_swe", text, Stage.MINI_JSON)
    assert result.document.text == text


def test_protected_diff_and_incomplete_terminal_action_are_rejected():
    source = "open_swe_traces/v1_0/sweagent/model/tasks"
    text = chat(
        ("user", "Fix"), ("tool", "diff --git a/tests/test_x.py b/tests/test_x.py\n+assert True"), ("assistant", "Done")
    )
    assert not apply(source, text, Stage.TEST_TAMPERING).keep
    text = chat(("user", "Fix"), ("assistant", '{"commands": [{"keystrokes": "pytest"}], "task_complete": false}'))
    assert not apply(source, text, Stage.TRACE_ENDING).keep


def test_warning_cleanup_preserves_command_output_and_genuine_missing_field_warning():
    action = '{"commands": [], "task_complete": true}'
    warning = (
        "Program output: OK\nWARNINGS:\n- Extra text detected before JSON object\n"
        "ERROR: Missing required fields: analysis"
    )
    result = apply("agenttrove", chat(("assistant", action), ("user", warning), ("assistant", "Done")), Stage.WARNINGS)
    content = turns(result.document.text)[1].content
    assert "Program output: OK" in content
    assert "Missing required fields: analysis" in content
    assert "Extra text detected" not in content


def test_raw_question_wrapper_preserves_notes_and_nested_quoted_prompts():
    question = 'Explain "quotes" and \\slashes literally.'
    wrapped = '"Explain \\"quotes\\" and \\slashes literally.'
    notes = '\n<note>Keep \\"quoted\\" evidence.</note>'
    assert unquote_opencode_prompt(wrapped + notes + '"', question) == question + notes
    subagent = '"Review the implementation and quote the task."'
    assert unquote_opencode_prompt(subagent, question) == subagent
    extra = wrapped + '\nextra instructions"'
    assert unquote_opencode_prompt(extra, question) == extra
    four_notes = wrapped + "<note>x</note>" * 4 + '"'
    assert unquote_opencode_prompt(four_notes, question) == four_notes


@pytest.mark.parametrize(
    "row",
    json.loads((Path(__file__).parent / "fixtures/opencode_raw_questions.json").read_text()),
    ids=lambda row: row["partition"],
)
def test_opencode_pinned_real_questions_repair_only_user_wrapper(row):
    quoted_subagent = '"Review the code independently; keep this quoted subagent request."'
    original = chat(("user", row["content"]), ("assistant", row["content"]), ("user", quoted_subagent))
    expected = chat(("user", row["expected"]), ("assistant", row["content"]), ("user", quoted_subagent))
    repaired, changed = repair_text(original, row["question"])
    assert repaired == expected
    assert changed == 1
    assert repair_text(repaired, row["question"]) == (expected, 0)
