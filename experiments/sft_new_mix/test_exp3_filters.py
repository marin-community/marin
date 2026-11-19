# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from marin.datakit.chat_render import render_chat_record

from experiments.sft_new_mix.exp3_filters import (
    BOS,
    END_HEADER,
    EOS,
    EOT,
    HEADER,
    MESSAGE,
    Decision,
    Document,
    FilterConfig,
    Stage,
    apply_stage,
    turns,
    unquote_opencode_prompt,
)
from experiments.sft_new_mix.opencode_questions import repair_text


def render(*messages: tuple[str, str]) -> str:
    """Build a native fixture through the real MARIN_CHAT_TEMPLATE framing."""
    record = {
        "id": "fixture",
        "messages": [
            (
                {"role": role, "content": text, "channel": "final"}
                if role == "assistant"
                else {"role": role, "content": text}
            )
            for role, text in messages
        ],
    }
    return BOS + render_chat_record(record)["text"] + EOS


def source_harness_text(*messages: tuple[str, str]) -> str:
    """Frame tool bodies as bare JSON the way the source harness caches them."""
    return (
        BOS
        + "".join(f"{HEADER}{role}{END_HEADER}\n{text}{EOT}\n" for role, text in messages[:-1])
        + (f"{HEADER}{messages[-1][0]}{END_HEADER}\n{messages[-1][1]}{EOT}")
        + EOS
    )


def apply(source: str, text: str, stage: Stage, *, passed: bool = False) -> Decision:
    config = FilterConfig(frozenset(Stage), {}, {})
    return apply_stage(Document(source, "fixture", text, passed), stage, config, len)


def assert_no_assistant_post_eot_newline(text: str) -> None:
    for match in MESSAGE.finditer(text):
        if match[1] == "assistant":
            assert text[match.end() : match.end() + 1] != "\n"


def test_outcome_requires_recorded_pass_even_for_confident_answer():
    text = render(("user", "Fix this"), ("assistant", "All tests passed. Done."))
    assert not apply("agenttrove", text, Stage.OUTCOME).keep
    assert apply("agenttrove", text, Stage.OUTCOME, passed=True).keep


def test_turn_edits_splice_into_template_framing_without_post_eot_newlines():
    text = render(("system", "Reasoning: /think"), ("system", "Be precise."), ("user", "2+2?"), ("assistant", "4"))
    result = apply("chat", text, Stage.SYSTEM_TURNS)
    expected = render(("system", "Reasoning: /think\n\nBe precise."), ("user", "2+2?"), ("assistant", "4"))
    assert result.document.text == expected
    assert_no_assistant_post_eot_newline(result.document.text)


def test_system_turn_merge_and_drop_splice_into_original_framing():
    text = render(("system", "First."), ("system", ""), ("system", "Second."), ("user", "Go"), ("assistant", "Ok"))
    result = apply("chat", text, Stage.SYSTEM_TURNS)
    expected = render(("system", "First.\n\nSecond."), ("user", "Go"), ("assistant", "Ok"))
    assert result.document.text == expected
    assert_no_assistant_post_eot_newline(result.document.text)
    assert [(turn.role, turn.content) for turn in turns(result.document.text)] == [
        ("system", "First.\n\nSecond."),
        ("user", "Go"),
        ("assistant", "Ok"),
    ]


def test_turn_edits_keep_role_order_when_a_turn_is_dropped_between_roles():
    text = render(("system", ""), ("user", "Task A"), ("system", "Keep this after Task A"), ("assistant", "Answer"))
    result = apply("chat", text, Stage.SYSTEM_TURNS)
    expected = render(("user", "Task A"), ("system", "Keep this after Task A"), ("assistant", "Answer"))
    assert result.document.text == expected
    assert [(turn.role, turn.content) for turn in turns(result.document.text)] == [
        ("user", "Task A"),
        ("system", "Keep this after Task A"),
        ("assistant", "Answer"),
    ]
    assert_no_assistant_post_eot_newline(result.document.text)


def test_system_cleanup_preserves_task_and_relabels_only_all_spanless_think():
    text = render(
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
    assert_no_assistant_post_eot_newline(result.document.text)


def test_mixed_think_mode_is_dropped_without_removing_reasoning():
    text = render(
        ("system", "Reasoning: /think"),
        ("assistant", "<|start_think|>Compute.<|end_think|>4"),
        ("user", "Again"),
        ("assistant", "4"),
    )
    result = apply("chat", text, Stage.THINK_MODE)
    assert not result.keep
    assert result.document.text == text


def test_think_mode_cleanup_does_not_rewrite_quoted_user_instructions():
    text = render(
        ("system", "Reasoning: /nothink"),
        ("user", "Explain the line Reasoning: /think"),
        ("assistant", "It enables reasoning."),
    )
    assert apply("chat", text, Stage.THINK_MODE).document.text == text


def test_lean_prompt_helpers_survive_but_duplicate_answer_declarations_do_not():
    source = "nemotron_sft_v3/math_proofs_v1/lean"
    task = "def helper := 2\nProve the statement."
    good = render(("user", task), ("assistant", "def helper := 2\ntheorem answer : helper = 2 := rfl"))
    bad = render(("user", task), ("assistant", "def helper := 2\ndef helper := 2"))
    assert apply(source, good, Stage.LEAN).keep
    assert not apply(source, bad, Stage.LEAN).keep


def test_native_miniswe_json_survives_without_unescaping_output():
    text = source_harness_text(
        ("user", "Fix"), ("assistant", "Run"), ("tool", '{"returncode": 0, "output": "a\\nb"}'), ("assistant", "Done")
    )
    result = apply("open_swe_traces/v1_1/minisweagent/qwen36_27b/scale_swe", text, Stage.MINI_JSON)
    assert result.document.text == text


def test_protected_diff_and_incomplete_terminal_action_are_rejected():
    source = "open_swe_traces/v1_0/sweagent/model/tasks"
    text = source_harness_text(
        ("user", "Fix"), ("tool", "diff --git a/tests/test_x.py b/tests/test_x.py\n+assert True"), ("assistant", "Done")
    )
    assert not apply(source, text, Stage.TEST_TAMPERING).keep
    text = render(("user", "Fix"), ("assistant", '{"commands": [{"keystrokes": "pytest"}], "task_complete": false}'))
    assert not apply(source, text, Stage.TRACE_ENDING).keep


def test_warning_cleanup_preserves_command_output_and_genuine_missing_field_warning():
    action = '{"commands": [], "task_complete": true}'
    warning = (
        "Program output: OK\nWARNINGS:\n- Extra text detected before JSON object\n"
        "ERROR: Missing required fields: analysis"
    )
    result = apply("agenttrove", render(("assistant", action), ("user", warning), ("assistant", "Done")), Stage.WARNINGS)
    content = turns(result.document.text)[1].content
    assert "Program output: OK" in content
    assert "Missing required fields: analysis" in content
    assert "Extra text detected" not in content
    assert_no_assistant_post_eot_newline(result.document.text)


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
    original = render(("user", row["content"]), ("assistant", row["content"]), ("user", quoted_subagent))
    expected = render(("user", row["expected"]), ("assistant", row["content"]), ("user", quoted_subagent))
    repaired, changed = repair_text(original, row["question"])
    assert repaired == expected
    assert changed == 1
    assert repair_text(repaired, row["question"]) == (expected, 0)
    assert_no_assistant_post_eot_newline(repaired)
