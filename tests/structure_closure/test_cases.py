# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from experiments.post_training.task_curation.datasets.bfcl_structure_closure.cases import (
    Family,
    Split,
    score_case,
    syntax_state,
    task_cases,
)


def test_reference_continuations_cover_each_family_and_preserve_content(tokenizer):
    answer = [{"search": {"query": 'quoted " slash \\ </tool_call>'}}]
    cases = task_cases("example_1234", answer, Split.VALIDATION, "source", tokenizer)
    assert {case.family for case in cases} == set(Family)
    assert len({case.case_id for case in cases}) == len(cases)
    for case in cases:
        if case.family == Family.OBJECT:
            assert json.loads(case.expected) == {"task": "example_1234"}
        elif case.family == Family.ARRAY:
            expected = ["example_1234", True] if case.control == "value_continue_required" else ["example_1234"]
            assert json.loads(case.expected) == expected
        elif case.family == Family.NESTED:
            assert json.loads(case.expected) == {"task": "example_1234", "answers": [answer]}
        elif case.family in (Family.TOOL, Family.REPEATED_OPENER):
            assert json.loads(case.expected.removeprefix("<tool_call>").removesuffix("</tool_call>")) == {
                "name": "write_result",
                "arguments": {"task": "example_1234", "result": answer},
            }
        elif case.family == Family.FENCE:
            assert json.loads("\n".join(case.expected.splitlines()[1:-1])) == answer
        assert score_case(case, case.expected[len(case.prefix) :]).success
        assert not score_case(case, "").success


def test_close_probes_supply_content_only_in_the_prefix_while_continue_controls_supply_a_reference(tokenizer):
    cases = task_cases("example_1234", [{"search": {"query": "unique payload"}}], Split.VALIDATION, "source", tokenizer)
    for case in cases:
        if case.control == "close_required":
            assert "example_1234" not in case.instruction
            assert case.expected not in case.instruction
            suffix = syntax_state(case.prefix, case.family).closing_suffix
            assert suffix is not None
            assert score_case(case, suffix).success
        else:
            assert case.expected in case.instruction


def test_premature_closure_cannot_pass_continue_controls(tokenizer):
    cases = task_cases("example_1234", [], Split.VALIDATION, "source", tokenizer)
    tested = set()
    for case in cases:
        if case.control != "continue_required":
            continue
        suffix = syntax_state(case.prefix, case.family).closing_suffix
        if suffix is None:
            continue
        result = score_case(case, suffix)
        assert result.syntax_complete
        assert not result.content_preserved
        assert not result.success
        tested.add(case.family)
    assert Family.STRING in tested and Family.REASONING in tested and Family.FENCE in tested


def test_tool_reference_is_a_serialization_probe_not_a_required_live_tool(tokenizer):
    cases = task_cases("example_1234", [{"f": {"x": 3}}], Split.TRAIN, "source", tokenizer)
    case = next(case for case in cases if case.family == Family.TOOL and case.control == "close_required")
    assert not score_case(case, "<tool_call>").success
    assert score_case(case, "</tool_call>\n").success


def test_escape_control_does_not_offer_a_closing_only_repair(tokenizer):
    case = next(
        case
        for case in task_cases("example_1234", [], Split.TRAIN, "source", tokenizer)
        if case.control == "escape_continue_required"
    )
    assert case.prefix.endswith("\\")
    assert syntax_state(case.prefix, case.family).closing_suffix is None
    assert not score_case(case, '"').success
    assert score_case(case, case.expected[len(case.prefix) :]).success


@pytest.mark.parametrize("family", [Family.OBJECT, Family.ARRAY, Family.NESTED, Family.TOOL])
def test_missing_value_controls_require_content_before_any_closing_suffix(family, tokenizer):
    case = next(
        case
        for case in task_cases("example_1234", [], Split.VALIDATION, "source", tokenizer)
        if case.family == family and case.control == "value_continue_required"
    )
    assert syntax_state(case.prefix, family).closing_suffix is None
    for suffix in ("}", "]", "]}", '"}', "</tool_call>"):
        assert not score_case(case, suffix).syntax_complete
    assert score_case(case, case.expected[len(case.prefix) :]).success


@pytest.mark.parametrize("task_id", ["case_1", "case_2", "case_3", "case_4"])
def test_varied_fences_require_the_opening_marker_length_and_character(task_id, tokenizer):
    case = next(
        case
        for case in task_cases(task_id, [], Split.VALIDATION, "source", tokenizer)
        if case.family == Family.FENCE and case.control == "close_required"
    )
    marker = case.expected.rsplit("\n", 1)[1]
    assert score_case(case, marker).success
    assert not score_case(case, marker[:-1]).syntax_complete
    wrong_character = "~" if marker[0] == "`" else "`"
    assert not score_case(case, wrong_character * len(marker)).syntax_complete


@pytest.mark.parametrize("task_id,ending", [("case_4", '\\"'), ("case_0", "\u263a"), ("case_1", "\n\t")])
def test_escaped_string_endings_are_complete_content_before_the_final_quote(task_id, ending, tokenizer):
    case = next(
        case
        for case in task_cases(task_id, [], Split.VALIDATION, "source", tokenizer)
        if case.family == Family.STRING and case.control == "close_required"
    )
    assert json.loads(case.prefix + '"').endswith(ending)
    assert score_case(case, '"').success
    assert syntax_state(case.prefix, case.family).closing_suffix == '"'
