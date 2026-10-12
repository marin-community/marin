# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.post_training.structure_closure.benchmark import Family, score_case, syntax_state, task_cases


def test_reference_continuations_cover_each_family_and_preserve_content():
    cases = task_cases("example_1234", [{"search": {"query": 'quoted " slash \\ </tool_call>'}}], "validation", "source")
    assert {case.family for case in cases} == set(Family)
    assert len({case.case_id for case in cases}) == len(cases)
    for case in cases:
        assert score_case(case, case.expected[len(case.prefix) :]).success
        assert not score_case(case, "").success


def test_close_probes_supply_content_only_in_the_prefix_while_continue_controls_supply_a_reference():
    cases = task_cases("example_1234", [{"search": {"query": "unique payload"}}], "validation", "source")
    for case in cases:
        if case.control == "close_required":
            assert "example_1234" not in case.instruction
            assert case.expected not in case.instruction
            suffix = syntax_state(case.prefix, case.family).closing_suffix
            assert suffix is not None
            assert score_case(case, suffix).success
        else:
            assert case.expected in case.instruction


def test_premature_closure_cannot_pass_continue_controls():
    cases = task_cases("example_1234", [], "validation", "source")
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


def test_tool_reference_is_a_serialization_probe_not_a_required_live_tool():
    cases = task_cases("example_1234", [{"f": {"x": 3}}], "train", "source")
    case = next(case for case in cases if case.family == Family.TOOL and case.control == "close_required")
    assert not score_case(case, "<tool_call>").success
    assert score_case(case, "</tool_call>\n").success


def test_escape_control_does_not_offer_a_closing_only_repair():
    case = next(
        case for case in task_cases("example_1234", [], "train", "source") if case.control == "escape_continue_required"
    )
    assert case.prefix.endswith("\\")
    assert syntax_state(case.prefix, case.family).closing_suffix is None
    assert not score_case(case, '"').success
    assert score_case(case, case.expected[len(case.prefix) :]).success
