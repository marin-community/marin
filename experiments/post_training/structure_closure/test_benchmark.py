# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from dataclasses import asdict

import pytest

from experiments.post_training.structure_closure.benchmark import Family, score_case, syntax_state, task_cases
from experiments.post_training.structure_closure.training_prefixes import prepare_training_prefixes


def test_reference_continuations_cover_each_family_and_preserve_content():
    cases = task_cases("example_1234", [{"search": {"query": 'quoted " slash \\ </tool_call>'}}], "validation", "source")
    assert {case.family for case in cases} == set(Family)
    assert len({case.case_id for case in cases}) == len(cases)
    for case in cases:
        assert score_case(case, case.expected[len(case.prefix) :]).success
        assert not score_case(case, "").success


def test_training_prefix_variant_preserves_validation_and_non_target_rows(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    cases = task_cases("example_1234", [], "train", "source")
    original = [json.dumps(asdict(case)) + "\n" for case in cases]
    (source / "train.jsonl").write_text("".join(original))
    validation = b' {"frozen": "validation bytes"}\r\n'
    (source / "validation.jsonl").write_bytes(validation)
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "hashes": {
                    split: hashlib.sha256((source / f"{split}.jsonl").read_bytes()).hexdigest()
                    for split in ("train", "validation")
                }
            }
        )
    )
    output = tmp_path / "variant"
    prepare_training_prefixes(source, output)
    assert (output / "validation.jsonl").read_bytes() == validation
    rows = (output / "train.jsonl").read_text().splitlines(keepends=True)
    assert len(rows) == len(cases)
    for case, old, new in zip(cases, original, rows, strict=True):
        if case.control != "continue_required":
            assert new == old
            continue
        changed = json.loads(new)
        assert {k: v for k, v in changed.items() if k != "prefix"} == {
            k: v for k, v in asdict(case).items() if k != "prefix"
        }
        assert changed["prefix"].startswith(case.prefix)
        prefix = changed["prefix"]
        suffix = syntax_state(prefix, case.family).closing_suffix
        assert suffix is not None
        assert not score_case(case, prefix[len(case.prefix) :] + suffix).content_preserved
        assert score_case(case, case.expected[len(case.prefix) :]).success
    array = next(
        row for row in map(json.loads, rows) if row["family"] == Family.ARRAY and row["control"] == "continue_required"
    )
    assert array["prefix"] == '["example_123'


def test_training_prefix_conversion_reports_all_failures_without_publishing_partial_data(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    case = next(
        case for case in task_cases("example_1234", [], "train", "source") if case.control == "continue_required"
    )
    rows = [asdict(case) | {"prefix": "unrelated"}, asdict(case) | {"split": "validation"}, asdict(case)]
    (source / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (source / "validation.jsonl").write_bytes(b"frozen")
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "hashes": {
                    split: hashlib.sha256((source / f"{split}.jsonl").read_bytes()).hexdigest()
                    for split in ("train", "validation")
                }
            }
        )
    )
    output = tmp_path / "variant"
    with pytest.raises(ValueError, match="2 conversion failures"):
        prepare_training_prefixes(source, output)
    failures = json.loads((output / "conversion-failures.json").read_text())
    assert [failure["line"] for failure in failures] == [1, 2]
    assert not (output / "train.jsonl").exists()
    assert not (output / "manifest.json").exists()


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


@pytest.mark.parametrize("family", [Family.OBJECT, Family.ARRAY, Family.NESTED, Family.TOOL])
def test_missing_value_controls_require_content_before_any_closing_suffix(family):
    case = next(
        case
        for case in task_cases("example_1234", [], "validation", "source")
        if case.family == family and case.control == "value_continue_required"
    )
    assert syntax_state(case.prefix, family).closing_suffix is None
    for suffix in ("}", "]", "]}", '"}', "</tool_call>"):
        assert not score_case(case, suffix).syntax_complete
    assert score_case(case, case.expected[len(case.prefix) :]).success


@pytest.mark.parametrize("task_id", ["case_1", "case_2", "case_3", "case_4"])
def test_varied_fences_require_the_opening_marker_length_and_character(task_id):
    case = next(
        case
        for case in task_cases(task_id, [], "validation", "source")
        if case.family == Family.FENCE and case.control == "close_required"
    )
    marker = case.expected.rsplit("\n", 1)[1]
    assert score_case(case, marker).success
    assert not score_case(case, marker[:-1]).syntax_complete
    wrong_character = "~" if marker[0] == "`" else "`"
    assert not score_case(case, wrong_character * len(marker)).syntax_complete


@pytest.mark.parametrize("task_id,ending", [("case_4", '\\"'), ("case_0", "\u263a"), ("case_1", "\n\t")])
def test_escaped_string_endings_are_complete_content_before_the_final_quote(task_id, ending):
    case = next(
        case
        for case in task_cases(task_id, [], "validation", "source")
        if case.family == Family.STRING and case.control == "close_required"
    )
    assert json.loads(case.prefix + '"').endswith(ending)
    assert score_case(case, '"').success
    assert syntax_state(case.prefix, case.family).closing_suffix == '"'
