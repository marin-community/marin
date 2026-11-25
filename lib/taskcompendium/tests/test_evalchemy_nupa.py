# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pytest
from rigging.filesystem.storage_path import StoragePath
from verifyit.grade import Status

from taskcompendium.importers.evalchemy.scorers.main import score
from taskcompendium.importers.evalchemy.scorers.nupa import grade_nupa_candidate
from taskcompendium.importers.evaluation_nupa import read_nupa_panel
from taskcompendium.pipeline.inputs import ConversionContext


@pytest.mark.parametrize(
    "candidate,expected,answer_format,digit_match,length_difference",
    [
        ("23", "123", "Integer", 2 / 3, 1),
        ("2.340", "12.345", "Float", 3 / 5, 1),
        ("12/3", "12/43", "Fraction", 3 / 4, 1),
        ("1.23e5", "1.23e45", "ScientificNotation", 4 / 5, 1),
    ],
)
def test_component_alignment_preserves_partial_digit_credit(
    candidate, expected, answer_format, digit_match, length_difference
):
    verdict = grade_nupa_candidate(expected, answer_format, candidate)
    assert verdict.status is Status.SCORED
    assert verdict.reward == 0
    assert verdict.detail["metrics"] == {
        "exact_match": 0.0,
        "digit_match": digit_match,
        "dlength": float(length_difference),
        "format_valid_rate": 1.0,
        "no_answer_rate": 0.0,
    }


def test_representation_equality_does_not_coerce_numeric_equivalents():
    assert grade_nupa_candidate("1.0", "Float", "1.00").reward == 0
    assert grade_nupa_candidate("1.0", "Float", "1.0").reward == 1


def test_source_component_metric_omits_scientific_exponent_sign():
    verdict = grade_nupa_candidate("1.23e5", "ScientificNotation", "1.23e-5")
    assert verdict.reward == 1
    assert verdict.detail["metrics"]["digit_match"] == 1


@pytest.mark.parametrize("candidate", [None, "missing", {"answer": "123"}])
def test_missing_answer_keeps_reference_length_and_zero_credit(candidate):
    verdict = grade_nupa_candidate("123", "Integer", candidate)
    assert verdict.reward == 0
    assert verdict.detail["metrics"] == {
        "exact_match": 0.0,
        "digit_match": 0.0,
        "dlength": 3.0,
        "format_valid_rate": 0.0,
        "no_answer_rate": 1.0,
    }


@pytest.mark.parametrize("expected", ["", "abc123", "nan", "inf", None])
def test_malformed_reference_cannot_be_reduced_to_matching_digits(expected):
    verdict = grade_nupa_candidate(expected, "Integer", "123")
    assert verdict.status is Status.INVALID_TASK
    assert verdict.reward == 0


@pytest.mark.parametrize(
    ("response", "answer_format", "expected"),
    [
        ("The result is 1046.1046", "Integer", "1046"),
        ("The result is 486.612.486.612", "Float", "486.612"),
        ("The result is 43/20.43/20", "Fraction", "43/20"),
        ("The result is 5.09786549e52.5.09786549e52", "ScientificNotation", "5.09786549e52"),
        ("The answer is $1046$.", "Integer", "1046"),
        ("Answer: **486.612**", "Float", "486.612"),
        ("42\nThis is because 6 times 7 is 42", "Integer", "42"),
        ("<|start_think|>Maybe 9<|end_think|>**1046**", "Integer", "1046"),
    ],
)
def test_packaged_nupa_scorer_accepts_final_representation(response, answer_format, expected):
    verdict = score("nupa", {"expected": expected, "answer_format": answer_format}, response)
    assert verdict.reward == 1
    assert verdict.detail["extracted"] == expected


@pytest.mark.parametrize(
    ("response", "expected", "answer_format"),
    [
        ("The answer is 18\n1818", "18", "Integer"),
        ("486.612486.612", "486.612", "Float"),
        ("The final answer is 123.42", "42", "Integer"),
        ("The final answer is 11046.1046", "1046", "Integer"),
        ("$1046$ but the answer is 9", "1046", "Integer"),
        ("<|start_think|>The answer is **1046**", "1046", "Integer"),
    ],
)
def test_packaged_nupa_scorer_rejects_repeats_fragments_and_unfinished_thinking(response, expected, answer_format):
    verdict = score("nupa", {"expected": expected, "answer_format": answer_format}, response)
    assert verdict.reward == 0


SOURCE = {
    "add_Integer_Integer_Integer": {"1": ["Add: 1 + 2 = 3", "Add: 1 + 2 = 3"]},
    "sub_Integer_Integer_Integer": {"1": ["Subtract: 3 - 1 = 2"]},
}
PANEL = [
    {
        "task_name": "sub_Integer_Integer_Integer",
        "digit": 1,
        "sha256": hashlib.sha256(b"Subtract: 3 - 1 = 2").hexdigest(),
    },
    {"task_name": "add_Integer_Integer_Integer", "digit": 1, "sha256": hashlib.sha256(b"Add: 1 + 2 = 3").hexdigest()},
]


@pytest.fixture
def staged_source(tmp_path):
    primary = tmp_path / "source"
    primary.mkdir()
    (primary / "test.json").write_text(json.dumps(SOURCE))
    panel = tmp_path / "panel"
    panel.mkdir()
    (panel / "nupa5k_manifest.jsonl").write_text("".join(json.dumps(identity) + "\n" for identity in PANEL))
    return primary, panel


def test_nupa_panel_resolves_manifest_order_and_rejects_missing_source_identity(staged_source):
    primary, panel = staged_source
    path = primary / "test.json"
    context = ConversionContext({"panel": StoragePath(str(panel))}, None)
    records = list(read_nupa_panel(StoragePath(str(path)), context))
    assert [
        {"task_name": row["task_name"], "digit": row["digit"], "sha256": row["source_sha256"]} for row in records
    ] == PANEL
    data = json.loads(path.read_text())
    data.clear()
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="missing"):
        list(read_nupa_panel(StoragePath(str(path)), context))


def test_nupa_panel_order_does_not_depend_on_source_task_order(staged_source):
    primary, panel = staged_source
    path = primary / "test.json"
    context = ConversionContext({"panel": StoragePath(str(panel))}, None)
    before = list(read_nupa_panel(StoragePath(str(path)), context))
    path.write_text(json.dumps(dict(reversed(list(SOURCE.items())))))
    after = list(read_nupa_panel(StoragePath(str(path)), context))
    assert after == before
