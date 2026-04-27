# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reference-parity test for the vendored IFBench scorer.

Runs `experiments.ifbench.verifiers.scoring.score_jsonl` against the upstream
`sample_output.jsonl` + `IFBench_test.jsonl` fixtures and asserts the result
matches the canonical numbers we recorded from upstream's own run_eval.py.

This is the hard gate from the verifier vendor plan: if these numbers
disagree, our port is broken. Re-run after every `tools/refresh_ifbench_verifiers.sh`.

Reference numbers were captured 2026-04-26 against:
- IFBench@cb932e352a505306ad0115272211df14bb8f628f
- The exact run_eval.py shipped at that SHA (with one local patch:
  rstrip-fallback on prompt lookup, since upstream sample_output.jsonl rows
  have trailing whitespace mismatches with IFBench_test.jsonl that crash
  upstream's untouched dict-key access).
"""

from __future__ import annotations

import math
import pathlib

import pytest

from experiments.ifbench.verifiers.scoring import score_jsonl
from experiments.ifbench.verifiers import IFBENCH_DICT, IFEVALG_DICT, INSTRUCTION_DICT_ALL

_FIXTURES = pathlib.Path(__file__).parent / "fixtures"


REFERENCE = {
    "strict": {
        "prompt_level_accuracy": 0.25333333333333335,
        "instruction_level_accuracy": 0.26744186046511625,
        "n_prompts": 300,
        "n_instructions": 344,
    },
    "loose": {
        "prompt_level_accuracy": 0.2866666666666667,
        "instruction_level_accuracy": 0.311046511627907,
        "n_prompts": 300,
        "n_instructions": 344,
    },
}


def test_registry_sizes() -> None:
    """Constraint coverage matches what the AI2 paper describes."""
    assert len(IFBENCH_DICT) == 58, "IFBench should have 58 OOD constraints"
    assert len(IFEVALG_DICT) == 54, "IFEvalG should have 25 IFEval + 29 IFTrain = 54"
    assert len(INSTRUCTION_DICT_ALL) == 112
    assert set(IFBENCH_DICT).isdisjoint(set(IFEVALG_DICT))


def test_ifbench_reference_parity() -> None:
    """Score IFBench's own sample_output.jsonl; numbers must match upstream exactly."""
    ours = score_jsonl(
        str(_FIXTURES / "IFBench_test.jsonl"),
        str(_FIXTURES / "sample_output.jsonl"),
    )
    for label, ref in REFERENCE.items():
        for key, ref_val in ref.items():
            actual = ours[label][key]
            assert math.isclose(actual, ref_val, abs_tol=1e-9), f"{label}.{key}: ours={actual} upstream={ref_val}"


@pytest.mark.parametrize(
    "instruction_id, kwargs, response, expected",
    [
        # IFEvalG-side spot checks (validates that the IFEvalG branch of the
        # registry actually loads + classifies; we don't have a full upstream
        # parity check for IFEvalG, so these are hand-crafted cases.)
        ("change_case:english_lowercase", {}, "the quick brown fox jumps over the lazy dog", True),
        ("change_case:english_lowercase", {}, "Mixed Case English Text Has Capitals", False),
        ("change_case:english_capital", {}, "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG", True),
        ("change_case:english_capital", {}, "mixed case english text", False),
        ("punctuation:no_comma", {}, "no commas here", True),
        ("punctuation:no_comma", {}, "yes, comma here", False),
        ("startend:end_checker", {"end_phrase": "the end."}, "story ends here. the end.", True),
        ("startend:end_checker", {"end_phrase": "the end."}, "story ends here.", False),
        # IFBench-side spot checks (no-arg constraint)
        ("format:no_whitespace", {}, "nowhitespace", True),
        ("format:no_whitespace", {}, "has whitespace", False),
    ],
)
def test_constraint_classification(instruction_id, kwargs, response, expected):
    """Each (instruction, response) classifies as expected when called manually."""
    cls = INSTRUCTION_DICT_ALL[instruction_id]
    instruction = cls(instruction_id)
    instruction.build_description(**kwargs)
    assert instruction.check_following(response) is expected, f"{instruction_id} on {response!r} expected {expected}"


def test_parse_ground_truth_roundtrip() -> None:
    """The parser handles the Python repr-style training-set ground_truth."""
    from experiments.ifbench.verifiers.parse import parse_ground_truth

    raw = (
        "[{'instruction_id': ['detectable_format:sentence_hyphens', "
        "'last_word:last_word_answer'], "
        "'kwargs': [None, {'last_word': 'brief'}]}]"
    )
    parsed = parse_ground_truth(raw)
    assert parsed.instruction_id_list == ["detectable_format:sentence_hyphens", "last_word:last_word_answer"]
    assert parsed.kwargs == [None, {"last_word": "brief"}]


def test_parse_ground_truth_rejects_malformed() -> None:
    from experiments.ifbench.verifiers.parse import parse_ground_truth

    with pytest.raises(ValueError):
        parse_ground_truth("")
    with pytest.raises(ValueError):
        parse_ground_truth("not python")
    with pytest.raises(ValueError):
        parse_ground_truth("[{'kwargs': []}]")  # missing instruction_id
