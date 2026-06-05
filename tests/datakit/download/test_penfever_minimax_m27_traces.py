# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for penfever MiniMax-M2.7 @ 131k trace rendering."""

import hashlib

import pytest
from marin.datakit.download.penfever_minimax_m27_traces import (
    _COHORT_ENTRIES,
    TEACHER,
    _outcome_tag,
    _row_to_doc,
    penfever_minimax_m27_traces_normalize_steps,
)


_SLUG = "inferredbugs-sandboxes-verifier"
_HF_REPO = f"penfever/{_SLUG}-minimax-m27-131k-traces"


def _valid_row(**overrides) -> dict:
    row = {
        "conversations": [
            {"role": "user", "content": "Fix the bug in foo().",},
            {"role": "assistant", "content": "Sure — let me read the file."},
            {"role": "tool", "content": "stdout: tests pass"},
            {"role": "assistant", "content": "Done."},
        ],
        "result": "1.0",
        "verifier_output": "VERIFIER: PASS",
    }
    row.update(overrides)
    return row


def test_outcome_tag_prefers_verifier_pass():
    assert _outcome_tag("VERIFIER: PASS", "0.0") == "This trajectory solved the task successfully."


def test_outcome_tag_prefers_verifier_fail():
    assert _outcome_tag("VERIFIER: FAIL", "1.0") == "This trajectory failed to solve the task."


def test_outcome_tag_falls_back_to_numeric_result():
    assert _outcome_tag(None, "1.0") == "This trajectory solved the task successfully."
    assert _outcome_tag("", "0.0") == "This trajectory failed to solve the task."


def test_outcome_tag_preserves_error_state():
    tag = _outcome_tag(None, "AgentTimeoutError")
    assert tag == "This trajectory failed to solve the task (AgentTimeoutError)."


def test_outcome_tag_none_when_no_signal():
    assert _outcome_tag(None, None) is None
    assert _outcome_tag("", "") is None


def test_row_to_doc_renders_trajectory_with_outcome_header():
    expected_text = (
        "This trajectory solved the task successfully.\n\n"
        "<user>\nFix the bug in foo().\n</user>\n\n"
        "<assistant>\nSure — let me read the file.\n</assistant>\n\n"
        "<tool>\nstdout: tests pass\n</tool>\n\n"
        "<assistant>\nDone.\n</assistant>"
    )
    [doc] = _row_to_doc(_SLUG)(_valid_row())
    assert doc == {
        "id": hashlib.sha256(expected_text.encode("utf-8")).hexdigest(),
        "text": expected_text,
        "source": _HF_REPO,
    }


def test_row_to_doc_renders_without_header_when_no_signal():
    row = _valid_row(verifier_output=None, result=None)
    [doc] = _row_to_doc(_SLUG)(row)
    # No leading outcome tag — text starts directly at the first role.
    assert doc["text"].startswith("<user>\nFix the bug in foo().\n</user>")


def test_row_to_doc_failure_tag_carries_error_name():
    row = _valid_row(verifier_output=None, result="AgentTimeoutError")
    [doc] = _row_to_doc(_SLUG)(row)
    assert doc["text"].startswith("This trajectory failed to solve the task (AgentTimeoutError).\n\n")


@pytest.mark.parametrize(
    "row",
    [
        {"conversations": None},
        {"conversations": []},
    ],
    ids=["missing-conversations", "empty-conversations"],
)
def test_row_to_doc_drops_rows_without_conversations(row):
    assert _row_to_doc(_SLUG)(row) == []


def test_normalize_steps_keyed_by_marin_name_for_every_cohort():
    chains = penfever_minimax_m27_traces_normalize_steps()
    expected = {f"penfever-traces/{TEACHER}/{slug}" for slug, _, _ in _COHORT_ENTRIES}
    assert set(chains) == expected
    # Each chain is (processed, normalize); processed pulls the raw download
    # in through its deps so the StepRunner can materialize the full DAG.
    for marin_name, chain in chains.items():
        assert len(chain) == 2
        processed, normalize = chain
        assert processed.name == f"processed/{marin_name}"
        assert normalize.name == f"normalized/{marin_name}"
        assert len(processed.deps) == 1
        assert processed.deps[0].name == f"raw/{marin_name}"
