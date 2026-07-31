# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for penfever MiniMax-M2.7 @ 131k trace rendering."""

import hashlib

import pytest
from marin.datakit.download.penfever_minimax_m27_traces import (
    TEACHER,
    outcome_tag,
    row_to_doc,
)
from marin.datakit.download.rollout_transforms import TRAJECTORY_UNVERIFIED_TAG

_SLUG = "inferredbugs-sandboxes-verifier"
_HF_REPO = f"penfever/{_SLUG}-minimax-m27-131k-traces"


def _valid_row(**overrides) -> dict:
    row = {
        "conversations": [
            {
                "role": "user",
                "content": "Fix the bug in foo().",
            },
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
    assert outcome_tag("VERIFIER: PASS", "0.0") == "This trajectory solved the task successfully."


def test_outcome_tag_prefers_verifier_fail():
    assert outcome_tag("VERIFIER: FAIL", "1.0") == "This trajectory failed to solve the task."


def test_outcome_tag_falls_back_to_numeric_result():
    assert outcome_tag(None, "1.0") == "This trajectory solved the task successfully."
    assert outcome_tag("", "0.0") == "This trajectory failed to solve the task."


def test_outcome_tag_treats_harness_error_as_unverified():
    assert outcome_tag(None, "AgentTimeoutError") == TRAJECTORY_UNVERIFIED_TAG


def test_outcome_tag_none_when_no_signal():
    assert outcome_tag(None, None) is None
    assert outcome_tag("", "") is None


def test_row_to_doc_renders_trajectory_with_outcome_header():
    expected_text = (
        "This trajectory solved the task successfully.\n\n"
        "<user>\nFix the bug in foo().\n</user>\n\n"
        "<assistant>\nSure — let me read the file.\n</assistant>\n\n"
        "<tool>\nstdout: tests pass\n</tool>\n\n"
        "<assistant>\nDone.\n</assistant>"
    )
    [doc] = row_to_doc(_SLUG)(_valid_row())
    assert doc == {
        "id": hashlib.sha256(expected_text.encode("utf-8")).hexdigest(),
        "text": expected_text,
        "source": _HF_REPO,
        "teacher": TEACHER,
        "task_source": _SLUG,
    }


def test_row_to_doc_renders_without_header_when_no_signal():
    row = _valid_row(verifier_output=None, result=None)
    [doc] = row_to_doc(_SLUG)(row)
    # No leading outcome tag — text starts directly at the first role.
    assert doc["text"].startswith("<user>\nFix the bug in foo().\n</user>")


def test_row_to_doc_harness_error_uses_unverified_tag():
    row = _valid_row(verifier_output=None, result="AgentTimeoutError")
    [doc] = row_to_doc(_SLUG)(row)
    assert doc["text"].startswith(f"{TRAJECTORY_UNVERIFIED_TAG}\n\n")


@pytest.mark.parametrize(
    "row",
    [
        {"conversations": None},
        {"conversations": []},
    ],
    ids=["missing-conversations", "empty-conversations"],
)
def test_row_to_doc_drops_rows_without_conversations(row):
    assert row_to_doc(_SLUG)(row) == []
