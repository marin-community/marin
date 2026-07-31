# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for issue #6191 rollout rendering."""

import hashlib
from collections import Counter

import pytest
from marin.datakit.download.penfever_rollouts import PENFEVER_ROLLOUTS, PenfeverRollout, outcome_tag, row_to_doc
from marin.datakit.download.rollout_transforms import (
    TRAJECTORY_FAILED_TAG,
    TRAJECTORY_SOLVED_TAG,
    TRAJECTORY_UNVERIFIED_TAG,
)

_DATASET = PenfeverRollout(
    cohort_name="minimax-m27-131k",
    teacher="MiniMax-M2.7",
    task_source="inferredbugs-sandboxes-verifier",
    hf_dataset_id="penfever/inferredbugs-sandboxes-verifier-minimax-m27-131k-traces",
    revision="d1e3350",
)


def test_manifest_covers_every_issue_cohort():
    assert Counter(dataset.cohort_name for dataset in PENFEVER_ROLLOUTS) == {
        "minimax-m27-131k": 45,
        "qwen35-122b-32k": 49,
        "qwen35-122b-131k-opencode": 35,
        "glm52-terminus2": 13,
    }
    assert len({dataset.hf_dataset_id for dataset in PENFEVER_ROLLOUTS}) == 142
    assert len({dataset.marin_name for dataset in PENFEVER_ROLLOUTS}) == 142


def _valid_row(**overrides) -> dict:
    row = {
        "conversations": [
            {"role": "user", "content": "Fix the bug in foo()."},
            {"role": "assistant", "content": "Sure — let me read the file."},
            {"role": "tool", "content": "stdout: tests pass"},
            {"role": "assistant", "content": "Done."},
        ],
        "result": "1.0",
        "verifier_output": "VERIFIER: PASS",
    }
    row.update(overrides)
    return row


@pytest.mark.parametrize(
    ("verifier_output", "result", "expected"),
    [
        ("VERIFIER: FAIL", "1.0", TRAJECTORY_SOLVED_TAG),
        ("VERIFIER: PASS", "0.0", TRAJECTORY_FAILED_TAG),
        (None, "AgentTimeoutError", TRAJECTORY_UNVERIFIED_TAG),
        ("VERIFIER: PASS", None, TRAJECTORY_SOLVED_TAG),
        ("VERIFIER: FAIL", None, TRAJECTORY_FAILED_TAG),
        (None, None, None),
        ("", "", None),
    ],
)
def test_outcome_tag_maps_available_result(verifier_output, result, expected):
    assert outcome_tag(verifier_output, result) == expected


def test_row_to_doc_renders_trajectory_with_provenance():
    expected_text = (
        f"{TRAJECTORY_SOLVED_TAG}\n\n"
        "<user>\nFix the bug in foo().\n</user>\n\n"
        "<assistant>\nSure — let me read the file.\n</assistant>\n\n"
        "<tool>\nstdout: tests pass\n</tool>\n\n"
        "<assistant>\nDone.\n</assistant>"
    )
    [doc] = row_to_doc(_DATASET)(_valid_row())
    assert doc == {
        "id": hashlib.sha256(expected_text.encode("utf-8")).hexdigest(),
        "text": expected_text,
        "source": _DATASET.hf_dataset_id,
        "teacher": _DATASET.teacher,
        "task_source": _DATASET.task_source,
    }


@pytest.mark.parametrize("conversations", [None, []])
def test_row_to_doc_drops_rows_without_conversations(conversations):
    assert row_to_doc(_DATASET)(_valid_row(conversations=conversations)) == []
