# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-task persistence for the Table 9 BPB evaluator.

A preempted eval attempt restarts from zero; these tests pin the contract that
each task's BPB is durable the moment it is scored, and that a later attempt
scores only the tasks that are still missing.
"""

import json
import os

import pytest
from marin.evaluation.olmo_base_eval.run import TASK_PROGRESS_FILENAME, score_tasks_resumable

TASKS = ["arc_easy", "hellaswag", "mmlu_anatomy", "piqa"]
BPB = {"arc_easy": 0.81, "hellaswag": 0.66, "mmlu_anatomy": 1.02, "piqa": 0.74}


class RecordingScorer:
    """Stands in for the model forward: returns fixed BPB, remembers what it was asked to score."""

    def __init__(self, fail_after: int | None = None):
        self.scored: list[str] = []
        self.fail_after = fail_after

    def __call__(self, task: str) -> float:
        if self.fail_after is not None and len(self.scored) >= self.fail_after:
            raise RuntimeError("preempted")
        self.scored.append(task)
        return BPB[task]


def test_first_attempt_scores_every_task_and_persists_each(tmp_path):
    scorer = RecordingScorer()

    scores = score_tasks_resumable(TASKS, scorer, output_path=str(tmp_path))

    assert scores == BPB
    assert scorer.scored == TASKS
    with open(os.path.join(tmp_path, TASK_PROGRESS_FILENAME)) as handle:
        assert json.load(handle) == BPB


def test_attempt_that_dies_midway_leaves_completed_tasks_on_disk(tmp_path):
    scorer = RecordingScorer(fail_after=2)

    with pytest.raises(RuntimeError, match="preempted"):
        score_tasks_resumable(TASKS, scorer, output_path=str(tmp_path))

    with open(os.path.join(tmp_path, TASK_PROGRESS_FILENAME)) as handle:
        assert json.load(handle) == {"arc_easy": 0.81, "hellaswag": 0.66}


def test_next_attempt_scores_only_the_missing_tasks(tmp_path):
    with pytest.raises(RuntimeError):
        score_tasks_resumable(TASKS, RecordingScorer(fail_after=2), output_path=str(tmp_path))

    second = RecordingScorer()
    scores = score_tasks_resumable(TASKS, second, output_path=str(tmp_path))

    assert second.scored == ["mmlu_anatomy", "piqa"]
    assert scores == BPB


def test_completed_progress_file_means_no_task_is_rescored(tmp_path):
    score_tasks_resumable(TASKS, RecordingScorer(), output_path=str(tmp_path))

    rerun = RecordingScorer()
    scores = score_tasks_resumable(TASKS, rerun, output_path=str(tmp_path))

    assert rerun.scored == []
    assert scores == BPB
