# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import json
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest

from experiments.post_training.bfcl_rl.data import BFCLPartition, DATASET_COMMIT, TaskIdentity
from experiments.post_training.bfcl_rl.preferences import PairDisposition, select_pair
from experiments.post_training.bfcl_rl.retained_preferences import (
    CollectionIdentity,
    TokenStep,
    causal_token_sequence,
    pretokenized_preference,
    read_retained_archives,
    retained_rollout,
)

TASK = TaskIdentity("bfcl-simple-python-13", "simple_python_13", "audited-task-digest")
HOLDOUT = TaskIdentity("bfcl-simple-python-12", "simple_python_12", "holdout-task-digest")
PARTITION = BFCLPartition(DATASET_COMMIT, (TASK,), (HOLDOUT,))


def _record(model: str, score: float, *, task: TaskIdentity = TASK) -> dict:
    completion = 10 if model == "teacher" else 30
    return {
        "schema_version": 6,
        "record_id": f"{model}-{task.name}",
        "run_id": f"{model}-collection",
        "trajectory": {
            "instance_id": task.name,
            "repetition_id": 0,
            "environment_extras": {"data_source": f"/staged/complement/{task.name}"},
        },
        "provenance": {"model_source_identity": f"{model}@pinned"},
        "verification_result": {"status": "verified", "score": score, "passed": None, "score_min": 0, "score_max": 1},
        "reward": {"outcome": score, "shaped": -0.25},
        "disposition": {"server_error": None, "exception_type": None, "error_treatment": None},
        "prompt": {"token_ids": [1, 2]},
        "response": {
            "token_ids": [completion, completion + 1, 20, 21],
            "loss_mask": [1, 1, 0, 1],
            "step_boundaries": [
                {"prompt_token_ids": [1, 2], "token_start": 0, "token_end": 2},
                {"prompt_token_ids": [1, 2, completion, completion + 1, 99], "token_start": 2, "token_end": 4},
            ],
        },
    }


def _identity(model: str) -> CollectionIdentity:
    return CollectionIdentity(f"{model}-collection", f"{model}@pinned", f"{model}-revision", "pi@0.87.0", DATASET_COMMIT)


def test_retained_archives_produce_exact_preferences_with_tool_context_masked(tmp_path: Path):
    records = []
    for model, score in (("teacher", 1.0), ("student", 0.0)):
        path = tmp_path / f"{model}.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("records/record.json.gz", gzip.compress(json.dumps(_record(model, score)).encode()))
        records.append(read_retained_archives([path], identity=_identity(model), partition=PARTITION)[0])
    teacher, student = records
    pair = select_pair(teacher.rollout, student.rollout).pair
    assert pair is not None and pair.chosen.model_revision == "teacher-revision"
    row = pretokenized_preference(teacher, student, max_length=16)
    assert row == {
        "chosen_input_ids": [1, 2, 10, 11, 99, 20, 21],
        "chosen_assistant_masks": [0, 0, 1, 1, 0, 0, 1],
        "rejected_input_ids": [1, 2, 30, 31, 99, 20, 21],
        "rejected_assistant_masks": [0, 0, 1, 1, 0, 0, 1],
    }
    assert pair.chosen.task_digest == TASK.digest


def test_retained_verdict_overrides_shaping_and_discards_infrastructure_failures():
    teacher_record = _record("teacher", 1.0)
    student_record = _record("student", 0.0)
    teacher = retained_rollout(teacher_record, identity=_identity("teacher"), partition=PARTITION, trajectory_uri="teacher")
    student_record["disposition"]["server_error"] = {"status_code": 503}
    student = retained_rollout(student_record, identity=_identity("student"), partition=PARTITION, trajectory_uri="student")
    assert select_pair(teacher.rollout, student.rollout).disposition == PairDisposition.UNSCORED
    teacher_record["reward"]["outcome"] = 0.0
    with pytest.raises(ValueError, match="differs from the BFCL verifier"):
        retained_rollout(teacher_record, identity=_identity("teacher"), partition=PARTITION, trajectory_uri="teacher")


def test_retained_holdout_and_changed_model_cannot_form_training_preferences():
    with pytest.raises(ValueError, match="outside the BFCL training complement"):
        retained_rollout(
            _record("teacher", 1.0, task=HOLDOUT),
            identity=_identity("teacher"),
            partition=PARTITION,
            trajectory_uri="holdout",
        )
    with pytest.raises(ValueError, match="different model source"):
        retained_rollout(
            _record("teacher", 1.0),
            identity=replace(_identity("teacher"), model_source_identity="unrelated@model"),
            partition=PARTITION,
            trajectory_uri="teacher",
        )


def test_context_forks_and_overlong_preferences_cannot_be_silently_rewritten():
    steps = (TokenStep((1, 2), (10, 11), (1, 1)), TokenStep((99, 98), (12,), (1,)))
    with pytest.raises(ValueError, match="context fork"):
        causal_token_sequence(steps, max_length=16)
    with pytest.raises(ValueError, match="truncation would change the rollout"):
        causal_token_sequence(steps[:1], max_length=3)
    teacher = retained_rollout(_record("teacher", 1.0), identity=_identity("teacher"), partition=PARTITION, trajectory_uri="t")
    student = retained_rollout(_record("student", 0.0), identity=_identity("student"), partition=PARTITION, trajectory_uri="s")
    changed_prompt = replace(student, steps=(TokenStep((3, 4), (20,), (1,)),))
    with pytest.raises(ValueError, match="exact initial prompt"):
        pretokenized_preference(teacher, changed_prompt, max_length=16)
