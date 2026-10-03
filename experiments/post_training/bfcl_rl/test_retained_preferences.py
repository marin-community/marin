# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import json
import zipfile
from dataclasses import replace
from pathlib import Path

import haliax as hax
import numpy as np
import pytest
from levanter.data.text.preference import PreferencePairDataset
from levanter.store.cache import TreeCache

from experiments.post_training.bfcl_rl.collect import DATA_URI, MODELS
from experiments.post_training.bfcl_rl.data import DATASET_COMMIT, BFCLPartition, TaskIdentity
from experiments.post_training.bfcl_rl.preferences import PairDisposition, select_pair
from experiments.post_training.bfcl_rl.recovery_data import (
    collection_receipt,
    recovery_preference_rows,
    write_recovery_cache,
)
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
        "phase": "eval",
        "global_step": 0,
        "trajectory": {
            "instance_id": task.name,
            "repetition_id": 0,
            "environment_extras": {"data_source": f"/staged/bfcl_complement/{task.name}"},
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
    return CollectionIdentity(
        f"{model}-collection",
        f"{model}@pinned",
        f"{model}-revision",
        "pi@0.87.0",
        DATASET_COMMIT,
        "/staged/bfcl_complement",
    )


def test_retained_archives_produce_exact_preferences_with_tool_context_masked(tmp_path: Path):
    records = []
    for model, score in (("teacher", 1.0), ("student", 0.0)):
        path = tmp_path / f"{model}.zip"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("records/record.json.gz", gzip.compress(json.dumps(_record(model, score)).encode()))
        records.append(read_retained_archives([str(path)], identity=_identity(model), partition=PARTITION)[0])
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
    teacher = retained_rollout(
        teacher_record, identity=_identity("teacher"), partition=PARTITION, trajectory_uri="teacher"
    )
    student_record["disposition"]["server_error"] = {"status_code": 503}
    student = retained_rollout(
        student_record, identity=_identity("student"), partition=PARTITION, trajectory_uri="student"
    )
    assert select_pair(teacher.rollout, student.rollout).disposition == PairDisposition.UNSCORED
    teacher_record["reward"]["outcome"] = 0.0
    with pytest.raises(ValueError, match="differs from the BFCL verifier"):
        retained_rollout(teacher_record, identity=_identity("teacher"), partition=PARTITION, trajectory_uri="teacher")


@pytest.mark.parametrize("score", [0.0, 1.0])
def test_zero_treated_agent_exit_is_excluded_from_verified_preferences(score):
    teacher_record = _record("teacher", score)
    teacher_record["reward"]["outcome"] = 0.0
    teacher_record["disposition"].update(error_treatment="zero", exception_type="NonZeroAgentExitCodeError")
    teacher = retained_rollout(
        teacher_record, identity=_identity("teacher"), partition=PARTITION, trajectory_uri="teacher"
    )
    student = retained_rollout(
        _record("student", 1.0 - score), identity=_identity("student"), partition=PARTITION, trajectory_uri="student"
    )
    selection = select_pair(teacher.rollout, student.rollout)
    assert selection.disposition == PairDisposition.UNSCORED
    assert selection.pair is None


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


def test_setup_failure_without_model_tokens_is_unscored_and_cannot_form_a_preference():
    record = _record("student", 0.0)
    record["verification_result"] = {"status": "unavailable", "reason": "NonZeroAgentExitCodeError"}
    record["disposition"]["exception_type"] = "NonZeroAgentExitCodeError"
    record["prompt"]["token_ids"] = []
    record["response"] = {"token_ids": [], "loss_mask": [], "step_boundaries": []}
    student = retained_rollout(record, identity=_identity("student"), partition=PARTITION, trajectory_uri="setup-failed")
    teacher = retained_rollout(
        _record("teacher", 1.0), identity=_identity("teacher"), partition=PARTITION, trajectory_uri="teacher"
    )
    selection = select_pair(teacher.rollout, student.rollout)
    assert selection.disposition == PairDisposition.UNSCORED
    assert selection.pair is None
    record["verification_result"] = _record("student", 0.0)["verification_result"]
    with pytest.raises(ValueError, match="verified rollout requires exact model-token evidence"):
        retained_rollout(record, identity=_identity("student"), partition=PARTITION, trajectory_uri="invalid-verified")


def test_context_forks_and_overlong_preferences_cannot_be_silently_rewritten():
    steps = (TokenStep((1, 2), (10, 11), (1, 1)), TokenStep((99, 98), (12,), (1,)))
    with pytest.raises(ValueError, match="context fork"):
        causal_token_sequence(steps, max_length=16)
    with pytest.raises(ValueError, match="truncation would change the rollout"):
        causal_token_sequence(steps[:1], max_length=3)
    teacher = retained_rollout(
        _record("teacher", 1.0), identity=_identity("teacher"), partition=PARTITION, trajectory_uri="t"
    )
    student = retained_rollout(
        _record("student", 0.0), identity=_identity("student"), partition=PARTITION, trajectory_uri="s"
    )
    changed_prompt = replace(student, steps=(TokenStep((3, 4), (20,), (1,)),))
    with pytest.raises(ValueError, match="exact initial prompt"):
        pretokenized_preference(teacher, changed_prompt, max_length=16)


def _receipts(model: str) -> tuple[dict, dict]:
    expected = MODELS[model]
    source = {
        "uri": DATA_URI,
        "identity": "complement@pinned",
        "relative_path": f"bfcl_complement/{TASK.name}",
        "local_path": "/staged",
    }
    terminal = {
        "result": {
            "state": "succeeded",
            "run_id": f"logical-{model}",
            "attempt_id": "attempt",
            "iris_job_id": f"iris-{model}",
        },
        "config": {
            "run": {"id": f"logical-{model}", "attempt_id": "attempt"},
            "runtime": {"entrypoint": "skyrl_train.entrypoints.terminal_bench_generate"},
            "ingress": {"record_literal": True},
            "inputs": {
                "model": {
                    "uri": expected.uri,
                    "identity": f"{model}@pinned",
                    "tokenizer_uri": expected.model,
                    "tokenizer_revision": expected.revision,
                },
                "train_data": [source],
                "validation_data": [],
            },
        },
    }
    resolved = {
        "train_data_sources": [dict(source)],
        "val_data_sources": [],
        "config": {
            "trainer": {
                "policy": {"model": {"source_uri": expected.uri, "source_identity": f"{model}@pinned"}},
                "algorithm": {"tito_full": True},
            },
            "generator": {
                "n_samples_per_prompt": 1,
                "max_input_length": 32768,
                "sampling_params": {"temperature": 1.0},
                "trajectory_retention": {
                    "enabled": True,
                    "required": True,
                    "sample_fraction": 1.0,
                    "phases": ["eval"],
                    "run_id": f"{model}-collection",
                    "output_path": f"/{model}/trajectories",
                },
            },
            "terminal_bench_config": {
                "model_info": {"max_input_tokens": 32768, "max_output_tokens": 8192},
                "harbor": {
                    "name": "pi",
                    "version": "0.87.0",
                    "thinking_format": "chat-template",
                    "container_profile": "gvisor",
                    "import_path": "marinskyrl.iris_harbor_environment:IrisEnvironment",
                },
            },
        },
    }
    resolved["config"] = {"skyrl": resolved["config"]}
    return terminal, resolved


def test_recovery_cache_roundtrip_preserves_causal_scoring_with_tool_context(tmp_path: Path):
    receipts = [
        collection_receipt(*_receipts(model), model=model, partition=PARTITION) for model in ("teacher", "student")
    ]
    teacher, student = [
        retained_rollout(_record(model, score), identity=receipt.identity, partition=PARTITION, trajectory_uri=model)
        for model, score, receipt in zip(("teacher", "student"), (1.0, 0.0), receipts, strict=True)
    ]
    rows, report = recovery_preference_rows(
        [teacher],
        [student],
        teacher_receipt=receipts[0],
        student_receipt=receipts[1],
        partition=PARTITION,
        max_length=16,
    )
    write_recovery_cache(rows, report, str(tmp_path / "cache"))
    exemplar = {key: np.zeros((0,), dtype=np.int32) for key in rows[0]}
    cache = TreeCache.load(str(tmp_path / "cache" / "train"), exemplar)
    example = PreferencePairDataset(cache, hax.Axis("position", 16)).as_sync_dataset()[0]
    chosen_targets = np.roll(np.asarray(example.chosen.tokens.array), -1)[
        np.asarray(example.chosen.loss_weight.array) > 0
    ]
    rejected_targets = np.roll(np.asarray(example.rejected.tokens.array), -1)[
        np.asarray(example.rejected.loss_weight.array) > 0
    ]
    np.testing.assert_array_equal(chosen_targets, [10, 11, 21])
    np.testing.assert_array_equal(rejected_targets, [30, 31, 21])
    assert report["dispositions"] == {"preference": 1}
    assert receipts[0].identity.run_id == "teacher-collection"
    assert (
        json.loads((tmp_path / "cache" / "selection.json").read_text())["preferences"][0]["chosen"]["model_revision"]
        == MODELS["teacher"].revision
    )


def test_incomplete_collections_cannot_publish_a_preference_cache(tmp_path: Path):
    receipts = [
        collection_receipt(*_receipts(model), model=model, partition=PARTITION) for model in ("teacher", "student")
    ]
    teacher = retained_rollout(
        _record("teacher", 1.0), identity=receipts[0].identity, partition=PARTITION, trajectory_uri="teacher"
    )
    student = retained_rollout(
        _record("student", 0.0), identity=receipts[1].identity, partition=PARTITION, trajectory_uri="student"
    )
    unseen = TaskIdentity("bfcl-simple-python-14", "simple_python_14", "unseen-digest")
    partition = replace(PARTITION, complement=(TASK, unseen))
    expanded = [replace(receipt, task_names=frozenset({TASK.name, unseen.name})) for receipt in receipts]
    with pytest.raises(ValueError, match="complete collection selection"):
        recovery_preference_rows(
            [teacher],
            [student],
            teacher_receipt=expanded[0],
            student_receipt=expanded[1],
            partition=partition,
            max_length=16,
        )
    assert not (tmp_path / "cache").exists()


def test_mismatched_initial_context_is_excluded_with_pair_provenance():
    receipts = [
        collection_receipt(*_receipts(model), model=model, partition=PARTITION) for model in ("teacher", "student")
    ]
    teacher, student = [
        retained_rollout(_record(model, score), identity=receipt.identity, partition=PARTITION, trajectory_uri=model)
        for model, score, receipt in zip(("teacher", "student"), (1.0, 0.0), receipts, strict=True)
    ]
    student = replace(student, steps=(TokenStep((3, 4), (30,), (1,)),))
    rows, report = recovery_preference_rows(
        [teacher],
        [student],
        teacher_receipt=receipts[0],
        student_receipt=receipts[1],
        partition=PARTITION,
        max_length=16,
    )
    assert rows == []
    assert report["preferences"] == []
    excluded = report["excluded_preferences"]
    assert len(excluded) == 1
    assert excluded[0]["reason"] == "initial_prompt_mismatch"
    assert excluded[0]["pair"]["chosen"]["trajectory_uri"] == "teacher"
    assert excluded[0]["pair"]["rejected"]["trajectory_uri"] == "student"


def test_two_wrong_rollouts_produce_no_optimizer_data(tmp_path: Path):
    receipts = [
        collection_receipt(*_receipts(model), model=model, partition=PARTITION) for model in ("teacher", "student")
    ]
    teacher, student = [
        retained_rollout(_record(model, 0.0), identity=receipt.identity, partition=PARTITION, trajectory_uri=model)
        for model, receipt in zip(("teacher", "student"), receipts, strict=True)
    ]
    rows, report = recovery_preference_rows(
        [teacher],
        [student],
        teacher_receipt=receipts[0],
        student_receipt=receipts[1],
        partition=PARTITION,
        max_length=16,
    )
    assert rows == [] and report["dispositions"] == {"both_incorrect": 1}
    with pytest.raises(ValueError, match="must perform no update"):
        write_recovery_cache(rows, report, str(tmp_path / "cache"))
    assert not (tmp_path / "cache").exists()


def test_collection_receipts_reject_holdout_sources_and_sampling_mismatches():
    terminal, resolved = _receipts("teacher")
    terminal["config"]["inputs"]["train_data"][0]["relative_path"] = f"bfcl_complement/{HOLDOUT.name}"
    with pytest.raises(ValueError, match="outside the BFCL complement"):
        collection_receipt(terminal, resolved, model="teacher", partition=PARTITION)
    teacher_receipt = collection_receipt(*_receipts("teacher"), model="teacher", partition=PARTITION)
    terminal, resolved = _receipts("student")
    resolved["config"]["skyrl"]["generator"]["sampling_params"]["temperature"] = 0.5
    student_receipt = collection_receipt(terminal, resolved, model="student", partition=PARTITION)
    with pytest.raises(ValueError, match="different harness or sampling conditions"):
        recovery_preference_rows(
            [], [], teacher_receipt=teacher_receipt, student_receipt=student_receipt, partition=PARTITION, max_length=16
        )
