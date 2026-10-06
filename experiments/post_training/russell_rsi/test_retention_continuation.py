# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Use synthetic completed records; no workers or models are started."""

import asyncio
import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import NamedTuple

import pytest
from marin.evaluation.records import EvalRunRecord, record_path
from marin.execution.lazy import StepContext, artifact_identity
from marin.experiment.cli import graph_handles
from marin.external_dependencies import MARIN_SKYRL

from experiments.post_training.russell_rsi import test_interrupted_calibration, test_teacher_four_pass
from experiments.post_training.russell_rsi.calibration_recovery import PinnedFile
from experiments.post_training.russell_rsi.coding_eval_feedback import collect_coding_eval_evidence
from experiments.post_training.russell_rsi.interrupted_calibration import coding_attempt, run_foreground_coding
from experiments.post_training.russell_rsi.launch_post_teacher_sft import StudySelectionConfig
from experiments.post_training.russell_rsi.retention_continuation import (
    FAILED_SOURCE_COMMIT,
    FAILURE_PROTOCOL,
    PROTOCOL,
    VERSION,
    prepare_retention_continuation,
    retention_continuation_stages,
    seal_retention_continuation,
    submit_continuation_retention,
)
from experiments.post_training.russell_rsi.sources import compact_json_sha256
from experiments.post_training.russell_rsi.teacher_four_pass import four_pass_post_workflow

continuation_inputs = test_teacher_four_pass.continuation_inputs
study_inputs = test_teacher_four_pass.study_inputs
interrupted_study = test_interrupted_calibration.interrupted_study


class CompletedCoding(NamedTuple):
    config: dict
    source: dict
    original: dict
    failure: dict
    record: dict
    saved: dict
    selection: StudySelectionConfig
    pin: Callable[..., dict]


@pytest.fixture
def completed_coding(interrupted_study, tmp_path):
    source, _, _, _ = interrupted_study
    original = four_pass_post_workflow(source, "evaluate-interrupted")
    source = {**source, "recovery_artifact_prefix": str(tmp_path / "artifacts"), "runtime_commit": MARIN_SKYRL.commit}
    coding, model = original["coding-sft"].deps
    coding_dir = tmp_path / "completed-coding"
    coding_dir.mkdir()
    bound = coding.build_config(
        StepContext.for_run(
            str(coding_dir), source["recovery_artifact_prefix"], deps=coding.deps, runtime_args=coding.runtime_args
        )
    )
    saved = {
        "path": str(coding_dir),
        "group_id": "completed",
        "records_prefix": str(tmp_path / "records"),
        "run_ids": ["he", "mbpp"],
        "results_paths": ["/existing-he", "/existing-mbpp"],
    }
    record = {
        "name": coding.name,
        "version": source["version"],
        "fingerprint": coding.fingerprint(),
        "config": asdict(bound),
        "output_path": saved["path"],
        "result": {key: value for key, value in saved.items() if key != "path"},
    }
    selection = (
        original["terminal"]
        .build_config(
            StepContext.for_run(
                str(tmp_path / "selection"), source["recovery_artifact_prefix"], deps=original["terminal"].deps
            )
        )
        .selection
    )

    def pin(path, payload):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(payload).encode()
        path.write_bytes(raw)
        return {"uri": str(path), "sha256": hashlib.sha256(raw).hexdigest()}

    source_pin = pin(tmp_path / "source.json", source)
    result_pin = pin(coding_dir / ".artifact.json", record)
    (coding_dir / ".executor_status").write_text("SUCCESS")
    journal = coding_attempt(bound)

    async def completed():
        return saved

    asyncio.run(journal.run(completed))
    journal_pin = pin(coding_dir / "journal/coding/result.json", {"binding": journal.binding, "result": saved})
    failure = {
        "protocol": FAILURE_PROTOCOL,
        "calibration_status": "incomplete_infrastructure",
        "signal_gate_passed": None,
        "rl_authorized": False,
        "source": {
            "config": source_pin,
            "model_identity": artifact_identity(model),
            "coding_identity": artifact_identity(coding),
            "retention_identity": artifact_identity(original["retention-sft"]),
            "panel_sha256": selection.record.panel_sha256,
            "source_commit": FAILED_SOURCE_COMMIT,
            "runtime_commit": MARIN_SKYRL.commit,
        },
        "retention": {"status": "never_issued", "reservations": 0, "submissions": 0, "model_requests": 0},
        "evidence": {
            name: pin(tmp_path / f"{name}.json", {"witness": name})
            for name in ("rejection", "not_found", "empty_controller_prefix", "absent_journal")
        },
        "coding": {"status": "original_v7_pending", "replay_authorized": False},
        "evaluation": {"version": VERSION, "conditions": ["sft"], "coding_reused": True, "retention_limit": 3},
    }
    failure["evidence"]["summary"] = pin(
        tmp_path / "admission.json",
        {
            "protocol": "russell-retention-never-admitted-launch-evidence-v1",
            "source": {"config_sha256": source_pin["sha256"]},
            "retention_artifact_prefix": original["retention-sft"].path(source["recovery_artifact_prefix"]),
            "conclusions": {
                "worker_admitted": False,
                "retention_journal_present": False,
                "retention_scientific_slots_issued": 0,
                "coding_replay_authorized": False,
            },
        },
    )
    config = {"protocol": PROTOCOL, "version": VERSION}
    for name, value in (
        ("source_config", source_pin),
        ("coding_result", result_pin),
        ("coding_journal_result", journal_pin),
    ):
        config.update({f"{name}_uri": value["uri"], f"{name}_sha256": value["sha256"]})
    return CompletedCoding(config, source, original, failure, record, saved, selection, pin)


def completed_retention(prepared, pin):
    path = Path(prepared.step.path(prepared.source["recovery_artifact_prefix"]))
    pin(path / ".artifact.json", {"fingerprint": prepared.step.fingerprint(), "output_path": str(path)})
    (path / ".executor_status").write_text("SUCCESS")


def test_retain_needs_no_completed_coding_and_selection_refuses_incomplete_retention(completed_coding, tmp_path):
    config, source, original, failure, _, _, _, pin = completed_coding
    amendment = pin(tmp_path / "failure.json", failure)
    retained_config = {
        "protocol": PROTOCOL,
        "version": VERSION,
        "source_config_uri": config["source_config_uri"],
        "source_config_sha256": config["source_config_sha256"],
        "launch_failure_uri": amendment["uri"],
        "launch_failure_sha256": amendment["sha256"],
    }
    retained_pin = pin(tmp_path / "retain.json", retained_config)
    prepared = prepare_retention_continuation(retained_config, source, original, PinnedFile(**retained_pin))
    coding_record = Path(config["coding_result_uri"])
    saved_record = coding_record.read_bytes()
    coding_record.unlink()
    repeated = prepare_retention_continuation(retained_config, source, original, PinnedFile(**retained_pin))
    assert artifact_identity(repeated.step) == artifact_identity(prepared.step)
    assert all(step.run is not run_foreground_coding for step in graph_handles([repeated.step]))
    coding_record.write_bytes(saved_record)
    selected = {key: value for key, value in config.items() if not key.startswith("source_config")}
    selected.update(retention_config_uri=retained_pin["uri"], retention_config_sha256=retained_pin["sha256"])
    with pytest.raises(ValueError, match="requires completed"):
        retention_continuation_stages(selected, prepared)


@pytest.mark.parametrize("saved_evidence", [False, True])
def test_retention_continuation_has_one_retention_and_no_coding_inference(completed_coding, tmp_path, saved_evidence):
    config, source, original, failure, _, _, old_selection, pin = completed_coding
    if saved_evidence:
        saved = completed_coding.saved
        record = completed_coding.record
        records = []
        for run_id, suite, path in zip(
            saved["run_ids"], ("humanevalplus", "mbppplus"), saved["results_paths"], strict=True
        ):
            value = EvalRunRecord.model_validate(
                {
                    "run_id": run_id,
                    "group_id": saved["group_id"],
                    "created_at": "2026-10-06T00:00:00Z",
                    "user": "fixture",
                    "model": {
                        "name": "sft",
                        "location": record["config"]["model"]["location"],
                        "backend": "vllm",
                        "config": record["config"]["model"],
                    },
                    "eval": {"name": suite, "mechanism": "evalchemy"},
                    "hardware": {
                        "platform": "coreweave",
                        "accelerator": "H100x8",
                        "region_or_cluster": "cw-us-east-02a",
                    },
                    "status": "succeeded",
                    "error": None,
                    "results_path": path,
                    "metrics": {},
                    "jobs": {},
                    "log_tails": {},
                    "provenance": {"git_sha": "synthetic", "eval_runtime": "synthetic", "launch_host": "fixture"},
                }
            ).model_dump(mode="json", by_alias=True)
            pin(record_path(saved["records_prefix"], run_id), value)
            records.append(value)
        evidence = pin(
            tmp_path / "coding-evidence/coding-evidence.json",
            {
                "model_identity": failure["source"]["model_identity"],
                "panel_sha256": old_selection.record.panel_sha256,
                "scores": {"humanevalplus": 25 / 32, "mbppplus": 27 / 32},
                "records_sha256": [compact_json_sha256(value) for value in records],
            },
        )
        pin(
            tmp_path / "coding-evidence/.artifact.json",
            {"fingerprint": original["coding-sft"].fingerprint(), "output_path": str(tmp_path / "coding-evidence")},
        )
        (tmp_path / "coding-evidence/.executor_status").write_text("SUCCESS")
        config.update(coding_evidence_uri=evidence["uri"], coding_evidence_sha256=evidence["sha256"])
    amendment = pin(tmp_path / "failure.json", failure)
    config.update(launch_failure_uri=amendment["uri"], launch_failure_sha256=amendment["sha256"])
    retained_config = {key: value for key, value in config.items() if not key.startswith("coding_")}
    retained_pin = pin(tmp_path / "retain.json", retained_config)
    prepared = prepare_retention_continuation(retained_config, source, original, PinnedFile(**retained_pin))
    completed_retention(prepared, pin)
    config = {key: value for key, value in config.items() if not key.startswith(("source_config", "launch_failure"))}
    config.update(retention_config_uri=retained_pin["uri"], retention_config_sha256=retained_pin["sha256"])
    outputs = retention_continuation_stages(config, prepared)
    graph = graph_handles([outputs["terminal"]])
    assert sum(step.run is submit_continuation_retention for step in graph) == 1
    assert all(step.run is not run_foreground_coding for step in graph)
    assert sum(step.run is collect_coding_eval_evidence for step in graph) == (0 if saved_evidence else 1)
    assert artifact_identity(outputs["retention-sft"]) == artifact_identity(prepared.step)
    retain_graph = graph_handles([prepared.step])
    assert sum(step.run is submit_continuation_retention for step in retain_graph) == 1
    assert all(step.run not in (run_foreground_coding, collect_coding_eval_evidence) for step in retain_graph)
    assert outputs["retention-sft"].version == VERSION
    assert artifact_identity(outputs["retention-sft"].deps[1]) == failure["source"]["model_identity"]
    final = outputs["terminal"].build_config(
        StepContext.for_run(str(tmp_path / "final"), str(tmp_path), deps=outputs["terminal"].deps)
    )
    assert final.selection.selection.record.parent == old_selection.record.parent
    assert final.selection.selection.original_parent == old_selection.original_parent
    assert outputs["terminal"].run is seal_retention_continuation
    if saved_evidence:
        pin(tmp_path / "coding-evidence/.artifact.json", {"fingerprint": "different-producer"})
        with pytest.raises(ValueError, match="original producer"):
            retention_continuation_stages(config, prepared)


@pytest.mark.parametrize("defect", ["model", "journal", "panel", "admitted"])
def test_retention_continuation_rejects_changed_coding_or_issued_retention(completed_coding, tmp_path, defect):
    config, source, original, failure, record, saved, _, pin = completed_coding
    if defect == "model":
        record["config"]["model"]["identity"] = "another-model"
        value = pin(config["coding_result_uri"], record)
        config["coding_result_sha256"] = value["sha256"]
    elif defect == "journal":
        value = pin(config["coding_journal_result_uri"], {**saved, "group_id": "another-group"})
        config["coding_journal_result_sha256"] = value["sha256"]
    elif defect == "panel":
        failure["source"]["panel_sha256"] = "another-panel"
    else:
        failure["retention"]["submissions"] = 1
    amendment = pin(tmp_path / "failure.json", failure)
    config.update(launch_failure_uri=amendment["uri"], launch_failure_sha256=amendment["sha256"])
    retained_config = {key: value for key, value in config.items() if not key.startswith("coding_")}
    retained_pin = pin(tmp_path / "retain.json", retained_config)
    if defect in {"panel", "admitted"}:
        with pytest.raises(ValueError, match="original transport failure"):
            prepare_retention_continuation(retained_config, source, original, PinnedFile(**retained_pin))
        return
    prepared = prepare_retention_continuation(retained_config, source, original, PinnedFile(**retained_pin))
    completed_retention(prepared, pin)
    config = {key: value for key, value in config.items() if not key.startswith(("source_config", "launch_failure"))}
    config.update(retention_config_uri=retained_pin["uri"], retention_config_sha256=retained_pin["sha256"])
    with pytest.raises(ValueError, match="Completed coding producer"):
        retention_continuation_stages(config, prepared)
