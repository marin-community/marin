# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Join synthetic terminal journals without starting workers or models."""

import asyncio
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import pytest
from marin.execution.lazy import StepContext
from marin.experiment.cli import graph_handles
from marin.external_dependencies import MARIN_SKYRL
from taskcompendium.parquet import read_tasks, write_tasks

from experiments.post_training.russell_rsi import completed_sft_selection as selection
from experiments.post_training.russell_rsi import test_coding_transport_replacement as coding_tests
from experiments.post_training.russell_rsi.calibration_recovery import PinnedFile
from experiments.post_training.russell_rsi.coding_eval_feedback import collect_coding_eval_evidence
from experiments.post_training.russell_rsi.coding_transport_replacement import (
    prepare_coding_replacement,
    replacement_coding_attempt,
)
from experiments.post_training.russell_rsi.contract_tasks import digest
from experiments.post_training.russell_rsi.interrupted_calibration import OUTPUT_PROTOCOL
from experiments.post_training.russell_rsi.sources import compact_json_sha256
from experiments.post_training.russell_rsi.token_preflight import PREFLIGHT_PROBES, preflight_task

continuation_inputs = coding_tests.continuation_inputs
study_inputs = coding_tests.study_inputs
interrupted_study = coding_tests.interrupted_study
completed_coding = coding_tests.completed_coding
replacement_inputs = coding_tests.replacement_inputs


def json_value(value):
    return json.loads(json.dumps(value))


@pytest.fixture
def completed_inputs(replacement_inputs, tmp_path):
    old, retention, _, coding_config, coding_pin = replacement_inputs
    pin = old.pin
    prepared = prepare_coding_replacement(coding_config, coding_pin, retention)
    coding_path = tmp_path / "coding-v9"
    bound = prepared.coding.build_config(
        StepContext.for_run(
            str(coding_path),
            old.source["recovery_artifact_prefix"],
            deps=prepared.coding.deps,
            runtime_args=prepared.coding.runtime_args,
        )
    )
    attempt = replacement_coding_attempt(bound)
    saved = {**old.saved, "path": str(coding_path)}

    async def finish():
        return saved

    asyncio.run(attempt.run(finish))
    source_files = {
        selection.CODING_SOURCE_FILE: attempt.binding["transport_replacement"]["source_sha256"],
        selection.RETENTION_SOURCE_FILE: "frozen-retention-worker",
        selection.JOURNAL_SOURCE_FILE: "frozen-retention-journal",
    }
    worker_provenance = {
        "branch_root": "/app",
        "modules": {
            "taskcompendium.models": {"path": "/app/lib/taskcompendium/src/taskcompendium/models.py", "sha256": "frozen"}
        },
        "skyrl": {"commit": MARIN_SKYRL.commit},
    }

    def producer(path, version, config_pin, bound_config, result):
        path.mkdir(exist_ok=True)
        head = selection.PRODUCER_SOURCE_HEADS[version]
        review = pin(path / "source-review.json", {"status": "approved", "source_head": head})
        request = pin(path / "request.json", {"review": review})
        preflight = pin(
            path / "preflight.json",
            {"exit_code": 0, "identity": {"source_head": head, "request_sha256": request["sha256"]}},
        )
        record = {
            "name": f"evals/synthetic-{version}",
            "version": version,
            "fingerprint": version,
            "config": json_value(bound_config),
            "output_path": str(path),
            "result": result,
            "provenance": {"base_commit": head, "dirty": False},
        }
        record_pin = pin(path / ".artifact.json", record)
        (path / ".executor_status").write_text("SUCCESS")
        launch = {
            "protocol": selection.LAUNCH_PROTOCOL,
            "source_head": head,
            "runtime_commit": MARIN_SKYRL.commit,
            "producer_identity": f"{record['name']}@{version}:{version}",
            "output_path": str(path),
            "config": config_pin,
            "bound_config": record["config"],
            "source_review": review,
            "request": request,
            "preflight": preflight,
            "source_files": source_files,
            "worker_provenance": worker_provenance,
            "calibration_status": "incomplete_infrastructure",
            "signal_gate_passed": None,
            "rl_authorized": False,
        }
        return {"config": config_pin, "producer_record": record_pin, "launch_proof": pin(path / "launch.json", launch)}

    coding = producer(
        coding_path,
        "2026.10.06.9",
        asdict(coding_pin),
        asdict(bound),
        {key: value for key, value in saved.items() if key != "path"},
    )
    coding.update(
        reservation=pin(coding_path / "journal/coding/reservation.json", attempt.binding),
        result=pin(coding_path / "journal/coding/result.json", {"binding": attempt.binding, "result": saved}),
    )
    retained_path = tmp_path / "retention-v10"
    retained_config = old.original["retention-sft"].build_config(
        StepContext.for_run(
            str(retained_path), old.source["recovery_artifact_prefix"], deps=old.original["retention-sft"].deps
        )
    )
    repair_pin = pin(tmp_path / "repair-amendment.json", {"status": "reviewed"})
    repaired = {
        "protocol": "russell-rsi-retention-preflight-hash-repair-v1",
        "version": "2026.10.06.10",
        "repair_amendment_uri": repair_pin["uri"],
        "repair_amendment_sha256": repair_pin["sha256"],
    }
    repaired_pin = pin(tmp_path / "retention-config.json", repaired)
    retained = producer(
        retained_path,
        "2026.10.06.10",
        repaired_pin,
        {"evaluation": asdict(retained_config), "retention_config": repaired_pin, "launch_failure": repair_pin},
        None,
    )
    task_ids = old.selection.record.retention_task_ids
    tasks = [
        preflight_task(index, "Return the value.", 23).model_copy(update={"id": task_id})
        for index, task_id in enumerate(task_ids, 101)
    ]
    Path(retained_config.tasks_path).parent.mkdir(parents=True, exist_ok=True)
    write_tasks(retained_config.tasks_path, tasks)
    fixtures = [
        preflight_task(i, instruction, value).model_dump(mode="json")
        for i, (instruction, value) in enumerate(PREFLIGHT_PROBES, 1)
    ]
    binding = {
        "protocol": OUTPUT_PROTOCOL,
        "config": json_value(asdict(retained_config)),
        "parquet_sha256": hashlib.sha256(Path(retained_config.tasks_path).read_bytes()).hexdigest(),
        "worker_provenance": worker_provenance,
        "worker_source_sha256": source_files[selection.JOURNAL_SOURCE_FILE],
        "job_policy": {"failure_retries": 0, "preemption_retries": 0, "timeout_hours": 6},
        "continuation": {
            "protocol": repaired["protocol"],
            "retention_config": repaired_pin,
            "launch_failure": repair_pin,
            "worker_source_sha256": source_files[selection.RETENTION_SOURCE_FILE],
        },
        "attempts": {
            "preflight": {str(i): compact_json_sha256(v) for i, v in enumerate(fixtures, 1)},
            "task": {
                f"{task.id}/0": digest(task.model_dump(mode="json")) for task in read_tasks(retained_config.tasks_path)
            },
        },
    }
    probes = [{"status": "passed"}, {"status": "failed"}]
    attempts = []
    for kind, keys in binding["attempts"].items():
        for key, task_digest in keys.items():
            reservation = {"evaluation": binding, "kind": kind, "key": key, "task_sha256": task_digest}
            result = (
                probes[int(key) - 1]
                if kind == "preflight"
                else {
                    "record": {
                        "task_id": key.removesuffix("/0"),
                        "grade": {"status": "graded", "reward": int(key != f"{task_ids[1]}/0")},
                    },
                    "startup_counts": {"retries": 0},
                }
            )
            directory = retained_path / "journal" / kind / key
            attempts.append(
                {
                    "kind": kind,
                    "key": key,
                    "reservation": pin(directory / "reservation.json", reservation),
                    "result": pin(directory / "result.json", {"binding": reservation, "result": result}),
                }
            )
    summary = {
        "model_identity": retained_config.model_identity,
        "tasks_path": retained_config.tasks_path,
        "tasks_identity": retained_config.tasks_identity,
        "count": 3,
        "samples_per_task": 1,
        "startup_attempts": 1,
        "startup_counts": {"retries": 0},
        "informative_groups": 0,
        "task_rewards": {task_id: [int(task_id != task_ids[1])] for task_id in task_ids},
        "categories": {"passed": 2, "incorrect": 1},
        "failed_task_ids": [task_ids[1]],
    }
    retained.update(
        journal_binding=pin(retained_path / "journal/binding.json", binding),
        attempts=attempts,
        summary=pin(retained_path / "failure_summary.json", summary),
        preflight_summary=pin(
            retained_path / "token-preflight.json", {"status": "passed", "attempts": probes, "fixtures": fixtures}
        ),
    )
    config = {
        "protocol": selection.PROTOCOL,
        "version": selection.VERSION,
        "source_config": {"uri": old.config["source_config_uri"], "sha256": old.config["source_config_sha256"]},
        "coding": coding,
        "retention": retained,
    }
    return old, config, pin


def test_completed_join_contains_only_evidence_and_selection(completed_inputs, tmp_path):
    old, config, pin = completed_inputs
    graph = selection.completed_sft_selection_stages(
        config, PinnedFile(**pin(tmp_path / "join.json", config)), old.original
    )
    handles = graph_handles([graph["terminal"]])
    assert {step.run.__name__ for step in handles if step.run} == {
        "collect_coding_eval_evidence",
        "seal_completed_selection",
        "_adopt_noop",
    }
    assert len(handles) == 3
    assert all(not step.deps for step in handles if step.run is collect_coding_eval_evidence)
    assert graph["terminal"].deps[1].path(old.source["recovery_artifact_prefix"]) == str(tmp_path / "retention-v10")


@pytest.mark.parametrize("defect", ["source", "summary", "missing", "ungraded", "contract", "binding", "dirty"])
def test_completed_join_rejects_changed_or_incomplete_evidence(completed_inputs, tmp_path, defect):
    old, config, pin = completed_inputs
    retained = config["retention"]
    target = retained["summary"]
    if defect == "source":
        target = retained["journal_binding"]
        value = PinnedFile(**target).read_json()
        value["worker_source_sha256"] = "current-integrated-source"
    elif defect == "missing":
        retained["attempts"].pop()
        value = None
    elif defect == "ungraded":
        target = next(item["result"] for item in retained["attempts"] if item["kind"] == "task")
        value = PinnedFile(**target).read_json()
        value["result"]["record"]["grade"]["status"] = "infra_error"
    elif defect == "contract":
        target = next(item["result"] for item in retained["attempts"] if item["kind"] == "preflight")
        value = PinnedFile(**target).read_json()
        value["result"]["contract_failure"] = "invalid completion"
    elif defect == "binding":
        target = config["coding"]["result"]
        value = PinnedFile(**target).read_json()
        value["binding"]["config"]["limit"] = 64
    elif defect == "dirty":
        target = retained["producer_record"]
        value = PinnedFile(**target).read_json()
        value["provenance"]["dirty"] = True
    else:
        value = PinnedFile(**target).read_json()
        value["task_rewards"][old.selection.record.retention_task_ids[0]] = [0]
    if value is not None:
        target.update(pin(target["uri"], value))
    with pytest.raises(ValueError):
        selection.completed_sft_selection_stages(config, PinnedFile(**pin(tmp_path / "join.json", config)), old.original)


def test_coding_extraction_does_not_require_retention(completed_inputs, tmp_path):
    old, joined, pin = completed_inputs
    Path(joined["retention"]["producer_record"]["uri"]).unlink()
    config = {key: joined[key] for key in ("version", "source_config", "coding")}
    config["protocol"] = selection.EXTRACTION_PROTOCOL
    stages = selection.completed_coding_extraction_stages(
        config, PinnedFile(**pin(tmp_path / "extract.json", config)), old.original
    )
    handles = graph_handles([stages["terminal"]])
    assert len(handles) == 1
    assert handles[0].run is collect_coding_eval_evidence
    assert handles[0].deps == ()


@pytest.mark.parametrize("scores,promotes", [((27 / 32, 25 / 32), False), ((1.0, 1.0), True)])
def test_completed_selection_keeps_original_per_suite_gate(completed_inputs, tmp_path, scores, promotes):
    old, config, pin = completed_inputs
    stages = selection.completed_sft_selection_stages(
        config, PinnedFile(**pin(tmp_path / "join.json", config)), old.original
    )
    terminal = stages["terminal"]
    bound = terminal.build_config(StepContext.for_run(str(tmp_path / "sealed"), str(tmp_path), deps=terminal.deps))
    record = bound.selection.selection.record
    evidence = {
        "model_identity": record.model_identities[0],
        "panel_sha256": record.panel_sha256,
        "scores": dict(zip(("humanevalplus", "mbppplus"), scores, strict=True)),
    }
    pin(Path(record.coding_paths[0]) / "coding-evidence.json", evidence)
    selection.seal_completed_selection(bound)
    result = json.loads((tmp_path / "sealed/post-sft-selection.json").read_text())
    assert result["protocol"] == selection.PROTOCOL
    assert result["promoted"]["checkpoint_identity"] == (
        record.model_identities[0] if promotes else record.parent.checkpoint_identity
    )
    assert result["incumbent"] == json_value(asdict(old.selection.record.parent))
    provenance = json.loads((tmp_path / "sealed/completed-producers.json").read_text())
    assert (provenance["coding_source"], provenance["retention_source"]) == ("replacement_v9", "repaired_v10")
    assert provenance["signal_gate_passed"] is None
    assert provenance["rl_authorized"] is False
