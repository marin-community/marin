# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the continuation preflight.

The preflight exists because hosted-v13 was submitted and only died later in
the campaign's first prepare call. These tests pin the refusal behavior: a
mismatched frozen receipt, a changed queue, a wrong serving digest or an
unsealed parent each refuse with the reason named.
"""

import json
from pathlib import Path

import pytest
from finestore.rl.difficulty import SampleTask
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.models import TaskSpec

from experiments.post_training.task_curation import difficulty_preflight
from experiments.post_training.task_curation.difficulty_contracts import (
    CONTRACT_VINTAGE_ABSENT_OMITTED,
    difficulty_task_contract,
)
from experiments.post_training.task_curation.difficulty_preflight import (
    SourcePreflight,
    load_queue_sources,
    preflight_continuation,
    preflight_source,
)
from experiments.post_training.task_curation.tests.test_difficulty_contracts import _task

QUEUE = "s3://queue/sources.json"
QUEUE_SHA = "q" * 64


def _frozen_source(difficulty_source, task: TaskSpec, task_sha256: str):
    """A queue row whose sample receipt binds the given digest for the single task."""
    source = difficulty_source(task)
    frozen_sample = source.sample.model_copy(update={"tasks": [SampleTask(task_id=task.id, task_sha256=task_sha256)]})
    return source.model_copy(update={"sample": frozen_sample})


def _fake_parent(tmp_path: Path, queue_sha256: str) -> Path:
    parent = tmp_path / "parent"
    (parent / "trials").mkdir(parents=True)
    (parent / "prepared.json").write_text(json.dumps({"source_queue_sha256": queue_sha256, "sources": []}))
    (parent / "serving.json").write_text(json.dumps({"solver": "hosted"}))
    (parent / "trials" / "model-requests-final.json").write_text("{}")
    return parent


def test_preflight_passes_a_matching_continuation(tmp_path, monkeypatch, difficulty_source):
    task = _task()
    contract = difficulty_task_contract(task, {}, CONTRACT_VINTAGE_ABSENT_OMITTED)
    source = _frozen_source(difficulty_source, task, contract["task_sha256"])
    parent = _fake_parent(tmp_path, QUEUE_SHA)
    monkeypatch.setattr(difficulty_preflight, "load_queue_sources", lambda *a, **k: [source])
    monkeypatch.setattr(
        difficulty_preflight,
        "preflight_source",
        lambda source, stage_root=None: SourcePreflight(source.declaration.id, CONTRACT_VINTAGE_ABSENT_OMITTED, 1, None),
    )
    receipt = preflight_continuation(
        QUEUE,
        QUEUE_SHA,
        None,
        str(parent),
        file_sha256(parent / "serving.json"),
    )
    assert receipt["sources"] == 1
    assert receipt["failures"] == []
    assert receipt["previous_campaign"] == str(parent)
    assert receipt["previous_serving_sha256"] == file_sha256(parent / "serving.json")


def test_preflight_names_a_drifted_frozen_receipt(tmp_path, monkeypatch, difficulty_source):
    task = _task()
    source = _frozen_source(difficulty_source, task, "f" * 64)
    parent = _fake_parent(tmp_path, QUEUE_SHA)
    monkeypatch.setattr(difficulty_preflight, "load_queue_sources", lambda *a, **k: [source])
    monkeypatch.setattr(difficulty_preflight, "panel_task_runtimes", lambda panel: ([task], [{}]))
    with pytest.raises(ValueError) as error:
        preflight_continuation(QUEUE, QUEUE_SHA, None, str(parent), file_sha256(parent / "serving.json"))
    message = str(error.value)
    assert "Fixture:one" in message or task.id in message
    assert "every recorded contract vintage" in message


def test_preflight_refuses_a_serving_digest_mismatch(tmp_path, monkeypatch):
    parent = _fake_parent(tmp_path, QUEUE_SHA)
    monkeypatch.setattr(difficulty_preflight, "load_queue_sources", lambda *a, **k: [])
    with pytest.raises(ValueError, match="serving receipt differs"):
        preflight_continuation(QUEUE, QUEUE_SHA, None, str(parent), "5" * 64)


def test_preflight_refuses_a_changed_frozen_queue(tmp_path, monkeypatch):
    parent = _fake_parent(tmp_path, "old" + "0" * 61)
    monkeypatch.setattr(difficulty_preflight, "load_queue_sources", lambda *a, **k: [])
    with pytest.raises(ValueError, match="cannot change the frozen source queue"):
        preflight_continuation(QUEUE, QUEUE_SHA, None, str(parent), "5" * 64)


def test_preflight_refuses_an_unsealed_parent(tmp_path, monkeypatch):
    parent = _fake_parent(tmp_path, QUEUE_SHA)
    serving_digest = file_sha256(parent / "serving.json")
    (parent / "trials" / "model-requests-final.json").unlink()
    monkeypatch.setattr(difficulty_preflight, "load_queue_sources", lambda *a, **k: [])
    with pytest.raises(ValueError, match="has not sealed"):
        preflight_continuation(QUEUE, QUEUE_SHA, None, str(parent), serving_digest)


def test_preflight_refuses_a_missing_parent_prepared_receipt(tmp_path, monkeypatch):
    parent = _fake_parent(tmp_path, QUEUE_SHA)
    (parent / "prepared.json").unlink()
    monkeypatch.setattr(difficulty_preflight, "load_queue_sources", lambda *a, **k: [])
    with pytest.raises(ValueError, match=r"no prepared\.json"):
        preflight_continuation(QUEUE, QUEUE_SHA, None, str(parent), "5" * 64)


def test_preflight_refuses_a_partial_continuation_binding(tmp_path, monkeypatch):
    monkeypatch.setattr(difficulty_preflight, "load_queue_sources", lambda *a, **k: [])
    with pytest.raises(ValueError, match="both the previous campaign and its serving receipt"):
        preflight_continuation(QUEUE, QUEUE_SHA, None, str(tmp_path / "parent"), None)


def test_preflight_names_a_truncated_serving_digest(tmp_path, monkeypatch):
    """small-v22 died on a hand-truncated digest; the preflight names it instead."""
    parent = _fake_parent(tmp_path, QUEUE_SHA)
    digest = file_sha256(parent / "serving.json")
    monkeypatch.setattr(difficulty_preflight, "load_queue_sources", lambda *a, **k: [])
    with pytest.raises(ValueError, match="not a full sha256"):
        preflight_continuation(QUEUE, QUEUE_SHA, None, str(parent), digest[:16])


_RELAY_HEALTHY_OUTPUT = (
    "Job: /muchanem/relay (/muchanem/relay)\n"
    "State: running  exit=0  failures=0  preemptions=0\n"
    "Tasks: 0/1 completed  running=1\n"
    "\n"
    "  TASK  STATE    EXIT    DURATION   PEAK MEM    CUR MEM    DIAGNOSTIC\n"
    "     0  running  -       3 days     -           -          -\n"
)

_RELAY_DEAD_OUTPUT = (
    "Job: /muchanem/relay (/muchanem/relay)\n"
    "State: running  exit=0  failures=0  preemptions=1\n"
    "Tasks: 0/1 completed  running=1\n"
    "\n"
    "  TASK  STATE    EXIT    DURATION                     PEAK MEM    CUR MEM    DIAGNOSTIC\n"
    "     0  running  -       1 week, 2 days and 13 hours  -           -          "
    "PodDeleted: pod was deleted while the attempt was active\n"
)


def _fake_iris_describe(monkeypatch, output: str):
    def fake_run(command, capture_output, text, timeout, check):
        return type("R", (), {"returncode": 0, "stdout": output, "stderr": ""})()

    monkeypatch.setattr(difficulty_preflight.subprocess, "run", fake_run)


def test_relay_liveness_accepts_a_live_relay(monkeypatch):
    _fake_iris_describe(monkeypatch, _RELAY_HEALTHY_OUTPUT)
    verdict = difficulty_preflight.relay_liveness("/muchanem/relay")
    assert verdict["alive"] is True
    assert verdict["dead_tasks"] == []
    assert verdict["preemptions"] == 0


def test_relay_liveness_refuses_a_pod_deleted_behind_a_running_record(monkeypatch):
    """hosted-v13c's failure: the job record says running, the serving pod is gone."""
    _fake_iris_describe(monkeypatch, _RELAY_DEAD_OUTPUT)
    with pytest.raises(ValueError, match=r"serving pod is gone.*preemptions=1.*PodDeleted"):
        difficulty_preflight.relay_liveness("/muchanem/relay")


def test_relay_liveness_refuses_a_failed_task_row(monkeypatch):
    output = _RELAY_HEALTHY_OUTPUT.replace(
        "     0  running  -       3 days     -           -          -",
        "     0  failed   1       3 days     -           -          boom",
    )
    _fake_iris_describe(monkeypatch, output)
    with pytest.raises(ValueError, match="serving pod is gone"):
        difficulty_preflight.relay_liveness("/muchanem/relay")


def test_load_queue_sources_filters_and_verifies(tmp_path, difficulty_source):
    source = difficulty_source(_task())
    queue_path = tmp_path / "sources.json"
    queue_path.write_text(json.dumps([source.model_dump(mode="json")]))
    digest = file_sha256(queue_path)
    selected = load_queue_sources(str(queue_path), digest, (source.declaration.id,))
    assert [row.declaration.id for row in selected] == [source.declaration.id]
    with pytest.raises(ValueError, match="absent from the frozen selection"):
        load_queue_sources(str(queue_path), digest, ("Other:source",))
    with pytest.raises(ValueError, match="differs from the frozen digest"):
        load_queue_sources(str(queue_path), "0" * 64, None)


def test_preflight_source_resolves_the_receipt_vintage(monkeypatch, difficulty_source):
    """The receipt stage reproduces prepare_source's comparison and names the vintage."""
    task = _task()
    source = difficulty_source(task)
    monkeypatch.setattr(difficulty_preflight, "panel_task_runtimes", lambda panel: ([task], [{}]))
    result = preflight_source(source)
    assert result.error is None
    assert result.vintage == CONTRACT_VINTAGE_ABSENT_OMITTED
    assert result.tasks == 1


def test_preflight_source_staging_catches_task_validation_failures(tmp_path, monkeypatch, difficulty_source):
    """Full local staging refuses the hosted-v13b class: a task that prepare rejects."""
    task = _task()
    source = difficulty_source(task)
    monkeypatch.setattr(difficulty_preflight, "panel_task_runtimes", lambda panel: ([task], [{}]))

    def refusing_prepare(source, root):
        raise ValueError("task_machine.storage_mb conflict between the runtime packet and the task specs")

    monkeypatch.setattr(difficulty_preflight, "prepare_source", refusing_prepare)
    result = preflight_source(source, stage_root=tmp_path / "tasks")
    assert result.vintage == CONTRACT_VINTAGE_ABSENT_OMITTED
    assert result.error is not None
    assert "storage_mb" in result.error


def test_preflight_receipt_stage_only_skips_staging(tmp_path, monkeypatch, difficulty_source):
    task = _task()
    source = difficulty_source(task)
    monkeypatch.setattr(difficulty_preflight, "panel_task_runtimes", lambda panel: ([task], [{}]))

    def unexpected_prepare(source, root):
        raise AssertionError("staging must not run in receipt-stage-only mode")

    monkeypatch.setattr(difficulty_preflight, "prepare_source", unexpected_prepare)
    result = preflight_source(source)
    assert result.error is None
