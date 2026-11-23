# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Frozen-contract, staging and judge-route preflight behavior."""

import json

import pytest
from finestore.rl.difficulty import SampleTask
from taskcompendium.evaluation.corpus import file_sha256
from taskcompendium.models import TaskSpec

from experiments.post_training.glm import GLM_MODEL
from experiments.post_training.task_curation import difficulty_preflight, difficulty_prepared
from experiments.post_training.task_curation.difficulty_contracts import (
    CONTRACT_VINTAGE_ABSENT_OMITTED,
)
from experiments.post_training.task_curation.difficulty_preflight import (
    load_queue_sources,
    preflight_batch,
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


def test_preflight_names_a_drifted_frozen_receipt(tmp_path, monkeypatch, difficulty_source):
    task = _task()
    source = _frozen_source(difficulty_source, task, "f" * 64)
    monkeypatch.setattr(difficulty_preflight, "load_queue_sources", lambda *a, **k: [source])
    monkeypatch.setattr(difficulty_preflight, "panel_task_runtimes", lambda panel: ([task], [{}]))
    with pytest.raises(ValueError) as error:
        preflight_batch(QUEUE, QUEUE_SHA, None)
    message = str(error.value)
    assert "Fixture:one" in message or task.id in message
    assert "every recorded contract vintage" in message


def test_preflight_probes_the_hosted_endpoint_and_refuses_a_dead_one(monkeypatch):
    """The hosted campaign now serves Together; the preflight probes the same endpoint
    before bindings, and a dead endpoint refuses the submission (the v13c class)."""
    calls = []

    def fake_probe(base_url, api_key, model):
        calls.append((base_url, model))
        raise RuntimeError("did not answer a liveness completion within 60s")

    monkeypatch.setattr(difficulty_preflight, "hosted_endpoint_liveness", fake_probe)
    monkeypatch.setattr(difficulty_preflight, "require_secret", lambda name: "fixture-key")
    monkeypatch.setattr(difficulty_preflight, "load_queue_sources", lambda *a, **k: [])
    with pytest.raises(ValueError, match=r"Hosted endpoint liveness failed.*liveness"):
        preflight_batch(QUEUE, QUEUE_SHA, None, probe_hosted=True)
    assert calls == [(difficulty_preflight.TOGETHER_BASE_URL, difficulty_preflight.TOGETHER_MODEL)]


def test_preflight_skips_the_hosted_probe_for_local_solvers(monkeypatch):
    calls = []
    monkeypatch.setattr(
        difficulty_preflight,
        "hosted_endpoint_liveness",
        lambda *a: calls.append(a),
    )
    monkeypatch.setattr(difficulty_preflight, "load_queue_sources", lambda *a, **k: [])
    preflight_batch(QUEUE, QUEUE_SHA, None)
    assert calls == []


def test_load_queue_sources_filters_and_verifies(tmp_path, difficulty_source):
    source = difficulty_source(_task())
    queue_path = tmp_path / "sources.json"
    unrelated = source.model_dump(mode="json")
    unrelated["declaration"].update(id="Unrelated:source", status="Pending", stage=None)
    queue_path.write_text(json.dumps([unrelated, source.model_dump(mode="json")]))
    digest = file_sha256(queue_path)
    selected = load_queue_sources(str(queue_path), digest, (source.declaration.id,))
    assert [row.declaration.id for row in selected] == [source.declaration.id]
    with pytest.raises(ValueError, match="Pending source carries no PendingStage"):
        load_queue_sources(str(queue_path), digest, ("Unrelated:source",))
    with pytest.raises(ValueError, match="Pending source carries no PendingStage"):
        load_queue_sources(str(queue_path), digest, None)
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


def _prepared_source(source, judge_model: str | None):
    """A prepared source whose single staged task declares the given judge."""
    return difficulty_prepared.PreparedSource(
        sample=source.sample,
        sample_receipt=source.sample_receipt,
        tasks=[
            difficulty_prepared.PreparedTask(
                task_id=source.sample.tasks[0].task_id,
                runtime_sha256="0" * 64,
                path="fixture/task",
                workload="task_spec",
                backend=source.backend,
                reason=None,
                judge_model=judge_model,
            )
        ],
    )


def test_preflight_refuses_a_queue_mixing_judge_tiers(tmp_path, monkeypatch, difficulty_source):
    """large-v21 spent a submission plus its prepare cost before the campaign's
    own tier check fired inside Iris. Staging knows every source's declared
    judge, so a queue mixing LIGHT and MEDIUM judges refuses before any submit."""
    task = _task()
    light_source = difficulty_source(task)
    medium_source = difficulty_source(task)
    prepared = {
        id(light_source): _prepared_source(light_source, "openai/gpt-oss-120b"),
        id(medium_source): _prepared_source(medium_source, "MiniMaxAI/MiniMax-M3-MXFP8"),
    }
    monkeypatch.setattr(difficulty_preflight, "load_queue_sources", lambda *a, **k: [light_source, medium_source])
    monkeypatch.setattr(difficulty_preflight, "panel_task_runtimes", lambda panel: ([task], [{}]))
    monkeypatch.setattr(difficulty_preflight, "prepare_source", lambda source, root: prepared[id(source)])
    with pytest.raises(ValueError, match="split the source queue"):
        preflight_batch(QUEUE, QUEUE_SHA, None, stage_root=tmp_path / "tasks")


def test_preflight_records_a_single_judge_tier(tmp_path, monkeypatch, difficulty_source):
    """A queue whose staged sources all declare one hosted tier passes and the
    receipt names the tier the campaign must host."""
    task = _task()
    source = difficulty_source(task)
    monkeypatch.setattr(difficulty_preflight, "load_queue_sources", lambda *a, **k: [source])
    monkeypatch.setattr(difficulty_preflight, "panel_task_runtimes", lambda panel: ([task], [{}]))
    monkeypatch.setattr(
        difficulty_preflight,
        "prepare_source",
        lambda source, root: _prepared_source(source, "MiniMaxAI/MiniMax-M3-MXFP8"),
    )
    receipt = preflight_batch(QUEUE, QUEUE_SHA, None, stage_root=tmp_path / "tasks")
    assert receipt["judge_tier"] == "MEDIUM"
    assert receipt["failures"] == []


def test_preflight_records_the_relay_route_without_a_hosted_tier(tmp_path, monkeypatch, difficulty_source):
    """GLM-judged and unjudged sources share the relay route; the receipt names it."""
    task = _task()
    source = difficulty_source(task)
    monkeypatch.setattr(difficulty_preflight, "load_queue_sources", lambda *a, **k: [source])
    monkeypatch.setattr(difficulty_preflight, "panel_task_runtimes", lambda panel: ([task], [{}]))
    monkeypatch.setattr(
        difficulty_preflight,
        "prepare_source",
        lambda source, root: _prepared_source(source, GLM_MODEL),
    )
    receipt = preflight_batch(QUEUE, QUEUE_SHA, None, stage_root=tmp_path / "tasks")
    assert receipt["judge_tier"] == "relay"


def test_preflight_skips_the_tier_gate_without_staging(tmp_path, monkeypatch, difficulty_source):
    """Receipt-stage-only mode reads no prepared tasks, so it records no tier."""
    task = _task()
    source = difficulty_source(task)
    monkeypatch.setattr(difficulty_preflight, "load_queue_sources", lambda *a, **k: [source])
    monkeypatch.setattr(difficulty_preflight, "panel_task_runtimes", lambda panel: ([task], [{}]))
    receipt = preflight_batch(QUEUE, QUEUE_SHA, None)
    assert "judge_tier" not in receipt
