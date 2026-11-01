# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for Harbor's native FineStore evaluation output."""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pytest
from finestore.eval import (
    ARCHIVE_ROLLOUTS_TABLE,
    ARCHIVE_SAMPLES_TABLE,
    ARCHIVE_STEPS_TABLE,
    EvaluationStore,
    RolloutRecord,
    sample_from_archive_row,
)
from finestore.reader import ReadView

from harbor.archive import ProcessTrialArchive, TrialArchive, open_archive
from harbor.models.agent.context import AgentContext
from harbor.models.trial.config import TaskConfig, TrialConfig
from harbor.models.trial.result import AgentInfo, ExceptionInfo, TrialResult
from harbor.models.verifier.result import VerifierResult
from harbor.trial.hooks import TrialEvent, TrialHookEvent
from harbor_config.models.job.config import ArchiveConfig, JobConfig


def _end_event(
    tmp_path: Path,
    *,
    reward: float | None,
    task_name: str = "prob-1",
    trial_id: str = "t0",
    trajectory: bool = True,
    exception_type: str | None = None,
    attempt: int | None = None,
) -> TrialHookEvent:
    trial_dir = tmp_path / f"trial-{trial_id}"
    if attempt is not None:
        trial_dir = trial_dir / "attempts" / f"{attempt:03d}"
    if trajectory:
        agent_dir = trial_dir / "agent"
        agent_dir.mkdir(parents=True)
        (agent_dir / "trajectory.json").write_text(
            json.dumps(
                {
                    "agent": {"name": "a", "version": "1.0"},
                    "steps": [
                        {
                            "step_id": 1,
                            "source": "agent",
                            "model_name": "m",
                            "message": "the answer",
                            "metrics": {"prompt_tokens": 5, "completion_tokens": 3},
                        }
                    ],
                }
            )
        )
    else:
        trial_dir.mkdir(parents=True)
    config = TrialConfig(
        task=TaskConfig(path=Path("/test/task")), trial_name=trial_id, job_id=uuid4()
    )
    result = TrialResult(
        task_name=task_name,
        trial_name=trial_id,
        trial_uri=trial_dir.as_uri(),
        trial_relpath=trial_dir.relative_to(tmp_path).as_posix(),
        task_id=config.task.get_task_id(),
        task_checksum="abc123",
        config=config,
        agent_info=AgentInfo(name="a", version="1.0"),
        verifier_result=(
            VerifierResult(rewards={"reward": reward}) if reward is not None else None
        ),
        exception_info=(
            ExceptionInfo(
                exception_type=exception_type,
                exception_message="trial failed",
                exception_traceback="",
                occurred_at=datetime.now(),
            )
            if exception_type
            else None
        ),
    )
    (trial_dir / "result.json").write_text(result.model_dump_json(indent=4))
    return TrialHookEvent(
        event=TrialEvent.END,
        trial_id=trial_id,
        task_name=task_name,
        config=config,
        result=result,
    )


def _open_archive(root: str, *, trials_dir: Path | None = None) -> ProcessTrialArchive:
    archive = open_archive(
        ArchiveConfig(root=root, dataset="mybench"), trials_dir=trials_dir
    )
    assert archive is not None
    return archive


def _sample_rows(root: str) -> list[dict]:
    table = ReadView(root).scan(ARCHIVE_SAMPLES_TABLE)
    if table is None:
        return []
    return table.to_pylist(maps_as_pydicts="strict")


def test_finalize_archives_sample_native_result_trajectory_and_steps(tmp_path: Path):
    root = str(tmp_path / "archive")
    archive = _open_archive(root)
    event = _end_event(tmp_path, reward=1.0)
    assert event.result is not None
    trial_dir = event.result.resolve_trial_path(None)
    (trial_dir / "result.json").write_text(
        json.dumps({"trial_name": "stale-resume-result"})
    )
    asyncio.run(archive.on_trial_end(event))
    asyncio.run(archive.finalize([event.result]))
    archive.seal()
    archive.close()

    (row,) = _sample_rows(root)
    sample = sample_from_archive_row(row)
    assert sample.task == "mybench"
    assert sample.doc_id == "prob-1"
    assert sample.correct is True
    assert sample.metrics["reward"] == 1.0
    assert sample.trajectory_uri is not None

    view = ReadView(root)
    steps_table = view.scan(ARCHIVE_STEPS_TABLE)
    assert steps_table is not None
    steps = steps_table.to_pylist()
    assert len(steps) == 1
    assert steps[0]["prompt_tokens"] == 5
    native_result = view.read_blob("sources/harbor/t0/result.json")
    assert native_result is not None
    assert json.loads(native_result)["trial_name"] == "t0"


def test_failed_trial_below_threshold_is_incorrect_without_trajectory(tmp_path: Path):
    root = str(tmp_path / "archive")
    archive = _open_archive(root)
    event = _end_event(tmp_path, reward=0.0, trajectory=False)
    asyncio.run(archive.on_trial_end(event))
    asyncio.run(archive.finalize([event.result]))
    archive.seal()
    archive.close()

    sample = sample_from_archive_row(_sample_rows(root)[0])
    assert sample.correct is False
    assert sample.metrics["reward"] == 0.0
    assert sample.trajectory_uri is None


@pytest.mark.parametrize("reward", [0.0, 1.0])
def test_archive_preserves_surrogate_trace_and_verifier_reward(tmp_path: Path, reward):
    root = str(tmp_path / "archive")
    archive = _open_archive(root)
    event = _end_event(
        tmp_path, reward=reward, exception_type="MalformedModelResponseError"
    )
    path = event.result.resolve_trial_path(None) / "agent" / "trajectory.json"
    trajectory = json.loads(path.read_bytes())
    trajectory["steps"][0]["message"] = "Analysis: \ud83c café 🦖"
    trajectory["steps"][0]["reasoning_content"] = "\ud842"
    raw = json.dumps(trajectory).encode()
    path.write_bytes(raw)

    asyncio.run(archive.on_trial_end(event))
    asyncio.run(archive.finalize([event.result]))
    archive.seal()
    archive.close()

    view = ReadView(root)
    assert view.is_sealed()
    sample = sample_from_archive_row(_sample_rows(root)[0])
    assert sample.correct is (reward == 1.0)
    assert sample.metrics["reward"] == reward
    assert view.read_blob("t0/trajectory.json") == raw
    steps = view.scan(ARCHIVE_STEPS_TABLE).to_pylist()
    assert steps[0]["message"] == "Analysis: \\ud83c café 🦖"
    assert steps[0]["reasoning_content"] == "\\ud842"


def test_archive_resolves_trajectory_from_relocated_trial_root(tmp_path: Path):
    root = str(tmp_path / "archive")
    archive = _open_archive(root, trials_dir=tmp_path)
    event = _end_event(tmp_path, reward=1.0)
    assert event.result is not None
    event.result = event.result.model_copy(
        update={
            "trial_uri": (tmp_path / "missing-original" / "trial-t0").as_uri(),
            "trial_relpath": "trial-t0",
        }
    )

    asyncio.run(archive.on_trial_end(event))
    asyncio.run(archive.finalize([event.result]))
    archive.seal()
    archive.close()

    sample = sample_from_archive_row(_sample_rows(root)[0])
    assert sample.trajectory_uri is not None


def test_finalize_archives_only_the_surviving_ungraded_attempt(tmp_path: Path):
    root = str(tmp_path / "archive")
    archive = _open_archive(root)
    abandoned = _end_event(
        tmp_path,
        reward=None,
        trial_id="abandoned",
        trajectory=False,
        exception_type="TrialTimeoutError",
    )
    survivor = _end_event(
        tmp_path,
        reward=None,
        trial_id="survivor",
        trajectory=False,
        exception_type="VerificationNotCompletedError",
    )

    asyncio.run(archive.on_trial_end(abandoned))
    assert survivor.result is not None
    asyncio.run(archive.finalize([survivor.result]))
    archive.seal()
    archive.close()

    (row,) = _sample_rows(root)
    sample = sample_from_archive_row(row)
    assert row["trial_id"] == "survivor"
    assert sample.grading is not None
    assert sample.grading.score is None
    assert sample.correct is None
    view = ReadView(root)
    assert view.read_blob("sources/harbor/abandoned/result.json") is None
    assert (
        view.read_blob(
            f"sources/harbor/abandoned/attempts/{abandoned.result.id}/result.json"
        )
        is not None
    )
    assert view.read_blob("sources/harbor/survivor/result.json") is not None


@pytest.mark.parametrize(
    "final_reward,final_error", [(1.0, None), (None, "TrialTimeoutError")]
)
def test_retry_preserves_native_attempts_and_normalizes_only_selected_result(
    tmp_path: Path, final_reward: float | None, final_error: str | None
):
    root = str(tmp_path / "archive")
    archive = _open_archive(root, trials_dir=tmp_path)
    abandoned = _end_event(
        tmp_path,
        reward=0.0,
        exception_type="NonZeroAgentExitCodeError",
        attempt=0,
    )
    selected = _end_event(
        tmp_path, reward=final_reward, exception_type=final_error, attempt=1
    )
    expected_sources = {}
    for event, count in ((abandoned, 35), (selected, 3)):
        result = event.result
        assert result is not None
        result.agent_result = AgentContext(metadata={"native_attempt": str(result.id)})
        result.secondary_exception_info = [
            ExceptionInfo(
                exception_type="CleanupError",
                exception_message="native cleanup evidence",
                exception_traceback="cleanup traceback",
                occurred_at=datetime.now(),
            )
        ]
        if result.verifier_result is not None:
            result.verifier_result.stdout = "native verifier output"
        path = result.resolve_trial_path(tmp_path) / "agent" / "trajectory.json"
        trajectory = json.loads(path.read_bytes())
        trajectory["steps"] = [
            {
                "step_id": step_id,
                "source": "agent",
                "model_name": "m",
                "message": f"{result.id}: step {step_id}",
            }
            for step_id in range(1, count + 1)
        ]
        raw = json.dumps(trajectory).encode()
        path.write_bytes(raw)
        expected_sources[result.id] = (result.model_dump_json(indent=4).encode(), raw)
        asyncio.run(archive.on_trial_end(event))

    assert selected.result is not None
    # Late filesystem writes and the queue's slimmed return value must not replace
    # the full native snapshot captured by END.
    path = selected.result.resolve_trial_path(tmp_path) / "agent" / "trajectory.json"
    path.write_text("{}")
    asyncio.run(archive.finalize([selected.result.slimmed()]))
    archive.seal()
    archive.close()

    view = ReadView(root)
    assert view.is_sealed()
    (row,) = _sample_rows(root)
    assert (row["doc_id"], row["trial_id"]) == ("prob-1", "t0")
    sample = sample_from_archive_row(row)
    assert sample.grading.score == final_reward
    assert sample.correct is (True if final_reward is not None else None)
    assert sample.metrics == ({"reward": final_reward} if final_reward is not None else {})
    if final_error is not None:
        assert json.loads(sample.grading.detail)["error"]["message"] == "trial failed"
    steps = view.scan(ARCHIVE_STEPS_TABLE).to_pylist()
    assert [(step["trial_id"], step["step_id"], step["message"]) for step in steps] == [
        ("t0", step_id, f"{selected.result.id}: step {step_id}")
        for step_id in range(1, 4)
    ]
    for attempt_id, (native, raw) in expected_sources.items():
        prefix = f"sources/harbor/t0/attempts/{attempt_id}"
        assert view.read_blob(f"{prefix}/result.json") == native
        assert view.read_blob(f"{prefix}/trajectory.json") == raw
    final_native, final_trajectory = expected_sources[selected.result.id]
    assert view.read_blob("sources/harbor/t0/result.json") == final_native
    assert view.read_blob("t0/trajectory.json") == final_trajectory


@pytest.mark.parametrize("reopened", [False, True])
def test_finalize_restores_full_native_resumed_result_before_normalizing(
    tmp_path: Path, reopened: bool
):
    root = str(tmp_path / "archive")
    event = _end_event(
        tmp_path, reward=0.0, exception_type="NonZeroAgentExitCodeError", attempt=1
    )
    result = event.result
    assert result is not None
    result.agent_result = AgentContext(metadata={"native_payload": "resumed"})
    native = result.model_dump_json(indent=4).encode()
    raw = (result.resolve_trial_path(tmp_path) / "agent" / "trajectory.json").read_bytes()
    (result.resolve_trial_path(tmp_path) / "result.json").write_bytes(native)
    if reopened:
        archive = _open_archive(root, trials_dir=tmp_path)
        asyncio.run(archive.on_trial_end(event))
        asyncio.run(archive.finalize([result]))
        archive.seal()
        archive.close()
        (result.resolve_trial_path(tmp_path) / "result.json").unlink()
        (result.resolve_trial_path(tmp_path) / "agent" / "trajectory.json").unlink()
    archive = _open_archive(root, trials_dir=tmp_path)

    asyncio.run(archive.finalize([result.slimmed()]))
    archive.seal()
    archive.close()

    view = ReadView(root)
    assert view.read_blob("sources/harbor/t0/result.json") == native
    assert view.read_blob(f"sources/harbor/t0/attempts/{result.id}/result.json") == native
    assert view.read_blob(f"sources/harbor/t0/attempts/{result.id}/trajectory.json") == raw
    assert view.read_blob("t0/trajectory.json") == raw
    assert len(view.scan(ARCHIVE_STEPS_TABLE).to_pylist()) == 1
    sample = sample_from_archive_row(_sample_rows(root)[0])
    assert json.loads(sample.grading.detail)["error"]["message"] == "trial failed"


@pytest.mark.parametrize("replacement_trial", ["t0", "replacement"])
def test_reopened_archive_rebuilds_selected_rows_and_preserves_native_attempts(
    tmp_path: Path, replacement_trial: str
):
    root = str(tmp_path / "archive")
    original = _end_event(tmp_path, reward=0.0, attempt=0)
    removed = _end_event(
        tmp_path, reward=0.0, task_name="old-doc", trial_id="removed", attempt=0
    )
    assert original.result is not None
    assert removed.result is not None
    path = original.result.resolve_trial_path(None) / "agent" / "trajectory.json"
    trajectory = json.loads(path.read_bytes())
    trajectory["steps"] = [
        {"step_id": index, "source": "agent", "message": f"old step {index}"}
        for index in range(1, 36)
    ]
    original_trace = json.dumps(trajectory).encode()
    path.write_bytes(original_trace)
    original_native = original.result.model_dump_json(indent=4).encode()
    archive = _open_archive(root, trials_dir=tmp_path)
    for event in (original, removed):
        asyncio.run(archive.on_trial_end(event))
    asyncio.run(archive.finalize([original.result, removed.result]))
    archive.seal()
    archive.close()
    with EvaluationStore.open(root, writer_id="derived-rollouts") as store:
        store.add_rollouts(
            [
                RolloutRecord(
                    task="mybench",
                    doc_id="old-doc",
                    trial_id="removed",
                    turn_id=0,
                    part_id=0,
                    conversation_type="agentic",
                    participant_type="assistant",
                    participant_id="m",
                    content_type="message",
                    content="stale derived rollout",
                )
            ]
        )
        store.seal()

    replacement = _end_event(
        tmp_path, reward=1.0, trial_id=replacement_trial, attempt=1
    )
    assert replacement.result is not None
    archive = _open_archive(root, trials_dir=tmp_path)
    reopened = ReadView(root)
    assert not reopened.is_sealed()
    assert _sample_rows(root) == []
    assert reopened.scan(ARCHIVE_STEPS_TABLE) is None
    assert reopened.scan(ARCHIVE_ROLLOUTS_TABLE) is None
    asyncio.run(archive.on_trial_end(replacement))
    asyncio.run(archive.finalize([replacement.result.slimmed()]))
    archive.seal()
    archive.close()

    view = ReadView(root)
    assert view.is_sealed()
    (row,) = _sample_rows(root)
    assert (row["doc_id"], row["trial_id"]) == ("prob-1", replacement_trial)
    assert sample_from_archive_row(row).metrics == {"reward": 1.0}
    steps = view.scan(ARCHIVE_STEPS_TABLE).to_pylist()
    assert [(step["trial_id"], step["step_id"], step["message"]) for step in steps] == [
        (replacement_trial, 1, "the answer")
    ]
    assert view.scan(ARCHIVE_ROLLOUTS_TABLE) is None
    prefix = f"sources/harbor/t0/attempts/{original.result.id}"
    assert view.read_blob(f"{prefix}/result.json") == original_native
    assert view.read_blob(f"{prefix}/trajectory.json") == original_trace
    assert view.read_blob(
        f"sources/harbor/removed/attempts/{removed.result.id}/result.json"
    ) == removed.result.model_dump_json(indent=4).encode()


@pytest.mark.parametrize("changed", [False, True])
def test_repeated_end_preserves_immutable_attempt_and_reports_changed_content(
    tmp_path: Path, changed: bool
):
    root = str(tmp_path / "archive")
    archive = _open_archive(root)
    event = _end_event(tmp_path, reward=1.0)
    result = event.result
    assert result is not None
    native = result.model_dump_json(indent=4).encode()
    asyncio.run(archive.on_trial_end(event))
    if changed:
        result.verifier_result = VerifierResult(rewards={"reward": 0.0})
    asyncio.run(archive.on_trial_end(event))

    if changed:
        with pytest.raises(ExceptionGroup) as exc_info:
            asyncio.run(archive.finalize([result]))
        assert any("immutable attempt" in str(error) for error in exc_info.value.exceptions)
    else:
        asyncio.run(archive.finalize([result]))
        archive.seal()
    archive.close()

    view = ReadView(root)
    assert view.is_sealed() is not changed
    assert view.read_blob(f"sources/harbor/t0/attempts/{result.id}/result.json") == native
    (row,) = _sample_rows(root)
    assert sample_from_archive_row(row).metrics == {"reward": 1.0}


@pytest.mark.parametrize("failure", ["missing", "different_id"])
def test_finalize_rejects_missing_or_mismatched_resumed_attempt(
    tmp_path: Path, failure: str
):
    root = str(tmp_path / "archive")
    event = _end_event(tmp_path, reward=1.0)
    result = event.result
    assert result is not None
    path = result.resolve_trial_path(None) / "result.json"
    if failure == "missing":
        path.unlink()
    else:
        path.write_text(result.model_copy(update={"id": uuid4()}).model_dump_json())
    archive = _open_archive(root)

    with pytest.raises(ValueError, match="attempt"):
        asyncio.run(archive.finalize([result.slimmed()]))
    archive.close()

    assert _sample_rows(root) == []
    assert not ReadView(root).is_sealed()


def test_finalize_preserves_harbor_job_files(tmp_path: Path):
    root = str(tmp_path / "archive")
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    for name in ("config.json", "lock.json", "result.json"):
        (job_dir / name).write_text(json.dumps({"source": name}))
    archive = _open_archive(root, trials_dir=job_dir)
    event = _end_event(job_dir, reward=1.0, trajectory=False)
    assert event.result is not None

    asyncio.run(archive.finalize([event.result]))
    archive.seal()
    archive.close()

    view = ReadView(root)
    for name in ("config.json", "lock.json", "result.json"):
        source = view.read_blob(f"sources/harbor/job/{name}")
        assert source is not None
        assert json.loads(source) == {"source": name}


def test_non_end_event_writes_nothing(tmp_path: Path):
    root = str(tmp_path / "archive")
    archive = _open_archive(root)
    event = _end_event(tmp_path, reward=1.0)
    event = event.model_copy(update={"event": TrialEvent.START})
    asyncio.run(archive.on_trial_end(event))
    archive.seal()
    archive.close()

    assert _sample_rows(root) == []


@pytest.mark.parametrize("operation", ["add_sample", "add_source_artifact"])
def test_archive_failure_is_reported_during_finalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
):
    archive = TrialArchive(
        EvaluationStore.open(str(tmp_path / "archive"), writer_id="failure-test"),
        "mybench",
        None,
        root=str(tmp_path / "archive"),
    )
    event = _end_event(tmp_path, reward=1.0, trajectory=False)
    assert event.result is not None

    def fail_archive_write(*_args, **_kwargs):
        raise OSError("archive unavailable")

    monkeypatch.setattr(EvaluationStore, operation, fail_archive_write)
    asyncio.run(archive.on_trial_end(event))

    with pytest.raises(
        ExceptionGroup, match="FineStore evaluation archive writes failed"
    ) as exc_info:
        asyncio.run(archive.finalize([event.result]))
    assert any(isinstance(error, OSError) for error in exc_info.value.exceptions)

    archive.close()


def test_invalid_trajectory_is_reported_during_finalization(tmp_path: Path):
    archive = _open_archive(str(tmp_path / "archive"))
    event = _end_event(tmp_path, reward=1.0)
    assert event.result is not None
    trial_dir = event.result.resolve_trial_path(None)
    (trial_dir / "agent" / "trajectory.json").write_text("{}")

    asyncio.run(archive.on_trial_end(event))

    with pytest.raises(
        ExceptionGroup, match="FineStore evaluation archive writes failed"
    ):
        asyncio.run(archive.finalize([event.result]))

    archive.close()


def test_archive_config_survives_job_config_round_trip():
    config = JobConfig(
        tasks=[],
        archive=ArchiveConfig(
            root="s3://eval-archives/run-1",
            dataset="mybench",
            writer_id="worker-1",
        ),
    )

    restored = JobConfig.model_validate_json(config.model_dump_json())

    assert restored.archive == config.archive
