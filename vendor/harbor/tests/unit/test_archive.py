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
    ARCHIVE_SAMPLES_TABLE,
    ARCHIVE_STEPS_TABLE,
    EvaluationStore,
    sample_from_archive_row,
)
from finestore.reader import ReadView

from harbor.archive import ProcessTrialArchive, TrialArchive, open_archive
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
) -> TrialHookEvent:
    trial_dir = tmp_path / f"trial-{trial_id}"
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


def test_end_hook_archives_sample_native_result_trajectory_and_steps(tmp_path: Path):
    root = str(tmp_path / "archive")
    archive = _open_archive(root)
    event = _end_event(tmp_path, reward=1.0)
    assert event.result is not None
    trial_dir = event.result.resolve_trial_path(None)
    (trial_dir / "result.json").write_text(
        json.dumps({"trial_name": "stale-resume-result"})
    )
    asyncio.run(archive.on_trial_end(event))
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
    asyncio.run(
        archive.on_trial_end(_end_event(tmp_path, reward=0.0, trajectory=False))
    )
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
    assert view.read_blob("sources/harbor/survivor/result.json") is not None


def test_finalize_preserves_harbor_job_files(tmp_path: Path):
    root = str(tmp_path / "archive")
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    for name in ("config.json", "lock.json", "result.json"):
        (job_dir / name).write_text(json.dumps({"source": name}))
    archive = _open_archive(root, trials_dir=job_dir)
    event = _end_event(tmp_path, reward=1.0, trajectory=False)
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


def test_archive_failure_is_reported_during_finalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    archive = TrialArchive(
        EvaluationStore.open(str(tmp_path / "archive"), writer_id="failure-test"),
        "mybench",
        None,
    )
    event = _end_event(tmp_path, reward=1.0, trajectory=False)
    assert event.result is not None

    def fail_sample_write(*_args, **_kwargs):
        raise OSError("archive unavailable")

    monkeypatch.setattr(EvaluationStore, "add_sample", fail_sample_write)
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
