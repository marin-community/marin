from pathlib import Path

import pytest
from pydantic import ValidationError

import harbor.hugging_face_dataset
from harbor.job import Job
from harbor.models.benchmark import BenchmarkMetadata, MetricKind, MetricMetadata
from harbor.models.job.config import DatasetConfig, JobConfig


def _write_task(root: Path, name: str) -> None:
    task = root / name
    task.mkdir()
    (task / "task.toml").write_text(f'[task]\nname = "test/{name}"\n')
    (task / "instruction.md").write_text("Complete the task.\n")
    environment = task / "environment"
    environment.mkdir()
    (environment / "Dockerfile").write_text("FROM ubuntu:22.04\n")
    tests = task / "tests"
    tests.mkdir()
    (tests / "test.sh").write_text("#!/bin/sh\nexit 0\n")


@pytest.mark.unit
@pytest.mark.asyncio
async def test_job_describes_dataset_before_and_after_task_cap(tmp_path: Path) -> None:
    dataset = tmp_path / "benchmark"
    dataset.mkdir()
    for name in ("task-a", "task-b", "task-c"):
        _write_task(dataset, name)

    config = JobConfig(
        job_name="metadata-test",
        jobs_dir=tmp_path / "jobs",
        datasets=[DatasetConfig(path=dataset, task_names=["task-*"], n_tasks=1)],
    )
    job = await Job.create(config)

    try:
        assert [
            metadata.model_dump(mode="json") for metadata in job.benchmark_metadata
        ] == [
            {
                "schema_version": 1,
                "task": "benchmark",
                "primary_metric": "reward",
                "metric_kind": "continuous",
                "metrics": [
                    {
                        "name": "reward",
                        "source_name": "reward",
                        "kind": "continuous",
                        "higher_is_better": True,
                    }
                ],
                "n_benchmark": 3,
                "n_attempted": 1,
            }
        ]
        assert job.benchmark_metadata[0].metric_kind is MetricKind.CONTINUOUS
        assert len(job) == 1
    finally:
        job._close_logger_handlers()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_job_resolves_hugging_face_task_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit = "a" * 40
    snapshot = tmp_path / "snapshots" / commit
    snapshot.mkdir(parents=True)
    for name in ("task-a", "task-b"):
        _write_task(snapshot, name)
    (snapshot / "README.md").write_text("Harbor tasks\n")

    def download(**kwargs: object) -> str:
        assert kwargs["repo_id"] == "example/tasks"
        assert kwargs["repo_type"] == "dataset"
        assert kwargs["revision"] == "main"
        return str(snapshot)

    monkeypatch.setattr(harbor.hugging_face_dataset, "snapshot_download", download)
    config = JobConfig(
        job_name="hf-metadata-test",
        jobs_dir=tmp_path / "jobs",
        datasets=[DatasetConfig(name="hf://example/tasks", ref="main", n_tasks=1)],
    )
    job = await Job.create(config)

    try:
        assert job.benchmark_metadata[0].task == "hf://example/tasks"
        assert job.benchmark_metadata[0].n_benchmark == 2
        assert job.benchmark_metadata[0].n_attempted == 1
        assert len(job) == 1
        assert config.datasets[0].ref == commit
    finally:
        job._close_logger_handlers()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_hugging_face_task_source_preserves_repository_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = tmp_path / ("b" * 40)
    snapshot.mkdir()
    _write_task(snapshot, "task-a")
    monkeypatch.setattr(
        harbor.hugging_face_dataset, "snapshot_download", lambda **_: str(snapshot)
    )

    tasks = await DatasetConfig(name="hf://example/tasks").get_task_configs()

    assert [task.source for task in tasks] == ["hf://example/tasks"]
    assert [task.path.name for task in tasks if task.path is not None] == ["task-a"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_hugging_face_task_dataset_materializes_snapshot_symlinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit = "c" * 40
    snapshot = tmp_path / "repository" / "snapshots" / commit
    snapshot.mkdir(parents=True)
    _write_task(snapshot, "task-a")

    task_config = snapshot / "task-a" / "environment" / "task_config.json"
    blob = tmp_path / "repository" / "blobs" / "task-config"
    blob.parent.mkdir()
    blob.write_text('{"model": "test-model"}\n')
    task_config.symlink_to(blob)

    monkeypatch.setattr(
        harbor.hugging_face_dataset, "snapshot_download", lambda **_: str(snapshot)
    )

    config = DatasetConfig(name="hf://example/tasks", ref=commit)
    tasks = await config.get_task_configs()

    assert config.ref == commit
    assert len(tasks) == 1
    assert tasks[0].path is not None
    materialized_config = tasks[0].path / "environment" / "task_config.json"
    assert not materialized_config.is_symlink()
    assert materialized_config.read_text() == '{"model": "test-model"}\n'


@pytest.mark.unit
@pytest.mark.parametrize("name", ["hf://", "hf://example", "hf://example/tasks/extra"])
def test_hugging_face_dataset_rejects_invalid_repository_name(name: str) -> None:
    with pytest.raises(ValidationError):
        DatasetConfig(name=name)


@pytest.mark.unit
def test_hugging_face_dataset_requires_ref_for_revision() -> None:
    with pytest.raises(ValidationError):
        DatasetConfig(name="hf://example/tasks", version="main")


def test_benchmark_metadata_rejects_inconsistent_primary_metric() -> None:
    with pytest.raises(
        ValidationError, match="primary_metric must name one of metrics"
    ):
        BenchmarkMetadata(
            task="benchmark",
            primary_metric="accuracy",
            metric_kind=MetricKind.BINARY,
            metrics=(
                MetricMetadata(
                    name="reward",
                    source_name="reward",
                    kind=MetricKind.CONTINUOUS,
                    higher_is_better=True,
                ),
            ),
            n_benchmark=3,
            n_attempted=1,
        )
