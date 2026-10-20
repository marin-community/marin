import asyncio
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from harbor.models.job.config import JobConfig
from harbor.models.job.result import JobResult, JobStats
from harbor.models.trial.config import TaskConfig, TrialConfig
from harbor.models.trial.result import AgentInfo, ExceptionInfo, TrialResult
from harbor.models.verifier.result import VerifierResult
from harbor.tasks.client import TaskClient

from skyrl_train.inline_harbor import HarborTaskCache, validation_metrics
from skyrl_train.inline_harbor_client import HarborTrialClient


@pytest.mark.asyncio
async def test_cohort_cache_shares_pinned_inputs_across_concurrent_trials(tmp_path, monkeypatch):
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    for index in range(3):
        task = repository / f"task-{index}"
        task.mkdir()
        (task / "instruction.md").write_text(f"original task {index}")
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "Original tasks",
        ],
        check=True,
    )
    revision = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    (repository / "task-0" / "instruction.md").write_text("changed task")
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qam",
            "Changed task",
        ],
        check=True,
    )
    config = JobConfig(
        tasks=[
            TaskConfig(
                path=f"task-{index}",
                git_url=repository.as_uri(),
                git_commit_id=revision,
                download_dir=tmp_path / "cache",
                overwrite=True,
            )
            for index in range(3)
        ]
    )
    original_exec = asyncio.create_subprocess_exec
    fetches = []

    async def git_process(*args, **kwargs):
        if args[:2] == ("git", "fetch"):
            fetches.append(args)
        return await original_exec(*args, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", git_process)
    cache = HarborTaskCache()
    prepared = await asyncio.gather(*(cache.prepare(config) for _ in range(3)))
    # These are real external Git operations, not internal helper calls.
    assert len(fetches) == 1
    repository.rename(tmp_path / "unavailable-repository")
    for index, ready in enumerate(prepared):
        selected = ready.tasks[index]
        result = await TaskClient().download_tasks(
            [selected.get_task_id()], overwrite=selected.overwrite, output_dir=selected.download_dir
        )
        assert result.results[0].cached
        assert (result.paths[0] / "instruction.md").read_text() == f"original task {index}"
        assert selected.get_task_id() == config.tasks[index].get_task_id()
    assert all(task.overwrite for task in config.tasks)


@pytest.mark.asyncio
async def test_worker_preserves_task_download_stderr(tmp_path):
    cohort = {
        "job": {
            "tasks": [
                {
                    "path": "task",
                    "git_url": (tmp_path / "missing-repository").as_uri(),
                    "git_commit_id": "a" * 40,
                    "download_dir": str(tmp_path / "cache"),
                }
            ]
        },
        "reward_key": "reward",
    }
    async with HarborTrialClient(os.environ.copy()) as client:
        with pytest.raises(RuntimeError, match="Subprocess stderr:") as caught:
            await client.run(cohort, 0)
    stderr = str(caught.value).split("Subprocess stderr:", 1)[1]
    assert "does not appear to be a git repository" in stderr


@pytest.mark.asyncio
async def test_worker_runs_outside_checkout_with_launcher_package_shadowed(tmp_path, monkeypatch):
    # The task image can supply an older launcher while the trainer comes from
    # the pinned source overlay. The worker must use that same trainer package.
    launcher = tmp_path / "marinskyrl"
    launcher.mkdir()
    (launcher / "__init__.py").write_text("")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join([str(tmp_path), str(Path(__file__).resolve().parents[2])])
    monkeypatch.chdir(tmp_path)
    async with asyncio.timeout(60), HarborTrialClient(environment) as client:
        with pytest.raises(RuntimeError, match="explicit tasks"):
            await client.run({"job": {"tasks": []}, "reward_key": "reward"}, 0)


def test_validation_scores_preserve_infrastructure_missingness():
    tasks = [
        TaskConfig(path=f"task-{index}", git_url="https://example.com/tasks", git_commit_id="a" * 40)
        for index in range(4)
    ]
    trials = [
        TrialResult(
            task_name=f"task-{index}",
            trial_name=f"trial-{index}",
            trial_uri=f"/tmp/trial-{index}",
            task_id=task.get_task_id(),
            task_checksum="checksum",
            config=TrialConfig(task=task),
            agent_info=AgentInfo(name="terminus-2", version="test"),
            verifier_result=VerifierResult(rewards={"reward": score}),
        )
        for index, (task, score) in enumerate(zip(tasks, [1, 0, 1, 0], strict=True))
    ]
    trials[2].exception_info = ExceptionInfo.from_exception(ConnectionError("sandbox unavailable"))
    trials[3].exception_info = ExceptionInfo(
        exception_type="AgentTimeoutError",
        exception_message="budget exhausted",
        exception_traceback="",
        occurred_at=datetime.now(timezone.utc),
    )
    result = JobResult(
        id=uuid4(), started_at=datetime.now(timezone.utc), n_total_trials=4, stats=JobStats(), trial_results=trials
    )
    config = JobConfig(tasks=tasks)
    metrics = validation_metrics(result, config, "reward")
    assert metrics == {
        "num_attempted": 4.0,
        "num_scored": 3.0,
        "reward_sum": 1.0,
        "avg_verifier_score": pytest.approx(1 / 3),
    }
    result.trial_results = trials[:-1] + [trials[0]]
    with pytest.raises(ValueError, match="duplicate"):
        validation_metrics(result, config, "reward")


@pytest.mark.asyncio
async def test_control_socket_multiplexes_errors_without_parsing_harbor_stdout():
    async with HarborTrialClient(os.environ.copy()) as client:
        results = await asyncio.gather(
            client.run({"job": {"tasks": []}, "reward_key": "reward"}, 0),
            client.run({"job": {"tasks": []}, "reward_key": "reward"}, 1),
            return_exceptions=True,
        )
        client.process.terminate()
        await client.process.wait()
        async with asyncio.timeout(5):
            with pytest.raises(RuntimeError):
                await client.run({"job": {"tasks": []}, "reward_key": "reward"}, 2)
    assert len(results) == 2
    assert all(isinstance(result, RuntimeError) and "explicit tasks" in str(result) for result in results)
