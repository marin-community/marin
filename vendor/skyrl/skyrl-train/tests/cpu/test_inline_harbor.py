import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from harbor.models.job.config import JobConfig
from harbor.models.job.result import JobResult, JobStats
from harbor.models.trial.config import TaskConfig, TrialConfig
from harbor.models.trial.result import AgentInfo, ExceptionInfo, TrialResult
from harbor.models.verifier.result import VerifierResult

from skyrl_train.inline_harbor import validation_metrics
from skyrl_train.inline_harbor_client import HarborTrialClient
from skyrl_train.callbacks.builtin import HarborEvaluationCallback


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


@pytest.mark.parametrize("agent_error", ["AgentTimeoutError", "TurnCapExhaustedError", "ModelInitiatedTmuxFailure"])
@pytest.mark.parametrize("unscored_error", ["ConnectionError", "VerifierRuntimeError", "VerifierTimeoutError"])
def test_validation_scores_preserve_infrastructure_missingness(agent_error, unscored_error):
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
    trials[2].exception_info = ExceptionInfo(
        exception_type=unscored_error,
        exception_message="unverified outcome",
        exception_traceback="",
        occurred_at=datetime.now(timezone.utc),
    )
    trials[3].exception_info = ExceptionInfo(
        exception_type=agent_error,
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
    # A terminal agent error never manufactures a zero or discards a verified success.
    trials[3].verifier_result.rewards["reward"] = 1.0
    assert validation_metrics(result, config, "reward")["reward_sum"] == 2.0
    trials[3].verifier_result = None
    assert validation_metrics(result, config, "reward")["num_scored"] == 2.0
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


@pytest.mark.asyncio
@pytest.mark.parametrize("delivery", ["file", "environment"])
async def test_callback_passes_secret_name_and_preserves_inherited_api_environment(tmp_path, monkeypatch, delivery):
    monkeypatch.setenv("RIGGING_SECRETS_DIR", str(tmp_path))
    monkeypatch.delenv("DAYTONA_EVAL_API_KEY", raising=False)
    monkeypatch.delenv("DAYTONA_API_KEY", raising=False)
    if delivery == "file":
        (tmp_path / "DAYTONA_EVAL_API_KEY").write_text("synthetic-callback-key")
    else:
        monkeypatch.setenv("DAYTONA_EVAL_API_KEY", "synthetic-callback-key")
    callback = HarborEvaluationCallback(
        cohorts=[{"name": "heldout", "job": {"agents": [{}], "environment": {"type": "daytona"}}}],
        timeout=10, minimum_coverage=1.0, daytona_api_key_env="DAYTONA_EVAL_API_KEY",
    )
    requests = []
    async def harbor_boundary(cohorts, pool, environment):
        requests.append((JobConfig.model_validate(cohorts[0]["job"]), environment))
        return {"heldout": {"num_scored": 1.0}}
    # Harbor collection owns the external trial subprocess and remote sandbox calls.
    monkeypatch.setattr("skyrl_train.callbacks.builtin.collect_harbor", harbor_boundary)
    client = SimpleNamespace(enable_http_endpoint=True, model_name="test", http_endpoint_host="127.0.0.1", http_endpoint_port=1234)
    trainer = SimpleNamespace(inference_engine_client=client, cfg=SimpleNamespace(trainer=SimpleNamespace(export_path=str(tmp_path))))
    metrics = await callback.collect_evaluation(SimpleNamespace(global_step=0), trainer=trainer, pool=None)
    config, environment = requests[0]
    assert config.environment.kwargs["api_key_env"] == "DAYTONA_EVAL_API_KEY"
    assert environment["RIGGING_SECRETS_DIR"] == str(tmp_path)
    assert "DAYTONA_API_KEY" not in environment
    if delivery == "file":
        assert "DAYTONA_EVAL_API_KEY" not in environment
    else:
        assert environment["DAYTONA_EVAL_API_KEY"] == os.environ["DAYTONA_EVAL_API_KEY"]
    assert "synthetic-callback-key" not in config.model_dump_json()
    assert metrics == {"eval/harbor/heldout/num_scored": 1.0}
