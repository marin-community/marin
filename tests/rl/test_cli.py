# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
import textwrap
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import click
import pytest
from click.testing import CliRunner
from fray.types import GpuConfig
from iris.client.client import IrisClient, IrisContext, iris_ctx_scope
from iris.cluster.client.job_info import JobInfo, set_job_info
from iris.cluster.types import JobName
from marin.execution.artifact import Artifact
from marin.execution.lazy import OUT, ArtifactStep, apply
from marin.rl.cli import _coordinator_request, rl_build_options
from marin.rl.skyrl import IrisSkyRLExecution
from rigging.timing import Duration


def _step(execution: IrisSkyRLExecution | None = None) -> ArtifactStep[Artifact]:
    runtime_args = {} if execution is None else {"skyrl_execution": execution}
    return ArtifactStep(
        name="checkpoints/test-rl",
        version="2026.09.21",
        artifact_type=Artifact,
        run=lambda _config: None,
        build_config=lambda _ctx: {},
        runtime_args=runtime_args,
    )


def _execution() -> IrisSkyRLExecution:
    return IrisSkyRLExecution(
        cluster="cw-rno2a",
        cluster_config="lib/iris/config/cw-rno2a.yaml",
        cpu=64,
        memory="128GB",
        disk="512GB",
        priority="interactive",
        max_retries=2,
        target_cluster="cw-rno2a",
        parent_cluster_config="lib/iris/config/marin.yaml",
        coordinator_timeout_hours=18,
    )


def test_coordinator_request_replays_the_experiment_main(tmp_path, monkeypatch) -> None:
    cluster_config, request = _coordinator_request(
        [_step(_execution())],
        "experiments.test_rl",
        ("--version", "2026.09.21", "--run", "--max-concurrent", "3"),
        Path("/workspace/marin"),
        {"DAYTONA_API_KEY": "secret"},
    )

    assert cluster_config == "lib/iris/config/marin.yaml"
    assert request.resources.target_cluster == "cw-rno2a"
    assert request.resources.cpu == 4
    assert request.resources.ram == "16GB"
    assert request.environment is not None
    assert request.environment.env_vars["DAYTONA_API_KEY"] == "secret"
    assert request.timeout == Duration.from_hours(18)
    assert request.entrypoint.binary_entrypoint is not None
    assert request.entrypoint.binary_entrypoint.command == "python"
    assert request.entrypoint.binary_entrypoint.args == [
        "-m",
        "experiments.test_rl",
        "--version",
        "2026.09.21",
        "--run",
        "--max-concurrent",
        "3",
    ]

    def native(output):
        Path(output).mkdir(parents=True, exist_ok=True)
        Path(output, "result.txt").write_text("completed")
        return Artifact(path=output)

    step = apply("tests/direct-in-job", native, version="2026.09.21", output=OUT)
    step = dataclasses.replace(step, runtime_args={"skyrl_execution": _execution()})
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    monkeypatch.delenv("MARIN_RL_COORDINATOR_GRAPH", raising=False)
    monkeypatch.setattr(
        "marin.rl.cli.open_iris_client", lambda **kwargs: pytest.fail("Direct job submitted a coordinator")
    )

    @click.command()
    @rl_build_options
    def direct():
        return [step]

    info = JobInfo(task_id=JobName.from_wire("/test-user/manual-rl/0"))
    set_job_info(info)
    try:
        with iris_ctx_scope(IrisContext.from_job_info(info)):
            result = CliRunner().invoke(direct, ["--version", "2026.09.21", "--run"])
        assert result.exit_code == 0, (result.output, result.exception)
        assert Path(step.path(str(tmp_path)), "result.txt").read_text() == "completed"
    finally:
        set_job_info(None)


def test_coordinator_request_requires_a_skyrl_step() -> None:
    with pytest.raises(ValueError, match="did not construct a SkyRL artifact step"):
        _coordinator_request(
            [_step()],
            "experiments.test_rl",
            ("--run",),
            Path("/workspace/marin"),
            {},
        )


@pytest.mark.parametrize("allow_submission", [False, True])
def test_prelaunch_check_finishes_before_coordinator_submission(tmp_path, monkeypatch, allow_submission):
    receipt = tmp_path / "prelaunch.txt"
    client = MagicMock(spec=IrisClient)
    submitted = []

    def submit(**kwargs):
        assert receipt.read_text() == "complete"
        submitted.append(kwargs)
        return SimpleNamespace(job_id="/tests/prelaunch-coordinator")

    client.submit.side_effect = submit
    monkeypatch.setattr("marin.rl.cli.open_iris_client", lambda **_kwargs: nullcontext(client))
    monkeypatch.delenv("IRIS_TASK_ID", raising=False)

    def prelaunch(_handles):
        if not allow_submission:
            raise RuntimeError("Live snapshot references unavailable")
        receipt.write_text("complete")

    @click.command()
    @partial(rl_build_options, before_submit=prelaunch)
    def main():
        return _step(_execution())

    dry = CliRunner().invoke(main, ["--version", "2026.11.26"])
    assert dry.exit_code == 0, dry.output
    assert not receipt.exists()
    client.submit.assert_not_called()

    result = CliRunner().invoke(main, ["--version", "2026.11.26", "--run"])
    if allow_submission:
        assert result.exit_code == 0, (result.output, result.exception)
        assert receipt.read_text() == "complete"
        assert len(submitted) == 1
    else:
        assert isinstance(result.exception, RuntimeError)
        assert not receipt.exists()
        client.submit.assert_not_called()


@pytest.mark.parametrize("coordinator_reentry", [False, True])
def test_in_job_check_runs_for_manual_launch_but_not_submitted_coordinator(tmp_path, monkeypatch, coordinator_reentry):
    receipt = tmp_path / "checked.txt"

    def native(output):
        Path(output).mkdir(parents=True, exist_ok=True)
        Path(output, "completed.txt").write_text("complete")
        return Artifact(path=output)

    step = apply("tests/prelaunch-in-job", native, version="2026.11.26", output=OUT)
    step = dataclasses.replace(step, runtime_args={"skyrl_execution": _execution()})
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    monkeypatch.delenv("MARIN_RL_COORDINATOR_GRAPH", raising=False)
    if coordinator_reentry:
        _, request = _coordinator_request([step], "test_cli", (), tmp_path, {})
        monkeypatch.setenv("MARIN_RL_COORDINATOR_GRAPH", request.environment.env_vars["MARIN_RL_COORDINATOR_GRAPH"])

    def prelaunch(_handles):
        receipt.write_text("checked")

    @click.command()
    @partial(rl_build_options, before_submit=prelaunch)
    def main():
        return step

    info = JobInfo(task_id=JobName.from_wire("/test-user/prelaunch/0"))
    set_job_info(info)
    try:
        with iris_ctx_scope(IrisContext.from_job_info(info)):
            result = CliRunner().invoke(main, ["--version", "2026.11.26", "--run"])
        assert result.exit_code == 0, (result.output, result.exception)
        assert Path(step.path(str(tmp_path)), "completed.txt").read_text() == "complete"
        assert receipt.exists() is not coordinator_reentry
    finally:
        set_job_info(None)


def test_rl_launch_resolves_explicit_credentials_before_remote_submission(tmp_path, monkeypatch) -> None:
    source = tmp_path / "versioned-judge-key"
    source.write_text("pinned-judge-key")
    client = MagicMock(spec=IrisClient)
    client.submit.return_value.job_id = "/tests/judge-key-coordinator"
    monkeypatch.setattr("marin.rl.cli.open_iris_client", lambda **_kwargs: nullcontext(client))
    monkeypatch.delenv("IRIS_TASK_ID", raising=False)
    monkeypatch.setenv("TOGETHER_API_KEY", "unrelated-ambient-key")

    @click.command()
    @rl_build_options
    def main() -> ArtifactStep[Artifact]:
        return _step(
            dataclasses.replace(_execution(), coordinator_secret_sources=(("JUDGE_API_KEY", f"file:{source}"),))
        )

    result = CliRunner().invoke(main, ["--version", "2026.11.20", "--run"])

    assert result.exit_code == 0, result.output
    environment = client.submit.call_args.kwargs["environment"]
    assert environment.secrets["JUDGE_API_KEY"] == "pinned-judge-key"
    assert "JUDGE_API_KEY" not in environment.env_vars
    assert "pinned-judge-key" not in result.output


@pytest.mark.parametrize("token", [None, "", "   ", "fixture-hf-token"])
def test_required_hf_secret_refuses_empty_values_before_submission(monkeypatch, token):
    client = MagicMock(spec=IrisClient)
    client.submit.return_value.job_id = "/tests/hf-coordinator"
    monkeypatch.setattr("marin.rl.cli.open_iris_client", lambda **_kwargs: nullcontext(client))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("IRIS_TASK_ID", raising=False)
    if token is not None:
        monkeypatch.setenv("HF_TOKEN", token)

    @click.command()
    @rl_build_options
    def main():
        return _step(dataclasses.replace(_execution(), coordinator_secret_sources=(("HF_TOKEN", "env:HF_TOKEN"),)))

    result = CliRunner().invoke(main, ["--version", "2026.11.23", "--run"])
    if not token or not token.strip():
        assert result.exit_code != 0
        assert "HF_TOKEN" in str(result.exception)
        client.submit.assert_not_called()
        return
    assert result.exit_code == 0, result.output
    environment = client.submit.call_args.kwargs["environment"]
    assert environment.secrets["HF_TOKEN"] == token
    assert "HF_TOKEN" not in environment.env_vars
    assert token not in result.output


@pytest.mark.parametrize(
    "missing_names", [(), ("JUDGE_KEY",), ("EVAL_KEY",), ("DAYTONA_EVAL_API_KEY", "DAYTONA_TARGET")]
)
def test_rl_launch_requires_and_forwards_credentials_before_submission(
    monkeypatch: pytest.MonkeyPatch, missing_names: tuple[str, ...]
) -> None:
    client = MagicMock(spec=IrisClient)
    client.submit.return_value.job_id = JobName.from_wire("/atqamar/submitted-coordinator")
    monkeypatch.setattr("marin.rl.cli.open_iris_client", lambda **_kwargs: nullcontext(client))
    monkeypatch.delenv("IRIS_TASK_ID", raising=False)
    expected_env = {
        "DAYTONA_RL_API_KEY": "test-rl-key",
        "DAYTONA_EVAL_API_KEY": "test-eval-key",
        "DAYTONA_API_KEY": "test-other-key",
        "TOGETHER_API_KEY": "test-together-key",
        "JUDGE_KEY": "test-judge-key",
        "EVAL_KEY": "test-custom-eval-key",
        "DAYTONA_TARGET": "us",
    }
    for name, value in expected_env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("HF_TOKEN", "test-hf-token")
    monkeypatch.setenv("WANDB_API_KEY", "test-wandb-key")
    for missing_name in missing_names:
        monkeypatch.delenv(missing_name)
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-forward")

    @click.command()
    @rl_build_options
    def main() -> list[ArtifactStep[Artifact]]:
        train = _step(dataclasses.replace(_execution(), required_env_vars=("JUDGE_KEY", "DAYTONA_TARGET")))
        evaluate = dataclasses.replace(
            _step(dataclasses.replace(_execution(), required_env_vars=("EVAL_KEY", "DAYTONA_EVAL_API_KEY"))),
            name="checkpoints/test-eval",
        )
        return [train, evaluate]

    result = CliRunner().invoke(main, ["--version", "2026.10.07", "--run"])

    if missing_names:
        assert result.exit_code != 0
        assert isinstance(result.exception, ValueError)
        assert all(name in str(result.exception) for name in missing_names)
        assert all(
            value not in str(result.exception) for name, value in expected_env.items() if name != "DAYTONA_TARGET"
        )
        client.submit.assert_not_called()
        return

    assert result.exit_code == 0, result.output
    environment = client.submit.call_args.kwargs["environment"]
    assert expected_env.items() <= environment.env_vars.items()
    assert {"HF_TOKEN", "WANDB_API_KEY", "UNRELATED_SECRET"}.isdisjoint(environment.env_vars)


def test_execution_requires_a_complete_federated_route() -> None:
    with pytest.raises(ValueError, match="target_cluster and parent_cluster_config"):
        dataclasses.replace(
            _execution(),
            parent_cluster_config=None,
        )


def test_rl_entrypoint_can_spawn_workers(tmp_path: Path) -> None:
    module = tmp_path / "spawnable_rl.py"
    module.write_text(
        textwrap.dedent(
            """\
        import multiprocessing
        import sys
        import click
        from marin.rl.cli import rl_build_options
        from experiments.post_training import rl

        assert "experiments.post_training.snowball_rl" not in sys.modules
        from experiments.post_training import snowball_rl

        assert "marin.processing.embeddings.harrier" not in sys.modules
        assert "levanter" not in sys.modules

        @click.command()
        @rl_build_options
        def main():
            return []

        def worker(queue):
            queue.put(42)

        if __name__ == "__main__":
            context = multiprocessing.get_context("spawn")
            queue = context.Queue()
            process = context.Process(target=worker, args=(queue,))
            process.start()
            try:
                assert queue.get(timeout=30) == 42
                process.join(timeout=10)
                assert process.exitcode == 0
            finally:
                if process.is_alive():
                    process.kill()
                    process.join()
                queue.close()
    """
        )
    )
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path), *sys.path])}
    result = subprocess.run(
        [sys.executable, "-m", "spawnable_rl"],
        env=environment,
        capture_output=True,
        text=True,
        timeout=50,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("module", "args"),
    (
        ("experiments.post_training.cat_count_canary.launcher", ()),
        ("experiments.post_training.mochi_rl", ("--help",)),
        ("experiments.post_training.snowball_rl", ("--help",)),
    ),
)
def test_launcher_startup_keeps_training_dependencies_unloaded(module: str, args: tuple[str, ...]) -> None:
    program = textwrap.dedent(
        """
        import importlib, runpy, sys

        sys.argv = sys.argv[1:]
        module = sys.argv[0]
        if len(sys.argv) > 1:
            try:
                runpy.run_module(module, run_name="__main__")
            except SystemExit as error:
                assert error.code == 0, error
        else:
            importlib.import_module(module)
        heavy = {"levanter", "transformers", "torch", "wandb"}
        loaded = heavy.intersection(name.split(".")[0] for name in sys.modules)
        assert not loaded, sorted(loaded)
        assert not any(
            name.startswith(("experiments.sft.", "experiments.post_training.mochi_sft",
                             "experiments.post_training.snowball_sft"))
            for name in sys.modules
        )
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", program, module, *args],
        env={},
        timeout=30,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    if args:
        assert "Show this message and exit." in result.stdout


def test_named_coordinator_repeated_submission_preserves_identity():
    execution = dataclasses.replace(_execution(), coordinator_name="long-loop-plumbing-2026.11.08")
    handles = [_step(execution)]
    requests = [
        _coordinator_request(handles, "experiments.test_rl", ("--run",), Path("/workspace/marin"), {})[1]
        for _ in range(2)
    ]
    assert requests[0].name == requests[1].name == "long-loop-plumbing-2026.11.08"


def test_gpu_coordinator_preserves_route_and_priority():
    execution = dataclasses.replace(_execution(), coordinator_gpu=GpuConfig(variant="H100", count=1))
    cluster, request = _coordinator_request(
        [_step(execution)], "experiments.test_rl", ("--run",), Path("/workspace/marin"), {}
    )
    assert request.resources.device == execution.coordinator_gpu
    cpu_cluster, cpu_request = _coordinator_request(
        [_step(_execution())], "experiments.test_rl", ("--run",), Path("/workspace/marin"), {}
    )
    assert cluster == cpu_cluster
    assert request.resources.target_cluster == cpu_request.resources.target_cluster
    assert request.priority == cpu_request.priority
