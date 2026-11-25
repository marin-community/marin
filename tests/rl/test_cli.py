# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
import textwrap
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import MagicMock

import click
import pytest
from click.testing import CliRunner
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


def test_coordinator_collects_missing_names_and_delivers_configured_secrets(tmp_path):
    required = ("MISSING_SETTING", "EMPTY_SETTING", "DAYTONA_API_KEY")
    execution = dataclasses.replace(_execution(), required_env_vars=required)
    with pytest.raises(ValueError) as error:
        _coordinator_request([_step(execution)], "experiments.test_rl", (), tmp_path, {"EMPTY_SETTING": " "})
    assert all(name in str(error.value) for name in required)

    source = tmp_path / "credential"
    execution = dataclasses.replace(execution, coordinator_secret_sources=(("DAYTONA_API_KEY", f"file:{source}"),))
    source.write_text("   ")
    ordinary_env = {"MISSING_SETTING": "present", "EMPTY_SETTING": "present", "DAYTONA_API_KEY": "fixture-decoy"}
    with pytest.raises(ValueError, match="DAYTONA_API_KEY"):
        _coordinator_request([_step(execution)], "experiments.test_rl", (), tmp_path, ordinary_env)

    source.write_text("fixture-coordinator-credential")
    _, request = _coordinator_request([_step(execution)], "experiments.test_rl", (), tmp_path, ordinary_env)
    assert request.environment is not None
    assert request.environment.secrets["DAYTONA_API_KEY"] == "fixture-coordinator-credential"
    assert "DAYTONA_API_KEY" not in request.environment.env_vars
    assert request.environment.env_vars["MISSING_SETTING"] == "present"


def test_coordinator_request_requires_a_skyrl_step() -> None:
    with pytest.raises(ValueError, match="did not construct a SkyRL artifact step"):
        _coordinator_request(
            [_step()],
            "experiments.test_rl",
            ("--run",),
            Path("/workspace/marin"),
            {},
        )


@pytest.mark.parametrize("missing_name", [None, "JUDGE_KEY", "EVAL_KEY"])
def test_rl_launch_requires_and_forwards_credentials_before_submission(
    monkeypatch: pytest.MonkeyPatch, missing_name: str | None
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
    }
    for name, value in expected_env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("HF_TOKEN", "test-hf-token")
    monkeypatch.setenv("WANDB_API_KEY", "test-wandb-key")
    if missing_name is not None:
        monkeypatch.delenv(missing_name)
    monkeypatch.setenv("UNRELATED_SECRET", "must-not-forward")

    @click.command()
    @rl_build_options
    def main() -> list[ArtifactStep[Artifact]]:
        train = _step(dataclasses.replace(_execution(), required_env_vars=("JUDGE_KEY",)))
        evaluate = dataclasses.replace(
            _step(dataclasses.replace(_execution(), required_env_vars=("EVAL_KEY",))),
            name="checkpoints/test-eval",
        )
        return [train, evaluate]

    result = CliRunner().invoke(main, ["--version", "2026.10.07", "--run"])

    if missing_name is not None:
        assert result.exit_code != 0
        assert isinstance(result.exception, ValueError)
        assert missing_name in str(result.exception)
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
