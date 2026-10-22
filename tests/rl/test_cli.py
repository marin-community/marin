# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import MagicMock

import click
import pytest
from click.testing import CliRunner
from iris.client.client import IrisClient
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep
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


def test_coordinator_request_replays_the_experiment_main() -> None:
    cluster_config, request = _coordinator_request(
        [_step(_execution())],
        "experiments.test_rl",
        ("--version", "2026.09.21", "--run"),
        3,
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


def test_coordinator_request_requires_a_skyrl_step() -> None:
    with pytest.raises(ValueError, match="did not construct a SkyRL artifact step"):
        _coordinator_request([_step()], "experiments.test_rl", ("--run",), 8, Path("/workspace/marin"), {})


@pytest.mark.parametrize("missing_name", [None, "JUDGE_KEY", "EVAL_KEY"])
def test_rl_launch_requires_and_forwards_credentials_before_submission(
    monkeypatch: pytest.MonkeyPatch, missing_name: str | None
) -> None:
    client = MagicMock(spec=IrisClient)
    monkeypatch.setattr("marin.rl.cli.open_iris_client", lambda **_kwargs: nullcontext(client))
    monkeypatch.delenv("IRIS_TASK_ID", raising=False)
    expected_env = {
        "DAYTONA_RL_API_KEY": "test-rl-key",
        "DAYTONA_EVAL_API_KEY": "test-eval-key",
        "DAYTONA_API_KEY": "test-other-key",
        "HF_TOKEN": "test-hf-token",
        "TOGETHER_API_KEY": "test-together-key",
        "WANDB_API_KEY": "test-wandb-key",
        "JUDGE_KEY": "test-judge-key",
        "EVAL_KEY": "test-custom-eval-key",
    }
    for name, value in expected_env.items():
        monkeypatch.setenv(name, value)
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
        assert isinstance(result.exception, KeyError)
        assert result.exception.args == (missing_name,)
        client.submit.assert_not_called()
        return

    assert result.exit_code == 0, result.output
    environment = client.submit.call_args.kwargs["environment"]
    assert expected_env.items() <= environment.env_vars.items()
    assert "UNRELATED_SECRET" not in environment.env_vars


def test_execution_requires_a_complete_federated_route() -> None:
    with pytest.raises(ValueError, match="target_cluster and parent_cluster_config"):
        dataclasses.replace(
            _execution(),
            parent_cluster_config=None,
        )
