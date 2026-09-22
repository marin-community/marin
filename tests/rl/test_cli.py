# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep
from marin.rl.cli import _coordinator_request
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


def test_execution_requires_a_complete_federated_route() -> None:
    with pytest.raises(ValueError, match="target_cluster and parent_cluster_config"):
        dataclasses.replace(
            _execution(),
            parent_cluster_config=None,
        )
