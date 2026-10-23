# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare the pinned task inventory and its fixed heldout set."""

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from fray.types import ResourceConfig
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.experiment.namespacing import user_owned_name
from marin.rl.skyrl import skyrl_cpu_command

VALIDATION_ROWS = 100


@dataclass(frozen=True)
class Source:
    uri: str
    sha256: str
    rows: int


TRAINING_SHA256 = "9a6dd45c753ac0180a2a8ea4bfaefa8a5e9c3c756dbcad65dbacc844dae8a9db"
VALIDATION_SHA256 = "a8db9a59591f38cb0d7973aa85c60ee9e9d297f9192750ba298860d961baf687"
TRAINING_ROWS = 100_000
FULL_VALIDATION_ROWS = 910


@dataclass(frozen=True)
class IngestConfig:
    training: Source
    validation: Source
    output_path: str


def prepare_data(config: IngestConfig) -> None:
    """Write the pinned training pool and held-out task Parquet files."""
    environment = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "UV_PROJECT_ENVIRONMENT", "VIRTUAL_ENV"):
        environment.pop(key, None)
    subprocess.run(
        skyrl_cpu_command(
            "python",
            "-m",
            "experiments.post_training.inventory._data",
            "--training",
            config.training.uri,
            "--training-sha256",
            config.training.sha256,
            "--training-rows",
            str(config.training.rows),
            "--validation",
            config.validation.uri,
            "--validation-sha256",
            config.validation.sha256,
            "--validation-rows",
            str(config.validation.rows),
            "--output",
            config.output_path,
        ),
        cwd=Path(__file__).resolve().parents[3],
        env=environment,
        check=True,
    )


def data_step(version: str, training_uri: str, validation_uri: str) -> ArtifactStep[Artifact]:
    """Bind the immutable 100k training source and original 910-row heldout source."""
    training = ArtifactStep.adopt("documents/snowball-inventory-training-source", "2026.10.19.2", training_uri)
    validation = ArtifactStep.adopt("documents/snowball-inventory-validation-source", "2026.10.20.2", validation_uri)

    def build_config(ctx: StepContext) -> IngestConfig:
        return IngestConfig(
            Source(ctx.artifact_path(training), TRAINING_SHA256, TRAINING_ROWS),
            Source(ctx.artifact_path(validation), VALIDATION_SHA256, FULL_VALIDATION_ROWS),
            ctx.output_path,
        )

    return ArtifactStep(
        name=user_owned_name("documents/snowball-inventory"),
        version=version,
        artifact_type=Artifact,
        deps=(training, validation),
        run=remote(prepare_data, resources=ResourceConfig.with_cpu(cpu=4, ram="16g", disk="8g")),
        build_config=build_config,
    )
