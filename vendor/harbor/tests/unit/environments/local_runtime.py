# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Construct the local adapter with offline task settings for environment tests."""

import os
from pathlib import Path

import pytest

from harbor.environments.local import LocalEnvironment
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths
from shellbox.backends.local.sandbox import SandboxUnavailable


def local_environment(tmp_path: Path, **kwargs) -> LocalEnvironment:
    config = kwargs.pop(
        "task_env_config",
        EnvironmentConfig(
            docker_image="offline.test/task@sha256:" + "a" * 64,
            allow_internet=False,
            workdir="/app",
        ),
    )
    try:
        return LocalEnvironment(
            environment_dir=tmp_path / "environment",
            environment_name="prepared-task",
            session_id="trial",
            trial_paths=TrialPaths(tmp_path / "trial"),
            task_env_config=config,
            **kwargs,
        )
    except SandboxUnavailable as error:
        if os.environ.get("CI"):
            pytest.fail(str(error))
        pytest.skip(str(error))
