# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compose authored presets using the checkout's native SkyRL CPU environment."""

import json
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from marin.execution.lazy import ArtifactStep, materialized_config
from marin.rl import skyrl
from marin.rl.skyrl import (
    ArtifactHfModel,
    ResolvedModelLocator,
    SkyRLModelSource,
    skyrl_cpu_command,
)

from experiments.post_training import rl_compose as recipe
from experiments.post_training.rl_presets import Preset, StoredResume


class InvalidComparison(ValueError):
    """Required configuration or measurement evidence is unavailable or inconsistent."""


def preset_launch_yaml(
    preset: Preset,
    *,
    name: str,
    tasks: ArtifactStep | None = None,
    init: SkyRLModelSource | ResolvedModelLocator | None = None,
    resume: StoredResume | None = None,
    settings: tuple[str, ...] = (),
) -> str:
    spec = recipe.run_spec(preset, name=name, tasks=tasks, init=init, resume=resume, settings=settings)
    if isinstance(spec.model, ArtifactHfModel) and spec.model.relative_path is None:
        spec = replace(spec, model=replace(spec.model, relative_path="<terminal-hf-export>"))
    step = recipe.run_step(preset, spec)
    return materialized_config(step, recipe.run_prefix(preset)).launch_config_yaml


def run_native_json(*command: str, timeout: int = 120, env: Mapping[str, str] | None = None):
    """Run a SkyRL CPU command that returns JSON on stdout."""
    try:
        result = subprocess.run(
            skyrl_cpu_command(*command),
            capture_output=True,
            env=env if env is not None else skyrl.skyrl_cpu_environment(),
            text=True,
            timeout=timeout,
        )
        result.check_returncode()
        return json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        cause = f"timed out after {error.timeout:g} s" if isinstance(error, subprocess.TimeoutExpired) else str(error)
        if isinstance(error, subprocess.CalledProcessError) and error.stderr and error.stderr.strip():
            cause = error.stderr.strip().splitlines()[-1]
        raise InvalidComparison(f"native configuration cannot be composed: {cause}") from error


def compose_configs(
    paths: Sequence[str], *, validate_driver: bool = False, env: Mapping[str, str] | None = None
) -> list[dict]:
    """Compose launch files and optionally check the driver configuration in SkyRL's CPU environment."""
    options = ("--validate-driver",) if validate_driver else ()
    with tempfile.TemporaryDirectory(prefix="rl-archive-config-") as directory:
        return run_native_json(
            "python",
            "-I",
            "-m",
            "cloud.iris.config_cli",
            "--checkout",
            str(recipe.REPO_ROOT / "vendor/skyrl"),
            "--progress-file",
            str(Path(directory) / "stage"),
            "compose",
            *options,
            "--",
            *paths,
            env=env,
        )
