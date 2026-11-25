# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compose authored presets using the checkout's native SkyRL CPU environment."""

import json
import subprocess
from dataclasses import replace
from pathlib import Path

from marin.execution.lazy import ArtifactStep, materialized_config
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


def compose_archived_config(path: Path) -> dict:
    """Compose a launch with the archived checkout's frozen SkyRL environment."""
    command = skyrl_cpu_command(
        "python",
        "-I",
        "-m",
        "cloud.iris.config_cli",
        "--checkout",
        str(recipe.REPO_ROOT / "vendor/skyrl"),
        "--progress-file",
        str(path.with_suffix(".progress")),
        "compose",
        "--",
        str(path),
    )
    result = subprocess.run(command, capture_output=True, check=True, text=True, timeout=120)
    (config,) = json.loads(result.stdout)
    return config
