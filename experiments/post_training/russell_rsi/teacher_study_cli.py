# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Select one validated stage from a teacher study graph."""

from collections.abc import Callable

import click
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep
from marin.external_dependencies import MARIN_SKYRL

TEACHER_STAGES = ("collect", "sft", "reload", "calibrate", "rl", "evaluate")


def teacher_study_stage(
    config: dict,
    stage: str,
    *,
    protocol: str,
    workflow: Callable[[dict], dict[str, ArtifactStep]],
    post_workflow: Callable[[dict, str], dict[str, ArtifactStep]],
    error_message: str,
) -> list[ArtifactStep]:
    """Validate study pins and return the requested collection or post-SFT step."""
    if (
        resolve_version(protocol, None) != config["version"]
        or config["protocol"] != protocol
        or config["runtime_commit"] != MARIN_SKYRL.commit
    ):
        raise click.UsageError(error_message)
    if stage in ("collect", "sft", "reload"):
        return [workflow(config)["train" if stage == "sft" else stage]]
    return [post_workflow(config, "train" if stage == "rl" else stage)["terminal"]]
