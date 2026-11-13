# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resolve and validate a pinned long-loop run spec before any allocation."""

import hashlib
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import click
from iris.cluster.client.job_info import get_job_info
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import graph_handles
from marin.rl.cli import rl_build_options
from marin.rl.skyrl import SkyRLRunConfig
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.secrets import require_secret

from experiments.post_training.long_loop.eval_runtime.judge import JudgedEvaluation
from experiments.post_training.long_loop.launch_validation import validate_launch
from experiments.post_training.long_loop.run_spec import resolve_spec, stage_blockers
from experiments.post_training.long_loop.spec_pipeline import build_spec_graph


def validate_graph(terminal: ArtifactStep, directory: Path, artifact_prefix: str) -> None:
    rows = []
    for step in graph_handles([terminal]):
        if step.adopt_source:
            rows.append({"stage": step.name, "status": "adopted", "source": step.adopt_source})
            continue
        context = StepContext.for_run(
            step.path(artifact_prefix), artifact_prefix, runtime_args=step.runtime_args, deps=step.deps
        )
        # Future checkpoint selection is unavailable before training. Use the
        # builder's identity branch, retaining actual runtime config and paths.
        config = step.build_config(replace(context, is_fingerprint=True))
        launch = config.skyrl if isinstance(config, JudgedEvaluation) else config
        row = {"stage": step.name, "status": "constructed; not a SkyRL launch"}
        if isinstance(launch, SkyRLRunConfig):
            digest = hashlib.sha256(launch.launch_config_yaml.encode()).hexdigest()
            row = {
                "stage": step.name,
                "status": "strictly composed; future checkpoint identity unresolved",
                **validate_launch(launch.launch_config_yaml, directory / digest),
            }
        rows.append(row)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "graph.json").write_text(json.dumps(rows, indent=2))


@click.command(help=__doc__)
@click.option("--spec", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--smoke", is_flag=True, help="Apply only the spec's explicit scale overrides.")
@click.option("--through", type=click.Choice(["sft", "rl", "archive"]), default="archive", show_default=True)
@click.option("--validation-output", type=click.Path(path_type=Path), required=True)
@click.option("--i-know-i-am-using-a-deprecated-config-format", is_flag=True)
@rl_build_options
def main(
    spec: Path,
    smoke: bool,
    through: str,
    validation_output: Path,
    i_know_i_am_using_a_deprecated_config_format: bool,
) -> ArtifactStep:
    resolved = resolve_spec(
        spec,
        smoke=smoke,
        acknowledge_deprecated_config_format=i_know_i_am_using_a_deprecated_config_format,
    )
    os.environ["MARIN_PREFIX"] = resolved["storage"]["output_prefix"]
    os.environ["MARIN_TEMP_PREFIX"] = resolved["storage"]["temporary_prefix"]
    resolved["through"] = through
    resolved["active_stage_blockers"] = stage_blockers(resolved, through)
    # Iris ships the validated workspace bundle without Git metadata.
    if get_job_info() is None:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        if revision != resolved["marin_revision"]:
            resolved["active_stage_blockers"].append("Run spec's Marin revision differs from the checked-out code")
    validation_output.mkdir(parents=True, exist_ok=True)
    plan = validation_output / "resolved-plan.json"
    plan.write_text(json.dumps(resolved, indent=2))
    click.echo(json.dumps(resolved, indent=2))
    if resolved["active_stage_blockers"]:
        raise click.ClickException("Run is not admitted: " + "; ".join(resolved["active_stage_blockers"]))
    configure_coreweave_s3()
    require_secret("HF_TOKEN")
    version = resolve_version(resolved["name"], None)
    terminal = build_spec_graph(resolved, version, through=through)
    validate_graph(terminal, validation_output / version, os.environ["MARIN_PREFIX"])
    return terminal


if __name__ == "__main__":
    main()
