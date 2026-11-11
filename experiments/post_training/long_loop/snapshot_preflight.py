# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the bounded snapshot lifecycle qualification through an artifact graph."""

import json
import os
from dataclasses import replace
from pathlib import Path

import click
import draccus
from fray.types import DeviceConfig, GpuConfig, ResourceConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import OUT, ArtifactStep, apply
from marin.execution.remote import remote
from marin.rl.cli import rl_build_options
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.post_training.long_loop.qualification import run_qualification
from experiments.post_training.long_loop.run_spec import resolve_spec
from experiments.post_training.long_loop.spec_pipeline import runtime_execution


def qualify(*, output: str, inputs: dict, probe_module: str) -> None:
    run_qualification(output=output, inputs=inputs, module=probe_module, dependency_group="harbor-test")


@click.command(help=__doc__)
@click.option("--spec", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--inputs", type=click.Path(exists=True, path_type=Path), required=True)
@click.option(
    "--probe-module",
    type=click.Choice(
        [
            "experiments.post_training.long_loop.snapshot_probe_worker",
            "experiments.post_training.long_loop.calibforge_probe_worker",
        ]
    ),
    required=True,
)
@rl_build_options
def main(spec: Path, inputs: Path, probe_module: str) -> ArtifactStep:
    resolved = resolve_spec(spec, smoke=False)
    os.environ["MARIN_PREFIX"] = resolved["storage"]["output_prefix"]
    os.environ["MARIN_TEMP_PREFIX"] = resolved["storage"]["temporary_prefix"]
    configure_coreweave_s3()
    values = resolved["execution"]["inputs"]
    resources = ResourceConfig(**{**values, "device": draccus.decode(DeviceConfig, values["device"])})
    resources = replace(resources, device=GpuConfig(variant="H100", count=1), cpu=4, ram="32GB")
    version = resolve_version("long-loop/snapshot-preflight", None)
    execution = replace(
        runtime_execution(resolved["execution"]["coordinator"]),
        coordinator_name=f"long-loop-snapshots-preflight-{version}",
        coordinator_timeout_hours=1,
        required_env_vars=("MARIN_PREFIX", "MARIN_TEMP_PREFIX", "HF_TOKEN", "DAYTONA_RL_API_KEY"),
    )
    step = apply(
        "long-loop/snapshot-preflight",
        remote(qualify, resources=resources),
        version=version,
        output=OUT,
        inputs=json.loads(inputs.read_text()),
        probe_module=probe_module,
    )
    return replace(step, runtime_args={**step.runtime_args, "skyrl_execution": execution})


if __name__ == "__main__":
    main()
