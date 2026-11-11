# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the bounded CP4 attention qualification through an artifact graph."""

import json
import os
from dataclasses import replace
from pathlib import Path

import click
import draccus
from fray.types import DeviceConfig, ResourceConfig
from marin.execution.build_context import resolve_version
from marin.execution.lazy import OUT, ArtifactStep, apply
from marin.execution.remote import remote
from marin.rl.cli import rl_build_options
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.post_training.long_loop.qualification import run_qualification
from experiments.post_training.long_loop.run_spec import resolve_spec
from experiments.post_training.long_loop.spec_pipeline import runtime_execution


def qualify(*, output: str, inputs: dict) -> None:
    run_qualification(
        output=output,
        inputs=inputs,
        module="experiments.post_training.long_loop.cp_numerical_probe",
        prefix_arguments=("-m", "torch.distributed.run", "--standalone", "--nproc-per-node=4"),
        extra="megatron",
        environment_overrides={"NVTE_FLASH_ATTN": "0", "NVTE_FUSED_ATTN": "1"},
    )


@click.command(help=__doc__)
@click.option("--spec", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--inputs", type=click.Path(exists=True, path_type=Path), required=True)
@rl_build_options
def main(spec: Path, inputs: Path) -> ArtifactStep:
    resolved = resolve_spec(spec, smoke=False)
    os.environ["MARIN_PREFIX"] = resolved["storage"]["output_prefix"]
    os.environ["MARIN_TEMP_PREFIX"] = resolved["storage"]["temporary_prefix"]
    configure_coreweave_s3()
    values = resolved["execution"]["inputs"]
    resources = ResourceConfig(**{**values, "device": draccus.decode(DeviceConfig, values["device"])})
    version = resolve_version("long-loop/cp-preflight", None)
    execution = replace(
        runtime_execution(resolved["execution"]["coordinator"]),
        coordinator_name=f"long-loop-cp-preflight-{version}",
        coordinator_timeout_hours=1,
        required_env_vars=("MARIN_PREFIX", "MARIN_TEMP_PREFIX", "HF_TOKEN"),
    )
    step = apply(
        "long-loop/cp-preflight",
        remote(qualify, resources=resources),
        version=version,
        output=OUT,
        inputs=json.loads(inputs.read_text()),
    )
    return replace(step, runtime_args={**step.runtime_args, "skyrl_execution": execution})


if __name__ == "__main__":
    main()
