# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a bounded serving-request qualification without loading model weights."""

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

from experiments.post_training.long_loop.launch_validation import tool_preflight
from experiments.post_training.long_loop.run_spec import resolve_spec
from experiments.post_training.long_loop.spec_pipeline import runtime_execution


@click.command(help=__doc__)
@click.option("--spec", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--model-uri", required=True)
@click.option("--request", type=click.Path(exists=True, path_type=Path), required=True)
@rl_build_options
def main(spec: Path, model_uri: str, request: Path) -> ArtifactStep:
    resolved = resolve_spec(spec, smoke=False)
    os.environ["MARIN_PREFIX"] = resolved["storage"]["output_prefix"]
    os.environ["MARIN_TEMP_PREFIX"] = resolved["storage"]["temporary_prefix"]
    configure_coreweave_s3()
    resources = resolved["execution"]["sandbox"]
    resources = ResourceConfig(**{**resources, "device": draccus.decode(DeviceConfig, resources["device"])})
    version = resolve_version("long-loop/tool-preflight", None)
    execution = replace(
        runtime_execution(resolved["execution"]["coordinator"]),
        coordinator_name=f"long-loop-tool-preflight-{version}",
        coordinator_timeout_hours=1,
        required_env_vars=("MARIN_PREFIX", "MARIN_TEMP_PREFIX", "HF_TOKEN"),
    )
    step = apply(
        "long-loop/tool-preflight",
        remote(tool_preflight, resources=resources),
        version=version,
        output=OUT,
        model_uri=model_uri,
        tokenizer=resolved["tokenizer"],
        generator=resolved["rl"]["config"]["generator"],
        window=resolved["rl"]["config"]["context_budget"]["request_window_tokens"],
        request=json.loads(request.read_text()),
    )

    return replace(step, runtime_args={**step.runtime_args, "skyrl_execution": execution})


if __name__ == "__main__":
    main()
