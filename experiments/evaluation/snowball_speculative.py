# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serve the frozen Snowball/EAGLE3 pair through Iris."""

import json
from dataclasses import asdict, replace
from enum import StrEnum
from pathlib import Path

import click
from iris.rpc import job_pb2
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.model_config import load_model_config
from marin.evaluation.serving_config import inference_config_for_model
from marin.inference.config import (
    RejectionSampleMethod,
    RemoteInferenceConfig,
    ResolvedModelLocator,
    SpeculativeMethod,
    SpeculativeServingConfig,
)
from marin.inference.iris import IrisServiceConfig, run_iris_service

MODEL_CONFIG = Path(__file__).parent / "serve/models/marin-community/grug-agentic-s3-step1903.yaml"
DRAFT = "laion/snowball-64k-eagle3-draft-r2egym@4bdb47c08e5b5190bea3c7a93c3e14470230e469"
TOKENIZER = "marin-community/marin-tokenizer"
TOKENIZER_REVISION = "a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2"
MODEL_NAME = "snowball-67b-a2b-step1903"
DEPTH = 3


class Mode(StrEnum):
    BASELINE = "baseline"
    EAGLE3 = "eagle3"
    SYNTHETIC = "synthetic"


def snowball_config(mode: Mode, acceptance_length: float | None = None) -> RemoteInferenceConfig:
    """Apply the frozen pair's pins and sampler to the catalog serving settings."""
    if mode is not Mode.SYNTHETIC and acceptance_length is not None:
        raise ValueError("acceptance_length requires synthetic mode")
    speculative = None
    if mode is not Mode.BASELINE:
        speculative = SpeculativeServingConfig(
            method=SpeculativeMethod.EAGLE3,
            model=ResolvedModelLocator(uri=DRAFT, identity=DRAFT),
            num_speculative_tokens=DEPTH,
            rejection_sample_method=(
                RejectionSampleMethod.SYNTHETIC if mode is Mode.SYNTHETIC else RejectionSampleMethod.STANDARD
            ),
            synthetic_acceptance_length=acceptance_length,
        )
    catalog = load_model_config(MODEL_CONFIG)
    model = replace(
        catalog,
        name=MODEL_NAME,
        identity=catalog.location,
        # The catalog uses a different tokenizer; this pin matches the target export.
        tokenizer=TOKENIZER,
        tokenizer_revision=TOKENIZER_REVISION,
        resource_hint=replace(catalog.resource_hint, cpu=16, disk="200g"),
        serve=replace(catalog.serve, speculative=speculative),
    )
    config = inference_config_for_model(
        model,
        AcceleratorChoice(platform=Platform.GPU, gpu_type="H100", gpu_count=8),
        env_vars={},
        priority=job_pb2.PRIORITY_BAND_INTERACTIVE,
        api_model=MODEL_NAME,
    )
    # A direct endpoint preserves the engine's ordinary streaming responses.
    return replace(config, broker=None)


@click.command()
@click.option("--mode", type=click.Choice([mode.value for mode in Mode]), default="baseline")
@click.option("--acceptance-length", type=float, default=None, help="Measured coding acceptance, including bonus token.")
@click.option("--dry-run", is_flag=True, help="Print configuration without starting an engine.")
@click.option("--endpoint-prefix", required=True, help="Unique absolute Iris endpoint prefix for this launch.")
@click.option("--timeout-hours", type=float, default=1.0, help="Persistent service lifetime after startup.")
def main(
    mode: str,
    acceptance_length: float | None,
    dry_run: bool,
    endpoint_prefix: str,
    timeout_hours: float,
) -> None:
    if not endpoint_prefix.startswith("/") or "." in endpoint_prefix:
        raise click.UsageError("--endpoint-prefix must be absolute and contain no periods")
    selected = Mode(mode)
    config = snowball_config(selected, acceptance_length)
    service = IrisServiceConfig(
        model=config.model,
        engine=config.engine,
        iris=config.iris,
        endpoint_name=f"{endpoint_prefix.rstrip('/')}-{selected.value}",
        timeout_hours=timeout_hours,
        port_name=None,
    )
    if dry_run:
        click.echo(json.dumps(asdict(service), default=list, indent=2))
    else:
        run_iris_service(service)


if __name__ == "__main__":
    main()
