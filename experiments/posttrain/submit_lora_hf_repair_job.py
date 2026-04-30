#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit an Iris CPU job that repairs a broken merged LoRA HF export in place.

The worker-side repair logic lives in `repair_lora_hf_export.py`. This script is
just the reproducible Iris submission wrapper so future repairs do not require
hand-written `iris job run` commands.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from iris.client import IrisClient
from iris.cluster.config import IrisConfig
from iris.cluster.constraints import Constraint, region_constraint, zone_constraint
from iris.cluster.types import Entrypoint, ResourceSpec

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_FILE = "lib/iris/examples/marin.yaml"
DEFAULT_SOURCE_MODEL_PATH = (
    "gs://marin-us-central1/checkpoints/dpo/tune_lora/"
    "bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d/hf/step-1699"
)
DEFAULT_REGION = "us-central1"
DEFAULT_CPU = 8.0
DEFAULT_MEMORY = "128GB"
DEFAULT_DISK = "200GB"


def find_workspace_root(start: Path) -> Path:
    """Find the repo root used for Iris workspace bundling."""
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise ValueError(f"Could not find workspace root from {start}")


@contextmanager
def controller_client(config_file: str) -> Iterator[IrisClient]:
    """Open a tunneled Iris client using the standard cluster config."""
    iris_config = IrisConfig.load(config_file)
    controller_address = iris_config.controller_address()
    providers = iris_config.provider_bundle()
    controller = providers.controller
    workspace = find_workspace_root(Path.cwd())
    if not controller_address:
        controller_address = controller.discover_controller(iris_config.proto.controller)
    with controller.tunnel(address=controller_address) as tunneled:
        client = IrisClient.remote(tunneled, workspace=workspace)
        try:
            yield client
        finally:
            client.shutdown()


def default_output_path(source_model_path: str, run_label: str) -> str:
    """Map `.../hf/step-N` to `.../hf-repair-direct-<label>/step-N`."""
    marker = "/hf/"
    if marker not in source_model_path:
        raise ValueError(f"Expected source path to contain {marker!r}: {source_model_path}")
    prefix, suffix = source_model_path.split(marker, maxsplit=1)
    return f"{prefix}/hf-repair-direct-{run_label}/{suffix}"


def default_job_name(source_model_path: str, run_label: str) -> str:
    """Derive a stable Iris job name from the checkpoint step."""
    step_name = source_model_path.rstrip("/").split("/")[-1].replace("-", "")
    return f"lora-vllm-direct-repair-{step_name}-cpu-{run_label}"


def build_constraints(region: str, zone: str | None) -> list[Constraint]:
    """Build explicit placement constraints for the repair job."""
    constraints = [region_constraint([region])]
    if zone is not None:
        constraints.append(zone_constraint(zone))
    return constraints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-file", default=DEFAULT_CONFIG_FILE)
    parser.add_argument("--source-model-path", default=DEFAULT_SOURCE_MODEL_PATH)
    parser.add_argument(
        "--output-path",
        help="Explicit repair destination. Defaults to .../hf-repair-direct-<run-label>/step-<n>.",
    )
    parser.add_argument("--run-label", default="r2")
    parser.add_argument("--job-name", help="Explicit Iris job name override.")
    parser.add_argument("--region", default=DEFAULT_REGION)
    parser.add_argument("--zone")
    parser.add_argument("--cpu", type=float, default=DEFAULT_CPU)
    parser.add_argument("--memory", default=DEFAULT_MEMORY)
    parser.add_argument("--disk", default=DEFAULT_DISK)
    parser.add_argument(
        "--skip-verify-shapes",
        action="store_true",
        help="Pass through to the worker repair script.",
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for the Iris job to finish before returning.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    output_path = args.output_path or default_output_path(args.source_model_path, args.run_label)
    job_name = args.job_name or default_job_name(args.source_model_path, args.run_label)
    constraints = build_constraints(args.region, args.zone)

    command = [
        "python",
        "experiments/posttrain/repair_lora_hf_export.py",
        "--source-model-path",
        args.source_model_path,
        "--output-path",
        output_path,
    ]
    if args.skip_verify_shapes:
        command.append("--skip-verify-shapes")

    logger.info("Submitting Iris repair job %s", job_name)
    logger.info("Source model path: %s", args.source_model_path)
    logger.info("Output path: %s", output_path)

    with controller_client(args.config_file) as client:
        job = client.submit(
            entrypoint=Entrypoint.from_command(*command),
            name=job_name,
            resources=ResourceSpec(cpu=args.cpu, memory=args.memory, disk=args.disk),
            constraints=constraints,
        )
        print(f"job_id={job.job_id}")
        print(f"hf_repair_path={output_path}")
        if args.wait:
            job.wait()
            status = client.status(job.job_id)
            print(f"job_state={status.state}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
