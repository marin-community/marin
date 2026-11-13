# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit one ``reference_hero`` job: four GB200 nodes, one JAX process per GPU, interactive priority.

The launch mirrors the native Hero sampler (``experiments/grug/moe_hero_ep/ops/vibe_check/jobs.py``).

    uv run python experiments/evaluation/hero_vllm_hillclimb/submit_reference.py --name ref-a1
"""

import argparse
from pathlib import Path

from fray.iris_backend import convert_constraints, convert_resources, resolve_coscheduling, wrap_multiprocess
from fray.types import GpuConfig, ResourceConfig
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.types import Entrypoint, EnvironmentSpec
from iris.rpc import job_pb2
from marin.training.training import resolve_training_env
from rigging.timing import Duration

CASES_URI = "s3://marin-us-east-02a/tmp/ttl=14d/hero-vllm-hillclimb/inputs/cases-0baf4938.json"
EXPORT_REQUEST = (
    "s3://marin-us-east-02a/marin/exports/mochi-sft/hero-sft-fullab-step393k-v2/step-400407/2af3f828aa-cpu-a1/"
    "export-request.json"
)
OUTPUT_ROOT = "s3://marin-us-east-02a/tmp/ttl=14d/hero-vllm-hillclimb/reference"
NODES = 4
GPUS_PER_NODE = 4
TIMEOUT = Duration.from_hours(3)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--long-lengths", default="4096,32768,65536")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    resources = ResourceConfig(
        cpu=120,
        ram="890g",
        disk="1t",
        device=GpuConfig(variant="GB200", count=GPUS_PER_NODE),
        replicas=NODES,
        target_cluster="cw-us-east-08a",
    )
    output = f"{OUTPUT_ROOT}/{args.name}"
    command = Entrypoint(
        command=[
            "python",
            "-m",
            "experiments.evaluation.hero_vllm_hillclimb.reference_hero",
            "--cases",
            CASES_URI,
            "--export-request",
            EXPORT_REQUEST,
            "--output",
            output,
            "--long-lengths",
            args.long_lengths,
        ]
    )
    environment = resolve_training_env(
        base_env={
            "JAX_PLATFORMS": "cuda",
            "JAX_ENABLE_PGLE": "false",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.75",
            "XLA_PYTHON_CLIENT_ALLOCATOR": "cuda_async",
            "WANDB_MODE": "disabled",
        },
        resources=resources,
    )
    native = convert_resources(resources)
    print(f"output={output} replicas={resources.replicas} coscheduling={resolve_coscheduling(resources, NODES)}")
    if args.dry_run:
        return
    with connect_controller(cluster_name="marin") as endpoint:
        with IrisClient.remote(endpoint.url, credentials=endpoint.credentials, workspace=Path.cwd()) as client:
            job = client.submit(
                wrap_multiprocess(command, native, GPUS_PER_NODE),
                name=f"hero-vllm-reference-{args.name}",
                resources=native,
                replicas=resources.replicas,
                environment=EnvironmentSpec(env_vars=environment, extras=["gpu"]),
                constraints=convert_constraints(resources),
                coscheduling=resolve_coscheduling(resources, resources.replicas),
                ports=["jax"],
                scheduling_timeout=Duration.from_hours(12),
                timeout=TIMEOUT,
                max_retries_failure=0,
                max_retries_preemption=2,
                max_task_failures=0,
                priority_band=job_pb2.PRIORITY_BAND_INTERACTIVE,
            )
            print(f"submitted {job}")


if __name__ == "__main__":
    main()
