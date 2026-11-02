# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run Snowball next-token probability and repeatability checks on H100 or TPU."""

import argparse
import dataclasses
import hashlib
import json
import os
import time
from pathlib import Path

from fray.iris_backend import convert_constraints, convert_resources, resolve_coscheduling
from fray.types import ResourceConfig
from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.client.job_info import get_job_info
from iris.cluster.types import Entrypoint, EnvironmentSpec
from marin.inference.iris_tpu_vllm import RAY_PORTS
from marin.testing.inference import snowball
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import Duration

H100_WEIGHTS = snowball.SNOWBALL.export_uri
V4_WEIGHTS = "gs://marin-us-central2/scratch/karan/task336/weights/june-42150-d819cbc63780bd86/"
V4_MANIFEST = "gs://marin-us-central2/scratch/karan/task336/staging/manifest.json"
RUNTIME_SECONDS = 3600


def run_arm(hardware: str, weights: str, output: str, cpus_per_host: float) -> None:
    """Own one backend lifecycle and publish a result only after all gates pass."""
    # This module includes optional JAX/Levanter scorer dependencies. Import on
    # the Iris CPU coordinator, never while constructing the local dry plan.
    from marin.testing.inference import snowball_backend_parity_jobs as jobs  # noqa: PLC0415

    configure_coreweave_s3()
    info = get_job_info()
    if info is None:
        raise RuntimeError("Run this check inside Iris")
    root = StoragePath(output)
    started = time.monotonic()
    if info.task_index == 0:
        if (root / "started.json").exists():
            raise ValueError("Use a fresh output prefix for each run")
        manifest = None
        if hardware == "v4":
            manifest = json.loads(StoragePath(V4_MANIFEST).read_bytes())
            assert manifest["verified"] and manifest["destination"] == weights
            assert manifest["source"] == snowball.SNOWBALL.export_uri
            assert manifest["export_sha256"] == snowball.SNOWBALL.export_sha256
        (root / "started.json").write_text(
            json.dumps(
                dict(
                    hardware=hardware,
                    job_id=str(info.job_id),
                    model=dataclasses.asdict(snowball.SNOWBALL),
                    staged_manifest=manifest,
                    scorer_sha256=hashlib.sha256(Path(jobs.__file__).read_bytes()).hexdigest(),
                ),
                indent=2,
            )
        )
    goldens = snowball.read_representative_goldens()
    if hardware == "h100":
        parities = jobs.score_vllm_against_goldens(goldens, "FLASH_ATTN", 1)
    else:
        parities = jobs.score_tpu_vllm_against_goldens(goldens, weights, cpus_per_host)
    if parities is None:
        return
    result = dict(
        hardware=hardware,
        export_sha256=snowball.SNOWBALL.export_sha256,
        max_probability_error=jobs.VLLM_MAX_PROBABILITY_ERROR,
        parities=[dataclasses.asdict(parity) for parity in parities],
        same_rank_repeatability=True,
        wall_seconds=time.monotonic() - started,
    )
    (root / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != "parities"}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", choices=("h100", "v4"), required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--weights", help="Exact June d819 export; TPU uses its verified regional mirror")
    parser.add_argument("--output", required=True, help="Fresh S3 (H100) or GCS (TPU) prefix")
    parser.add_argument("--submit", action="store_true", help="Otherwise print the plan only")
    parser.add_argument("--scheduling-timeout", type=int, default=3600, help="Queue bound in seconds")
    parser.add_argument("--job-id-file", type=Path, help="Write submitted job ID for workflow monitoring")
    args = parser.parse_args()
    gpu = args.hardware == "h100"
    weights = H100_WEIGHTS if gpu else V4_WEIGHTS
    if args.weights is not None and args.weights != weights:
        parser.error("Use the verified June export for the selected hardware")
    resources = (
        ResourceConfig.with_gpu("H100", count=8, cpu=64, ram="512g", disk="128g", target_cluster="cw-us-east-02a")
        if gpu
        else ResourceConfig.with_tpu(
            "v4-32", cpu=32, ram="360g", disk="100g", regions=["us-central2"], zone="us-central2-b", preemptible=False
        )
    )
    ports = [*RAY_PORTS] if args.hardware == "v4" else []
    coscheduling = resolve_coscheduling(resources, resources.replicas)
    print(
        json.dumps(
            dict(
                vars(args),
                weights=weights,
                cpus_per_host=resources.cpu,
                replicas=resources.replicas,
                ports=ports,
                coscheduling=str(coscheduling),
            ),
            indent=2,
            default=str,
        )
    )
    if not args.submit:
        return
    with connect_controller(cluster_name="marin") as endpoint:
        with IrisClient.remote(endpoint.url, workspace=Path.cwd(), credentials=endpoint.credentials) as client:
            job = client.submit(
                name=args.name,
                entrypoint=Entrypoint.from_callable(run_arm, args.hardware, weights, args.output, resources.cpu),
                replicas=resources.replicas,
                coscheduling=coscheduling,
                ports=ports,
                resources=convert_resources(resources),
                constraints=convert_constraints(resources),
                environment=EnvironmentSpec(
                    extras=("cpu",),
                    env_vars={
                        **{
                            key: os.environ[key]
                            for key in ("CW_KEY_ID", "CW_KEY_SECRET", "HF_TOKEN")
                            if key in os.environ
                        },
                        "JAX_PLATFORMS": "cpu",
                        "VLLM_BATCH_INVARIANT": "1",
                        "VLLM_USE_FLASHINFER_SAMPLER": "0",
                    },
                ),
                timeout=Duration.from_seconds(RUNTIME_SECONDS),
                scheduling_timeout=Duration.from_seconds(args.scheduling_timeout),
                max_retries_failure=0,
                max_retries_preemption=0,
                max_task_failures=0,
            )
            if args.job_id_file:
                args.job_id_file.write_text(str(job))
            print(job)


if __name__ == "__main__":
    main()
