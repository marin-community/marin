# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit a fixed point without holding a local control process open."""

import argparse
import dataclasses
import hashlib
import json
import subprocess
from pathlib import Path

from iris.cli.connect import connect_controller
from iris.client.client import IrisClient
from iris.cluster.constraints import Constraint, ConstraintOp
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec, gpu_device
from iris.rpc import job_pb2
from marin.external_dependencies import VLLM_GPU_RELEASE
from rigging.timing import Duration

SERIES_PATH = Path(__file__).with_name("series.json")


def point_manifest(profile: str, concurrency: int, source: dict) -> dict:
    series = json.loads(SERIES_PATH.read_text())
    requirements = SERIES_PATH.with_name("requirements-harness.txt").read_bytes()
    recipe = {"series": series, "harness_requirements_sha256": hashlib.sha256(requirements).hexdigest()}
    fingerprint = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    return {
        "series_sha256": hashlib.sha256(SERIES_PATH.read_bytes()).hexdigest(),
        "recipe_fingerprint": fingerprint,
        "profile": profile,
        "concurrency": concurrency,
        "vllm_release": dataclasses.asdict(VLLM_GPU_RELEASE),
        "source": source,
    }


def submit_point(client: IrisClient, name: str, manifest: dict, cluster: str, priority: str):
    """Use Iris's stable name and native retry budgets; never recreate a completed job."""
    series = json.loads(SERIES_PATH.read_text())
    if hashlib.sha256(SERIES_PATH.read_bytes()).hexdigest() != manifest["series_sha256"]:
        raise ValueError("The submission checkout does not match the saved series")
    if json.dumps(dataclasses.asdict(VLLM_GPU_RELEASE), sort_keys=True) != json.dumps(
        manifest["vllm_release"], sort_keys=True
    ):
        raise ValueError("The submission checkout changed the saved vLLM adoption")
    profile = series["profiles"][manifest["profile"]]
    entrypoint = Entrypoint(
        command=[
            "timeout",
            "--signal=TERM",
            "--kill-after=60s",
            str(series["task_timeout"]),
            "uv",
            "run",
            "--frozen",
            "--python",
            "3.12",
            "--package",
            "marin-core",
            "--extra",
            "cpu",
            "--no-default-groups",
            "python",
            "-m",
            "experiments.evaluation.agentx.run",
            "--manifest",
            "agentx-point.json",
        ],
        workdir_files={"agentx-point.json": json.dumps(manifest, sort_keys=True).encode()},
    )
    return client.submit(
        entrypoint=entrypoint,
        name=name,
        resources=ResourceSpec(
            cpu=profile["cpu"],
            memory=profile["memory"],
            disk=profile["disk"],
            device=gpu_device(profile["gpu"], profile["gpus"]),
        ),
        environment=EnvironmentSpec(setup_scripts=[]),
        constraints=[Constraint.create(key="cluster", op=ConstraintOp.EQ, value=cluster)],
        priority_band={"batch": job_pb2.PRIORITY_BAND_BATCH, "interactive": job_pb2.PRIORITY_BAND_INTERACTIVE}[priority],
        scheduling_timeout=Duration.from_seconds(series["scheduling_timeout"]),
        timeout=Duration.from_seconds(series["scheduling_timeout"] + series["task_timeout"]),
        max_retries_failure=series["max_retries_failure"],
        max_task_failures=series["max_retries_failure"],
        max_retries_preemption=series["max_retries_preemption"],
        existing_job_policy=job_pb2.EXISTING_JOB_POLICY_KEEP,
        task_image=series["task_image"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--profile", choices=["h100", "gb200"], required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--cluster", choices=["cw-rno2a", "cw-us-east-02a", "cw-us-east-08a"], required=True)
    parser.add_argument("--priority", choices=["batch", "interactive"], default="batch")
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Real GitHub run provenance, or explicitly marked local validation metadata",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = point_manifest(args.profile, args.concurrency, json.loads(args.source.read_text()))
    manifest["workspace_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest["workspace_diff_sha256"] = hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"])).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Persist intent before the submission RPC. The deterministic name survives a lost response.
    args.output.write_text(json.dumps({"name": args.name, "manifest": manifest}, indent=2))
    with connect_controller(cluster_name="marin") as endpoint:
        with IrisClient.remote(endpoint.url, credentials=endpoint.credentials, workspace=Path.cwd()) as client:
            job = submit_point(client, args.name, manifest, args.cluster, args.priority)
            args.output.write_text(json.dumps({"job_id": str(job.job_id), "manifest": manifest}, indent=2))
            print(job.job_id)


if __name__ == "__main__":
    main()
