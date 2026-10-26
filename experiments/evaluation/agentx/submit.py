# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit a fixed point without holding a local control process open."""

import dataclasses
import hashlib
import json

from iris.client.client import IrisClient
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec, gpu_device
from iris.rpc import job_pb2
from marin.external_dependencies import VLLM_GPU_RELEASE
from rigging.timing import Duration

from experiments.evaluation.agentx.recipe import SERIES_PATH


def point_manifest(profile: str, concurrency: int, source: dict, workload: dict, release: dict) -> dict:
    series = json.loads(SERIES_PATH.read_text())
    requirements = SERIES_PATH.with_name("requirements-harness.txt").read_bytes()
    recipe = {
        "series": series,
        "harness_requirements_sha256": hashlib.sha256(requirements).hexdigest(),
        "preparation_key": workload["key"],
    }
    fingerprint = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    return {
        "series_sha256": hashlib.sha256(SERIES_PATH.read_bytes()).hexdigest(),
        "recipe_fingerprint": fingerprint,
        "profile": profile,
        "concurrency": concurrency,
        "vllm_release": release,
        "workload": workload,
        "source": source,
    }


def submit_point(client: IrisClient, name: str, manifest: dict, priority: str):
    """Submit a child on the coordinator's peer with native whole-attempt retries."""
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
        # Cluster pins route roots through federation; a peer cannot route its own
        # children again. Clear inherited routing pins; Iris retains the peer region.
        constraints=[],
        priority_band={"batch": job_pb2.PRIORITY_BAND_BATCH, "interactive": job_pb2.PRIORITY_BAND_INTERACTIVE}[priority],
        scheduling_timeout=Duration.from_seconds(series["scheduling_timeout"]),
        timeout=Duration.from_seconds(series["scheduling_timeout"] + series["task_timeout"]),
        max_retries_failure=series["max_retries_failure"],
        max_task_failures=series["max_retries_failure"],
        max_retries_preemption=series["max_retries_preemption"],
        existing_job_policy=job_pb2.EXISTING_JOB_POLICY_KEEP,
        task_image=series["task_image"],
    )
