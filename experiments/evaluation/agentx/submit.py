# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit a fixed point without holding a local control process open."""

import hashlib
import json

from iris.client.client import IrisClient
from iris.cluster.types import CoschedulingConfig, Entrypoint, EnvironmentSpec, ResourceSpec, gpu_device
from iris.rpc import job_pb2
from rigging.timing import Duration

from experiments.evaluation.agentx.recipe import SERIES_PATH, PointManifest, load_series, series_path
from experiments.evaluation.agentx.run import run
from experiments.evaluation.agentx.runtime import runtime_metadata


def point_manifest(
    profile: str, concurrency: int, source: dict, workload: dict, release: dict, placement: dict
) -> PointManifest:
    path = series_path(profile)
    series = load_series(path)
    requirements = SERIES_PATH.with_name("requirements-harness.txt").read_bytes()
    recipe = {
        "series": (
            series
            | {
                "profiles": {
                    profile: {key: value for key, value in series["profiles"][profile].items() if key != "concurrencies"}
                }
            }
        ),
        "harness_requirements_sha256": hashlib.sha256(requirements).hexdigest(),
        "preparation_key": workload["key"],
    }
    fingerprint = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    return {
        "series_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "recipe_fingerprint": fingerprint,
        "profile": profile,
        "concurrency": concurrency,
        "vllm_release": release,
        "workload": workload,
        "source": source,
        "placement": placement,
    }


def submit_point(client: IrisClient, name: str, manifest: PointManifest, priority: str):
    """Submit a child on the coordinator's peer with native whole-attempt retries."""
    path = series_path(manifest["profile"])
    series = load_series(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["series_sha256"]:
        raise ValueError("The submission checkout does not match the saved series")
    if json.dumps(runtime_metadata(manifest["profile"]), sort_keys=True) != json.dumps(
        manifest["vllm_release"], sort_keys=True
    ):
        raise ValueError("The submission checkout changed the saved profile runtime")
    profile = series["profiles"][manifest["profile"]]
    return client.submit(
        entrypoint=Entrypoint.from_callable(run, manifest),
        name=name,
        resources=ResourceSpec(
            cpu=profile["cpu"],
            memory=profile["memory"],
            disk=profile["disk"],
            device=gpu_device(profile["gpu"], profile["gpus"]),
        ),
        environment=EnvironmentSpec(sync_packages=["marin-core"], extras=["cpu"]),
        # Cluster pins route roots through federation; a peer cannot route its own
        # children again. Clear inherited routing pins; Iris retains the peer region.
        constraints=[],
        replicas=profile["hosts"],
        coscheduling=CoschedulingConfig(group_by="nvlink.domain") if profile["hosts"] > 1 else None,
        priority_band={"batch": job_pb2.PRIORITY_BAND_BATCH, "interactive": job_pb2.PRIORITY_BAND_INTERACTIVE}[priority],
        scheduling_timeout=Duration.from_seconds(series["scheduling_timeout"]),
        timeout=Duration.from_seconds(series["task_timeout"]),
        max_retries_failure=series["max_retries_failure"],
        max_task_failures=series["max_retries_failure"],
        max_retries_preemption=series["max_retries_preemption"],
        existing_job_policy=job_pb2.EXISTING_JOB_POLICY_KEEP,
        task_image=series["task_image"],
    )
