# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris job submission for pinned native sampler requests."""

import logging
import subprocess
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Protocol

from fray.iris_backend import convert_constraints, convert_resources, resolve_coscheduling, wrap_multiprocess
from fray.types import GpuConfig, ResourceConfig
from iris.cli.connect import ControllerEndpoint
from iris.client.client import IrisClient, JobAlreadyExists
from iris.cluster.types import Entrypoint, EnvironmentSpec
from iris.resources.state import TERMINAL_JOB_STATES, JobState
from iris.rpc import job_pb2
from marin.training.training import resolve_training_env
from rigging.timing import Duration

from experiments.grug.moe_hero_ep.ops.vibe_check.completions import SampleRequest, SampleStore

logger = logging.getLogger(__name__)

JOB_USER = "hero-completions"
MAX_ATTEMPTS = 3


class Jobs(Protocol):
    def states(self) -> dict[str, JobState]: ...

    def submit(self, request: SampleRequest, name: str) -> None: ...


def submit_pending(store: SampleStore, jobs: Jobs, requests: list[SampleRequest]) -> None:
    """Start at most one full-checkpoint attempt. The workflow serializes callers."""
    for request in requests:
        store.save_request(request)
    # Read job states before results. Process zero can save a result during this RPC.
    states = jobs.states()
    if any(state not in TERMINAL_JOB_STATES for state in states.values()):
        return  # Wait for teardown even if the active job already wrote its result.
    pending = []
    for request in sorted(store.requests(), key=lambda row: (row.checkpoint.step, row.sample_id), reverse=True):
        if store.result(request) is not None or store.failed(request):
            continue
        for attempt in range(1, MAX_ATTEMPTS + 1):
            name = f"hero-completions-{request.sample_id}-a{attempt}"
            if name not in states:
                pending.append((request, name))
                break
        else:
            error = f"All {MAX_ATTEMPTS} attempts ended without a completed result"
            store.save_failure(request, error)  # Retain the stop marker after Iris prunes terminal jobs.
            logger.error("Sample %s: %s", request.sample_id, error)
    if pending:
        request, name = pending[0]
        jobs.submit(request, name)
        logger.info("Submitted %s for step %d", name, request.checkpoint.step)


class IrisSamplingJobs:
    """Address deterministic jobs and submit their original source snapshot."""

    def __init__(
        self,
        client: IrisClient,
        endpoint: ControllerEndpoint,
        repository: Path,
        store_root: str,
        resources: ResourceConfig,
        processes_per_task: int,
        sampler_module: str,
    ):
        self.client = client
        self.endpoint = endpoint
        self.repository = repository
        self.store_root = store_root
        self.resources = resources
        self.processes_per_task = processes_per_task
        self.sampler_module = sampler_module

    def states(self) -> dict[str, JobState]:
        return {job.job_id.name: job.state for job in self.client.list_jobs(prefix=f"/{JOB_USER}/")}

    def submit(self, request: SampleRequest, name: str) -> None:
        resources = replace(self.resources, target_cluster=request.target_cluster)
        if not isinstance(resources.device, GpuConfig):
            raise ValueError("Native sampling requires GPU resources")
        if resources.device.count * resources.replicas != request.spec.batch_size:
            raise ValueError("Sampling batch size must match the GPU count")
        native_resources = convert_resources(resources)
        command = Entrypoint(
            command=[
                "python",
                "-m",
                self.sampler_module,
                "--request",
                "completion-request.json",
                "--store-root",
                self.store_root,
            ],
            workdir_files={"completion-request.json": request.model_dump_json().encode()},
        )
        environment = resolve_training_env(
            base_env={
                "JAX_PLATFORMS": "cuda",
                "JAX_ENABLE_PGLE": "false",
                "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.75",
                "XLA_PYTHON_CLIENT_ALLOCATOR": "cuda_async",
                "WANDB_MODE": "disabled",
                "GIT_COMMIT": request.source_revision,
            },
            resources=resources,
        )
        with TemporaryDirectory(prefix="hero-completion-source-") as directory:
            snapshot = Path(directory) / "source"
            subprocess.run(
                ["git", "worktree", "add", "--detach", str(snapshot), request.source_revision],
                cwd=self.repository,
                check=True,
            )
            try:
                with IrisClient.remote(
                    self.endpoint.url,
                    credentials=self.endpoint.credentials,
                    workspace=snapshot,
                ) as client:
                    try:
                        client.submit(
                            wrap_multiprocess(command, native_resources, self.processes_per_task),
                            name=name,
                            user=JOB_USER,
                            resources=native_resources,
                            replicas=resources.replicas,
                            environment=EnvironmentSpec(env_vars=environment, extras=["gpu"]),
                            constraints=convert_constraints(resources),
                            coscheduling=resolve_coscheduling(resources, resources.replicas),
                            ports=["jax"],
                            scheduling_timeout=Duration.from_hours(24),
                            timeout=Duration.from_hours(4),
                            max_retries_failure=0,
                            max_retries_preemption=0,
                            max_task_failures=0,
                            priority_band=job_pb2.PRIORITY_BAND_BATCH,
                            existing_job_policy=job_pb2.EXISTING_JOB_POLICY_ERROR,
                        )
                    except JobAlreadyExists:
                        # A lost submit response can address this same attempt.
                        logger.info("Attempt already exists: %s", name)
            finally:
                subprocess.run(["git", "worktree", "remove", "--force", str(snapshot)], cwd=self.repository, check=True)
