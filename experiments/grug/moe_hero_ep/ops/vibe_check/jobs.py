# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris job submission for pinned native sampler requests."""

import logging
import subprocess
from dataclasses import replace
from enum import StrEnum
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
from iris.rpc.proto_display import priority_band_rank
from marin.training.training import resolve_training_env
from rigging.timing import Duration

from experiments.grug.moe_hero_ep.ops.vibe_check.completions import SampleRequest, SampleStore, SamplingSpec

logger = logging.getLogger(__name__)

JOB_USER = "hero-completions"
MAX_ATTEMPTS = 3


class SubmissionMode(StrEnum):
    NEXT = "next"
    ALL = "all"


class Jobs(Protocol):
    def states(self) -> dict[str, JobState]: ...

    def submit(self, request: SampleRequest, name: str, priority_band: int) -> None:
        """Submit with ERROR-on-exists. Never replace an existing job."""
        ...


def sample_job_names(request: SampleRequest) -> list[str]:
    return [f"hero-completions-{request.sample_id}-a{attempt}" for attempt in range(1, MAX_ATTEMPTS + 1)]


def submit_pending(
    store: SampleStore,
    jobs: Jobs,
    requests: list[SampleRequest],
    *,
    spec: SamplingSpec,
    priority_band: int | None = None,
    submission: SubmissionMode = SubmissionMode.NEXT,
) -> None:
    """Submit the next request or all discovered requests. The workflow serializes callers."""
    if priority_band is not None:
        priority_band_rank(priority_band)
    for request in requests:
        store.save_request(request)
    if priority_band is not None:
        store.set_priorities([request.sample_id for request in requests], priority_band)
    # Read job states before results. Process zero can save a result during this RPC.
    states = jobs.states()
    saved_requests = store.requests(spec)
    current_names = {name for request in saved_requests for name in sample_job_names(request)}
    active = {name for name, state in states.items() if name in current_names and state not in TERMINAL_JOB_STATES}
    if active and submission == SubmissionMode.NEXT:
        logger.info("Waiting for active jobs: %s", active)
        return  # Wait for teardown even if the active job already wrote its result.
    completed = store.completed_ids()
    attempts = store.attempt_names()
    priorities = store.priorities()
    if submission == SubmissionMode.ALL:
        discovered_ids = {request.sample_id for request in requests}
        saved_requests = [request for request in saved_requests if request.sample_id in discovered_ids]
    pending = []
    for request in sorted(
        saved_requests,
        key=lambda row: (
            priority_band_rank(priorities.get(row.sample_id, job_pb2.PRIORITY_BAND_BATCH)),
            -row.checkpoint.step,
            row.sample_id,
        ),
    ):
        if request.sample_id in completed or store.retries_exhausted(request):
            continue
        names = sample_job_names(request)
        if active.intersection(names):
            continue
        for name in names:
            if name not in attempts and name not in states:
                pending.append((request, name))
                break
        else:
            error = "\n".join(f"{name}: {states.get(name, 'missing')}" for name in names)
            store.save_failure(request, error)  # Retain the stop marker after Iris prunes terminal jobs.
            logger.error("Sample %s: %s", request.sample_id, error)
    if not pending:
        logger.info("No pending sample sets; %d completed", len(completed))
        return
    selected = pending[:1] if submission == SubmissionMode.NEXT else pending
    for request, name in selected:
        store.save_attempt(name)  # A lost or pruned job still consumes this attempt.
        jobs.submit(request, name, priorities.get(request.sample_id, job_pb2.PRIORITY_BAND_BATCH))
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
        return {
            job.job_id.name: job.state for job in self.client.list_jobs(prefix=f"/{JOB_USER}/") if job.job_id.is_root
        }

    def submit(self, request: SampleRequest, name: str, priority_band: int) -> None:
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
                            max_retries_preemption=1000,
                            max_task_failures=0,
                            priority_band=priority_band,
                            existing_job_policy=job_pb2.EXISTING_JOB_POLICY_ERROR,
                        )
                    except JobAlreadyExists:
                        # A lost submit response can address this same attempt.
                        logger.warning("Attempt already exists: %s", name)
            finally:
                subprocess.run(["git", "worktree", "remove", "--force", str(snapshot)], cwd=self.repository, check=True)
