# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris job submission for pinned native sampler requests."""

import subprocess
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from fray.iris_backend import convert_constraints, convert_resources, resolve_coscheduling, wrap_multiprocess
from fray.types import GpuConfig, ResourceConfig
from iris.cli.connect import ControllerEndpoint
from iris.client.client import IrisClient, JobAlreadyExists
from iris.cluster.types import Entrypoint, EnvironmentSpec, JobName
from iris.resources.state import TERMINAL_JOB_STATES, JobState
from iris.rpc import job_pb2
from marin.training.training import resolve_training_env
from rigging.timing import Duration

from experiments.grug.moe_hero_ep.ops.vibe_check.completions import Entry, JobStatus

JOB_USER = "hero-completions"


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

    def status(self, name: str) -> JobStatus:
        try:
            state = self.client.job(JobName.root(JOB_USER, name)).state
        except ConnectError as error:
            if error.code != Code.NOT_FOUND:
                raise
            return JobStatus.MISSING
        if state == JobState.SUCCEEDED:
            return JobStatus.SUCCEEDED
        if state == JobState.UNSCHEDULABLE:
            return JobStatus.DEFERRED
        if state in TERMINAL_JOB_STATES:
            return JobStatus.FAILED
        return JobStatus.RUNNING

    def submit(self, entry: Entry) -> None:
        resources = replace(self.resources, target_cluster=entry.request.target_cluster)
        if not isinstance(resources.device, GpuConfig):
            raise ValueError("Native sampling requires GPU resources")
        if resources.device.count * resources.replicas != entry.request.spec.batch_size:
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
            workdir_files={"completion-request.json": entry.request.model_dump_json().encode()},
        )
        environment = resolve_training_env(
            base_env={
                "JAX_PLATFORMS": "cuda",
                "JAX_ENABLE_PGLE": "false",
                "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.75",
                "XLA_PYTHON_CLIENT_ALLOCATOR": "cuda_async",
                "WANDB_MODE": "disabled",
                "GIT_COMMIT": entry.request.source_revision,
            },
            resources=resources,
        )
        with TemporaryDirectory(prefix="hero-completion-source-") as directory:
            snapshot = Path(directory) / "source"
            subprocess.run(
                ["git", "worktree", "add", "--detach", str(snapshot), entry.request.source_revision],
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
                            name=entry.job_name,
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
                            priority_band=job_pb2.PRIORITY_BAND_BATCH,
                            existing_job_policy=job_pb2.EXISTING_JOB_POLICY_ERROR,
                        )
                    except JobAlreadyExists:
                        # A lost submit response or concurrent tick can address this same attempt.
                        if self.status(entry.job_name) == JobStatus.MISSING:
                            raise
            finally:
                subprocess.run(["git", "worktree", "remove", "--force", str(snapshot)], cwd=self.repository, check=True)
