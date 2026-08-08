# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed Job specifications, queries, and read records."""

from dataclasses import dataclass

from rigging.timing import Duration, Timestamp

from iris.cluster.constraints import Constraint
from iris.cluster.resources.identity import JobIdentity, ResourceKey
from iris.cluster.types import CoschedulingConfig, JobState, ResourceSpec
from iris.rpc import job_pb2


@dataclass(frozen=True, slots=True)
class JobSpec:
    version: int
    name: str
    entrypoint: job_pb2.RuntimeEntrypoint
    resources: ResourceSpec
    environment: job_pb2.EnvironmentConfig
    bundle_id: str
    scheduling_timeout: Duration | None
    ports: tuple[str, ...]
    max_task_failures: int
    max_retries_failure: int
    max_retries_preemption: int
    constraints: tuple[Constraint, ...]
    coscheduling: CoschedulingConfig | None
    replicas: int
    timeout: Duration | None
    fail_if_exists: bool
    preemption_policy: job_pb2.JobPreemptionPolicy
    existing_job_policy: job_pb2.ExistingJobPolicy
    priority_band: job_pb2.PriorityBand
    task_image: str
    submit_argv: tuple[str, ...]
    client_revision_date: str
    container_profile: job_pb2.ContainerProfile


@dataclass(frozen=True, slots=True)
class JobQuery:
    owner_id: str | None = None
    parent: ResourceKey | None = None
    job_id_prefix: str | None = None
    states: frozenset[JobState] = frozenset()
    backend_id: str | None = None
    execution_cluster_id: str | None = None
    page_size: int = 50
    page_token: str | None = None


@dataclass(frozen=True, slots=True)
class JobSummary:
    identity: JobIdentity
    owner_id: str
    parent: JobIdentity | None
    state: JobState
    execution_cluster_id: str
    backend_id: str
    num_tasks: int
    submitted_at: Timestamp
    started_at: Timestamp | None
    finished_at: Timestamp | None
    error_message: str
    pending_reason: str


@dataclass(frozen=True, slots=True)
class JobDetail:
    summary: JobSummary
    spec: JobSpec
