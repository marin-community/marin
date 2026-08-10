# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical resource operations over controller state and backend observations."""

from collections.abc import Mapping, Sequence

from rigging.timing import Duration

from iris.backends.protocol import TaskBackend
from iris.cluster.bundle import BundleStore
from iris.cluster.config import BackendConfig
from iris.cluster.controller.activity import ActivityResources
from iris.cluster.controller.attempt import AttemptResources
from iris.cluster.controller.auth import (
    ControllerAuth,
)
from iris.cluster.controller.capacity import CapacityResources
from iris.cluster.controller.dependencies import (
    CapabilityUrlConfig,
    EndpointRegistry,
    ResourceDependencies,
    ResourceRuntime,
)
from iris.cluster.controller.endpoint import EndpointResources
from iris.cluster.controller.job import FederationSubmission, JobService
from iris.cluster.controller.node import NodeDetails, NodeResources
from iris.cluster.controller.persistence.database import ControllerDB
from iris.cluster.controller.profile import ProfileResources
from iris.cluster.controller.slice import SliceResources
from iris.cluster.controller.task import TaskResources
from iris.cluster.controller.user import UserResources
from iris.cluster.types import (
    LOCAL_ADMIN_SUBMITTER,
    UserBudgetDefaults,
)
from iris.resources.action import ActionReceipt
from iris.resources.activity import ActivityEntry, ActivityQuery
from iris.resources.attempt import AttemptDetail
from iris.resources.capacity import CapacityStatus
from iris.resources.endpoint import (
    EndpointDetail,
    EndpointQuery,
    EndpointSummary,
    EndpointToken,
    ExecResult,
    ProfileConfiguration,
    ProfileResult,
)
from iris.resources.identity import (
    AttemptIdentity,
    AttemptLocator,
    JobIdentity,
    NodeLocator,
    ResourceKey,
    SliceLocator,
    TaskIdentity,
)
from iris.resources.job import (
    JobDetail,
    JobInventoryPage,
    JobInventoryQuery,
    JobObservation,
    JobQuery,
    JobSpec,
    JobSummary,
)
from iris.resources.log import LogPage, LogQuery, LogReader
from iris.resources.names import JobName
from iris.resources.node import (
    NodeDetail,
    NodeQuery,
    NodeSummary,
)
from iris.resources.slice import (
    SliceDetail,
    SliceQuery,
    SliceSummary,
)
from iris.resources.source import (
    Page,
)
from iris.resources.state import JobState
from iris.resources.task import TaskDetail, TaskQuery, TaskSummary
from iris.resources.user import UserSummary


class Controller:
    """Typed controller application service composed from resource nouns."""

    def __init__(
        self,
        *,
        cluster_id: str,
        db: ControllerDB,
        runtime: ResourceRuntime,
        bundle_store: BundleStore,
        endpoint_registry: EndpointRegistry,
        auth: ControllerAuth,
        user_budget_defaults: UserBudgetDefaults,
        capability_url_config: CapabilityUrlConfig,
        backends: Mapping[str, TaskBackend],
        backend_configs: Mapping[str, BackendConfig],
        log_reader: LogReader | None = None,
    ) -> None:
        dependencies = ResourceDependencies(
            cluster_id=cluster_id,
            db=db,
            runtime=runtime,
            endpoint_registry=endpoint_registry,
            auth=auth,
            capability_url_config=capability_url_config,
            backends=backends,
            backend_configs=backend_configs,
            log_reader=log_reader,
        )
        self._dependencies = dependencies
        self._jobs = JobService(
            dependencies,
            bundle_store=bundle_store,
            auth=auth,
            user_budget_defaults=user_budget_defaults,
        )
        self._tasks = TaskResources(dependencies)
        self._attempts = AttemptResources(dependencies, self._tasks)
        self._profiles = ProfileResources(dependencies, self._attempts)
        self._capacity = CapacityResources(dependencies)
        self._endpoints = EndpointResources(dependencies)
        self._nodes = NodeResources(dependencies)
        self._slices = SliceResources(dependencies)
        self._activity = ActivityResources(dependencies, self._jobs, self._tasks, self._attempts)
        self._users = UserResources(dependencies)

    @property
    def cluster_id(self) -> str:
        return self._dependencies.cluster_id

    def received_job_from_peer(self, root_job: JobName, peer_id: str) -> bool:
        return self._jobs.received_job_from_peer(root_job, peer_id)

    def submit_job(
        self, spec: JobSpec, bundle_blob: bytes = b"", *, enforce_client_freshness: bool = True
    ) -> JobIdentity:
        return self._jobs.submit_job(spec, bundle_blob, enforce_client_freshness=enforce_client_freshness)

    def submit_federated_job(self, spec: JobSpec, bundle_blob: bytes, federation: FederationSubmission) -> JobIdentity:
        return self._jobs.submit_federated_job(spec, bundle_blob, federation)

    def list_jobs(self, query: JobQuery = JobQuery()) -> Page[JobSummary]:
        return self._jobs.list_jobs(query)

    def list_users(self) -> tuple[UserSummary, ...]:
        return self._users.list_users()

    def describe_job(self, key: ResourceKey) -> JobDetail:
        return self._jobs.describe_job(key)

    def job_states(self, resource_ids: Sequence[str]) -> dict[str, JobState]:
        return self._jobs.job_states(resource_ids)

    def job_state(self, identity: JobIdentity) -> JobState:
        return self._jobs.job_state(identity)

    def list_job_inventory(self, query: JobInventoryQuery) -> JobInventoryPage:
        return self._jobs.list_job_inventory(query)

    def observe_jobs(self, summaries: Sequence[JobSummary]) -> tuple[JobObservation, ...]:
        return self._jobs.observe_jobs(summaries)

    def cancel_job(
        self, identity: JobIdentity, *, idempotency_key: str, principal_id: str = LOCAL_ADMIN_SUBMITTER
    ) -> ActionReceipt:
        return self._jobs.cancel_job(identity, idempotency_key=idempotency_key, principal_id=principal_id)

    def list_tasks(self, query: TaskQuery = TaskQuery()) -> Page[TaskSummary]:
        return self._tasks.list_tasks(query)

    def describe_task(self, key: ResourceKey) -> TaskDetail:
        return self._tasks.describe_task(key)

    def describe_tasks(self, keys: Sequence[ResourceKey]) -> tuple[TaskDetail, ...]:
        return self._tasks.describe_tasks(keys)

    def require_task(self, identity: TaskIdentity) -> TaskDetail:
        return self._tasks.require_task(identity)

    def describe_attempt(self, locator: AttemptLocator) -> AttemptDetail:
        return self._attempts.describe_attempt(locator)

    def require_attempt(self, identity: AttemptIdentity) -> AttemptDetail:
        return self._attempts.require_attempt(identity)

    def retry_task(
        self,
        identity: TaskIdentity,
        *,
        expected_attempt_uid: str,
        idempotency_key: str,
        principal_id: str = LOCAL_ADMIN_SUBMITTER,
    ) -> ActionReceipt:
        return self._attempts.retry_task(
            identity,
            expected_attempt_uid=expected_attempt_uid,
            idempotency_key=idempotency_key,
            principal_id=principal_id,
        )

    def terminate_attempt(
        self, identity: AttemptIdentity, *, idempotency_key: str, principal_id: str = LOCAL_ADMIN_SUBMITTER
    ) -> ActionReceipt:
        return self._attempts.terminate_attempt(identity, idempotency_key=idempotency_key, principal_id=principal_id)

    def get_action_receipt(self, action_id: str) -> ActionReceipt:
        return self._attempts.get_action_receipt(action_id)

    def list_endpoints(self, query: EndpointQuery = EndpointQuery()) -> Page[EndpointSummary]:
        return self._endpoints.list_endpoints(query)

    def describe_endpoint(self, key: ResourceKey) -> EndpointDetail:
        return self._endpoints.describe_endpoint(key)

    def describe_endpoints(self, keys: Sequence[ResourceKey]) -> tuple[EndpointDetail, ...]:
        return self._endpoints.describe_endpoints(keys)

    def mint_endpoint_token(self, key: ResourceKey, ttl: Duration | None) -> EndpointToken:
        return self._endpoints.mint_endpoint_token(key, ttl)

    def fetch_logs(self, target: JobIdentity | TaskIdentity | AttemptIdentity, query: LogQuery = LogQuery()) -> LogPage:
        return self._activity.fetch_logs(target, query)

    def list_activity(self, query: ActivityQuery) -> Page[ActivityEntry]:
        return self._activity.list_activity(query)

    def exec_attempt(self, identity: AttemptIdentity, command: tuple[str, ...], timeout: Duration | None) -> ExecResult:
        return self._profiles.exec_attempt(identity, command, timeout)

    def profile_attempt(
        self, identity: AttemptIdentity, profile: ProfileConfiguration | None, duration: Duration | None
    ) -> ProfileResult:
        return self._profiles.profile_attempt(identity, profile, duration)

    def list_nodes(self, query: NodeQuery = NodeQuery()) -> Page[NodeSummary]:
        return self._nodes.list_nodes(query)

    def list_nodes_with_details(
        self, query: NodeQuery = NodeQuery()
    ) -> tuple[Page[NodeSummary], Mapping[tuple[str, str], NodeDetails]]:
        return self._nodes.list_nodes_with_details(query)

    def describe_node(self, locator: NodeLocator) -> NodeDetail:
        return self._nodes.describe_node(locator)

    def list_slices(self, query: SliceQuery = SliceQuery()) -> Page[SliceSummary]:
        return self._slices.list_slices(query)

    def describe_slice(self, locator: SliceLocator) -> SliceDetail:
        return self._slices.describe_slice(locator)

    def capacity_status(self) -> CapacityStatus:
        return self._capacity.status()
