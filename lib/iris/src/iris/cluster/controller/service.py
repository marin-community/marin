# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Thin protobuf adapter for the controller's operation modules."""

from connectrpc.request import RequestContext
from finelog.client import LogClient
from rigging.timing import Timer

from iris.cluster.bundle import BundleStore
from iris.cluster.controller import (
    accounts,
    artifacts,
    attempts,
    backend_status,
    checkpoint,
    diagnostics,
    federation_service,
    jobs,
    tasks,
    workers,
)
from iris.cluster.controller.auth import ControllerAuth
from iris.cluster.controller.backend import TaskBackend
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.endpoint_service import EndpointServiceImpl
from iris.cluster.controller.runtime_interface import ControllerRuntime
from iris.cluster.stats.tables import (
    PROFILE_NAMESPACE,
    TASK_EVENT_NAMESPACE,
    TASK_EVENT_STORAGE_POLICY,
    IrisProfile,
    TaskEventRow,
)
from iris.cluster.types import UserBudgetDefaults
from iris.rpc import controller_pb2, job_pb2, query_pb2


class ControllerServiceImpl:
    """ControllerService RPC implementation.

    Args:
        controller: Controller runtime for scheduling and worker management
        bundle_store: Bundle store for zip storage.
        log_client: Finelog client for task summaries and profile records.
        db: Underlying database connection.
    """

    def __init__(
        self,
        controller: ControllerRuntime,
        bundle_store: BundleStore,
        log_client: LogClient,
        *,
        db: ControllerDB,
        endpoint_service: EndpointServiceImpl,
        auth: ControllerAuth | None = None,
        user_budget_defaults: UserBudgetDefaults | None = None,
        capability_url_config: attempts.CapabilityUrlConfig | None = None,
    ):
        # Every cursor this DB mints carries the per-controller cache registry as
        # ``tx.caches``, so cache-touching reads/writes reach the derived-count memo
        # and the endpoint projection through the cursor — no cache reference held.
        self._db = db
        self._controller = controller
        self._endpoint_service = endpoint_service
        self._bundle_store = bundle_store
        self._log_client = log_client
        self._timer = Timer()
        self._auth = auth or ControllerAuth()
        self._accounts = accounts.AccountDependencies(db=self._db, auth=self._auth)
        self._artifacts = artifacts.ArtifactDependencies(bundles=self._bundle_store)
        self._diagnostics = diagnostics.DiagnosticDependencies(db=self._db)
        self._checkpoint = checkpoint.CheckpointDependencies(runtime=controller)
        self._workers = workers.WorkerDependencies(db=self._db, runtime=controller, auth=self._auth)
        self._tasks = tasks.TaskDependencies(db=self._db, logs=self._log_client, runtime=controller, auth=self._auth)
        self._user_budget_defaults = user_budget_defaults or UserBudgetDefaults()
        self._backend_status = backend_status.BackendStatusDependencies(
            db=self._db,
            runtime=controller,
            user_budget_defaults=self._user_budget_defaults,
        )
        self._jobs = jobs.JobDependencies(
            db=self._db,
            runtime=controller,
            bundles=self._bundle_store,
            auth=self._auth,
            user_budget_defaults=self._user_budget_defaults,
        )
        self._federation = federation_service.FederationDependencies(db=self._db, runtime=controller)
        self._capability_url_config = capability_url_config or attempts.CapabilityUrlConfig()
        self._profile_table = self._log_client.get_table(PROFILE_NAMESPACE, IrisProfile)
        self._attempts = attempts.AttemptDependencies(
            db=self._db,
            runtime=controller,
            auth=self._auth,
            endpoints=endpoint_service,
            profile_table=self._profile_table,
            capability_urls=self._capability_url_config,
            timer=self._timer,
        )
        self._db.attach_task_event_table(
            self._log_client.get_table(
                TASK_EVENT_NAMESPACE,
                TaskEventRow,
                storage_policy=TASK_EVENT_STORAGE_POLICY,
            )
        )

    def bundle_zip(self, bundle_id: str) -> bytes:
        return artifacts.bundle_zip(self._artifacts, bundle_id)

    def blob_data(self, blob_id: str) -> bytes:
        return artifacts.blob_data(self._artifacts, blob_id)

    def probe_database(self) -> int | None:
        return diagnostics.probe_database(self._diagnostics)

    def launch_job(
        self,
        request: controller_pb2.Controller.LaunchJobRequest,
        ctx: RequestContext,
    ) -> controller_pb2.Controller.LaunchJobResponse:
        return jobs.launch_job(self._jobs, request, ctx)

    def get_job_status(
        self,
        request: controller_pb2.Controller.GetJobStatusRequest,
        ctx: RequestContext,
    ) -> controller_pb2.Controller.GetJobStatusResponse:
        return jobs.get_job_status(self._jobs, request, ctx)

    def get_job_state(
        self,
        request: controller_pb2.Controller.GetJobStateRequest,
        ctx: RequestContext,
    ) -> controller_pb2.Controller.GetJobStateResponse:
        return jobs.get_job_state(self._jobs, request, ctx)

    def terminate_job(
        self,
        request: controller_pb2.Controller.TerminateJobRequest,
        ctx: RequestContext,
    ) -> job_pb2.Empty:
        return jobs.terminate_job(self._jobs, request, ctx)

    def complete_job(
        self,
        request: controller_pb2.Controller.CompleteJobRequest,
        _ctx: RequestContext | None,
    ) -> job_pb2.Empty:
        return jobs.complete_job(self._jobs, request, _ctx)

    def list_jobs(
        self,
        request: controller_pb2.Controller.ListJobsRequest,
        ctx: RequestContext,
    ) -> controller_pb2.Controller.ListJobsResponse:
        return jobs.list_jobs(self._jobs, request, ctx)

    def get_task_status(
        self,
        request: controller_pb2.Controller.GetTaskStatusRequest,
        ctx: RequestContext,
    ) -> controller_pb2.Controller.GetTaskStatusResponse:
        return tasks.get_task_status(self._tasks, request, ctx)

    def list_tasks(
        self,
        request: controller_pb2.Controller.ListTasksRequest,
        ctx: RequestContext,
    ) -> controller_pb2.Controller.ListTasksResponse:
        return tasks.list_tasks(self._tasks, request, ctx)

    def kick_tasks(
        self,
        request: controller_pb2.Controller.KickTasksRequest,
        ctx: RequestContext,
    ) -> controller_pb2.Controller.KickTasksResponse:
        return tasks.kick_tasks(self._tasks, request, ctx)

    def register(
        self,
        request: controller_pb2.Controller.RegisterRequest,
        ctx: RequestContext,
    ) -> controller_pb2.Controller.RegisterResponse:
        return workers.register(self._workers, request, ctx)

    def list_workers(
        self,
        request: controller_pb2.Controller.ListWorkersRequest,
        ctx: RequestContext,
    ) -> controller_pb2.Controller.ListWorkersResponse:
        return workers.list_workers(self._workers, request, ctx)

    @property
    def backend(self) -> TaskBackend:
        return self._controller.backend

    @property
    def endpoint_service(self) -> EndpointServiceImpl:
        return self._endpoint_service

    def get_autoscaler_status(
        self,
        request: controller_pb2.Controller.GetAutoscalerStatusRequest,
        ctx: RequestContext,
    ) -> controller_pb2.Controller.GetAutoscalerStatusResponse:
        return backend_status.get_autoscaler_status(self._backend_status, request, ctx)

    def get_kubernetes_cluster_status(
        self,
        request: controller_pb2.Controller.GetKubernetesClusterStatusRequest,
        ctx: RequestContext,
    ) -> controller_pb2.Controller.GetKubernetesClusterStatusResponse:
        return backend_status.get_kubernetes_cluster_status(self._backend_status, request, ctx)

    def profile_task(
        self,
        request: job_pb2.ProfileTaskRequest,
        ctx: RequestContext,
    ) -> job_pb2.ProfileTaskResponse:
        return attempts.profile_task(self._attempts, request, ctx)

    def list_users(
        self,
        request: controller_pb2.Controller.ListUsersRequest,
        ctx: RequestContext,
    ) -> controller_pb2.Controller.ListUsersResponse:
        return accounts.list_users(self._accounts, request, ctx)

    def get_worker_status(
        self,
        request: controller_pb2.Controller.GetWorkerStatusRequest,
        ctx: RequestContext,
    ) -> controller_pb2.Controller.GetWorkerStatusResponse:
        return workers.get_worker_status(self._workers, request, ctx)

    def begin_checkpoint(
        self,
        request: controller_pb2.Controller.BeginCheckpointRequest,
        ctx: RequestContext,
    ) -> controller_pb2.Controller.BeginCheckpointResponse:
        return checkpoint.begin_checkpoint_request(self._checkpoint, request, ctx)

    def get_process_status(
        self,
        request: job_pb2.GetProcessStatusRequest,
        ctx: RequestContext,
    ) -> job_pb2.GetProcessStatusResponse:
        return attempts.get_process_status(self._attempts, request, ctx)

    def mint_endpoint_token(
        self,
        request: controller_pb2.Controller.MintEndpointTokenRequest,
        ctx: RequestContext,
    ) -> controller_pb2.Controller.MintEndpointTokenResponse:
        return attempts.mint_endpoint_token(self._attempts, request, ctx)

    def get_current_user(
        self,
        request: job_pb2.GetCurrentUserRequest,
        ctx: RequestContext,
    ) -> job_pb2.GetCurrentUserResponse:
        return accounts.get_current_user(request, ctx)

    def exec_in_container(
        self,
        request: controller_pb2.Controller.ExecInContainerRequest,
        ctx: RequestContext,
    ) -> controller_pb2.Controller.ExecInContainerResponse:
        return attempts.exec_in_container(self._attempts, request, ctx)

    def execute_raw_query(
        self,
        request: query_pb2.RawQueryRequest,
        ctx: RequestContext,
    ) -> query_pb2.RawQueryResponse:
        return diagnostics.execute_raw_query(self._diagnostics, request, ctx)

    def set_user_budget(
        self,
        request: controller_pb2.Controller.SetUserBudgetRequest,
        ctx: RequestContext,
    ) -> controller_pb2.Controller.SetUserBudgetResponse:
        return accounts.set_user_budget(self._accounts, request, ctx)

    def get_user_budget(
        self,
        request: controller_pb2.Controller.GetUserBudgetRequest,
        ctx: RequestContext,
    ) -> controller_pb2.Controller.GetUserBudgetResponse:
        return accounts.get_user_budget(self._accounts, request, ctx)

    def list_user_budgets(
        self,
        request: controller_pb2.Controller.ListUserBudgetsRequest,
        ctx: RequestContext,
    ) -> controller_pb2.Controller.ListUserBudgetsResponse:
        return accounts.list_user_budgets(self._accounts, request, ctx)

    def get_scheduler_state(
        self,
        request: controller_pb2.Controller.GetSchedulerStateRequest,
        ctx: RequestContext,
    ) -> controller_pb2.Controller.GetSchedulerStateResponse:
        return backend_status.get_scheduler_state(self._backend_status, request, ctx)

    def list_backends(
        self,
        request: controller_pb2.Controller.ListBackendsRequest,
        ctx: RequestContext,
    ) -> controller_pb2.Controller.ListBackendsResponse:
        return backend_status.list_backends(self._backend_status, request, ctx)

    def list_peers(
        self,
        request: controller_pb2.Controller.ListPeersRequest,
        ctx: RequestContext,
    ) -> controller_pb2.Controller.ListPeersResponse:
        return federation_service.list_peers(self._federation, request, ctx)

    def federation_sync(
        self,
        request: controller_pb2.Controller.FederationSyncRequest,
        ctx: RequestContext,
    ) -> controller_pb2.Controller.FederationSyncResponse:
        return federation_service.federation_sync(self._federation, request, ctx)
