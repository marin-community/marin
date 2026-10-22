# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Attempt-bound task sessions and on-demand storage credentials."""

import base64
import hashlib
import hmac
import json
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from rigging.filesystem.s3_compat import needs_virtual_host_addressing
from rigging.server_auth import VerifiedIdentity, require_identity
from rigging.timing import Timestamp
from sqlalchemy import delete, func, insert, or_, select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from iris.cluster.config import CredentialBrokerConfig, user_admitted
from iris.cluster.controller.auth import CONTROL_PLANE_AUDIENCE, JwtTokenManager
from iris.cluster.controller.db import ControllerDB, Tx
from iris.cluster.controller.projections.endpoints import EndpointsProjection
from iris.cluster.controller.projections.run_templates import RunTemplatesProjection
from iris.cluster.controller.schema import (
    endpoints_table,
    jobs_table,
    task_attempts_table,
    task_credential_audit_table,
    task_credential_blocks_table,
    task_sessions_table,
    tasks_table,
)
from iris.cluster.controller.storage_credentials import StorageCredentials, exchange_credentials
from iris.cluster.credential_process import REFRESH_MARGIN
from iris.cluster.runtime.env import TASK_OUTPUT_FINALIZING_STATUS
from iris.cluster.types import UNVERIFIED_SUBMITTERS, JobName
from iris.rpc import controller_pb2, job_pb2
from iris.time_proto import timestamp_to_proto

TASK_ACCESS_LIFETIME = 900
_READ_RPCS = frozenset(
    {
        "GetCurrentUser",
        "GetTaskCredentials",
        "ListEndpoints",
        "ListWorkers",
        "GetWorkerStatus",
        "ListBackends",
        "GetAutoscalerStatus",
        "GetSchedulerState",
    }
)
_RESOURCE_RPCS = frozenset(
    {
        "LaunchJob",
        "GetJobStatus",
        "GetJobState",
        "TerminateJob",
        "CompleteJob",
        "GetTaskStatus",
        "ListTasks",
        "GetProcessStatus",
        "ProfileTask",
        "ExecInContainer",
        "KickTasks",
        "RegisterEndpoint",
        "UnregisterEndpoint",
        "MintEndpointToken",
        "ListJobs",
    }
)


@dataclass(frozen=True)
class TaskSession:
    attempt_uid: str
    generation: int
    principal: str
    job_id: JobName
    root_job_id: str
    deadline: float
    finalizing: bool


class TaskSessionBroker:
    """Owns durable authorization decisions; raw capabilities and S3 keys are never persisted."""

    def __init__(
        self, db: ControllerDB, config: CredentialBrokerConfig, jwt: JwtTokenManager, finalization_timeout: float
    ):
        if not isinstance(config.capability_key, str) or len(config.capability_key.encode()) < 32:
            raise ValueError("credential_broker.capability_key must resolve to at least 32 bytes")
        self._db = db
        self._config = config
        self._jwt = jwt
        self._finalization_timeout = finalization_timeout
        self._lock = threading.RLock()
        self._publish: Callable[[str], None] | None = None
        self._cache: dict[tuple[str, int, str], StorageCredentials] = {}
        self._active_sessions: dict[str, TaskSession] = {}

    def attach_projection(self, publish: Callable[[str], None]) -> None:
        with self._lock:
            self._publish = publish
            self.publish()

    def _capability(self, attempt_uid: str) -> str:
        assert isinstance(self._config.capability_key, str)
        digest = hmac.digest(
            self._config.capability_key.encode(), b"iris-task-refresh-v1\0" + attempt_uid.encode(), hashlib.sha256
        )
        return base64.urlsafe_b64encode(digest).decode().rstrip("=")

    def _sessions(self, tx: Tx, attempt_uid: str | None = None) -> dict[str, TaskSession]:
        attempts = task_attempts_table
        jobs = jobs_table
        tasks = tasks_table
        sessions = task_sessions_table
        blocks = {(r.scope, r.target) for r in tx.execute(select(task_credential_blocks_table)).all()}
        rows = tx.execute(
            select(
                attempts.c.attempt_uid,
                sessions.c.generation,
                jobs.c.submitting_user,
                jobs.c.job_id,
                jobs.c.root_job_id,
                attempts.c.finished_at_ms,
                sessions.c.finalizing_at_ms,
                tasks.c.status_message,
            )
            .select_from(
                sessions.join(attempts, sessions.c.attempt_uid == attempts.c.attempt_uid)
                .join(tasks, tasks.c.task_id == attempts.c.task_id)
                .join(jobs, jobs.c.job_id == tasks.c.job_id)
            )
            .where(tasks.c.current_attempt_id == attempts.c.attempt_id)
            .where(sessions.c.attempt_uid == attempt_uid if attempt_uid is not None else True)
        ).all()
        now = Timestamp.now().epoch_seconds()
        result = {}
        for row in rows:
            job_id = row.job_id.to_wire()
            blocked = (
                ("attempt", row.attempt_uid) in blocks
                or ("principal", row.submitting_user) in blocks
                or any(
                    scope == "job" and (job_id == target or job_id.startswith(target + "/")) for scope, target in blocks
                )
            )
            finalization_started_at = row.finalizing_at_ms
            deadline = (
                finalization_started_at.epoch_seconds() + self._finalization_timeout
                if finalization_started_at is not None
                else 253402300799.0
            )
            if (
                blocked
                or deadline <= now
                or row.finished_at_ms is not None
                or row.submitting_user in UNVERIFIED_SUBMITTERS
            ):
                continue
            result[row.attempt_uid] = TaskSession(
                row.attempt_uid,
                row.generation,
                row.submitting_user,
                row.job_id,
                row.root_job_id,
                deadline,
                finalization_started_at is not None or row.status_message == TASK_OUTPUT_FINALIZING_STATUS,
            )
        return result

    def _current(self, attempt_uid: str) -> TaskSession:
        self._record_finalization(attempt_uid)
        with self._db.read_snapshot() as tx:
            session = self._sessions(tx, attempt_uid).get(attempt_uid)
        if session is None:
            raise ConnectError(Code.UNAUTHENTICATED, "Task session is disabled, superseded or closed")
        return session

    def _record_finalization(self, attempt_uid: str | None = None) -> None:
        sessions, attempts, tasks = task_sessions_table, task_attempts_table, tasks_table
        with self._db.read_snapshot() as tx:
            uids = (
                tx.execute(
                    select(sessions.c.attempt_uid)
                    .select_from(
                        sessions.join(attempts, sessions.c.attempt_uid == attempts.c.attempt_uid).join(
                            tasks, tasks.c.task_id == attempts.c.task_id
                        )
                    )
                    .where(
                        sessions.c.finalizing_at_ms.is_(None),
                        tasks.c.status_message == TASK_OUTPUT_FINALIZING_STATUS,
                        tasks.c.current_attempt_id == attempts.c.attempt_id,
                    )
                    .where(sessions.c.attempt_uid == attempt_uid if attempt_uid is not None else True)
                )
                .scalars()
                .all()
            )
        if uids:
            with self._db.transaction() as tx:
                tx.execute(
                    update(sessions)
                    .where(sessions.c.attempt_uid.in_(uids), sessions.c.finalizing_at_ms.is_(None))
                    .values(finalizing_at_ms=Timestamp.now())
                )

    def publish(self) -> None:
        """Replace native authorization state, including after terminal attempt transitions."""
        with self._lock:
            self._record_finalization()
            with self._db.read_snapshot() as tx:
                sessions = self._sessions(tx)
            projection = {
                uid: {"generation": s.generation, "deadline": s.deadline, "finalizing": s.finalizing}
                for uid, s in sessions.items()
            }
            self._cache = {
                key: value
                for key, value in self._cache.items()
                if key[0] in sessions and key[1] == sessions[key[0]].generation
            }
            self._active_sessions = sessions
            if self._publish is not None:
                self._publish(json.dumps(projection))

    def prepare(self, requests: Sequence[job_pb2.RunTaskRequest]) -> list[job_pb2.RunTaskRequest | None]:
        """Return attempt-authenticated dispatches, with None for each denied dispatch."""
        if not requests:
            return []
        with self._lock:
            with self._db.transaction() as tx:
                tx.execute(
                    sqlite_insert(task_sessions_table)
                    .from_select(
                        ["attempt_uid"],
                        select(task_attempts_table.c.attempt_uid).where(
                            task_attempts_table.c.attempt_uid.in_([r.attempt_uid for r in requests])
                        ),
                    )
                    .on_conflict_do_nothing()
                )
            with self._db.read_snapshot() as tx:
                sessions = self._sessions(tx)
            prepared = []
            for request in requests:
                session = sessions.get(request.attempt_uid)
                if session is None:
                    prepared.append(None)
                    continue
                stamped = job_pb2.RunTaskRequest()
                stamped.CopyFrom(request)
                stamped.task_token = ""
                stamped.task_refresh_capability = self._capability(request.attempt_uid)
                stamped.task_storage_profiles_json = json.dumps(
                    {
                        name: {
                            "endpoint": profile.endpoint,
                            "region": profile.region,
                            "buckets": profile.buckets,
                            "addressing_style": "virtual" if needs_virtual_host_addressing(profile.endpoint) else "auto",
                        }
                        for name, profile in self._config.profiles.items()
                        if user_admitted(profile.allowed_principals, session.principal)
                    }
                )
                prepared.append(stamped)
            return prepared

    def renew(self, request: controller_pb2.Controller.RenewTaskSessionRequest):
        if not request.attempt_uid or not hmac.compare_digest(
            request.refresh_capability, self._capability(request.attempt_uid)
        ):
            raise ConnectError(Code.UNAUTHENTICATED, "Invalid task refresh capability")
        with self._lock:
            session = self._current(request.attempt_uid)
            lifetime = TASK_ACCESS_LIFETIME
            token = self._jwt.signer.mint(
                {
                    "sub": session.principal,
                    "role": "task",
                    "owner_id": session.job_id.user,
                    "job_id": session.job_id.to_wire(),
                    "root_job_id": session.root_job_id,
                    "session_id": session.attempt_uid,
                    "session_generation": session.generation,
                },
                audience=CONTROL_PLANE_AUDIENCE,
                ttl_seconds=lifetime,
            )
            return controller_pb2.Controller.RenewTaskSessionResponse(
                token=token,
                expires_at=timestamp_to_proto(Timestamp.from_seconds(Timestamp.now().epoch_seconds() + lifetime)),
            )

    def authorize_request(self, identity: VerifiedIdentity, method: str, request: object) -> None:
        if identity.role != "task":
            return
        session = self._active_sessions.get(identity.session_id or "")
        if session is None or session.deadline <= Timestamp.now().epoch_seconds():
            raise ConnectError(Code.UNAUTHENTICATED, "Task session is disabled, superseded or closed")
        if (
            session.generation != identity.session_generation
            or session.principal != identity.user_id
            or session.job_id.to_wire() != identity.job_id
            or session.job_id.user != identity.owner_id
        ):
            raise ConnectError(Code.UNAUTHENTICATED, "Task session has rolled over")
        if method in _READ_RPCS:
            return
        if session.finalizing:
            raise ConnectError(Code.PERMISSION_DENIED, "Finalizing task sessions may only renew storage credentials")
        if method not in _RESOURCE_RPCS:
            raise ConnectError(Code.PERMISSION_DENIED, f"Task session cannot call {method}")
        rpc = controller_pb2.Controller
        if isinstance(request, rpc.UnregisterEndpointRequest):
            with self._db.read_snapshot() as tx:
                job = tx.execute(
                    select(endpoints_table.c.job_id).where(endpoints_table.c.endpoint_id == request.endpoint_id)
                ).scalar_one_or_none()
            if job is None:
                return
            resources = [job.to_wire()]
        elif isinstance(request, rpc.RegisterEndpointRequest):
            # Names may be absolute aliases (Marin inference uses /serve/...).
            # Authority follows the owning task, also checked by the native proxy.
            resources = [request.task_id]
        elif isinstance(request, rpc.MintEndpointTokenRequest):
            with self._db.read_snapshot() as tx:
                endpoint = tx.caches[EndpointsProjection].resolve(request.endpoint_name)
            if endpoint is None:
                raise ConnectError(Code.NOT_FOUND, "Endpoint not found")
            resources = [endpoint.task_id.require_task()[0].to_wire()]
        elif isinstance(request, rpc.LaunchJobRequest):
            resources = [request.name]
        elif isinstance(request, rpc.GetJobStateRequest):
            resources = list(request.job_ids)
        elif isinstance(request, rpc.KickTasksRequest):
            resources = list(request.targets)
        elif isinstance(request, rpc.ListJobsRequest):
            # Exact children queries and an anchored root prefix are both bounded.
            query = request.query
            resources = (
                [query.parent_job_id]
                if query.scope == rpc.JOB_QUERY_SCOPE_CHILDREN
                else [query.job_id_prefix.rstrip("/")]
            )
            if query.scope != rpc.JOB_QUERY_SCOPE_CHILDREN and not query.job_id_prefix.endswith("/"):
                raise ConnectError(Code.PERMISSION_DENIED, "ListJobs requires a root-bounded query")
        elif isinstance(
            request, (rpc.GetJobStatusRequest, rpc.TerminateJobRequest, rpc.CompleteJobRequest, rpc.ListTasksRequest)
        ):
            resources = [request.job_id]
        elif isinstance(request, (rpc.GetTaskStatusRequest, rpc.ExecInContainerRequest)):
            resources = [request.task_id]
        elif isinstance(request, (job_pb2.GetProcessStatusRequest, job_pb2.ProfileTaskRequest)):
            resources = [request.target]
        else:
            raise ConnectError(Code.PERMISSION_DENIED, f"Task session cannot authorize {method}")
        if not resources or any(
            not isinstance(resource, str)
            or not (resource == session.root_job_id or resource.startswith(session.root_job_id + "/"))
            for resource in resources
        ):
            raise ConnectError(Code.PERMISSION_DENIED, "Task session is limited to its root job")
        if isinstance(request, rpc.LaunchJobRequest) and request.name == session.root_job_id:
            raise ConnectError(Code.PERMISSION_DENIED, "Task sessions may submit child jobs only")

    def authorize_download(self, identity: VerifiedIdentity, content_id: str) -> None:
        if identity.role in {"admin", "worker"}:
            return
        self.authorize_request(identity, "GetCurrentUser", job_pb2.Empty())
        if identity.role != "task" or identity.job_id is None:
            raise ConnectError(Code.PERMISSION_DENIED, "Downloads require an active task session")
        with self._db.read_snapshot() as tx:
            template = tx.caches[RunTemplatesProjection].get(tx, JobName.from_wire(identity.job_id))
        if template is None or (
            content_id != template.bundle_id and content_id not in template.entrypoint.workdir_file_refs.values()
        ):
            raise ConnectError(Code.PERMISSION_DENIED, "Download is not part of this task's job")

    def credentials(self, request: controller_pb2.Controller.GetTaskCredentialsRequest):
        identity = require_identity()
        self.authorize_request(identity, "GetTaskCredentials", request)
        if identity.role != "task":
            raise ConnectError(Code.PERMISSION_DENIED, "Storage vending requires a task session")
        profile = self._config.profiles.get(request.profile)
        if profile is None or not user_admitted(profile.allowed_principals, identity.user_id):
            raise ConnectError(Code.PERMISSION_DENIED, "Storage profile is not granted to this principal")
        key = (identity.session_id or "", identity.session_generation, request.profile)
        with self._lock:
            credentials = self._cache.get(key)
        needs_exchange = (
            credentials is None
            or credentials.expires_at.epoch_seconds() <= Timestamp.now().epoch_seconds() + REFRESH_MARGIN
        )
        if needs_exchange:
            credentials = exchange_credentials(profile, identity.user_id)
        assert credentials is not None
        # A provider exchange may race revocation. Recheck before releasing any secret.
        with self._lock:
            current = self._current(identity.session_id or "")
            if current.generation != identity.session_generation:
                raise ConnectError(Code.UNAUTHENTICATED, "Task session has rolled over")
            self.authorize_request(identity, "GetTaskCredentials", request)
            if needs_exchange:
                with self._db.transaction() as tx:
                    tx.execute(
                        insert(task_credential_audit_table).values(
                            at_ms=Timestamp.now(),
                            actor=identity.user_id,
                            action="ISSUE_STORAGE_CREDENTIALS",
                            scope="attempt-profile",
                            target=f"{identity.session_id}:{request.profile}",
                        )
                    )
            self._cache[key] = credentials
            return controller_pb2.Controller.GetTaskCredentialsResponse(
                access_key_id=credentials.access_key,
                secret_access_key=credentials.secret_key,
                session_token=credentials.session_token,
                expires_at=timestamp_to_proto(credentials.expires_at),
            )

    def manage(self, request: controller_pb2.Controller.ManageTaskSessionsRequest):
        identity = require_identity()
        if identity.role != "admin":
            raise ConnectError(Code.PERMISSION_DENIED, "Managing task sessions requires admin access")
        selectors = [("attempt", request.attempt_uid), ("job", request.job_id), ("principal", request.principal)]
        selected = [(scope, target) for scope, target in selectors if target]
        actions = controller_pb2.Controller.ManageTaskSessionsRequest
        if len(selected) != 1 or request.action not in {actions.ROLLOVER, actions.DISABLE, actions.ENABLE}:
            raise ConnectError(Code.INVALID_ARGUMENT, "Select one scope and a session action")
        scope, target = selected[0]
        with self._lock:
            with self._db.transaction() as tx:
                sessions, attempts, tasks, jobs = task_sessions_table, task_attempts_table, tasks_table, jobs_table
                if scope == "attempt":
                    selection = sessions.c.attempt_uid == target
                elif scope == "principal":
                    selection = jobs.c.submitting_user == target
                else:
                    selection = or_(
                        jobs.c.job_id == target, func.substr(jobs.c.job_id, 1, len(target) + 1) == target + "/"
                    )
                affected = (
                    tx.execute(
                        select(sessions.c.attempt_uid)
                        .select_from(
                            sessions.join(attempts, sessions.c.attempt_uid == attempts.c.attempt_uid)
                            .join(tasks, tasks.c.task_id == attempts.c.task_id)
                            .join(jobs, jobs.c.job_id == tasks.c.job_id)
                        )
                        .where(
                            selection,
                            tasks.c.current_attempt_id == attempts.c.attempt_id,
                            attempts.c.finished_at_ms.is_(None),
                        )
                    )
                    .scalars()
                    .all()
                )
                if request.action == actions.DISABLE:
                    tx.execute(
                        sqlite_insert(task_credential_blocks_table)
                        .values(scope=scope, target=target)
                        .on_conflict_do_nothing()
                    )
                elif request.action == actions.ENABLE:
                    tx.execute(
                        delete(task_credential_blocks_table).where(
                            task_credential_blocks_table.c.scope == scope,
                            task_credential_blocks_table.c.target == target,
                        )
                    )
                if affected:
                    tx.execute(
                        update(task_sessions_table)
                        .where(task_sessions_table.c.attempt_uid.in_(affected))
                        .values(generation=task_sessions_table.c.generation + 1)
                    )
                tx.execute(
                    insert(task_credential_audit_table).values(
                        at_ms=Timestamp.now(),
                        actor=identity.user_id,
                        action=actions.Action.Name(request.action),
                        scope=scope,
                        target=target,
                    )
                )
            self._cache.clear()
            # No acknowledgement until the committed decision reaches the native listener.
            self.publish()
        return controller_pb2.Controller.ManageTaskSessionsResponse(affected=len(affected))
