# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded shared credentials and restart-safe retirement for exact attempts."""

import logging
import math
import threading
from collections import OrderedDict
from typing import Protocol

import requests
from pydantic import BaseModel, ConfigDict, Field
from rigging.grants import GrantDenied
from rigging.shared_grants import SHARED_GRANT_TOKEN_SECRET, SharedGrantConfig, SharedGrantUnavailable
from rigging.timing import Timestamp
from sqlalchemy import and_, delete, insert, or_, select

from iris.cluster.config import SharedAttemptConfig
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.reconcile.snapshot import TaskUpdate
from iris.cluster.controller.schema import (
    jobs_table,
    shared_attempt_grants_table,
    task_attempts_table,
    tasks_table,
    workers_table,
)
from iris.cluster.types import TERMINAL_JOB_STATES, TERMINAL_TASK_STATES, JobName
from iris.rpc import job_pb2

logger = logging.getLogger(__name__)
RENEWAL_INTERVAL = 20
MAX_RENEWAL_CACHE_ENTRIES = 1024


class AttemptCredentials(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, hide_input_in_errors=True)
    grant_id: str
    parent_grant_id: str
    actor: str
    attempt_uid: str
    access_token: str = Field(repr=False)
    expires_at: float
    auth_url: str


class AttemptBroker(Protocol):
    def issue(self, *, grant_id: str, resource: str, task_id: str, attempt_uid: str) -> AttemptCredentials: ...

    def renew(self, *, grant_id: str, resource: str, task_id: str, attempt_uid: str) -> AttemptCredentials | None: ...

    def retire(self, *, grant_id: str, resource: str, task_id: str, attempt_uid: str) -> None: ...


class SharedAttemptClient:
    """Use a controller-only identity at the explicitly configured authority."""

    def __init__(self, trust: SharedGrantConfig, config: SharedAttemptConfig):
        self.trust = trust
        self.config = config
        self._renew_after: OrderedDict[tuple[str, str, str, str], float] = OrderedDict()
        self._renew_lock = threading.Lock()

    def _post(self, endpoint: str, payload: dict) -> dict:
        try:
            token = self.config.identity_token_file.read_text().strip()
        except OSError as error:
            raise SharedGrantUnavailable("The controller workload identity is unavailable") from error
        if not token:
            raise GrantDenied("A controller workload identity is required")
        try:
            response = requests.post(
                self.trust.authority_url.rstrip("/") + endpoint,
                headers={"Authorization": f"Bearer {token}"},
                json=payload,
                timeout=self.trust.timeout,
                verify=self.trust.ca_bundle or True,
                allow_redirects=False,
            )
        except requests.RequestException as error:
            raise SharedGrantUnavailable("Shared attempt authority is unavailable") from error
        if response.status_code in {401, 403}:
            raise GrantDenied("Shared attempt authority denied the operation")
        if response.status_code != 200:
            raise SharedGrantUnavailable("Shared attempt authority returned no decision")
        try:
            result = response.json()
        except ValueError:
            raise SharedGrantUnavailable("Malformed shared attempt response") from None
        if not isinstance(result, dict):
            raise SharedGrantUnavailable("Malformed shared attempt response")
        return result

    def issue(self, *, grant_id: str, resource: str, task_id: str, attempt_uid: str) -> AttemptCredentials:
        result = self._post(
            "/v1/iris/attempts/credentials",
            {"grant_id": grant_id, "resource": resource, "task_id": task_id, "attempt_uid": attempt_uid},
        )
        credentials = self._credentials(result, grant_id=grant_id, attempt_uid=attempt_uid)
        self._remember_renewal((grant_id, resource, task_id, attempt_uid), credentials.expires_at)
        return credentials

    def renew(self, *, grant_id: str, resource: str, task_id: str, attempt_uid: str) -> AttemptCredentials | None:
        key = (grant_id, resource, task_id, attempt_uid)
        with self._renew_lock:
            if Timestamp.now().epoch_ms() / 1000 < self._renew_after.get(key, 0):
                return None
        result = self._post(
            "/v1/iris/attempts/renew",
            {"grant_id": grant_id, "resource": resource, "task_id": task_id, "attempt_uid": attempt_uid},
        )
        credentials = self._credentials(result, grant_id=grant_id, attempt_uid=attempt_uid)
        self._remember_renewal(key, credentials.expires_at)
        return credentials

    def _remember_renewal(self, key: tuple[str, str, str, str], expires_at: float) -> None:
        with self._renew_lock:
            now = Timestamp.now().epoch_ms() / 1000
            self._renew_after[key] = now + min(RENEWAL_INTERVAL, (expires_at - now) / 3)
            self._renew_after.move_to_end(key)
            while len(self._renew_after) > MAX_RENEWAL_CACHE_ENTRIES:
                self._renew_after.popitem(last=False)

    def _credentials(self, result: dict, *, grant_id: str, attempt_uid: str) -> AttemptCredentials:
        try:
            credentials = AttemptCredentials.model_validate(result)
        except ValueError:
            raise SharedGrantUnavailable("Malformed shared attempt credentials") from None
        now = Timestamp.now().epoch_ms() / 1000
        if (
            credentials.parent_grant_id != grant_id
            or credentials.attempt_uid != attempt_uid
            or credentials.auth_url.rstrip("/") != self.trust.authority_url.rstrip("/")
            or not credentials.access_token
            or not credentials.grant_id
            or not math.isfinite(credentials.expires_at)
            or not now < credentials.expires_at <= now + 60
        ):
            raise SharedGrantUnavailable("Shared attempt credentials do not match the request")
        return credentials

    def retire(self, *, grant_id: str, resource: str, task_id: str, attempt_uid: str) -> None:
        result = self._post(
            "/v1/iris/attempts/retire",
            {"grant_id": grant_id, "resource": resource, "task_id": task_id, "attempt_uid": attempt_uid},
        )
        if result.get("retired") is not True:
            raise SharedGrantUnavailable("Shared attempt retirement was not confirmed")
        with self._renew_lock:
            self._renew_after.pop((grant_id, resource, task_id, attempt_uid), None)


def with_shared_attempt_credentials(
    request: job_pb2.RunTaskRequest,
    *,
    db: ControllerDB,
    broker: AttemptBroker | None,
    job_prefix: str,
) -> job_pb2.RunTaskRequest | None:
    """Stamp a shared attempt, returning None for a native job."""
    job_id, _ = JobName.from_wire(request.task_id).require_task()
    with db.read_snapshot() as tx:
        grant_id = tx.execute(select(jobs_table.c.shared_grant_id).where(jobs_table.c.job_id == job_id)).scalar()
    if not grant_id:
        return None
    if broker is None or not job_prefix or not request.attempt_uid:
        raise GrantDenied("Shared attempts require an explicit credential broker")
    resource = job_prefix + job_id.to_wire() + "/"
    # Persist retirement intent before issuance so a lost response cannot orphan authority.
    with db.transaction() as tx:
        existing = tx.execute(
            select(shared_attempt_grants_table).where(shared_attempt_grants_table.c.attempt_uid == request.attempt_uid)
        ).first()
        if existing is None:
            tx.execute(
                insert(shared_attempt_grants_table).values(
                    attempt_uid=request.attempt_uid,
                    job_id=job_id,
                    task_id=request.task_id,
                    grant_id=grant_id,
                    resource=resource,
                )
            )
    credentials = broker.issue(
        grant_id=grant_id, resource=resource, task_id=request.task_id, attempt_uid=request.attempt_uid
    )
    stamped = job_pb2.RunTaskRequest()
    stamped.CopyFrom(request)
    stamped.ClearField("task_token")
    stamped.environment.secrets[SHARED_GRANT_TOKEN_SECRET] = credentials.access_token
    stamped.environment.env_vars.update(
        {
            "MARIN_GRANT_ID": credentials.grant_id,
            "MARIN_AUTH_URL": credentials.auth_url,
            "IRIS_ATTEMPT_UID": request.attempt_uid,
        }
    )
    return stamped


def retire_shared_attempts(db: ControllerDB, broker: AttemptBroker | None) -> None:
    """Retry terminal attempt retirement; unavailable authorities retain the outbox."""
    if broker is None:
        return
    bindings = shared_attempt_grants_table
    with db.read_snapshot() as tx:
        pending = tx.execute(
            select(bindings)
            .select_from(
                bindings.outerjoin(jobs_table, bindings.c.job_id == jobs_table.c.job_id).outerjoin(
                    task_attempts_table, bindings.c.attempt_uid == task_attempts_table.c.attempt_uid
                )
            )
            .where(
                or_(
                    jobs_table.c.job_id.is_(None),
                    jobs_table.c.state.in_(TERMINAL_JOB_STATES),
                    task_attempts_table.c.attempt_uid.is_(None),
                    task_attempts_table.c.state.in_(TERMINAL_TASK_STATES),
                )
            )
        ).all()
    for row in pending:
        try:
            broker.retire(grant_id=row.grant_id, resource=row.resource, task_id=row.task_id, attempt_uid=row.attempt_uid)
        except (GrantDenied, SharedGrantUnavailable):
            logger.exception("Shared attempt retirement remains pending for %s", row.attempt_uid)
            continue
        with db.transaction() as tx:
            tx.execute(delete(bindings).where(bindings.c.attempt_uid == row.attempt_uid))


def renew_shared_attempts(db: ControllerDB, broker: AttemptBroker | None, observations: list[TaskUpdate]) -> None:
    """Renew exact current attempts only after a fresh live backend observation.

    A cached RUNNING state supplies no liveness. The post-commit snapshot fences
    cancellation, terminal transitions, retries and stale provider identities.
    An authority outage leaves at most the existing minute-long lease.
    """
    if broker is None:
        return
    live = {
        (update.task_id, update.attempt_id, update.attempt_uid): update
        for update in observations
        if update.attempt_uid and update.new_state in {job_pb2.TASK_STATE_BUILDING, job_pb2.TASK_STATE_RUNNING}
    }
    if not live:
        return
    bindings = shared_attempt_grants_table
    with db.read_snapshot() as tx:
        rows = tx.execute(
            select(bindings, task_attempts_table.c.attempt_id, task_attempts_table.c.worker_id)
            .select_from(
                bindings.join(jobs_table, bindings.c.job_id == jobs_table.c.job_id)
                .join(task_attempts_table, bindings.c.attempt_uid == task_attempts_table.c.attempt_uid)
                .join(
                    tasks_table,
                    and_(
                        tasks_table.c.task_id == task_attempts_table.c.task_id,
                        tasks_table.c.current_attempt_id == task_attempts_table.c.attempt_id,
                    ),
                )
                .outerjoin(workers_table, task_attempts_table.c.worker_id == workers_table.c.worker_id)
            )
            .where(
                or_(task_attempts_table.c.worker_id.is_(None), workers_table.c.worker_id.is_not(None)),
                jobs_table.c.shared_grant_id == bindings.c.grant_id,
                jobs_table.c.state.not_in(TERMINAL_JOB_STATES),
                task_attempts_table.c.state.not_in(TERMINAL_TASK_STATES),
                tasks_table.c.state.not_in(TERMINAL_TASK_STATES),
            )
        ).all()
    for row in rows:
        observed = live.get((JobName.from_wire(row.task_id), row.attempt_id, row.attempt_uid))
        if observed is None or (observed.worker_id is not None and observed.worker_id != row.worker_id):
            continue
        try:
            broker.renew(grant_id=row.grant_id, resource=row.resource, task_id=row.task_id, attempt_uid=row.attempt_uid)
        except (GrantDenied, SharedGrantUnavailable):
            logger.warning("Shared attempt lease renewal failed for %s", row.attempt_uid)
