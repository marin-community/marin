# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded shared credentials and restart-safe retirement for exact attempts."""

import logging
import math
from typing import Protocol

import requests
from pydantic import BaseModel, ConfigDict, Field
from rigging.grants import GrantDenied
from rigging.shared_grants import SHARED_GRANT_TOKEN_SECRET, SharedGrantConfig, SharedGrantUnavailable
from rigging.timing import Timestamp
from sqlalchemy import delete, insert, or_, select

from iris.cluster.config import SharedAttemptConfig
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.schema import jobs_table, shared_attempt_grants_table, task_attempts_table
from iris.cluster.types import TERMINAL_JOB_STATES, TERMINAL_TASK_STATES, JobName
from iris.rpc import job_pb2

logger = logging.getLogger(__name__)


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

    def retire(self, *, grant_id: str, resource: str, task_id: str, attempt_uid: str) -> None: ...


class SharedAttemptClient:
    """Use a controller-only identity at the explicitly configured authority."""

    def __init__(self, trust: SharedGrantConfig, config: SharedAttemptConfig):
        self.trust = trust
        self.config = config

    def _post(self, endpoint: str, payload: dict) -> dict:
        token = self.config.identity_token_file.read_text().strip()
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


def with_shared_attempt_credentials(
    request: job_pb2.RunTaskRequest,
    *,
    db: ControllerDB,
    broker: AttemptBroker | None,
    job_prefix: str,
) -> job_pb2.RunTaskRequest | None:
    """Stamp a shared attempt, returning None for a native job.

    The outbox precedes issuance so response loss, process death and job deletion
    cannot lose the retirement request. Bearer credentials never enter the DB.
    """
    job_id, _ = JobName.from_wire(request.task_id).require_task()
    with db.read_snapshot() as tx:
        grant_id = tx.execute(select(jobs_table.c.shared_grant_id).where(jobs_table.c.job_id == job_id)).scalar()
    if not grant_id:
        return None
    if broker is None or not job_prefix or not request.attempt_uid:
        raise GrantDenied("Shared attempts require an explicit credential broker")
    resource = job_prefix + job_id.to_wire() + "/"
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
