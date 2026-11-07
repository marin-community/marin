# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Trusted controller delegation for short, sandboxed Iris attempts."""

import hashlib
import re
import time
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, model_validator
from rigging.grant_auth import GrantAuthService
from rigging.grants import GrantDenied, Permission
from rigging.server_auth import extract_bearer_token
from sqlalchemy import text
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route


class TaskPermissionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    service: str
    actions: frozenset[str]
    resource: str

    def permission(self) -> Permission:
        return Permission(self.service, self.actions, self.resource)


class IrisBrokerConfig(BaseModel):
    """One deployment-selected task profile, capped at a minute of authority."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    controller_clusters: dict[str, str]
    permissions: tuple[TaskPermissionConfig, ...]
    audience: str
    max_attempt_ttl: int = Field(gt=0, le=60)

    @model_validator(mode="after")
    def validate_profile(self):
        if not self.controller_clusters or any(
            not principal or not cluster or "/" in cluster for principal, cluster in self.controller_clusters.items()
        ):
            raise ValueError("Iris delegation requires explicit controller principals and cluster names")
        if not self.permissions or any(item.service != self.audience for item in self.permissions):
            raise ValueError("The task profile must select one nonempty token audience")
        for item in self.permissions:
            item.permission()
        return self


class IrisAttemptRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    grant_id: str
    resource: str
    task_id: str
    attempt_uid: str


@dataclass(frozen=True)
class AttemptBinding:
    key: str
    actor: str
    request: IrisAttemptRequest


class IrisAttemptBroker:
    """Delegate only configured permissions to a controller-derived identity.

    The controller is an explicitly trusted deputy for its named cluster. Every
    issuance checks the submitted grant's exact job.submit authority. Durable
    retirement tombstones prevent an in-flight or retried issue from resurrecting
    a completed attempt, including across broker replicas.
    """

    def __init__(
        self,
        authority: GrantAuthService,
        config: IrisBrokerConfig,
        *,
        auth_url: str,
        now: Callable[[], float] = time.time,
    ):
        self.authority = authority
        self.config = config
        self.auth_url = auth_url
        self.now = now
        self.database = authority.store.database
        with self.database.transaction() as db:
            db.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS iris_grant_attempts ("
                    "request_key TEXT PRIMARY KEY, parent_id TEXT NOT NULL, resource TEXT NOT NULL, "
                    "task_id TEXT NOT NULL, attempt_uid TEXT NOT NULL, retired INTEGER NOT NULL)"
                )
            )

    def _binding(self, identity_token: str, request: IrisAttemptRequest) -> AttemptBinding:
        try:
            principal = self.authority.identity_verifier.verify(identity_token)
        except ValueError as error:
            raise GrantDenied("Invalid controller identity") from error
        cluster = self.config.controller_clusters.get(principal)
        if cluster is None:
            raise GrantDenied("No Iris cluster is assigned to this controller")
        parts = request.task_id.split("/")
        if (
            len(parts) < 4
            or parts[0] != ""
            or any(not part or part in {".", ".."} for part in parts[1:])
            or re.fullmatch("0|[1-9][0-9]*", parts[-1]) is None
        ):
            raise GrantDenied("A canonical task identity is required")
        job = "/".join(parts[:-1])
        try:
            Permission("iris", frozenset({"job.submit"}), request.resource)
        except ValueError as error:
            raise GrantDenied("A canonical job resource is required") from error
        if (
            request.resource != f"/clusters/{cluster}/jobs{job}/"
            or re.fullmatch("[0-9a-f]{16}", request.attempt_uid) is None
            or not request.grant_id
        ):
            raise GrantDenied("The attempt is outside this controller's cluster")
        digest = hashlib.sha256(f"{cluster}:{request.attempt_uid}".encode()).hexdigest()
        actor_digest = hashlib.sha256(f"{request.resource}:{request.task_id}".encode()).hexdigest()[:16]
        return AttemptBinding(f"iris:{digest}", f"iris:{cluster}:{request.attempt_uid}:{actor_digest}", request)

    def _record(self, binding: AttemptBinding, *, retiring: bool) -> None:
        request = binding.request
        with self.database.transaction() as db:
            row = (
                db.execute(text("SELECT * FROM iris_grant_attempts WHERE request_key = :key"), {"key": binding.key})
                .mappings()
                .first()
            )
            if row is not None:
                if any(
                    row[name] != value
                    for name, value in (
                        ("parent_id", request.grant_id),
                        ("resource", request.resource),
                        ("task_id", request.task_id),
                        ("attempt_uid", request.attempt_uid),
                    )
                ):
                    raise GrantDenied("The attempt identity is already bound to another grant or task")
                if row["retired"] and not retiring:
                    raise GrantDenied("The attempt has retired")
                if retiring:
                    db.execute(
                        text("UPDATE iris_grant_attempts SET retired = 1 WHERE request_key = :key"),
                        {"key": binding.key},
                    )
                return
            db.execute(
                text("INSERT INTO iris_grant_attempts VALUES (:key, :parent, :resource, :task, :uid, :retired)"),
                {
                    "key": binding.key,
                    "parent": request.grant_id,
                    "resource": request.resource,
                    "task": request.task_id,
                    "uid": request.attempt_uid,
                    "retired": int(retiring),
                },
            )

    def issue(self, identity_token: str, request: IrisAttemptRequest) -> dict:
        binding = self._binding(identity_token, request)
        with self.database.serialize(binding.key):
            parent = self.authority.store.active(request.grant_id)
            self.authority.store.authorize(
                parent.id, actor=parent.actor, service="iris", action="job.submit", resource=request.resource
            )
            self._record(binding, retiring=False)
            # This privileged deputy acts only after authenticating the configured
            # controller and checking its cluster and the exact parent's authority.
            child = self.authority.store.create(
                parent.actor,
                actor=binding.actor,
                permissions=tuple(item.permission() for item in self.config.permissions),
                expires_at=min(parent.expires_at, self.now() + self.config.max_attempt_ttl),
                parent_id=parent.id,
                idempotency_key=binding.key,
            )
            minted = self.authority.mint_for(child, audience=self.config.audience)
            # A disconnected replica can lose its session lock while its signer
            # is still working. Recheck the durable tombstone before releasing
            # credentials; retirement never permits a late issuance to revive it.
            try:
                self._record(binding, retiring=False)
            except GrantDenied:
                self.authority.store.revoke(child.id, principal=child.actor)
                raise
            return {
                "grant_id": child.id,
                "parent_grant_id": parent.id,
                "actor": child.actor,
                "attempt_uid": request.attempt_uid,
                "access_token": minted["access_token"],
                "expires_at": child.expires_at,
                "auth_url": self.auth_url,
            }

    def retire(self, identity_token: str, request: IrisAttemptRequest) -> dict:
        binding = self._binding(identity_token, request)
        with self.database.serialize(binding.key):
            self._record(binding, retiring=True)
            child = self.authority.store.grant_for_request(parent_id=request.grant_id, idempotency_key=binding.key)
            if child is not None:
                if child.actor != binding.actor:
                    raise GrantDenied("The attempt grant has another actor")
                self.authority.store.revoke(child.id, principal=child.actor)
        return {"retired": True}


def mount_iris_attempts(app: Starlette, broker: IrisAttemptBroker) -> None:
    async def operate(request: Request) -> JSONResponse:
        token = extract_bearer_token(dict(request.headers))
        if token is None:
            raise GrantDenied("Controller authentication is required")
        payload = IrisAttemptRequest.model_validate(await request.json())
        operation = broker.retire if request.url.path.endswith("/retire") else broker.issue
        result = await run_in_threadpool(operation, token, payload)
        return JSONResponse(result, headers={"Cache-Control": "no-store", "Pragma": "no-cache"})

    app.router.routes.extend(
        [
            Route("/v1/iris/attempts/credentials", operate, methods=["POST"]),
            Route("/v1/iris/attempts/retire", operate, methods=["POST"]),
        ]
    )
