# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch through Loom's shared-grant API and reconcile runtime grant lifetime."""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import requests
from rigging.auth import TokenProvider
from rigging.grant_database import GrantDatabase
from rigging.grants import Grant, GrantDenied
from rigging.shared_grants import authority_origin
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError


class LoomGrantService(Protocol):
    def authorize(self, token: str, *, audience: str, action: str, resource: str) -> Grant: ...

    def active(self, grant_id: str, actor: str) -> bool: ...

    def revoke(self, grant_id: str, principal: str) -> None: ...


@dataclass(frozen=True)
class LoomRun:
    grant_id: str
    actor: str
    principal: str
    template: str
    title: str
    goal: str
    requested_at: float
    run_id: str
    session_id: str
    status: str


_TERMINAL = frozenset({"completed", "failed", "cancelled"})
_NATIVE_STATUS = _TERMINAL | {"creating", "waiting", "delivering", "running"}
UNBOUND_GRANT_TTL = 60
logger = logging.getLogger(__name__)


class InvalidLoomRun(ValueError):
    """Loom cannot establish the recorded grant's run identity or state."""


class LoomLauncher:
    """Persist launch intent and stop credential refresh when the actual run ends."""

    def __init__(
        self,
        database: GrantDatabase,
        *,
        base_url: str,
        observer_tokens: TokenProvider,
        grants: LoomGrantService,
        session: requests.Session,
        timeout: float = 30,
        now: Callable[[], float] = time.time,
    ):
        self.database = database
        self.base_url = authority_origin(base_url)
        self.observer_tokens = observer_tokens
        self.grants = grants
        self.session = session
        self.timeout = timeout
        self._now = now
        with self.database.transaction() as db:
            db.execute(
                text(
                    "CREATE TABLE IF NOT EXISTS loom_grant_runs ("
                    "grant_id TEXT PRIMARY KEY, actor TEXT NOT NULL, principal TEXT NOT NULL, "
                    "template TEXT NOT NULL, title TEXT NOT NULL, goal TEXT NOT NULL, "
                    "requested_at DOUBLE PRECISION NOT NULL, "
                    "run_id TEXT NOT NULL DEFAULT '', session_id TEXT NOT NULL DEFAULT '', "
                    "status TEXT NOT NULL DEFAULT 'pending')"
                )
            )

    def _post(self, operation: str, payload: dict, *, token: str | None = None):
        bearer = token if token is not None else self.observer_tokens.get_token()
        if not bearer:
            raise ValueError("A Loom lifecycle observer credential is required")
        response = self.session.post(
            f"{self.base_url}/api/{operation}",
            json=payload,
            headers={"Authorization": f"Bearer {bearer}"},
            timeout=self.timeout,
            allow_redirects=False,
        )
        if 300 <= response.status_code < 400:
            raise requests.HTTPError("Loom redirected a credential-bearing request", response=response)
        response.raise_for_status()
        return response.json()

    def launch(self, token: str, *, template: str, title: str, goal: str) -> LoomRun:
        """Authorize before delivery; Loom rechecks the signed grant before launch."""
        grant = self.grants.authorize(token, audience="loom", action="session.create", resource=f"/{template}/")
        with self.database.serialize(f"loom-grant:{grant.id}"):
            with self.database.transaction() as db:
                db.execute(
                    text(
                        "INSERT INTO loom_grant_runs "
                        "(grant_id, actor, principal, template, title, goal, requested_at) "
                        "VALUES (:id, :actor, :principal, :template, :title, :goal, :now) "
                        "ON CONFLICT (grant_id) DO NOTHING"
                    ),
                    {
                        "id": grant.id,
                        "actor": grant.actor,
                        "principal": grant.created_by,
                        "template": template,
                        "title": title,
                        "goal": goal,
                        "now": self._now(),
                    },
                )
                row = (
                    db.execute(text("SELECT * FROM loom_grant_runs WHERE grant_id = :id"), {"id": grant.id})
                    .mappings()
                    .one()
                )
                run = LoomRun(**dict(row))
                if (run.template, run.title, run.goal, run.actor) != (template, title, goal, grant.actor):
                    raise ValueError("A grant already belongs to a different Loom launch")
            if run.status != "pending":
                return run
            result = self._post(
                "runs/create",
                {
                    "profile": template,
                    "idempotency_key": f"marin-grant:{grant.id}",
                    "source": "ops",
                    "session": {"title": title, "goal": goal},
                },
                token=token,
            )
            return self._record_result(run, result)

    def observe(self, grant: Grant, template: str, run_id: str, session_id: str) -> None:
        """Bind credentials to an actual, authenticated native launch reservation."""
        try:
            result = self._post("runs/get", {"id": run_id})
        except requests.HTTPError as error:
            if error.response.status_code == 404:
                raise GrantDenied("Loom has no matching launch reservation") from None
            raise
        expected = {
            "id": run_id,
            "shared_grant_id": grant.id,
            "actor_subject": grant.actor,
            "profile": template,
            "status": "creating",
            "session_id": session_id,
        }
        if (
            not run_id
            or not session_id
            or not isinstance(result, dict)
            or any(result.get(key) != value for key, value in expected.items())
        ):
            raise GrantDenied("Loom has no matching launch reservation")
        # Do not take the launch operation lock: the launch thread may be waiting
        # for Loom, which calls this endpoint before returning that response.
        with self.database.transaction() as db:
            db.execute(
                text(
                    "INSERT INTO loom_grant_runs "
                    "(grant_id, actor, principal, template, title, goal, requested_at) "
                    "VALUES (:id, :actor, :principal, :template, '', '', :now) ON CONFLICT (grant_id) DO NOTHING"
                ),
                {
                    "id": grant.id,
                    "actor": grant.actor,
                    "principal": grant.created_by,
                    "template": template,
                    "now": self._now(),
                },
            )
            row = (
                db.execute(text("SELECT * FROM loom_grant_runs WHERE grant_id = :id"), {"id": grant.id}).mappings().one()
            )
            if (
                (row["actor"], row["principal"], row["template"]) != (grant.actor, grant.created_by, template)
                or (row["run_id"] and row["run_id"] != run_id)
                or (row["session_id"] and row["session_id"] != session_id)
                or row["status"] not in {"pending", "creating"}
            ):
                raise GrantDenied("The grant is already bound to a different Loom launch")
            db.execute(
                text(
                    "UPDATE loom_grant_runs SET run_id = :run_id, session_id = :session_id, status = 'creating' "
                    "WHERE grant_id = :id"
                ),
                {"run_id": run_id, "session_id": session_id, "id": grant.id},
            )

    def _record_result(self, run: LoomRun, result: dict) -> LoomRun:
        fields = ("shared_grant_id", "actor_subject", "profile", "id", "session_id", "status")
        if not isinstance(result, dict) or any(not isinstance(result.get(name), str) for name in fields):
            raise InvalidLoomRun("Loom returned malformed run metadata")
        if not all(result[name] for name in fields if name != "session_id") or (
            not result["session_id"] and result["status"] not in {"creating", "failed", "cancelled"}
        ):
            raise InvalidLoomRun("Loom returned incomplete run metadata")
        if result["status"] not in _NATIVE_STATUS:
            raise InvalidLoomRun("Loom returned an unknown run status")
        if (
            result["shared_grant_id"] != run.grant_id
            or result["actor_subject"] != run.actor
            or result["profile"] != run.template
        ):
            raise InvalidLoomRun("Loom returned a run belonging to another grant, actor, or profile")
        if (run.run_id and result["id"] != run.run_id) or (run.session_id and result["session_id"] != run.session_id):
            raise InvalidLoomRun("Loom changed the recorded run or session identity")
        with self.database.transaction() as db:
            current = (
                db.execute(text("SELECT * FROM loom_grant_runs WHERE grant_id = :id"), {"id": run.grant_id})
                .mappings()
                .one()
            )
            if current["status"] == "cleaned":
                return LoomRun(**dict(current))
            if (current["run_id"] and current["run_id"] != result["id"]) or (
                current["session_id"] and current["session_id"] != result["session_id"]
            ):
                raise InvalidLoomRun("Loom changed the recorded run or session identity")
            status = current["status"] if current["status"] in _TERMINAL else result["status"]
            db.execute(
                text(
                    "UPDATE loom_grant_runs SET run_id = :run_id, session_id = :session_id, status = :status "
                    "WHERE grant_id = :id"
                ),
                {"run_id": result["id"], "session_id": result["session_id"], "status": status, "id": run.grant_id},
            )
            row = (
                db.execute(text("SELECT * FROM loom_grant_runs WHERE grant_id = :id"), {"id": run.grant_id})
                .mappings()
                .one()
            )
        return LoomRun(**dict(row))

    def reconcile(self) -> list[LoomRun]:
        """Revoke finished grants and archive runs under one replica's leadership."""
        with self.database.leadership("loom-lifecycle") as leader:
            if not leader:
                return []
            with self.database.transaction() as db:
                rows = (
                    db.execute(
                        text("SELECT * FROM loom_grant_runs WHERE status != 'cleaned' ORDER BY requested_at, grant_id")
                    )
                    .mappings()
                    .all()
                )
            pending = []
            for row in rows:
                with self.database.serialize(f"loom-grant:{row['grant_id']}"):
                    with self.database.transaction() as db:
                        current = (
                            db.execute(
                                text("SELECT * FROM loom_grant_runs WHERE grant_id = :id"), {"id": row["grant_id"]}
                            )
                            .mappings()
                            .one()
                        )
                    run = LoomRun(**dict(current))
                    if run.status == "cleaned":
                        continue
                    try:
                        self._retire_local_authority(run)
                        pending.append(run)
                    except (ValueError, KeyError, TypeError, OSError, SQLAlchemyError):
                        logger.exception("Loom local retirement failed for grant %s; retrying later", run.grant_id)
            observed = []
            for run in pending:
                with self.database.serialize(f"loom-grant:{run.grant_id}"):
                    with self.database.transaction() as db:
                        row = (
                            db.execute(text("SELECT * FROM loom_grant_runs WHERE grant_id = :id"), {"id": run.grant_id})
                            .mappings()
                            .one()
                        )
                    run = LoomRun(**dict(row))
                    if run.status == "cleaned":
                        continue
                    try:
                        observed.append(self._reconcile_run(run))
                    except (requests.RequestException, ValueError, KeyError, TypeError, OSError, SQLAlchemyError):
                        logger.exception("Loom lifecycle reconciliation failed for grant %s", run.grant_id)
            return observed

    def _retire_local_authority(self, run: LoomRun) -> bool:
        """Revoke local authority if retirement is due; return whether cleanup is needed."""
        unbound = (not run.session_id or run.status in {"pending", "creating"}) and (
            self._now() - run.requested_at >= UNBOUND_GRANT_TTL
        )
        retiring = unbound or run.status in _TERMINAL or not self.grants.active(run.grant_id, run.actor)
        if retiring:
            # Credential authority must end even while discovery or archival is unavailable.
            self.grants.revoke(run.grant_id, run.principal)
        return retiring

    def _reconcile_run(self, run: LoomRun) -> LoomRun:
        retiring = self._retire_local_authority(run)
        try:
            if not run.run_id:
                candidates = self._post("runs/list", {})
                if not isinstance(candidates, list) or any(
                    not isinstance(candidate, dict) or "shared_grant_id" not in candidate for candidate in candidates
                ):
                    raise InvalidLoomRun("Loom returned malformed run discovery metadata")
                matches = [candidate for candidate in candidates if candidate["shared_grant_id"] == run.grant_id]
                if len(matches) > 1:
                    raise InvalidLoomRun("Loom returned multiple runs for one grant")
                if matches:
                    run = self._record_result(run, matches[0])
            elif not retiring:
                run = self._record_result(run, self._post("runs/get", {"id": run.run_id}))
        except requests.HTTPError as error:
            if error.response.status_code != 404 or not run.run_id:
                raise
            logger.warning("Loom run is missing for grant %s; retiring its authority", run.grant_id)
            retiring = True
        except (InvalidLoomRun, requests.exceptions.JSONDecodeError):
            self.grants.revoke(run.grant_id, run.principal)
            if run.session_id:
                self._retire_run(run)
            raise
        if retiring or run.status in _TERMINAL:
            self._retire_run(run)
        return run

    def _retire_run(self, run: LoomRun) -> None:
        self.grants.revoke(run.grant_id, run.principal)
        if run.session_id and run.status not in _TERMINAL:
            try:
                self._post("sessions/archive", {"session": run.session_id})
            except requests.HTTPError as error:
                if error.response.status_code != 404:
                    raise
        with self.database.transaction() as db:
            db.execute(text("UPDATE loom_grant_runs SET status = 'cleaned' WHERE grant_id = :id"), {"id": run.grant_id})
