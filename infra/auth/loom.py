# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch through Loom's shared-grant API and reconcile runtime grant lifetime."""

import logging
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

import requests
from rigging.auth import TokenProvider
from rigging.grants import Grant, GrantDenied


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
        database: Path,
        *,
        base_url: str,
        observer_tokens: TokenProvider,
        grants: LoomGrantService,
        session: requests.Session,
        timeout: float = 30,
        now: Callable[[], float] = time.time,
    ):
        parsed = urlsplit(base_url)
        if (
            not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path not in {"", "/"}
            or not (
                parsed.scheme == "https"
                or (parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"})
            )
        ):
            raise ValueError("Loom requires an HTTPS origin outside loopback")
        self.database = database
        self.base_url = base_url.rstrip("/")
        self.observer_tokens = observer_tokens
        self.grants = grants
        self.session = session
        self.timeout = timeout
        self._now = now
        self._lock = threading.RLock()
        with self._connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS loom_grant_runs ("
                "grant_id TEXT PRIMARY KEY, actor TEXT NOT NULL, principal TEXT NOT NULL, "
                "template TEXT NOT NULL, title TEXT NOT NULL, goal TEXT NOT NULL, requested_at REAL NOT NULL, "
                "run_id TEXT NOT NULL DEFAULT '', session_id TEXT NOT NULL DEFAULT '', "
                "status TEXT NOT NULL DEFAULT 'pending')"
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.database, timeout=self.timeout)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

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
        with self._lock:
            grant = self.grants.authorize(token, audience="loom", action="session.create", resource=f"/{template}/")
            with self._connect() as db:
                db.execute("BEGIN IMMEDIATE")
                db.execute(
                    "INSERT OR IGNORE INTO loom_grant_runs "
                    "(grant_id, actor, principal, template, title, goal, requested_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (grant.id, grant.actor, grant.created_by, template, title, goal, self._now()),
                )
                row = db.execute("SELECT * FROM loom_grant_runs WHERE grant_id = ?", (grant.id,)).fetchone()
                run = LoomRun(**dict(row))
                if (run.template, run.title, run.goal, run.actor) != (template, title, goal, grant.actor):
                    raise ValueError("A grant already belongs to a different Loom launch")
                db.commit()
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

    def observe(self, grant: Grant, template: str) -> None:
        """Register a native launch before issuing its runtime credential."""
        # Do not take the HTTP session lock: the launch thread may be waiting
        # for Loom, which calls this endpoint before returning that response.
        with self._connect() as db:
            db.execute(
                "INSERT OR IGNORE INTO loom_grant_runs "
                "(grant_id, actor, principal, template, title, goal, requested_at) VALUES (?, ?, ?, ?, '', '', ?)",
                (grant.id, grant.actor, grant.created_by, template, self._now()),
            )
            row = db.execute("SELECT * FROM loom_grant_runs WHERE grant_id = ?", (grant.id,)).fetchone()
            if (row["actor"], row["principal"], row["template"]) != (grant.actor, grant.created_by, template):
                raise GrantDenied("The grant is already bound to a different Loom launch")

    def _record_result(self, run: LoomRun, result: dict) -> LoomRun:
        fields = ("shared_grant_id", "actor_subject", "profile", "id", "session_id", "status")
        if not isinstance(result, dict) or any(not isinstance(result.get(name), str) for name in fields):
            raise InvalidLoomRun("Loom returned malformed run metadata")
        if not all(result[name] for name in fields):
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
        with self._connect() as db:
            db.execute(
                "UPDATE loom_grant_runs SET run_id = ?, session_id = ?, status = ? WHERE grant_id = ?",
                (result["id"], result["session_id"], result["status"], run.grant_id),
            )
            row = db.execute("SELECT * FROM loom_grant_runs WHERE grant_id = ?", (run.grant_id,)).fetchone()
        return LoomRun(**dict(row))

    def reconcile(self) -> list[LoomRun]:
        """Revoke finished grants and archive revoked runs, including after restart."""
        with self._lock:
            with self._connect() as db:
                rows = db.execute("SELECT * FROM loom_grant_runs WHERE status != 'cleaned'").fetchall()
            pending = []
            for row in rows:
                run = LoomRun(**dict(row))
                try:
                    self._retire_local_authority(run)
                    pending.append(run)
                except (ValueError, KeyError, TypeError, OSError, sqlite3.Error):
                    logger.exception("Loom local retirement failed for grant %s; retrying later", run.grant_id)
            observed = []
            for run in pending:
                try:
                    observed.append(self._reconcile_run(run))
                except (requests.RequestException, ValueError, KeyError, TypeError, OSError, sqlite3.Error):
                    logger.exception("Loom lifecycle reconciliation failed for grant %s", run.grant_id)
            return observed

    def _retire_local_authority(self, run: LoomRun) -> bool:
        unbound = not run.run_id and self._now() - run.requested_at >= UNBOUND_GRANT_TTL
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
        with self._connect() as db:
            db.execute("UPDATE loom_grant_runs SET status = 'cleaned' WHERE grant_id = ?", (run.grant_id,))
