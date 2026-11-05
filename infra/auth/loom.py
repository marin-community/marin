# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch through Loom's shared-grant API and reconcile runtime grant lifetime."""

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
UNBOUND_GRANT_TTL = 60


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
        """Register a native launch before issuing its runtime credential.

        This callback performs no HTTP and takes no session lock: Loom calls it
        while a launch request can be waiting for Loom on a different thread.
        """
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
        if (
            result["shared_grant_id"] != run.grant_id
            or result["actor_subject"] != run.actor
            or result["profile"] != run.template
        ):
            raise ValueError("Loom returned a run belonging to another grant, actor, or profile")
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
            observed = []
            for row in rows:
                run = LoomRun(**dict(row))
                if not run.run_id:
                    matches = [
                        candidate
                        for candidate in self._post("runs/list", {})
                        if candidate["shared_grant_id"] == run.grant_id
                    ]
                    if len(matches) > 1:
                        raise ValueError("Loom returned multiple runs for one grant")
                    if matches:
                        run = self._record_result(run, matches[0])
                elif run.status not in _TERMINAL:
                    run = self._record_result(run, self._post("runs/get", {"id": run.run_id}))
                unbound = not run.run_id and self._now() - run.requested_at >= UNBOUND_GRANT_TTL
                if unbound or run.status in _TERMINAL or not self.grants.active(run.grant_id, run.actor):
                    self.grants.revoke(run.grant_id, run.principal)
                    if run.session_id and run.status not in _TERMINAL:
                        self._post("sessions/archive", {"session": run.session_id})
                    with self._connect() as db:
                        db.execute("UPDATE loom_grant_runs SET status = 'cleaned' WHERE grant_id = ?", (run.grant_id,))
                observed.append(run)
            return observed
