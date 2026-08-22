# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""HTTP status page for a Levanter training process running on Iris."""

from __future__ import annotations

import html
import hmac
import json
import logging
import os
import secrets
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import Generic, TypeVar, cast
from urllib.parse import parse_qs, urlsplit

import draccus
import jax
from rigging.connect import proxy_path
from rigging.redaction import REDACTED_VALUE, redact_value

from iris.client.client import get_iris_ctx
from iris.cluster.client.job_info import get_job_info
from iris.cluster.health import HEALTH_PATH, publish_task_health, task_health_enabled, task_health_port
from iris.cluster.types import EndpointAccess
from iris.hooks.multigpu import IRIS_MULTIGPU_LOCAL_PROCESS_INDEX_ENV
from levanter.callbacks.progress_watchdog import ProgressState, ProgressWatchdog


logger = logging.getLogger(__name__)

TRAINING_CONTROL_ENDPOINT = "training-control"
_REDACTED_ENVIRONMENT_VARIABLES = ("IRIS_JOB_ENV", "IRIS_JOB_SETUP_SCRIPTS", "MARIN_PROVENANCE")
_PROGRAMMATIC_ACTION_HEADER = "X-Levanter-Training-Control"
_PROGRAMMATIC_ACTION_VALUE = "request-checkpoint"
_PAGE_CSP = "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; base-uri 'none'"
_JSON_CSP = "default-src 'none'; base-uri 'none'"
ConfigT = TypeVar("ConfigT")


@dataclass(frozen=True)
class _TrainingSnapshot:
    run_id: str
    job_id: str
    task_id: str
    environment: dict[str, str]
    configuration: str

    @classmethod
    def capture(
        cls,
        config: ConfigT,
        *,
        run_id: str,
        job_id: str,
        task_id: str,
    ) -> _TrainingSnapshot:
        environment = dict(os.environ)
        for name in _REDACTED_ENVIRONMENT_VARIABLES:
            if name in environment:
                environment[name] = REDACTED_VALUE
        redacted_environment = cast(dict[str, str], redact_value(environment))
        encoded_config = redact_value(draccus.encode(config))
        return cls(
            run_id=run_id,
            job_id=job_id,
            task_id=task_id,
            environment=dict(sorted(redacted_environment.items())),
            configuration=json.dumps(encoded_config, indent=2, sort_keys=True),
        )


@dataclass(frozen=True)
class _HealthReport:
    """Fixed-shape report for the task health route."""

    run_id: str
    job_id: str
    task_id: str
    monitored: bool
    state: str | None
    event: str | None
    elapsed: float | None
    timeout: float | None

    @classmethod
    def capture(cls, snapshot: _TrainingSnapshot, watchdog: ProgressWatchdog | None) -> _HealthReport:
        health = watchdog.health() if watchdog is not None else None
        return cls(
            run_id=snapshot.run_id,
            job_id=snapshot.job_id,
            task_id=snapshot.task_id,
            monitored=health is not None,
            state=str(health.state) if health is not None else None,
            event=str(health.event) if health is not None and health.event is not None else None,
            elapsed=health.elapsed if health is not None else None,
            timeout=health.timeout if health is not None else None,
        )

    @property
    def status(self) -> HTTPStatus:
        if self.state == ProgressState.STALLED:
            return HTTPStatus.SERVICE_UNAVAILABLE
        return HTTPStatus.OK


def _render_page(snapshot: _TrainingSnapshot, action_token: str, health: _HealthReport) -> str:
    configuration = html.escape(snapshot.configuration)
    environment = html.escape(json.dumps(snapshot.environment, indent=2))
    progress = html.escape(json.dumps(asdict(health), indent=2))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Levanter training: {html.escape(snapshot.run_id)}</title>
  <style>
    body {{ font-family: sans-serif; margin: 2rem; }}
    pre {{ white-space: pre-wrap; }}
  </style>
</head>
<body>
  <h1>Levanter training</h1>
  <p>Detected secret values are shown as [REDACTED].</p>
  <ul>
    <li>Run ID: {html.escape(snapshot.run_id)}</li>
    <li>Iris job: {html.escape(snapshot.job_id)}</li>
    <li>Iris task: {html.escape(snapshot.task_id)}</li>
  </ul>
  <form method="post">
    <input type="hidden" name="token" value="{html.escape(action_token, quote=True)}">
    <button type="submit">Save checkpoint</button>
  </form>
  <h2>Progress</h2>
  <pre>{progress}</pre>
  <h2>Resolved configuration</h2>
  <pre>{configuration}</pre>
  <h2>Environment</h2>
  <pre>{environment}</pre>
</body>
</html>
"""


class _TrainingDashboardRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def __init__(
        self,
        *args,
        snapshot: _TrainingSnapshot,
        request_checkpoint: Callable[[], None] | None,
        watchdog: ProgressWatchdog | None,
        action_token: str,
        **kwargs,
    ):
        self._snapshot = snapshot
        self._request_checkpoint = request_checkpoint
        self._watchdog = watchdog
        self._action_token = action_token
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        report = _HealthReport.capture(self._snapshot, self._watchdog)
        if urlsplit(self.path).path == HEALTH_PATH:
            self._send(json.dumps(asdict(report)).encode(), "application/json", report.status, _JSON_CSP)
            return
        body = _render_page(self._snapshot, self._action_token, report).encode()
        self._send(body, "text/html; charset=utf-8", HTTPStatus.OK, _PAGE_CSP)

    def _send(self, body: bytes, content_type: str, status: HTTPStatus, csp: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy", csp)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self._request_checkpoint is None:
            self.send_error(404)
            return
        path = urlsplit(self.path).path.rstrip("/")
        if path == "/checkpoint":
            if self.headers.get(_PROGRAMMATIC_ACTION_HEADER) != _PROGRAMMATIC_ACTION_VALUE:
                self.send_error(403)
                return
        elif path == "":
            content_length = int(self.headers.get("Content-Length", "0"))
            token = parse_qs(self.rfile.read(content_length).decode()).get("token", [""])[0]
            if not hmac.compare_digest(token, self._action_token):
                self.send_error(403)
                return
        else:
            self.send_error(404)
            return

        self._request_checkpoint()
        body = b"Checkpoint requested. The save will start after the current training step.\n"
        self.send_response(202)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        logger.debug("Training dashboard HTTP request: " + format, *args)


@contextmanager
def _serve_status_page(
    snapshot: _TrainingSnapshot,
    request_checkpoint: Callable[[], None] | None,
    watchdog: ProgressWatchdog | None,
    *,
    bind_port: int,
) -> Iterator[int]:
    action_token = secrets.token_urlsafe(32)
    handler = partial(
        _TrainingDashboardRequestHandler,
        snapshot=snapshot,
        request_checkpoint=request_checkpoint,
        watchdog=watchdog,
        action_token=action_token,
    )
    with ThreadingHTTPServer(("0.0.0.0", bind_port), handler) as server:
        thread = Thread(
            target=server.serve_forever, name=f"training-dashboard-{server.server_address[1]}", daemon=True
        )
        thread.start()
        try:
            yield int(server.server_address[1])
        finally:
            server.shutdown()
            thread.join()


class TrainingDashboard(Generic[ConfigT]):
    """Publish the process-zero training status page through Iris."""

    def __init__(
        self,
        config: ConfigT,
        request_checkpoint: Callable[[], None] | None,
        run_id: str,
        watchdog: ProgressWatchdog | None = None,
    ):
        self._config = config
        self._request_checkpoint = request_checkpoint
        self._run_id = run_id
        self._watchdog = watchdog
        self._server_stack: ExitStack | None = None
        self._registry_stack: ExitStack | None = None

    def __enter__(self) -> TrainingDashboard:
        health_enabled = task_health_enabled()
        if health_enabled and self._watchdog is None:
            raise RuntimeError("Iris task health requires a progress watchdog")
        local_process_index = int(os.environ.get(IRIS_MULTIGPU_LOCAL_PROCESS_INDEX_ENV, "0"))
        if jax.process_index() != 0 and (not health_enabled or local_process_index != 0):
            return self

        try:
            self._publish()
        except Exception:
            if health_enabled:
                raise
            logger.warning("Training dashboard failed to start; training will continue", exc_info=True)
        return self

    def _publish(self) -> None:
        job_info = get_job_info()
        if job_info is None:
            if task_health_enabled():
                raise RuntimeError("Iris task health requires task metadata")
            return
        snapshot = _TrainingSnapshot.capture(
            self._config,
            run_id=self._run_id,
            job_id=str(job_info.job_id),
            task_id=str(job_info.task_id),
        )
        health_enabled = task_health_enabled()
        local_process_index = int(os.environ.get(IRIS_MULTIGPU_LOCAL_PROCESS_INDEX_ENV, "0"))
        serves_health = health_enabled and local_process_index == 0
        server_stack = ExitStack()
        try:
            port = server_stack.enter_context(
                _serve_status_page(
                    snapshot,
                    self._request_checkpoint if jax.process_index() == 0 else None,
                    self._watchdog,
                    bind_port=task_health_port() if serves_health else 0,
                )
            )
            if serves_health:
                publish_task_health(port)
        except Exception:
            server_stack.close()
            raise

        self._server_stack = server_stack
        if jax.process_index() != 0:
            return

        context = get_iris_ctx()
        if context is None:
            return
        endpoint_name = f"{job_info.job_id}/{TRAINING_CONTROL_ENDPOINT}"
        address = f"http://{job_info.advertise_host}:{port}"
        registry_stack = ExitStack()
        try:
            registry_stack.enter_context(
                context.registry.registered(endpoint_name, address, access=EndpointAccess.ENDPOINT_ACCESS_LINK)
            )
        except Exception:
            registry_stack.close()
            logger.warning("Training dashboard endpoint registration failed", exc_info=True)
            return

        self._registry_stack = registry_stack
        logger.info("Training dashboard: %s/", proxy_path(endpoint_name))

    def __exit__(self, *_: object) -> None:
        for name in ("_registry_stack", "_server_stack"):
            stack = getattr(self, name)
            if stack is None:
                continue
            setattr(self, name, None)
            try:
                stack.close()
            except Exception:
                logger.warning("Training dashboard failed to stop", exc_info=True)
