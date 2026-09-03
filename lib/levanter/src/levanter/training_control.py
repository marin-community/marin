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
from dataclasses import dataclass
from functools import partial
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
from iris.cluster.types import EndpointAccess


logger = logging.getLogger(__name__)

TRAINING_CONTROL_ENDPOINT = "training-control"
_REDACTED_ENVIRONMENT_VARIABLES = ("IRIS_JOB_ENV", "IRIS_JOB_SETUP_SCRIPTS", "MARIN_PROVENANCE")
_PROGRAMMATIC_ACTION_HEADER = "X-Levanter-Training-Control"
_PROGRAMMATIC_ACTION_VALUE = "request-checkpoint"
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


def _render_page(snapshot: _TrainingSnapshot, action_token: str) -> str:
    configuration = html.escape(snapshot.configuration)
    environment = html.escape(json.dumps(snapshot.environment, indent=2))
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
        request_checkpoint: Callable[[], None],
        action_token: str,
        **kwargs,
    ):
        self._snapshot = snapshot
        self._request_checkpoint = request_checkpoint
        self._action_token = action_token
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        body = _render_page(self._snapshot, self._action_token).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; base-uri 'none'",
        )
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
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
def _serve_status_page(snapshot: _TrainingSnapshot, request_checkpoint: Callable[[], None]) -> Iterator[int]:
    action_token = secrets.token_urlsafe(32)
    handler = partial(
        _TrainingDashboardRequestHandler,
        snapshot=snapshot,
        request_checkpoint=request_checkpoint,
        action_token=action_token,
    )
    with ThreadingHTTPServer(("0.0.0.0", 0), handler) as server:
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

    def __init__(self, config: ConfigT, request_checkpoint: Callable[[], None], run_id: str):
        self._config = config
        self._request_checkpoint = request_checkpoint
        self._run_id = run_id
        self._stack: ExitStack | None = None

    def __enter__(self) -> TrainingDashboard:
        if jax.process_index() != 0:
            return self

        try:
            self._publish()
        except Exception:
            logger.warning("Training dashboard failed to start; training will continue", exc_info=True)
        return self

    def _publish(self) -> None:
        context = get_iris_ctx()
        job_info = get_job_info()
        if context is None or job_info is None:
            return
        snapshot = _TrainingSnapshot.capture(
            self._config,
            run_id=self._run_id,
            job_id=str(job_info.job_id),
            task_id=str(job_info.task_id),
        )
        stack = ExitStack()
        try:
            port = stack.enter_context(_serve_status_page(snapshot, self._request_checkpoint))
            endpoint_name = f"{job_info.job_id}/{TRAINING_CONTROL_ENDPOINT}"
            address = f"http://{job_info.advertise_host}:{port}"
            stack.enter_context(
                context.registry.registered(
                    endpoint_name,
                    address,
                    access=EndpointAccess.ENDPOINT_ACCESS_LINK,
                )
            )
        except Exception:
            stack.close()
            raise

        self._stack = stack
        logger.info("Training dashboard: %s/", proxy_path(endpoint_name))

    def __exit__(self, *_: object) -> None:
        if self._stack is None:
            return
        stack = self._stack
        self._stack = None
        try:
            stack.close()
        except Exception:
            logger.warning("Training dashboard failed to stop", exc_info=True)
