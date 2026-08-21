# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""HTTP status page for a Levanter training process running on Iris."""

from __future__ import annotations

import html
import json
import logging
import os
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event
from types import TracebackType
from typing import cast
from urllib.parse import urlsplit

import draccus
import jax
from rigging.connect import proxy_path
from rigging.redaction import REDACTED_VALUE, redact_value

from iris.client.client import get_iris_ctx
from iris.cluster.client.job_info import get_job_info
from iris.cluster.types import EndpointAccess
from iris.managed_thread import ThreadContainer
from levanter.trainer import AllConfig

logger = logging.getLogger(__name__)

TRAINING_CONTROL_ENDPOINT = "training-control"
TRAINING_CONTROL_PORT = "training_control"
_REDACTED_ENVIRONMENT_VARIABLES = frozenset({"IRIS_JOB_ENV", "IRIS_JOB_SETUP_SCRIPTS", "MARIN_PROVENANCE"})


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
        config: AllConfig,
        *,
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
            run_id=config.trainer.id or "unknown",
            job_id=job_id,
            task_id=task_id,
            environment=dict(sorted(redacted_environment.items())),
            configuration=json.dumps(encoded_config, indent=2, sort_keys=True),
        )


def _table_row(name: str, value: str) -> str:
    return f"<tr><th>{html.escape(name)}</th><td>{html.escape(value)}</td></tr>"


def _render_page(snapshot: _TrainingSnapshot) -> str:
    metadata = "".join(
        (
            _table_row("Run ID", snapshot.run_id),
            _table_row("Iris job", snapshot.job_id),
            _table_row("Iris task", snapshot.task_id),
        )
    )
    environment = "".join(_table_row(name, value) for name, value in snapshot.environment.items())
    configuration = html.escape(snapshot.configuration)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Levanter training: {html.escape(snapshot.run_id)}</title>
  <style>
    body {{ color: #1f2933; font: 14px/1.45 system-ui, sans-serif; margin: 2rem auto; max-width: 1100px; padding: 0 1rem; }}
    h1, h2 {{ line-height: 1.2; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #d8dee4; padding: .45rem .6rem; text-align: left; vertical-align: top; }}
    th {{ width: 16rem; }}
    td, pre {{ overflow-wrap: anywhere; white-space: pre-wrap; }}
    pre {{ background: #f6f8fa; border: 1px solid #d8dee4; border-radius: 4px; padding: 1rem; }}
  </style>
</head>
<body>
  <h1>Levanter training</h1>
  <p>Secret-like values are shown as [REDACTED].</p>
  <h2>Run</h2>
  <table>{metadata}</table>
  <h2>Resolved configuration</h2>
  <pre>{configuration}</pre>
  <h2>Environment</h2>
  <table>{environment}</table>
</body>
</html>
"""


class _TrainingControlRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def __init__(self, *args, snapshot: _TrainingSnapshot, **kwargs):
        self._snapshot = snapshot
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        if urlsplit(self.path).path != "/":
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return

        body = _render_page(self._snapshot).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:
        logger.debug("Training control HTTP request: " + format, *args)


class _TrainingControlHttpServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


@contextmanager
def _serve_status_page(snapshot: _TrainingSnapshot, port: int) -> Iterator[int]:
    handler = partial(_TrainingControlRequestHandler, snapshot=snapshot)
    server = _TrainingControlHttpServer(("0.0.0.0", port), handler)
    server.timeout = 0.2
    threads = ThreadContainer("levanter-training-control")

    def serve(stop_event: Event) -> None:
        while not stop_event.is_set():
            server.handle_request()

    try:
        threads.spawn(serve, name=f"training-control-{server.server_address[1]}")
        yield int(server.server_address[1])
    finally:
        threads.stop()
        server.server_close()


class TrainingControl:
    """Publish the process-zero training status page through Iris."""

    def __init__(self, config: AllConfig):
        self._config = config
        self._stack: ExitStack | None = None

    def __enter__(self) -> TrainingControl:
        if jax.process_index() != 0:
            return self

        try:
            self._publish()
        except Exception:
            logger.warning("Training control page failed to start; training will continue", exc_info=True)
        return self

    def _publish(self) -> None:
        context = get_iris_ctx()
        job_info = get_job_info()
        if context is None or job_info is None:
            return
        if TRAINING_CONTROL_PORT not in job_info.ports:
            logger.info("Training control page is disabled because the Iris job has no %s port", TRAINING_CONTROL_PORT)
            return

        snapshot = _TrainingSnapshot.capture(
            self._config,
            job_id=str(job_info.job_id),
            task_id=str(job_info.task_id),
        )
        stack = ExitStack()
        try:
            port = stack.enter_context(_serve_status_page(snapshot, job_info.ports[TRAINING_CONTROL_PORT]))
            endpoint_name = f"{job_info.job_id}/{TRAINING_CONTROL_ENDPOINT}"
            address = f"http://{job_info.advertise_host}:{port}"
            stack.enter_context(
                context.registry.registered(
                    endpoint_name,
                    address,
                    access=EndpointAccess.ENDPOINT_ACCESS_PRIVATE,
                )
            )
        except Exception:
            stack.close()
            raise

        self._stack = stack
        logger.info("Training control page: %s/", proxy_path(endpoint_name))

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        if self._stack is None:
            return False
        stack = self._stack
        self._stack = None
        try:
            return bool(stack.__exit__(exc_type, exc_value, traceback))
        except Exception:
            logger.warning("Training control page failed to stop", exc_info=True)
            return False
