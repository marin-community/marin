# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Paced marina-applet SQL client with a shared bounded retry budget.

The per-release workers of the previous release cycles each carried a copy of this
harness (retry control, paced inserts, IPv4 resolution quirk). One maintained copy
serves every SQL-writing release stage: history staging, catalog load and curriculum
load. The retry budget is shared across all clients of one control so a failing run
halts its peers instead of retrying independently.
"""

import errno
import http.client
import json
import socket
import sys
import threading
import time
from collections import deque

import requests
import urllib3.util.connection as urllib3_connection
from google.auth.exceptions import TransportError
from marina.client import MarinaRequestError, bearer_token, marina_token_provider

import infra.marina.apps.rl_data_catalog.load as load_module

MAX_RETRIES_PER_RUN = 10
MAX_ATTEMPTS_PER_STATEMENT = 3
RETRY_WAIT_SECONDS = 300
PACED_INSERT_TABLES = ("INSERT INTO ATLAS_ITEMS", "INSERT INTO ATLAS_PACKETS")


def is_connection_reset(error: BaseException) -> bool:
    """True when an exception chain is a connection reset/abort/broken pipe."""
    pending: list[BaseException] = [error]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        reset_types = ConnectionResetError | ConnectionAbortedError | BrokenPipeError | http.client.RemoteDisconnected
        if isinstance(current, reset_types):
            return True
        if isinstance(current, OSError) and current.errno in (errno.ECONNRESET, errno.ECONNABORTED, errno.EPIPE):
            return True
        pending.extend(arg for arg in current.args if isinstance(arg, BaseException))
        pending.extend(cause for cause in (current.__cause__, current.__context__) if isinstance(cause, BaseException))
    return False


class RetryControl:
    """A shared budget of retries across all clients of one run."""

    def __init__(self, max_retries: int = MAX_RETRIES_PER_RUN) -> None:
        self.stop = threading.Event()
        self.max_retries = max_retries
        self._lock = threading.Lock()
        self._used = 0

    def claim(self) -> int | None:
        """Consume one retry from the budget, or ``None`` when exhausted."""
        with self._lock:
            if self._used >= self.max_retries:
                return None
            self._used += 1
            return self._used


class PacedQuery:
    """Wraps an applet query with retries, pacing and progress logging."""

    def __init__(self, inner, control: RetryControl) -> None:
        self.inner = inner
        self.control = control
        self.statements = 0
        self.pace = 0.1
        self.copy_pace = 0.0
        self.latencies: deque[float] = deque(maxlen=200)

    def _halt(self, label: str, error: BaseException, context: str) -> None:
        self.control.stop.set()
        print(f"{label}: {str(error)[:600]} | while: {context}", file=sys.stderr, flush=True)
        raise SystemExit(42)

    def _retry(self, label: str, error: BaseException, context: str, attempt: int) -> int:
        used = self.control.claim() if attempt < MAX_ATTEMPTS_PER_STATEMENT else None
        if used is None:
            self._halt(label + "-STOP", error, context)
        print(
            f"{label}-RETRY run-total {used} of {self.control.max_retries},"
            f" attempt {attempt + 1} of {MAX_ATTEMPTS_PER_STATEMENT}:"
            f" waiting {RETRY_WAIT_SECONDS}s - {str(error)[:300]} | while: {context}",
            file=sys.stderr,
            flush=True,
        )
        if self.control.stop.wait(RETRY_WAIT_SECONDS):
            raise RuntimeError("Stopped after another client failed")
        return attempt + 1

    def __call__(self, sql: str, parameters: dict):
        batch = parameters.get("rows")
        context = " ".join(sql.split())[:700]
        if batch is not None:
            context += f" | rows={len(json.loads(batch))} bytes={len(batch.encode())}"
        attempt = 0
        while True:
            if self.control.stop.is_set():
                raise RuntimeError("Stopped after another client failed")
            started = time.monotonic()
            try:
                result = self.inner(sql, parameters)
                break
            except MarinaRequestError as error:
                transient_iap = error.status_code >= 500 and (
                    "Error code 1" in str(error) or "authorizing your request" in str(error)
                )
                if transient_iap:
                    attempt = self._retry("IAP", error, context, attempt)
                    continue
                if error.status_code == 504:
                    attempt = self._retry("READ-504", error, context, attempt)
                    continue
                if error.status_code >= 500:
                    self._halt(f"5XX-STOP {error.status_code}", error, context)
                raise
            except (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout) as error:
                label = "READ" if isinstance(error, requests.exceptions.ReadTimeout) else "CONNECT"
                attempt = self._retry(label, error, context, attempt)
            except TransportError as error:
                self._halt("TOKEN-STOP", error, context)
            except (requests.ConnectionError, requests.exceptions.ChunkedEncodingError) as error:
                if is_connection_reset(error):
                    attempt = self._retry("RESET", error, context, attempt)
                    continue
                self._halt("CONNECT-STOP", error, context)
            except requests.HTTPError as error:
                status = error.response.status_code if error.response is not None else 0
                if status == 504:
                    attempt = self._retry("READ-504", error, context, attempt)
                    continue
                if status >= 500:
                    body = (error.response.text or "")[:600] if error.response is not None else ""
                    self._halt(f"5XX-STOP {status}", error, context if not body else body)
                raise
        elapsed = time.monotonic() - started
        self.latencies.append(elapsed)
        ordered = sorted(self.latencies)
        p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))] if len(ordered) >= 20 else 0
        if elapsed > 2 or p95 > 1.5:
            if self.copy_pace == 0:
                print(
                    f"PACE-BACKOFF: client={threading.current_thread().name} retained-copy 0 -> 0.25 s",
                    file=sys.stderr,
                    flush=True,
                )
            self.pace = 0.25
            self.copy_pace = 0.25
        stripped = sql.lstrip().upper()
        paced_insert = stripped.startswith(PACED_INSERT_TABLES)
        server_copy = paced_insert and batch is None
        if paced_insert:
            delay = self.copy_pace if server_copy else self.pace
            if delay and self.control.stop.wait(delay):
                raise RuntimeError("Stopped after another client failed")
        self.statements += 1
        if elapsed > 2:
            print(f"SLOW {round(elapsed, 2)}s {context[:200]}", file=sys.stderr, flush=True)
        if self.statements % 200 == 0:
            client = threading.current_thread().name
            print(f"PROGRESS client={client} statements={self.statements}", file=sys.stderr, flush=True)
        return result


_original_getaddrinfo = socket.getaddrinfo


def _ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):  # noqa: A002
    # The keyword name must match the stdlib getaddrinfo signature exactly.
    return _original_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)


def open_paced_query(service_url: str, applet_id: str, control: RetryControl) -> PacedQuery:
    """Open one paced applet query client for the given service and applet."""
    # The marina endpoint resolves to both address families; IPv4-only resolution avoids
    # stalled IPv6 attempts inside the regional workers (r39-era quirk, kept deliberately).
    socket.getaddrinfo = _ipv4_getaddrinfo
    urllib3_connection.allowed_gai_family = lambda: socket.AF_INET
    session = requests.Session()
    provider = marina_token_provider()
    bearer_token(provider)
    inner = load_module.applet_query(service_url, applet_id, session=session, token_provider=provider)
    return PacedQuery(inner, control)
