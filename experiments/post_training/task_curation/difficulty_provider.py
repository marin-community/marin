# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bound shared-provider traffic and preserve admission and retry waits."""

import json
import logging
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import BoundedSemaphore, Lock

import httpx
from rigging.timing import ExponentialBackoff
from verifyit.grade import transient_model_auth

LOGGER = logging.getLogger(__name__)
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
ADMISSION_POLL_SECONDS = 1.0
MODEL_REQUEST_TIMEOUT = 600
DECLARED_JUDGE_REQUEST_TIMEOUT = 30


@dataclass(frozen=True)
class ProviderLimits:
    max_in_flight: int
    attempts: int
    request_timeout: float

    def __post_init__(self):
        if self.max_in_flight < 1 or self.attempts < 1 or self.request_timeout <= 0:
            raise ValueError("Provider capacity, attempts and request timeout must be positive")


class ProviderUnavailable(Exception):
    """The request remained outside provider capacity without reaching the model."""


class ProviderGate:
    def __init__(self, limits: ProviderLimits):
        self.limits = limits
        self.slots = BoundedSemaphore(limits.max_in_flight)
        self.lock = Lock()
        self.active = 0
        self.peak = 0

    @contextmanager
    def admission(self, root: Path, request_alive: Callable[[], bool]):
        started = time.monotonic()
        (root / "provider-queued.json").write_text(json.dumps({"started_at": time.time()}))
        polls = 0
        while not self.slots.acquire(timeout=ADMISSION_POLL_SECONDS):
            polls += 1
            if polls == 1:
                (root / "provider-capacity-wait.json").write_text(
                    json.dumps({"kind": "capacity_queue", "task_attempt": False})
                )
            if not request_alive():
                wait = root / "provider-waits" / "disconnected.json"
                wait.parent.mkdir(exist_ok=True)
                wait.write_text(
                    json.dumps({"kind": "provider_wait", "reason": "client_disconnected", "task_attempt": False})
                )
                raise ProviderUnavailable("The waiting client disconnected")
        try:
            with self.lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
                admission = {
                    "active": self.active,
                    "peak": self.peak,
                    "cap": self.limits.max_in_flight,
                    "capacity_wait_seconds": time.monotonic() - started,
                    "capacity_polls": polls,
                }
            (root / "provider-admitted.json").write_text(json.dumps(admission))
            LOGGER.info(
                "Provider admitted request active=%d peak=%d cap=%d capacity_wait=%.3f",
                admission["active"],
                admission["peak"],
                admission["cap"],
                admission["capacity_wait_seconds"],
            )
            try:
                yield
            finally:
                with self.lock:
                    self.active -= 1
                    LOGGER.info("Provider completed request active=%d cap=%d", self.active, self.limits.max_in_flight)
        finally:
            self.slots.release()

    def response(
        self, client: httpx.Client, endpoint: str, body: dict, root: Path, request_alive: Callable[[], bool]
    ) -> httpx.Response:
        """Retry transient model requests, retaining upstream replies and timeout evidence."""
        backoff = ExponentialBackoff(initial=1, maximum=32, factor=2, jitter=0.25)
        with self.admission(root, request_alive):
            for number in range(self.limits.attempts):
                if not request_alive():
                    wait = root / "provider-waits" / "disconnected.json"
                    wait.parent.mkdir(exist_ok=True)
                    wait.write_text(
                        json.dumps({"kind": "provider_wait", "reason": "client_disconnected", "task_attempt": False})
                    )
                    raise ProviderUnavailable("The waiting client disconnected")
                attempt = root / "upstream-attempts" / f"{number:02d}"
                attempt.mkdir(parents=True, exist_ok=False)
                try:
                    response = client.post(endpoint.rstrip("/") + "/chat/completions", json=body)
                except httpx.TimeoutException as error:
                    reason = type(error).__name__
                    (attempt / "transport-error.json").write_text(json.dumps({"error_identity": reason}))
                    if number + 1 == self.limits.attempts:
                        raise
                    status = None
                else:
                    status = response.status_code
                    (attempt / "response.body").write_bytes(response.content)
                    (attempt / "status.json").write_text(json.dumps({"status": status}))
                    auth_wait = transient_model_auth(status, response.content)
                    if status not in RETRYABLE_STATUS and not auth_wait:
                        return response
                    reason = "transient_model_auth" if auth_wait else None
                delay = backoff.next_interval() if number + 1 < self.limits.attempts else 0
                wait = root / "provider-waits" / f"{number:02d}.json"
                wait.parent.mkdir(exist_ok=True)
                wait.write_text(
                    json.dumps(
                        {
                            "status": status,
                            "backoff_seconds": delay,
                            "kind": "provider_wait",
                            "reason": reason,
                            "task_attempt": False,
                        }
                    )
                )
                LOGGER.warning(
                    "Provider wait status=%s reason=%s delay=%.3f active_cap=%d",
                    status,
                    reason,
                    delay,
                    self.limits.max_in_flight,
                )
                if delay:
                    time.sleep(delay)
            return response
