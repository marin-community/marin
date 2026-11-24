# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read-only Hublot API served behind Marina's named-app authentication."""

import logging
import os
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from finelog.client import LogClient, StatsError
from marina.apps import Services
from marina.discovery import resolve_internal_ip
from marina.process_cache import process_cache
from rigging.connect import IapAuth
from rigging.credentials import iap_provider_for

from . import queries
from .project import driver_facts, run_configuration, telemetry_rows
from .queries import Execution, Window
from .query_contract import MAX_WINDOW_MS, MIN_WINDOW_MS, SPAN_LOOKAHEAD_MS

logger = logging.getLogger(__name__)

MINUTE_MS = 60_000
LOOKBACK_HOURS = 72
MAX_WINDOW_ROWS = 108_000
MAX_PAGE_QUERIES = 24
MAX_RESPONSE_BYTES = 24 * 1024 * 1024
# Leave time for Marina to encode the response before Cloud Run's 60-second deadline.
READ_BUDGET = 50
RPC_TIMEOUT = 22
NO_STORE = {"Cache-Control": "private, no-store"}

type ReadRows = Callable[[str, int], list[dict[str, Any]]]
type RunChoice = dict[str, str | int | None]


@dataclass(frozen=True)
class RunMatch:
    cluster: str
    execution_uid: str
    job_id: str
    run_id: str | None
    first_ms: int
    last_ms: int
    wait_ms: int | None
    configurations: int
    configuration: str | None
    grading_ms: int | None
    start_ms: int | None
    end_ms: int | None
    inference_gpu_min: float | None
    inference_gpu_max: float | None
    policy_gpu_min: float | None
    policy_gpu_max: float | None
    buffer_capacity_min: float | None
    buffer_capacity_max: float | None

    @property
    def execution(self) -> Execution:
        return Execution(self.cluster, self.execution_uid, self.job_id)

    def choice(self) -> RunChoice:
        return {
            "id": self.execution.link_id,
            "cluster": self.cluster,
            "execution_uid": self.execution_uid,
            "job_id": self.job_id,
            "run_id": self.run_id,
            "last_ms": self.last_ms,
        }


class RunError(Exception):
    def __init__(self, status: int, message: str, *, choices: list[RunChoice] | None = None):
        super().__init__(message)
        self.status = status
        self.choices = choices


def _connect(url: str | None) -> LogClient:
    # Same hub and internal discovery as Zephyr. The override can use devbox's existing Marin IAP login.
    if url:
        interceptors = IapAuth(iap_provider_for("marin")).interceptors() if url.startswith("iap+") else ()
        return LogClient.connect(url.removeprefix("iap+"), timeout_ms=RPC_TIMEOUT * 1000, interceptors=interceptors)

    def resolve(_endpoint: str) -> str:
        ip = resolve_internal_ip("hai-gcp-models", "us-central1-a", "name = finelog-marin", timeout=5.0)
        return f"http://{ip}:10001"

    return LogClient.connect("finelog-marin", resolver=resolve, timeout_ms=RPC_TIMEOUT * 1000)


def _requested_window(start: str | None, end: str | None) -> Window | None:
    if start is None and end is None:
        return None
    if start is not None and end is not None and re.fullmatch(r"[0-9]{13}", start) and re.fullmatch(r"[0-9]{13}", end):
        window = Window(int(start), int(end))
        if MIN_WINDOW_MS <= window.end - window.start <= MAX_WINDOW_MS:
            return window
    seconds, minutes = MIN_WINDOW_MS // 1000, MAX_WINDOW_MS // MINUTE_MS
    raise RunError(400, f"Enter two UTC times {seconds} seconds to {minutes} minutes apart.")


def _resolve_execution(read: ReadRows, run: str, requested: Window | None) -> RunMatch:
    now = int(time.time() * 1000)
    search = (
        Window(requested.start - MAX_WINDOW_MS, requested.end + SPAN_LOOKAHEAD_MS)
        if requested
        else Window(now - LOOKBACK_HOURS * 60 * MINUTE_MS, now)
    )
    matches = [RunMatch(**row) for row in read(queries.resolution_sql(run, search), queries.CHOICE_LIMIT + 1)]
    if not matches:
        if requested:
            raise RunError(404, "No matching execution in or near this UTC window.")
        raise RunError(
            404, f"No matching execution in the last {LOOKBACK_HOURS} hours. For an older run, enter its UTC window."
        )
    if len(matches) > 1:
        choices = [match.choice() for match in matches[: queries.CHOICE_LIMIT]]
        raise RunError(409, "This ID matches more than one execution. Choose one.", choices=choices)
    return matches[0]


def _window_too_large() -> RunError:
    return RunError(422, "This window has too many records. Load a shorter window.")


def _read_timeout() -> RunError:
    return RunError(504, "Finelog took too long. Try again or load a shorter window.")


def _read_telemetry(read: ReadRows, execution: Execution, visible: Window) -> list[dict[str, Any]]:
    after: tuple[int, int] | None = None
    page_size = queries.PAGE_ROWS
    rows: list[dict[str, Any]] = []
    for _ in range(MAX_PAGE_QUERIES):
        remaining = MAX_WINDOW_ROWS - len(rows)
        # Read one extra row to distinguish a complete result from a full page.
        limit = min(page_size, remaining) + 1
        try:
            samples = read(queries.telemetry_sql(execution, visible, limit=limit, after=after), limit)
        except StatsError as error:
            cause = error.__cause__
            if not (
                isinstance(cause, ConnectError)
                and cause.code == Code.RESOURCE_EXHAUSTED
                and "message limit" in str(cause)
            ):
                raise
            if limit <= 2:
                raise _window_too_large() from error
            page_size = (limit - 1) // 2
            continue
        if len(samples) > remaining:
            raise _window_too_large()
        if len(samples) < limit:
            rows.extend(samples)
            return rows
        samples.pop()
        rows.extend(samples)
        last = samples[-1]
        after = (last["timestamp_ms"], last["seq"])
    raise _window_too_large()


def _read_driver(
    read: ReadRows, execution: Execution, visible: Window, start: int | None, end: int | None
) -> list[dict[str, Any]] | None:
    if execution.cluster == "local" or not execution.job.startswith("/") or start is None:
        return []
    search_end = visible.end + SPAN_LOOKAHEAD_MS
    start = max(start, visible.start - SPAN_LOOKAHEAD_MS)
    window = Window(start, min(end, search_end) if end is not None else search_end)
    if window.end <= window.start:
        return []
    try:
        rows = read(queries.driver_sql(execution, window), queries.DRIVER_LIMIT)
    except (StatsError, ConnectError, OSError, RunError) as error:
        logger.warning("Finelog did not return driver timing facts (%s)", type(error).__name__)
        return None
    if len(rows) == queries.DRIVER_LIMIT or len({row["key"] for row in rows}) != 1:
        return []
    return driver_facts(rows)


def _stable_count(low: float | None, high: float | None) -> int | None:
    if low is not None and low == high and 0 < low < 2**53 and int(low) == low:
        return int(low)
    return None


def _load_run(read: ReadRows, run: str, requested: Window | None) -> dict[str, Any]:
    resolved = _resolve_execution(read, run, requested)
    execution = resolved.execution
    start, end = resolved.start_ms, resolved.end_ms
    if start is None or end is None or end < start:
        end = None
    full = None
    if start is not None and end is not None and MIN_WINDOW_MS <= end - start + 1 <= MAX_WINDOW_MS:
        full = Window(start, end + 1)
    newest = resolved.wait_ms or resolved.last_ms
    default_start = max(resolved.first_ms - MINUTE_MS, newest - 20 * MINUTE_MS) // MIN_WINDOW_MS * MIN_WINDOW_MS
    default_end = (newest + 2 * MINUTE_MS + MIN_WINDOW_MS - 1) // MIN_WINDOW_MS * MIN_WINDOW_MS
    visible = requested or full or Window(default_start, max(default_start + MINUTE_MS, default_end))
    rows = _read_telemetry(read, execution, visible)
    driver = _read_driver(read, execution, visible, start, end)
    return {
        "deployment": "marin",
        "namespace": queries.NAMESPACE,
        "window": visible.as_dict(),
        "run": {
            "id": execution.link_id,
            "cluster": execution.cluster,
            "execution_uid": execution.uid,
            "job_id": execution.job,
            "run_id": resolved.run_id,
            "start_ms": start,
            "end_ms": end,
            "fullRun": full.as_dict() if full else None,
            "inferenceGpus": _stable_count(resolved.inference_gpu_min, resolved.inference_gpu_max),
            "policyGpus": _stable_count(resolved.policy_gpu_min, resolved.policy_gpu_max),
            "bufferCapacity": _stable_count(resolved.buffer_capacity_min, resolved.buffer_capacity_max),
            "separateGrading": resolved.grading_ms is not None,
        },
        "rows": telemetry_rows(rows),
        "configuration": run_configuration(resolved.configuration) if resolved.configurations == 1 else None,
        "driverRows": driver,
    }


def create_api(_services: Services) -> FastAPI:
    api = FastAPI(openapi_url=None)
    url = os.environ.get("HUBLOT_FINELOG_URL")
    client = process_cache.get_or_create("hublot.finelog.v1", (url, RPC_TIMEOUT), lambda: _connect(url))
    slot = process_cache.get_or_create("hublot.read_slot", (), lambda: threading.BoundedSemaphore(1))

    @api.get("/run")
    def get_run(
        run: str = "", start: str | None = Query(None, alias="from"), end: str | None = Query(None, alias="to")
    ) -> JSONResponse:
        try:
            run = run.strip()
            if not run or len(run) > 1024 or re.search(r"[\x00-\x1f\x7f]", run):
                raise RunError(400, "Enter an exact Finelog run ID, Iris job path, or execution UID.")
            requested = _requested_window(start, end)
            if not slot.acquire(blocking=False):
                raise RunError(503, "Hublot is reading another window. Try again in a moment.")
            try:
                deadline = time.monotonic() + READ_BUDGET

                def read(sql: str, max_rows: int) -> list[dict[str, Any]]:
                    if time.monotonic() + RPC_TIMEOUT > deadline:
                        raise _read_timeout()
                    try:
                        return client.query(sql, max_rows=max_rows).to_pylist()
                    except TimeoutError as error:
                        raise _read_timeout() from error
                    except StatsError as error:
                        cause = error.__cause__
                        if isinstance(cause, ConnectError) and cause.code == Code.DEADLINE_EXCEEDED:
                            raise _read_timeout() from error
                        raise

                response = JSONResponse(_load_run(read, run, requested), headers=NO_STORE)
                if len(response.body) > MAX_RESPONSE_BYTES:
                    raise _window_too_large()
                return response
            finally:
                slot.release()
        except RunError as error:
            payload: dict[str, object] = {"error": str(error)}
            if error.choices is not None:
                payload["choices"] = error.choices
            return JSONResponse(payload, error.status, headers=NO_STORE)
        except (StatsError, ConnectError, OSError, RuntimeError) as error:
            logger.warning("Hublot Finelog read failed (%s)", type(error).__name__)
            return JSONResponse({"error": "Finelog could not complete the read."}, 502, headers=NO_STORE)

    return api
