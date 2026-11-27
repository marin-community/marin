# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read-only Hublot API served behind Marina's named-app authentication."""

import logging
import math
import os
import re
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, TypedDict, cast

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse
from finelog.client import LogClient, StatsError
from marina.apps import Services
from marina.discovery import resolve_internal_ip
from marina.process_cache import process_cache
from rigging.connect import IapAuth
from rigging.credentials import iap_provider_for

from . import queries
from .project import driver_facts, run_facts, telemetry_rows
from .queries import Execution, Window
from .query_contract import (
    BACKGROUND_READ_PAUSE_MS,
    BUCKET_MS,
    DETAIL_NAMES,
    DETAIL_SERIES_COUNTS,
    FLEET_CARD,
    INFERENCE_TAB_GROUPS,
    MAX_OVERVIEW_MS,
    METRIC_POINTS,
    MIN_SUMMARY_BUCKET_MS,
    MIN_WINDOW_MS,
    RAW_METRIC_WINDOW_MS,
    SAMPLE_BATCH_SIZE,
    SAMPLE_SLICE_MS,
    SPAN_LOOKAHEAD_MS,
)

logger = logging.getLogger(__name__)

MINUTE_MS = 60_000
DAY_MS = 24 * 60 * MINUTE_MS
LOOKBACK_DAYS = 7
SUMMARY_SCAN_MS = 4 * 60 * MINUTE_MS
BACKGROUND_SUMMARY_SCAN_MS = 60 * MINUTE_MS
INFERENCE_TAB_FIRST_PAGE_MS = 20 * MINUTE_MS
INFERENCE_TAB_SCAN_MS = 120 * MINUTE_MS
MAX_WINDOW_ROWS = 108_000
MAX_PAGE_QUERIES = 24
MAX_RESPONSE_BYTES = 24 * 1024 * 1024
# Leave time for Marina to encode the response before Cloud Run's 60-second deadline.
READ_BUDGET = 50
RPC_TIMEOUT = 22
NO_STORE = {"Cache-Control": "private, no-store"}

type ReadRows = Callable[[str, int], list[dict[str, Any]]]
type RunChoice = dict[str, str | int | None]


class SummaryRow(TypedDict):
    bucket_start: int
    name: str
    reason: str | None
    value: float
    requests: float | None
    engines: int | None


class ReadPace(TypedDict):
    next_start: float


@dataclass(frozen=True)
class RunMatch:
    cluster: str
    execution_uid: str
    job_id: str
    run_id: str | None
    first_ms: int
    last_ms: int
    start_ms: int | None
    end_ms: int | None

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
    def __init__(
        self, status: int, message: str, *, choices: list[RunChoice] | None = None, retry_after: int | None = None
    ):
        super().__init__(message)
        self.status = status
        self.choices = choices
        self.retry_after = retry_after


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
        if MIN_WINDOW_MS <= window.end - window.start <= MAX_OVERVIEW_MS:
            return window
    raise RunError(
        400, f"Enter two UTC times {MIN_WINDOW_MS // 1000} seconds to {MAX_OVERVIEW_MS // DAY_MS} days apart."
    )


def _validated_run_lookup(run: str) -> str:
    run = run.strip()
    if not run or len(run) > 1024 or re.search(r"[\x00-\x1f\x7f]", run):
        raise RunError(400, "Enter an exact Finelog run ID, Iris job path, or execution UID.")
    return run


def _metrics_scope(run: str, start: str | None, end: str | None, card: str | None) -> tuple[Execution, Window]:
    execution = Execution.from_link_id(_validated_run_lookup(run))
    visible = _requested_window(start, end)
    if execution is None or visible is None:
        raise RunError(400, "Open a run before loading its metrics.")
    if card is not None and card not in DETAIL_NAMES:
        raise RunError(400, "Unknown metrics card.")
    return execution, visible


def _search_window(date_hint: str | None) -> Window:
    now = int(time.time() * 1000)
    if date_hint is None:
        return Window(now - LOOKBACK_DAYS * DAY_MS, now)
    try:
        if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", date_hint):
            raise ValueError("expected an ISO date")
        day = int(datetime.fromisoformat(date_hint).replace(tzinfo=UTC).timestamp() * 1000)
    except ValueError as error:
        raise RunError(400, "Enter the UTC date when the run started.") from error
    if day >= now:
        raise RunError(400, "Enter a run start date up to today.")
    return Window(day, min(now, day + DAY_MS + MAX_OVERVIEW_MS))


def _resolve_execution(read: ReadRows, run: str, search: Window, date_hint: str | None) -> RunMatch:
    limit = queries.CHOICE_LIMIT + 1
    matches: list[RunMatch] = []
    if queries.is_execution_id(run):
        matches = [RunMatch(**row) for row in read(queries.lifecycle_sql(run, search), limit)]
        if len(matches) == 1:
            match = matches[0]
            if match.start_ms is not None and match.end_ms is not None:
                return match
            run = match.execution.link_id
            if match.start_ms is not None:
                search = Window(match.start_ms, search.end)
    if len(matches) < 2:
        matches = [RunMatch(**row) for row in read(queries.activity_sql(run, search), limit)]
    if not matches:
        message = (
            f"No matching run starting near {date_hint} UTC."
            if date_hint
            else (f"No matching run in the last {LOOKBACK_DAYS} days.")
        )
        raise RunError(404, message)
    if len(matches) > 1:
        choices = [match.choice() for match in matches[: queries.CHOICE_LIMIT]]
        raise RunError(409, "This ID matches more than one execution. Choose one.", choices=choices)
    return matches[0]


def _window_too_large() -> RunError:
    return RunError(422, "This window has too many records. Load a shorter window.")


def _read_timeout() -> RunError:
    return RunError(504, "Finelog took too long. Try again or load a shorter window.")


def _read_telemetry(read: ReadRows, execution: Execution, visible: Window, card: str | None) -> list[dict[str, Any]]:
    after: tuple[int, int] | None = None
    page_size = queries.PAGE_ROWS
    rows: list[dict[str, Any]] = []
    for _ in range(MAX_PAGE_QUERIES):
        remaining = MAX_WINDOW_ROWS - len(rows)
        # Read one extra row to distinguish a complete result from a full page.
        limit = min(page_size, remaining) + 1
        try:
            samples = read(queries.telemetry_sql(execution, visible, card=card, limit=limit, after=after), limit)
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


def _run_metadata(read: ReadRows, run: str, search: Window, date_hint: str | None) -> dict[str, Any]:
    resolved = _resolve_execution(read, run, search, date_hint)
    execution = resolved.execution
    start, end = resolved.start_ms, resolved.end_ms
    if start is None or end is None or end < start:
        end = None
    newest = (end if end is not None else resolved.last_ms) + 1
    earliest = max(start if start is not None else resolved.first_ms, newest - MAX_OVERVIEW_MS)
    visible = Window(earliest, max(newest, earliest + MIN_WINDOW_MS))
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
        },
    }


def _window_response(payload: dict[str, Any]) -> JSONResponse:
    response = JSONResponse(payload, headers=NO_STORE)
    if len(response.body) > MAX_RESPONSE_BYTES:
        raise _window_too_large()
    return response


def _summary_rows(read: ReadRows, sql: str, max_rows: int) -> list[SummaryRow]:
    return [
        cast(SummaryRow, row)
        for row in read(sql, max_rows)
        if math.isfinite(row["value"]) and (row["requests"] is None or math.isfinite(row["requests"]))
    ]


def _read_metrics(
    read: ReadRows, execution: Execution, visible: Window, card: str, after: int | None, *, background: bool
) -> JSONResponse:
    duration = visible.end - visible.start
    summary_only = card == FLEET_CARD or card in INFERENCE_TAB_GROUPS
    if not summary_only and duration <= RAW_METRIC_WINDOW_MS and after is None:
        rows = _read_telemetry(read, execution, visible, card)
        return _window_response({"rows": telemetry_rows(rows)})

    bucket_ms = max(MIN_SUMMARY_BUCKET_MS, math.ceil(duration / (METRIC_POINTS * BUCKET_MS)) * BUCKET_MS)
    start = visible.start if after is None else after
    if not visible.start <= start < visible.end or (start - visible.start) % bucket_ms:
        raise RunError(400, "Use Hublot's next time range to continue loading metrics.")
    # Finish on a bucket boundary so adjacent requests can share counter samples.
    scan_limit = BACKGROUND_SUMMARY_SCAN_MS if background else SUMMARY_SCAN_MS
    if card in INFERENCE_TAB_GROUPS:
        # A short first page lets charts draw while the remaining pages load.
        first_long_page = after is None and duration > INFERENCE_TAB_SCAN_MS
        scan_limit = INFERENCE_TAB_FIRST_PAGE_MS if first_long_page else INFERENCE_TAB_SCAN_MS
    scan_ms = max(1, scan_limit // bucket_ms) * bucket_ms
    end = min(start + scan_ms, visible.end)
    rows = _summary_rows(
        read,
        queries.detail_summary_sql(execution, Window(start, end), card, bucket_ms),
        METRIC_POINTS * DETAIL_SERIES_COUNTS[card],
    )
    return _window_response({"buckets": rows, "bucketMs": bucket_ms, "next": end if end < visible.end else None})


def create_api(_services: Services) -> FastAPI:
    api = FastAPI(openapi_url=None)
    url = os.environ.get("HUBLOT_FINELOG_URL")
    client = process_cache.get_or_create("hublot.finelog.v1", (url, RPC_TIMEOUT), lambda: _connect(url))
    slot = process_cache.get_or_create("hublot.read_slot", (), lambda: threading.BoundedSemaphore(1))
    # Keep a plain, typed record that all loaded app revisions can share.
    pace = process_cache.get_or_create("hublot.sample_pace", (url,), lambda: ReadPace(next_start=0.0))

    @contextmanager
    def finelog_reader(*, background: bool = False) -> Iterator[ReadRows]:
        if not slot.acquire(blocking=False):
            raise RunError(503, "Hublot is reading another window. Try again in a moment.", retry_after=1)
        started = time.monotonic()
        deadline = started + READ_BUDGET

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

        admitted = False
        try:
            if background:
                remaining = pace["next_start"] - started
                if remaining > 0:
                    raise RunError(429, "Hublot is pacing background reads.", retry_after=math.ceil(remaining))
            admitted = True
            yield read
        except (StatsError, ConnectError, OSError, RuntimeError) as error:
            logger.warning("Hublot Finelog read failed (%s)", type(error).__name__)
            raise RunError(502, "Finelog could not complete the read.") from error
        finally:
            if background and admitted:
                pace["next_start"] = time.monotonic() + BACKGROUND_READ_PAUSE_MS / 1000
            slot.release()

    @api.exception_handler(RunError)
    def run_error(_request: Request, error: RunError) -> JSONResponse:
        payload: dict[str, object] = {"error": str(error)}
        if error.choices is not None:
            payload["choices"] = error.choices
        headers = NO_STORE if error.retry_after is None else {**NO_STORE, "Retry-After": str(error.retry_after)}
        return JSONResponse(payload, error.status, headers=headers)

    @api.get("/run")
    def get_run(run: str = "", date: str | None = None) -> JSONResponse:
        run = _validated_run_lookup(run)
        search = _search_window(date)
        with finelog_reader() as read:
            return JSONResponse(_run_metadata(read, run, search, date), headers=NO_STORE)

    @api.get("/window")
    def get_window(
        run: str = "",
        start: str | None = Query(None, alias="from"),
        end: str | None = Query(None, alias="to"),
        card: str | None = None,
        after: int | None = None,
        run_start: int | None = None,
        run_end: int | None = None,
        background: bool = False,
    ) -> JSONResponse:
        execution, visible = _metrics_scope(run, start, end, card)
        with finelog_reader(background=background) as read:
            if card is not None:
                return _read_metrics(read, execution, visible, card, after, background=background)
            rows = _read_telemetry(read, execution, visible, None)
            driver = _read_driver(read, execution, visible, run_start, run_end)
            return _window_response({"rows": telemetry_rows(rows), "driverRows": driver, **run_facts(rows)})

    @api.get("/samples")
    def get_samples(
        run: str = "",
        start: str | None = Query(None, alias="from"),
        end: str | None = Query(None, alias="to"),
        card: str = "",
        until: str = "",
    ) -> JSONResponse:
        execution, visible = _metrics_scope(run, start, end, card)
        times = until.split(",")
        if len(times) > SAMPLE_BATCH_SIZE or any(re.fullmatch(r"[0-9]{13}", time) is None for time in times):
            raise RunError(400, "Choose sample times within the selected range.")
        ends = sorted(set(map(int, times)))
        if ends[0] <= visible.start or ends[-1] > visible.end:
            raise RunError(400, "Choose sample times within the selected range.")
        windows = [Window(max(visible.start, sample_end - SAMPLE_SLICE_MS), sample_end) for sample_end in ends]
        with finelog_reader(background=True) as read:
            rows = _summary_rows(
                read,
                queries.sampled_summary_sql(execution, windows, card, SAMPLE_SLICE_MS),
                len(windows) * DETAIL_SERIES_COUNTS[card],
            )
        return _window_response({"buckets": rows, "samples": [window.as_dict() for window in windows]})

    return api
