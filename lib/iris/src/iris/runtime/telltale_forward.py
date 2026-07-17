# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Persist a process's ``rigging.telltale`` registry into finelog.

A telltale registry is a live, process-local Prometheus view — a levanter
trainer's ``levanter_train_loss`` gauge, a zephyr counter — that dies when the
process exits. This module runs a background thread that snapshots the registry
every :data:`DEFAULT_INTERVAL` seconds and appends the samples to the shared
``telltale`` finelog namespace, so the series outlive the job and drive
dashboards over the grafana bridge.

The metrics carry no run/process identity of their own, so each row is stamped
with the Iris job identity (``job_id``/``task_id``/``process_index``/…) and any
process-global labels a producer declared via
``rigging.telltale.set_global_labels`` (levanter sets ``run``; zephyr sets
``source``/``run``). See the ``telltale`` schema below.

The namespace keys its parquet segments on ``name`` (the series name), so after
compaction a metric's rows cluster into contiguous row groups and
``WHERE name = 'levanter_train_loss'`` prunes on parquet stats + bloom filters
rather than scanning every series. Query time series with ``ORDER BY ts`` —
``seq`` is an ingest tiebreaker, not a clock.

Forwarding is best-effort: a resolve/connect/write failure is logged and the job
is never affected. Outside an in-cluster Iris job (no controller to resolve the
finelog endpoint) it is a no-op.
"""

import atexit
import logging
import os
import random
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar

from finelog.client import LogClient, Table
from rigging import telltale
from rigging.timing import Timestamp

from iris.client.client import get_iris_ctx
from iris.cluster.client.job_info import JobInfo, get_job_info
from iris.cluster.endpoints import LOG_SERVER_ENDPOINT_NAME
from iris.cluster.types import Namespace
from iris.runtime.multigpu import IRIS_MULTIGPU_PROCESS_INDEX_ENV

logger = logging.getLogger(__name__)

#: Finelog namespace every process's telltale metrics land in.
TELLTALE_NAMESPACE = "telltale"

#: Seconds between registry snapshots. Finelog seals at most one L0 per second,
#: so this stays well clear of L0 churn while keeping dashboards near-live.
DEFAULT_INTERVAL = 15.0

#: Producer prefixes that name their own source; anything else is "process"
#: (the default Prometheus process/platform collectors).
_KNOWN_SOURCES = frozenset({"levanter", "zephyr", "iris"})

_LABEL_SOURCE = "source"
_LABEL_RUN = "run"

_started = False
_started_lock = threading.Lock()


@dataclass
class TelltaleMetric:
    """One Prometheus sample, persisted as a finelog row.

    Segments key on ``name`` so a single metric's rows cluster together. ``run``
    and ``source`` are lifted out of the label map into top-level columns for
    clean, prunable dashboard SQL; the remaining identity and Prometheus labels
    (``le``, ``job_id``, …) stay in ``labels``.
    """

    key_column: ClassVar[str] = "name"

    name: str
    value: float
    labels: dict[str, str]
    kind: str
    source: str
    run: str | None
    ts: datetime


def _identity_labels(job_info: JobInfo) -> dict[str, str]:
    """The Iris identity stamped onto every row this process forwards.

    Authoritative: it overrides any same-named Prometheus or global label so a
    metric can never spoof the job it came from.
    """
    # The job root (``/user/job``), not JobInfo.job_id — the latter is the task's
    # immediate parent, which for a nested ``.../worker/3`` task is ``.../worker``.
    identity = {
        "job_id": str(Namespace.from_job_id(job_info.task_id)),
        "task_id": str(job_info.task_id),
        "attempt": str(job_info.attempt_id),
    }
    if job_info.worker_id:
        identity["worker"] = job_info.worker_id
    if job_info.worker_region:
        identity["region"] = job_info.worker_region
    process_index = os.environ.get(IRIS_MULTIGPU_PROCESS_INDEX_ENV)
    if process_index is not None:
        identity["process_index"] = process_index
    return identity


def _source_for(name: str, labels: Mapping[str, str]) -> str:
    """Resolve a row's ``source``: an explicit label wins, else the name prefix."""
    declared = labels.get(_LABEL_SOURCE)
    if declared:
        return declared
    head = name.split("_", 1)[0]
    return head if head in _KNOWN_SOURCES else "process"


def scrape_rows(
    identity: Mapping[str, str],
    global_labels: Mapping[str, str],
    ts: datetime,
) -> list[TelltaleMetric]:
    """Convert the current telltale registry into rows. Pure; no I/O.

    Label precedence on a key collision is ``sample < global < identity`` — the
    job identity always wins. Prometheus ``_created`` series (a counter's start
    time, not a metric) are dropped.
    """
    rows: list[TelltaleMetric] = []
    for family in telltale.samples():
        sample = family.sample
        if sample.name.endswith("_created"):
            continue
        merged = {**sample.labels, **global_labels, **identity}
        source = _source_for(sample.name, merged)
        run = merged.pop(_LABEL_RUN, None)
        merged.pop(_LABEL_SOURCE, None)
        rows.append(
            TelltaleMetric(
                name=sample.name,
                value=float(sample.value),
                labels=merged,
                kind=family.kind,
                source=source,
                run=run,
                ts=ts,
            )
        )
    return rows


class TelltaleForwarder:
    """Periodically appends the telltale registry to finelog on a daemon thread.

    Constructed with a live :class:`~finelog.client.Table`; the caller resolves
    the endpoint and identity (see :func:`start_forwarding`). ``stop`` flushes a
    final batch so a clean exit does not drop the last window.
    """

    def __init__(
        self,
        table: Table,
        identity: Mapping[str, str],
        *,
        interval: float = DEFAULT_INTERVAL,
        client: LogClient | None = None,
    ) -> None:
        self._table = table
        self._identity = dict(identity)
        self._interval = interval
        self._client = client
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="telltale-forward", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        # A random first delay desynchronizes the fleet so N processes don't all
        # write on the same tick and stack L0 flushes.
        if self._stop.wait(random.uniform(0.0, self._interval)):
            return
        while not self._stop.is_set():
            self._scrape_once()
            self._stop.wait(self._interval)

    def _scrape_once(self) -> None:
        rows = scrape_rows(self._identity, telltale.get_global_labels(), Timestamp.now().as_naive_utc())
        if not rows:
            return
        try:
            self._table.write(rows)
        except Exception:
            logger.warning("telltale forward: write failed", exc_info=True)

    def stop(self, *, flush_timeout: float = 5.0) -> None:
        self._stop.set()
        try:
            self._scrape_once()
            self._table.flush(timeout=flush_timeout)
        except Exception:
            logger.debug("telltale forward: final flush failed", exc_info=True)
        if self._client is not None:
            self._client.close()


def start_forwarding(*, interval: float = DEFAULT_INTERVAL) -> TelltaleForwarder | None:
    """Start the telltale→finelog forwarder for this process. Idempotent.

    Returns the forwarder, or ``None`` when nothing was started — either a repeat
    call, or the process is outside an in-cluster Iris job (no controller to
    resolve the finelog endpoint, so nothing to write to).
    """
    global _started
    with _started_lock:
        if _started:
            return None

        job_info = get_job_info()
        ctx = get_iris_ctx()
        if job_info is None or ctx is None or ctx.client is None:
            logger.debug("no in-cluster Iris job context; skipping telltale forwarding")
            return None

        try:
            endpoint = ctx.client.resolve_endpoint(LOG_SERVER_ENDPOINT_NAME)
        except Exception:
            logger.warning("telltale forward: could not resolve finelog endpoint", exc_info=True)
            return None

        try:
            client = LogClient.connect(endpoint)
            table = client.get_table(TELLTALE_NAMESPACE, TelltaleMetric)
        except Exception:
            logger.warning("telltale forward: could not connect to finelog at %s", endpoint, exc_info=True)
            return None

        forwarder = TelltaleForwarder(table, _identity_labels(job_info), interval=interval, client=client)
        forwarder.start()
        atexit.register(forwarder.stop)
        _started = True
        logger.info("telltale forwarding to finelog namespace %r via %s", TELLTALE_NAMESPACE, endpoint)
        return forwarder
