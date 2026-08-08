# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Boot a candidate image against a copy of a real store and query it.

Covers what the schema pre-flight cannot: catalog adoption, the ``.fidx`` and
Parquet layout revisions in a deployment's segments, and the planner's
substitution of covering projections.

The image serves in shadow mode, so it runs no maintenance and cannot reach the
archive the copy came from.
"""

import os
import re
import socket
import sqlite3
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from finelog.benchmarks.grafana_dashboard_corpus import load_dashboard_corpus, sqlstring_variables
from finelog.benchmarks.query_measurement import query_table, stats_client
from finelog.client.log_client import LogClient
from finelog.deploy.bootstrap import CACHE_DIR, HEALTH_OK
from finelog.deploy.config import INTRA_CLUSTER_CIDRS, CidrAuthLayer, auth_policy_json

# The snapshot mounts where the store was taken from: the catalog records
# absolute segment paths and boot adoption matches them exactly.
STORE_DIR = CACHE_DIR

CATALOG_FILENAME = "_finelog_catalog.sqlite"

# Opening a snapshot runs catalog adoption and per-namespace recovery.
BOOT_TIMEOUT = 300.0

# A dashboard variable the snapshot cannot supply a value for. The query still
# plans, substitutes projections, and scans; it just matches no rows.
UNMATCHED_VALUE = "__finelog_shadow_no_match__"

# `${__interval_ms}` is a panel's pixel width in Grafana; here it only has to
# divide the snapshot's window more than once.
DASHBOARD_INTERVAL_MS = 60_000

# The origin a row carries when its `cluster` column is unset, which happens
# only on the deployment that wrote it.
HOME_CLUSTER = "marin"

_MISSING_TABLE = re.compile(r"table '([^']+)' not found")
_DATAFUSION_PREFIX = "datafusion.public."


def namespaces_in_catalog(catalog: Path) -> set[str]:
    """Every namespace the catalog has registered."""
    connection = sqlite3.connect(f"file:{catalog}?mode=ro", uri=True)
    try:
        return {row[0] for row in connection.execute("SELECT namespace FROM namespaces")}
    finally:
        connection.close()


def missing_namespace(error: str) -> str | None:
    """The namespace a DataFusion planning error says does not exist, if that is the error."""
    match = _MISSING_TABLE.search(error)
    if match is None:
        return None
    return match.group(1).removeprefix(_DATAFUSION_PREFIX)


@dataclass
class ShadowReport:
    """What the rehearsal found. ``failures`` empty means the image is good."""

    namespaces_expected: tuple[str, ...] = ()
    namespaces_rehydrated: tuple[str, ...] = ()
    queries_run: int = 0
    queries_skipped: dict[str, str] = field(default_factory=dict)
    dashboards_run: tuple[str, ...] = ()
    dashboards_skipped: dict[str, str] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    def passed(self) -> bool:
        return not self.failures

    def describe(self) -> str:
        lines = [
            f"namespaces rehydrated: {len(self.namespaces_rehydrated)}/{len(self.namespaces_expected)}",
            f"dashboard queries green: {self.queries_run} across {', '.join(self.dashboards_run) or 'nothing'}",
        ]
        if self.queries_skipped:
            absent = sorted(set(self.queries_skipped.values()))
            lines.append(
                f"dashboard queries not run: {len(self.queries_skipped)}, "
                f"over namespaces this deployment does not have: {', '.join(absent)}"
            )
        for dashboard, reason in sorted(self.dashboards_skipped.items()):
            lines.append(f"not run: {dashboard} ({reason})")
        lines.extend(f"FAIL {failure}" for failure in self.failures)
        lines.append("SHADOW PASS" if self.passed() else "SHADOW FAIL")
        return "\n".join(lines)


def _unused_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


class ShadowServer:
    """The candidate image serving a snapshot in shadow mode."""

    def __init__(self, image: str, snapshot: Path, *, container_port: int = 10001) -> None:
        self.address = f"http://127.0.0.1:{_unused_port()}"
        host_port = self.address.rsplit(":", 1)[1]
        # The port publishes on loopback only, so the sole reachable caller is
        # this process — but it arrives over the container's bridge, not
        # loopback, and the server's default policy admits loopback alone.
        policy = auth_policy_json((CidrAuthLayer(cidrs=INTRA_CLUSTER_CIDRS),))
        # An explicitly empty FINELOG_REMOTE_DIR and FINELOG_FORWARDING: the
        # image inherits neither today, and if it ever did, shadow mode would
        # refuse to start rather than act on them.
        self._container = subprocess.run(
            [
                "docker",
                "run",
                "--detach",
                "--rm",
                "--volume",
                f"{snapshot.resolve()}:{STORE_DIR}",
                # The snapshot is owned by whoever extracted it, not by the
                # image's `finelog` user; the store opens read-write.
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "--publish",
                f"127.0.0.1:{host_port}:{container_port}",
                "--env",
                "FINELOG_MODE=shadow",
                "--env",
                "FINELOG_REMOTE_DIR=",
                "--env",
                "FINELOG_FORWARDING=",
                "--env",
                f"FINELOG_PORT={container_port}",
                "--env",
                f"FINELOG_AUTH_POLICY={policy}",
                image,
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    def wait_until_serving(self, timeout: float = BOOT_TIMEOUT) -> str:
        """Return the ``/health`` body once the store has opened and the server binds."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self._running():
                raise RuntimeError(f"the shadow server exited during boot:\n{self.logs()}")
            try:
                response = httpx.get(f"{self.address}/health", timeout=5)
                if response.is_success:
                    return response.text.strip()
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        raise TimeoutError(f"the shadow server did not open the snapshot within {timeout:.0f}s:\n{self.logs()}")

    def _running(self) -> bool:
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Running}}", self._container],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() == "true"

    def logs(self, tail: int = 100) -> str:
        result = subprocess.run(
            ["docker", "logs", "--tail", str(tail), self._container],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout + result.stderr

    def stop(self) -> None:
        subprocess.run(["docker", "stop", "--time", "10", self._container], capture_output=True, check=False)

    def __enter__(self) -> "ShadowServer":
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


def snapshot_window(client: LogClient, namespace: str = "telemetry_v1") -> tuple[int, int]:
    """The time range the snapshot's rows actually cover, in epoch ms."""
    table = client.query(f'SELECT min(timestamp_ms) AS lo, max(timestamp_ms) AS hi FROM "{namespace}"')
    low = table.column("lo")[0].as_py()
    high = table.column("hi")[0].as_py()
    if low is None or high is None:
        raise ValueError(f"{namespace} in this snapshot has no rows to query over")
    # A dashboard's window is half-open and its `date_bin` needs room for at
    # least one bucket, so widen a snapshot that landed inside a single ms.
    return int(low), max(int(high) + 1, int(low) + 2)


def snapshot_clusters(client: LogClient, namespace: str = "telemetry_v1") -> tuple[str, ...]:
    """The origin clusters present in the snapshot, as the dashboards filter on them."""
    table = client.query(
        f"SELECT DISTINCT COALESCE(NULLIF(\"cluster\", ''), '{HOME_CLUSTER}') AS c FROM \"{namespace}\" LIMIT 32"
    )
    return tuple(sorted(value for value in table.column("c").to_pylist() if value))


def dashboard_variables(dashboard: Path, clusters: Sequence[str]) -> dict[str, list[str]]:
    """Values for every ``${name:sqlstring}`` variable ``dashboard`` reads."""
    variables: dict[str, list[str]] = {}
    for name in sqlstring_variables(dashboard):
        variables[name] = list(clusters) if name == "cluster" else [UNMATCHED_VALUE]
    return variables


def run_dashboard_corpus(
    address: str,
    dashboards: Sequence[Path],
    *,
    start_ms: int,
    end_ms: int,
    interval_ms: int,
    clusters: Sequence[str],
    report: ShadowReport,
) -> None:
    """Run every renderable dashboard query against the shadow server.

    The dashboards read every namespace any Marin service writes, while a
    deployment holds only what its own clients registered. A query over a
    namespace absent from this catalog is recorded as not run; one over a
    namespace that did rehydrate and still fails to plan is a failure.
    """
    client = stats_client(address)
    rehydrated = set(report.namespaces_rehydrated)
    run: list[str] = []
    for dashboard in dashboards:
        try:
            corpus = load_dashboard_corpus(
                dashboard,
                start_ms=start_ms,
                end_ms=end_ms,
                interval_ms=interval_ms,
                variables=dashboard_variables(dashboard, clusters),
            )
        except ValueError as exc:
            # Recorded in full: an unresolved macro names the query it is in.
            report.dashboards_skipped[dashboard.name] = str(exc)
            continue
        green = 0
        for query in corpus.queries:
            try:
                query_table(client, query.sql)
            except Exception as exc:
                absent = missing_namespace(str(exc))
                if absent is not None and absent not in rehydrated:
                    report.queries_skipped[f"{dashboard.name}:{query.name}"] = absent
                    continue
                report.failures.append(f"{dashboard.name}:{query.name}: {exc}")
                continue
            green += 1
        report.queries_run += green
        if green:
            run.append(dashboard.name)
        else:
            report.dashboards_skipped[dashboard.name] = "every panel reads a namespace this deployment does not have"
    report.dashboards_run = tuple(run)


def check_snapshot(image: str, snapshot: Path, dashboards: Sequence[Path]) -> ShadowReport:
    """Boot ``image`` against ``snapshot`` and assert it serves what it adopted."""
    expected = namespaces_in_catalog(snapshot / CATALOG_FILENAME)
    report = ShadowReport(namespaces_expected=tuple(sorted(expected)))
    with ShadowServer(image, snapshot) as server:
        health = server.wait_until_serving()
        if health != HEALTH_OK:
            # The server-owned namespaces are registered on this same boot, so a
            # body that is not `ok` means the snapshot's catalog rejected them.
            report.failures.append(f"the server is serving but not ingesting: {health}")
        client = LogClient.connect(server.address)
        try:
            rehydrated = set(client.list_namespaces())
            report.namespaces_rehydrated = tuple(sorted(rehydrated))
            missing = expected - rehydrated
            if missing:
                report.failures.append(f"namespaces in the catalog that did not rehydrate: {sorted(missing)}")
            start_ms, end_ms = snapshot_window(client)
            clusters = snapshot_clusters(client)
        finally:
            client.close()
        run_dashboard_corpus(
            server.address,
            dashboards,
            start_ms=start_ms,
            end_ms=end_ms,
            interval_ms=DASHBOARD_INTERVAL_MS,
            clusters=clusters or (HOME_CLUSTER,),
            report=report,
        )
        if report.failures:
            report.failures.append(f"server log tail:\n{server.logs()}")
    return report
