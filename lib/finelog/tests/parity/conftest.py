# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dual-backend RPC parity harness.

The finelog server is being rewritten in Rust (see
``.agents/projects/2026-06-02_finelog_rust.md``). To keep the Python test
surface valid against either implementation, parity tests talk to the server
**only over HTTP/RPC** — they never import ``DuckDBLogStore`` or any store
internals. A single fixture spawns the chosen backend as a subprocess (both
share the same ``--port``/``--log-dir`` CLI), waits for ``/health``, and yields
a base URL. Tests drive it through the real :class:`finelog.client.LogClient`.

Backends are parametrized: ``python`` always runs; ``rust`` runs only when the
``finelog-server`` binary has been built (skipped otherwise so the suite still
passes on a machine without the Rust toolchain). As each migration phase lands
an RPC family in Rust, the corresponding ``rust`` parametrization flips from
xfail/skip to green — no test-body changes required.
"""

from __future__ import annotations

import io
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.ipc as paipc
import pytest
from finelog.client import LogClient
from finelog.rpc import finelog_stats_pb2 as stats_pb2
from finelog.rpc.finelog_stats_connect import StatsServiceClientSync

# lib/finelog/tests/parity/conftest.py -> repo root is four parents up from
# the lib dir: parents[4] == repo root.
_REPO_ROOT = Path(__file__).resolve().parents[4]


def _rust_binary() -> Path | None:
    """Locate a built ``finelog-server`` binary, or None if unavailable."""
    override = os.environ.get("FINELOG_RUST_BIN")
    if override:
        p = Path(override)
        return p if p.exists() else None
    for profile in ("release", "debug"):
        candidate = _REPO_ROOT / "rust" / "target" / profile / "finelog-server"
        if candidate.exists():
            return candidate
    return None


@dataclass(frozen=True)
class Backend:
    """One server implementation under test."""

    name: str

    def command(self, *, port: int, log_dir: Path, remote_log_dir: str = "") -> list[str]:
        # ``--remote-log-dir`` configures the offload target; empty disables sync.
        # The offload / eviction parity families pass a tmp remote dir; the rest
        # leave it empty. Both backends accept the same flag.
        remote_args = ["--remote-log-dir", remote_log_dir] if remote_log_dir else []
        if self.name == "python":
            return [
                sys.executable,
                "-m",
                "finelog.server.main",
                "--port",
                str(port),
                "--log-dir",
                str(log_dir),
                "--log-level",
                "INFO",
                # Phase 4: the non-proto test-only admin surface, on BOTH
                # backends, so /debug/maintain + /debug/segments + /debug/backdate
                # drive the same parity test body.
                "--debug-admin",
                *remote_args,
            ]
        if self.name == "rust":
            binary = _rust_binary()
            assert binary is not None  # guarded by the fixture skip
            return [
                str(binary),
                "--port",
                str(port),
                "--log-dir",
                str(log_dir),
                "--debug-admin",
                *remote_args,
            ]
        raise ValueError(f"unknown backend: {self.name}")


_BACKENDS = ("python", "rust")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_health(base_url: str, proc: subprocess.Popen, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode} before /health came up")
        try:
            resp = httpx.get(f"{base_url}/health", timeout=1.0)
            if resp.status_code == 200:
                return
        except httpx.HTTPError as exc:  # not up yet
            last_err = exc
        time.sleep(0.05)
    raise TimeoutError(f"{base_url}/health did not come up within {timeout}s: {last_err}")


@pytest.fixture(params=_BACKENDS)
def server_backend(request: pytest.FixtureRequest) -> Backend:
    """Parametrized server backend. Skips ``rust`` when the binary is absent."""
    name = request.param
    if name == "rust" and _rust_binary() is None:
        pytest.skip("finelog-server Rust binary not built (run `cargo build -p finelog`)")
    return Backend(name)


@pytest.fixture
def finelog_url(server_backend: Backend, tmp_path: Path) -> Iterator[str]:
    """Spawn the backend on an ephemeral port; yield its base URL."""
    port = _free_port()
    log_dir = tmp_path / "store"
    log_dir.mkdir(parents=True, exist_ok=True)
    base_url = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen(
        server_backend.command(port=port, log_dir=log_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_for_health(base_url, proc, timeout=20.0)
        yield base_url
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5.0)


@pytest.fixture
def client(finelog_url: str) -> Iterator[LogClient]:
    c = LogClient.connect(finelog_url)
    try:
        yield c
    finally:
        c.close()


@dataclass(frozen=True)
class RemoteServer:
    """A spawned backend plus the on-disk remote (offload) directory it syncs to.

    The offload / eviction parity families need to observe the bucket directly
    (a segment uploaded, an orphan deleted, a durable archive preserved), so the
    remote dir is a local tmp path both backends `put`/`rm` into. The Rust
    backend roots an ``object_store`` ``LocalFileSystem`` there; the Python
    backend uses ``fsspec``'s ``LocalFileSystem`` — same on-disk layout
    ``{remote}/{namespace}/{basename}``.
    """

    base_url: str
    remote_dir: Path

    def remote_files(self, namespace: str) -> list[str]:
        """Basenames of every parquet object the bucket holds for ``namespace``."""
        ns_dir = self.remote_dir / namespace
        if not ns_dir.is_dir():
            return []
        return sorted(p.name for p in ns_dir.glob("*.parquet"))


@pytest.fixture
def finelog_url_remote(server_backend: Backend, tmp_path: Path) -> Iterator[RemoteServer]:
    """Spawn the backend with a configured local remote dir; yield both."""
    port = _free_port()
    log_dir = tmp_path / "store"
    log_dir.mkdir(parents=True, exist_ok=True)
    remote_dir = tmp_path / "remote"
    remote_dir.mkdir(parents=True, exist_ok=True)
    base_url = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen(
        server_backend.command(port=port, log_dir=log_dir, remote_log_dir=str(remote_dir)),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        _wait_for_health(base_url, proc, timeout=20.0)
        yield RemoteServer(base_url=base_url, remote_dir=remote_dir)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5.0)


class RestartableServer:
    """Start/stop one backend repeatedly on a FIXED ``log_dir``.

    The default :func:`finelog_url` fixture spawns a fresh backend on a throwaway
    tmp dir, which cannot exercise restart-survival. This helper keeps the
    ``log_dir`` constant so a second process boots over the first one's parquet
    segments + catalog — the externally-observable durability gate. Phase 4/6
    (boot reconcile, cross-backend cutover) reuse this; keep it backend-agnostic.
    """

    def __init__(self, backend: Backend, log_dir: Path, remote_dir: Path | None = None) -> None:
        self._backend = backend
        self._log_dir = log_dir
        self._remote_dir = remote_dir
        self._proc: subprocess.Popen | None = None
        self._base_url: str | None = None

    @property
    def base_url(self) -> str:
        assert self._base_url is not None, "server is not running"
        return self._base_url

    @property
    def remote_dir(self) -> Path:
        assert self._remote_dir is not None, "no remote dir configured"
        return self._remote_dir

    @property
    def log_dir(self) -> Path:
        return self._log_dir

    def log_dir_sidecars(self) -> list[Path]:
        """Catalog sidecar files present in ``log_dir`` (backend-agnostic).

        Python uses ``_finelog_registry.duckdb``; Rust uses
        ``_finelog_catalog.sqlite``. The wiped-catalog gate deletes whichever
        exists to model a lost local catalog while keeping the remote bucket.
        """
        candidates = ["_finelog_registry.duckdb", "_finelog_catalog.sqlite"]
        return [self._log_dir / name for name in candidates if (self._log_dir / name).exists()]

    def remote_files(self, namespace: str) -> list[str]:
        """Basenames of every parquet object the bucket holds for ``namespace``."""
        ns_dir = self.remote_dir / namespace
        if not ns_dir.is_dir():
            return []
        return sorted(p.name for p in ns_dir.glob("*.parquet"))

    def start(self, *, health_timeout: float = 20.0) -> str:
        """Spawn the backend on a fresh port over the fixed log_dir; wait healthy."""
        assert self._proc is None, "server already running; call stop() first"
        port = _free_port()
        base_url = f"http://127.0.0.1:{port}"
        remote_log_dir = str(self._remote_dir) if self._remote_dir is not None else ""
        proc = subprocess.Popen(
            self._backend.command(port=port, log_dir=self._log_dir, remote_log_dir=remote_log_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            _wait_for_health(base_url, proc, timeout=health_timeout)
        except BaseException:
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
            raise
        self._proc = proc
        self._base_url = base_url
        return base_url

    def stop(self) -> None:
        """Terminate the running process and wait for it to exit."""
        if self._proc is None:
            return
        self._proc.terminate()
        try:
            self._proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait(timeout=5.0)
        self._proc = None
        self._base_url = None


@pytest.fixture
def restartable_server(server_backend: Backend, tmp_path: Path) -> Iterator[RestartableServer]:
    """A :class:`RestartableServer` over a fixed log_dir; auto-stops on teardown."""
    log_dir = tmp_path / "store"
    log_dir.mkdir(parents=True, exist_ok=True)
    server = RestartableServer(server_backend, log_dir)
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture
def restartable_remote_server(server_backend: Backend, tmp_path: Path) -> Iterator[RestartableServer]:
    """A :class:`RestartableServer` over fixed log_dir + remote_dir; auto-stops.

    Used by the boot-reconcile gate (wipe local catalog + parquet, keep the
    remote bucket, respawn on the same dirs -> remote files adopted as REMOTE).
    """
    log_dir = tmp_path / "store"
    log_dir.mkdir(parents=True, exist_ok=True)
    remote_dir = tmp_path / "remote"
    remote_dir.mkdir(parents=True, exist_ok=True)
    server = RestartableServer(server_backend, log_dir, remote_dir=remote_dir)
    try:
        yield server
    finally:
        server.stop()


def rust_pending(backend: Backend, phase: str) -> None:
    """Mark the current test xfail when an RPC family isn't yet in Rust.

    Call at the top of a parity test that exercises an RPC the Rust server does
    not implement yet. Delete the call (per phase) once the family lands.
    """
    if backend.name == "rust":
        pytest.xfail(f"RPC not yet implemented in Rust server ({phase})")


@dataclass(frozen=True)
class CutoverHarness:
    """Two-phase cross-backend harness over one shared ``log_dir`` (Phase 6f).

    The cutover gate is intrinsically NOT a single-backend, parametrized test:
    it writes with the PYTHON server, stops it, then boots the RUST server on
    the SAME ``log_dir`` (no Rust sidecar, no adoption sentinel) and asserts the
    Rust server rebuilt the catalog from the on-disk parquet layout + footers.
    This helper bundles a :class:`RestartableServer` per backend over the shared
    dir; the writer is always Python, the reader always Rust.
    """

    writer: RestartableServer  # python
    reader: RestartableServer  # rust
    log_dir: Path

    def rust_sidecar(self) -> Path:
        return self.log_dir / "_finelog_catalog.sqlite"

    def rust_sentinel(self) -> Path:
        return self.log_dir / ".finelog-rust-catalog"


@pytest.fixture
def cutover_harness(tmp_path: Path) -> Iterator[CutoverHarness]:
    """A Python-writes -> Rust-reads harness over one shared ``log_dir``.

    Skips when the Rust binary is absent (the reader cannot run). Both servers
    auto-stop on teardown. Unlike :func:`restartable_server`, this fixture is NOT
    parametrized by ``server_backend`` — the cutover identity is specifically
    Python-writer to Rust-reader.
    """
    if _rust_binary() is None:
        pytest.skip("finelog-server Rust binary not built (run `cargo build -p finelog`)")
    log_dir = tmp_path / "store"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = RestartableServer(Backend("python"), log_dir)
    reader = RestartableServer(Backend("rust"), log_dir)
    try:
        yield CutoverHarness(writer=writer, reader=reader, log_dir=log_dir)
    finally:
        writer.stop()
        reader.stop()


@dataclass(frozen=True)
class DebugSegment:
    """One row from ``GET /debug/segments`` (the non-proto admin surface)."""

    path: str
    level: int
    min_seq: int
    max_seq: int
    row_count: int
    byte_size: int
    location: str
    created_at_ms: int


def maintain(base_url: str, namespace: str, *, force_compact_l0: bool = False) -> None:
    """Force one synchronous flush -> compact -> sync -> evict cycle via the
    ``--debug-admin`` ``POST /debug/maintain`` route. Raises on non-200."""
    resp = httpx.post(
        f"{base_url}/debug/maintain",
        json={"namespace": namespace, "force_compact_l0": force_compact_l0},
        timeout=30.0,
    )
    resp.raise_for_status()


def segments(base_url: str, namespace: str) -> list[DebugSegment]:
    """Read per-segment level/location/seq-bounds via ``GET /debug/segments``,
    ordered by ``min_seq``."""
    resp = httpx.get(
        f"{base_url}/debug/segments",
        params={"namespace": namespace},
        timeout=30.0,
    )
    resp.raise_for_status()
    return [DebugSegment(**row) for row in resp.json()]


def backdate(base_url: str, namespace: str, path: str, created_at_ms: int) -> None:
    """Set a segment's ``created_at_ms`` via ``POST /debug/backdate`` so
    age-eviction tests run without a wall-clock sleep. ``path`` is the segment
    filename (basename)."""
    resp = httpx.post(
        f"{base_url}/debug/backdate",
        json={"namespace": namespace, "path": path, "created_at_ms": created_at_ms},
        timeout=30.0,
    )
    resp.raise_for_status()


# --- Shared RPC helpers for the Phase-4 parity families -------------------
#
# The compaction / eviction / offload / policy tests all register the same
# "worker" table, write single-row batches (each WriteRows acks after sealing one
# L0 segment, so N writes => N L0 segments), and read back via Query. Defining
# these once here keeps the test bodies focused on the maintenance assertions.


def stats_client(url: str) -> StatsServiceClientSync:
    return StatsServiceClientSync(address=url)


def worker_schema(key_column: str = "") -> stats_pb2.Schema:
    return stats_pb2.Schema(
        columns=[
            stats_pb2.Column(name="worker_id", type=stats_pb2.COLUMN_TYPE_STRING, nullable=False),
            stats_pb2.Column(name="mem_bytes", type=stats_pb2.COLUMN_TYPE_INT64, nullable=False),
            stats_pb2.Column(name="timestamp_ms", type=stats_pb2.COLUMN_TYPE_INT64, nullable=False),
        ],
        key_column=key_column,
    )


def _worker_arrow_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("worker_id", pa.string(), nullable=False),
            pa.field("mem_bytes", pa.int64(), nullable=False),
            pa.field("timestamp_ms", pa.int64(), nullable=False),
        ]
    )


def write_worker_row(client: StatsServiceClientSync, namespace: str, worker_id: str, mem: int, ts: int) -> None:
    """Write one worker row; WriteRows acks only after the row is on a sealed L0
    segment, so each call produces exactly one L0 segment (no manual flush)."""
    batch = pa.RecordBatch.from_pydict(
        {"worker_id": [worker_id], "mem_bytes": [mem], "timestamp_ms": [ts]},
        schema=_worker_arrow_schema(),
    )
    sink = io.BytesIO()
    with paipc.new_stream(sink, batch.schema) as writer:
        writer.write_batch(batch)
    client.write_rows(stats_pb2.WriteRowsRequest(namespace=namespace, arrow_ipc=sink.getvalue()))


def query_table(client: StatsServiceClientSync, sql: str) -> pa.Table:
    resp = client.query(stats_pb2.QueryRequest(sql=sql))
    return paipc.open_stream(io.BytesIO(resp.arrow_ipc)).read_all()


def namespace_info(client: StatsServiceClientSync, namespace: str):
    """The ListNamespaces NamespaceInfo for ``namespace`` (or None)."""
    listed = client.list_namespaces(stats_pb2.ListNamespacesRequest())
    return next((n for n in listed.namespaces if n.namespace == namespace), None)
