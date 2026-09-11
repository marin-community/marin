#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stand up an in-process finelog server, seed it with a few namespaces + log
streams, and exercise the same JSON RPC paths the dashboard uses.

Usage:
    uv run python lib/finelog/dashboard/scripts/demo.py [--keep]

Boots the native ``finelog_server.EmbeddedServer`` (the same axum app the
``finelog-server`` binary serves) on port 10001. Without ``--keep`` the script
tears the server down on exit. With ``--keep`` the server stays up so the
dashboard at http://localhost:10001/ can be inspected in a browser. The server
serves the built dashboard from ``lib/finelog/dashboard/dist`` itself, so
``npm run build`` must have run at least once.

Requires the native extension (the marin-finelog-server wheel, or a
rust_mode.py dev build).
"""

import argparse
import base64
import dataclasses
import json
import logging
import signal
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as paipc
from finelog.client.log_client import LogClient
from finelog.embedded import require_embedded_server
from finelog.rpc import logging_pb2

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
log = logging.getLogger("finelog.demo")

PORT = 10001
HOST = f"http://localhost:{PORT}"


@dataclasses.dataclass
class WorkerCpu:
    worker_id: str
    cpu_pct: float
    timestamp_ms: int


@dataclasses.dataclass
class WorkerMem:
    worker_id: str
    used_mb: int
    timestamp_ms: int


def seed(client: LogClient) -> None:
    now = int(time.time() * 1000)

    cpu_table = client.get_table("metrics.cpu", WorkerCpu)
    mem_table = client.get_table("metrics.mem", WorkerMem)
    cpu_table.write(WorkerCpu(worker_id=f"w-{i}", cpu_pct=10.0 + i, timestamp_ms=now - 1000 * i) for i in range(8))
    mem_table.write(WorkerMem(worker_id=f"w-{i}", used_mb=100 + 17 * i, timestamp_ms=now - 1000 * i) for i in range(5))
    cpu_table.flush()
    mem_table.flush()
    log.info("wrote rows to metrics.cpu (8) and metrics.mem (5)")

    for key, lines in [
        ("/user/job-a/task-1", ["[INFO] starting task-1", "[WARNING] retry 1", "[ERROR] boom"]),
        ("/user/job-a/task-2", ["[INFO] starting task-2", "[INFO] done"]),
        ("/system/worker/w-3", ["[INFO] worker w-3 online"]),
    ]:
        entries = [
            logging_pb2.LogEntry(
                timestamp=logging_pb2.Timestamp(epoch_ms=now + idx),
                source="stdout",
                data=line,
            )
            for idx, line in enumerate(lines)
        ]
        client.write_batch(key, entries)
    client.flush(timeout=5.0)
    log.info("pushed log entries to 3 keys")


def post_json(path: str, body: dict) -> dict:
    req = urllib.request.Request(
        f"{HOST}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        raise RuntimeError(f"{e.code} {e.reason}: {body}") from None


def decode_arrow(b64_payload: str | None) -> pa.Table:
    if not b64_payload:
        return pa.table({})
    raw = base64.b64decode(b64_payload)
    return paipc.open_stream(pa.BufferReader(raw)).read_all()


def exercise() -> None:
    log.info("--- StatsService.ListNamespaces ---")
    resp = post_json("/finelog.stats.StatsService/ListNamespaces", {})
    namespaces = resp.get("namespaces", [])
    for ns in namespaces:
        cols = ns.get("schema", {}).get("columns", [])
        print(f"  {ns['namespace']}: {len(cols)} cols ({', '.join(c['name'] for c in cols)})")

    log.info("--- StatsService.GetTableSchema: metrics.cpu ---")
    resp = post_json(
        "/finelog.stats.StatsService/GetTableSchema",
        {"namespace": "metrics.cpu"},
    )
    print(json.dumps(resp.get("schema", {}), indent=2))

    log.info("--- StatsService.Query: row counts per namespace ---")
    for ns in namespaces:
        r = post_json(
            "/finelog.stats.StatsService/Query",
            {"sql": f'SELECT count(*) AS n FROM "{ns["namespace"]}"'},
        )
        rt = decode_arrow(r.get("arrowIpc"))
        print(f"  {ns['namespace']}: {rt.to_pylist()[0]}")

    log.info("--- StatsService.Query: SELECT * FROM metrics.cpu ORDER BY timestamp_ms DESC LIMIT 3 ---")
    r = post_json(
        "/finelog.stats.StatsService/Query",
        {"sql": 'SELECT * FROM "metrics.cpu" ORDER BY timestamp_ms DESC LIMIT 3'},
    )
    print(decode_arrow(r.get("arrowIpc")).to_pandas())

    log.info("--- LogService.FetchLogs: tail of /user/job-a/.* ---")
    r = post_json(
        "/finelog.logging.LogService/FetchLogs",
        {"source": "/user/job-a/.*", "tail": True, "maxLines": 10},
    )
    for e in r.get("entries", []):
        print(f"  {e.get('key')} [{e.get('level', '?')}] {e.get('data')}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="leave server running after seeding")
    args = ap.parse_args()

    if args.keep:
        dist = Path(__file__).resolve().parent.parent / "dist"
        if not (dist / "index.html").is_file():
            log.warning(
                "dashboard not built (%s missing); the server will serve a placeholder. "
                "Run `npm run build` in lib/finelog/dashboard to build the SPA.",
                dist / "index.html",
            )

    embedded_server = require_embedded_server()
    with tempfile.TemporaryDirectory(prefix="finelog-demo-") as tmpdir:
        log.info("starting in-process finelog server on :%d (log_dir=%s)", PORT, tmpdir)
        server = embedded_server(log_dir=tmpdir, host="127.0.0.1", port=PORT, debug_admin=True)
        try:
            client = LogClient.connect(("localhost", PORT))
            try:
                seed(client)
            finally:
                client.close()
            exercise()
            if args.keep:
                log.info("dashboard ready at http://localhost:%d/ (Ctrl-C to stop)", PORT)
                signal.pause()
        finally:
            log.info("stopping server")
            server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
