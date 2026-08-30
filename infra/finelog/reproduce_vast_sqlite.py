# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run an isolated, opt-in SQLite commit workload across CoreWeave VAST PVCs."""

import argparse
import json
import os
import sqlite3
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_NAMESPACE = "finelog-vast-sqlite-repro"
DEFAULT_REPLICAS = 100
DEFAULT_STORAGE_CLASS = "shared-vast"
DEFAULT_VOLUME_SIZE = "1Gi"
LABELS = {"app.kubernetes.io/name": "finelog-vast-sqlite-repro"}
SQLITE_FILENAME = "catalog.sqlite"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


@dataclass
class WorkloadState:
    """Thread-safe snapshot of SQLite commit progress."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    thread_id: int | None = None
    phase: str = "starting"
    commits: int = 0
    last_commit_at: float | None = None
    last_latency_ms: float | None = None
    max_latency_ms: float = 0.0
    error: str | None = None


def _set_phase(state: WorkloadState, phase: str) -> None:
    with state.lock:
        state.phase = phase


def _emit(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, "time_ns": time.time_ns(), **fields}, sort_keys=True), flush=True)


def _thread_wait_channel(thread_id: int | None) -> str | None:
    if thread_id is None:
        return None
    try:
        return Path(f"/proc/self/task/{thread_id}/wchan").read_text().strip()
    except OSError:
        return None


def _sqlite_commits(
    state: WorkloadState,
    *,
    data_dir: Path,
    commit_interval: float,
    retained_rows: int,
    payload_bytes: int,
    slow_commit_seconds: float,
    max_commits: int,
    ready_file: Path | None,
) -> None:
    with state.lock:
        state.thread_id = threading.get_native_id()
    try:
        if ready_file is not None:
            ready_file.unlink(missing_ok=True)
        _set_phase(state, "data_dir")
        data_dir.mkdir(parents=True, exist_ok=True)
        _set_phase(state, "open")
        connection = sqlite3.connect(data_dir / SQLITE_FILENAME, timeout=30, isolation_level=None)
        _set_phase(state, "journal_mode")
        journal_mode = connection.execute("PRAGMA journal_mode = PERSIST").fetchone()[0]
        _set_phase(state, "synchronous")
        connection.execute("PRAGMA synchronous = FULL")
        synchronous = connection.execute("PRAGMA synchronous").fetchone()[0]
        _set_phase(state, "create_table")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS commits ("
            "id INTEGER PRIMARY KEY, committed_at_ns INTEGER NOT NULL, payload BLOB NOT NULL)"
        )
        _emit("sqlite_ready", journal_mode=journal_mode, synchronous=synchronous)
        if ready_file is not None:
            ready_file.touch()
        payload = os.urandom(payload_bytes)

        while max_commits <= 0 or state.commits < max_commits:
            _set_phase(state, "commit")
            started = time.monotonic()
            connection.execute("BEGIN IMMEDIATE")
            try:
                cursor = connection.execute(
                    "INSERT INTO commits (committed_at_ns, payload) VALUES (?, ?)",
                    (time.time_ns(), payload),
                )
                newest_id = cursor.lastrowid
                assert newest_id is not None
                connection.execute("DELETE FROM commits WHERE id <= ?", (newest_id - retained_rows,))
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise

            latency_ms = (time.monotonic() - started) * 1000
            with state.lock:
                state.commits += 1
                state.last_commit_at = time.monotonic()
                state.last_latency_ms = latency_ms
                state.max_latency_ms = max(state.max_latency_ms, latency_ms)
            if latency_ms >= slow_commit_seconds * 1000:
                _emit("slow_commit", commit=state.commits, latency_ms=round(latency_ms, 3))
            if commit_interval > 0:
                _set_phase(state, "sleep")
                time.sleep(commit_interval)
    except Exception as error:
        with state.lock:
            state.error = repr(error)


def run_workload(args: argparse.Namespace) -> None:
    """Run the SQLite writer and emit heartbeats even when its NFS thread blocks."""
    state = WorkloadState()
    worker = threading.Thread(
        target=_sqlite_commits,
        kwargs={
            "state": state,
            "data_dir": args.data_dir,
            "commit_interval": args.commit_interval,
            "retained_rows": args.retained_rows,
            "payload_bytes": args.payload_bytes,
            "slow_commit_seconds": args.slow_commit_seconds,
            "max_commits": args.max_commits,
            "ready_file": args.ready_file,
        },
        daemon=True,
        name="sqlite-commits",
    )
    worker.start()

    while worker.is_alive():
        with state.lock:
            now = time.monotonic()
            last_commit_age = None if state.last_commit_at is None else now - state.last_commit_at
            snapshot = {
                "commits": state.commits,
                "phase": state.phase,
                "last_commit_age_seconds": None if last_commit_age is None else round(last_commit_age, 3),
                "last_latency_ms": None if state.last_latency_ms is None else round(state.last_latency_ms, 3),
                "max_latency_ms": round(state.max_latency_ms, 3),
                "sqlite_thread_wchan": _thread_wait_channel(state.thread_id),
            }
        _emit("heartbeat", **snapshot)
        worker.join(args.heartbeat_interval)

    with state.lock:
        if state.error is not None:
            raise RuntimeError(state.error)
        _emit("complete", commits=state.commits, max_latency_ms=round(state.max_latency_ms, 3))


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    """Build the isolated Namespace, ConfigMap, Service, and StatefulSet list."""
    script = Path(__file__).read_text(encoding="utf-8")
    workload_args = [
        "python",
        "/opt/repro/reproduce_vast_sqlite.py",
        "workload",
        "--data-dir",
        "/data",
        "--commit-interval",
        str(args.commit_interval),
        "--heartbeat-interval",
        str(args.heartbeat_interval),
        "--slow-commit-seconds",
        str(args.slow_commit_seconds),
        "--ready-file",
        "/tmp/sqlite-ready",
    ]
    return {
        "apiVersion": "v1",
        "kind": "List",
        "items": [
            {
                "apiVersion": "v1",
                "kind": "Namespace",
                "metadata": {"name": args.namespace, "labels": LABELS},
            },
            {
                "apiVersion": "v1",
                "kind": "ConfigMap",
                "metadata": {"name": "workload", "namespace": args.namespace, "labels": LABELS},
                "data": {"reproduce_vast_sqlite.py": script},
            },
            {
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {"name": "sqlite", "namespace": args.namespace, "labels": LABELS},
                "spec": {"clusterIP": "None", "selector": LABELS, "ports": [{"name": "unused", "port": 1}]},
            },
            {
                "apiVersion": "apps/v1",
                "kind": "StatefulSet",
                "metadata": {"name": "sqlite", "namespace": args.namespace, "labels": LABELS},
                "spec": {
                    "serviceName": "sqlite",
                    "replicas": args.replicas,
                    "podManagementPolicy": "Parallel",
                    "selector": {"matchLabels": LABELS},
                    "template": {
                        "metadata": {"labels": LABELS},
                        "spec": {
                            "terminationGracePeriodSeconds": 5,
                            "securityContext": {"fsGroup": 1000, "fsGroupChangePolicy": "OnRootMismatch"},
                            "topologySpreadConstraints": [
                                {
                                    "maxSkew": 1,
                                    "topologyKey": "kubernetes.io/hostname",
                                    "whenUnsatisfiable": "ScheduleAnyway",
                                    "labelSelector": {"matchLabels": LABELS},
                                }
                            ],
                            "containers": [
                                {
                                    "name": "sqlite",
                                    "image": args.image,
                                    "imagePullPolicy": "IfNotPresent",
                                    "args": workload_args,
                                    "securityContext": {
                                        "runAsNonRoot": True,
                                        "runAsUser": 1000,
                                        "runAsGroup": 1000,
                                        "allowPrivilegeEscalation": False,
                                        "capabilities": {"drop": ["ALL"]},
                                    },
                                    "resources": {
                                        "requests": {"cpu": "10m", "memory": "32Mi"},
                                        "limits": {"cpu": "100m", "memory": "128Mi"},
                                    },
                                    "readinessProbe": {
                                        "exec": {"command": ["test", "-f", "/tmp/sqlite-ready"]},
                                        "periodSeconds": 2,
                                    },
                                    "volumeMounts": [
                                        {"name": "data", "mountPath": "/data"},
                                        {"name": "workload", "mountPath": "/opt/repro", "readOnly": True},
                                    ],
                                }
                            ],
                            "volumes": [{"name": "workload", "configMap": {"name": "workload"}}],
                        },
                    },
                    "volumeClaimTemplates": [
                        {
                            "metadata": {"name": "data", "labels": LABELS},
                            "spec": {
                                "accessModes": ["ReadWriteOnce"],
                                "storageClassName": args.storage_class,
                                "resources": {"requests": {"storage": args.volume_size}},
                            },
                        }
                    ],
                },
            },
        ],
    }


def _kubectl(args: argparse.Namespace, *command: str, stdin: str | None = None) -> None:
    subprocess.run(
        ["kubectl", "--kubeconfig", str(args.kubeconfig), "--context", args.context, *command],
        input=stdin,
        text=True,
        check=True,
    )


def apply_manifest(args: argparse.Namespace) -> None:
    """Apply the isolated stress workload without waiting for all claims to bind."""
    manifest = json.dumps(build_manifest(args))
    _kubectl(args, "apply", "-f", "-", stdin=manifest)
    print(f"Applied {args.replicas} SQLite writers in namespace {args.namespace}.")
    print(f"Run `{Path(__file__)} status --kubeconfig {args.kubeconfig} --context {args.context}` to inspect them.")


def show_status(args: argparse.Namespace) -> None:
    """Show resource state and the latest heartbeat from every running writer."""
    _kubectl(args, "get", "statefulset,pods,pvc", "-n", args.namespace, "-o", "wide")
    _kubectl(
        args,
        "logs",
        "-n",
        args.namespace,
        "-l",
        ",".join(f"{key}={value}" for key, value in LABELS.items()),
        "--tail=1",
        "--prefix",
        "--ignore-errors=true",
        f"--max-log-requests={args.max_log_requests}",
    )


def delete_workload(args: argparse.Namespace) -> None:
    """Delete the isolated namespace only after exact-name confirmation."""
    if args.confirm != args.namespace:
        raise ValueError(f"pass --confirm {args.namespace} to delete the namespace and its PVCs")
    _kubectl(args, "delete", "namespace", args.namespace)


def _manifest_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument("--replicas", type=_positive_int, default=DEFAULT_REPLICAS)
    parser.add_argument("--storage-class", default=DEFAULT_STORAGE_CLASS)
    parser.add_argument("--volume-size", default=DEFAULT_VOLUME_SIZE)
    parser.add_argument("--image", default="python:3.12-slim")
    parser.add_argument("--commit-interval", type=_nonnegative_float, default=1.0)
    parser.add_argument("--heartbeat-interval", type=_nonnegative_float, default=10.0)
    parser.add_argument("--slow-commit-seconds", type=_nonnegative_float, default=5.0)


def _cluster_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--kubeconfig", type=Path, required=True)
    parser.add_argument("--context", required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    workload = subparsers.add_parser("workload", help="run one writer (used by the generated pods)")
    workload.add_argument("--data-dir", type=Path, required=True)
    workload.add_argument("--commit-interval", type=_nonnegative_float, default=1.0)
    workload.add_argument("--heartbeat-interval", type=_nonnegative_float, default=10.0)
    workload.add_argument("--retained-rows", type=_positive_int, default=1024)
    workload.add_argument("--payload-bytes", type=_positive_int, default=4096)
    workload.add_argument("--slow-commit-seconds", type=_nonnegative_float, default=5.0)
    workload.add_argument("--max-commits", type=int, default=0)
    workload.add_argument("--ready-file", type=Path)
    workload.set_defaults(handler=run_workload)

    manifest = subparsers.add_parser("manifest", help="print the Kubernetes resource list as JSON")
    _manifest_arguments(manifest)
    manifest.set_defaults(handler=lambda args: print(json.dumps(build_manifest(args), indent=2)))

    apply = subparsers.add_parser("apply", help="create the isolated namespace and stress workload")
    _cluster_arguments(apply)
    _manifest_arguments(apply)
    apply.set_defaults(handler=apply_manifest)

    status = subparsers.add_parser("status", help="show pods, claims, and latest writer heartbeats")
    _cluster_arguments(status)
    status.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    status.add_argument("--max-log-requests", type=int, default=DEFAULT_REPLICAS)
    status.set_defaults(handler=show_status)

    delete = subparsers.add_parser("delete", help="delete the namespace and all reproduction PVCs")
    _cluster_arguments(delete)
    delete.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    delete.add_argument("--confirm", required=True)
    delete.set_defaults(handler=delete_workload)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
