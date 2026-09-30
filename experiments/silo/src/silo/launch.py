# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bring a silo deployment up or down on Iris.

    uv run python -m silo.launch up --hosts 4 --host-cpu 96 --host-memory-gb 384
    uv run python -m silo.launch status
    uv run python -m silo.launch worker-env        # what a pipeline worker needs
    uv run python -m silo.launch down

Replacing just the broker (hosts and their sandboxes keep running; they and the
workers re-resolve the broker by its endpoint name):

    uv run python -m silo.launch --name NAME export-state --out URL   # only if the old broker predates --state-url
    iris job cancel <old broker job>
    uv run python -m silo.launch --name NAME up --hosts 0 --broker-job-name NAME-broker-v2 --state-url URL

A deployment is one broker job plus N host jobs, all federated to
cw-us-east-02a through the marin hub. Secrets are generated here, kept in a
mode-0600 state file, and handed to jobs through the Iris API -- never on a
process command line, where `ps` would show them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from iris.cli.connect import IRIS_CLUSTER_CONFIG_DIRS, connect_controller
from iris.cli.job import run_iris_job
from iris.client.client import IrisClient
from iris.cluster.client.job_info import resolve_job_user
from iris.cluster.config import load_config
from iris.cluster.types import JobName
from rigging.config_discovery import resolve_cluster_config

from silo.auth import new_secret

STATE_DIR = Path(os.environ.get("SILO_STATE_DIR", Path.home() / ".config" / "silo"))
# A deployment lives on ONE cluster: CoreWeave clusters cannot route to each
# other's node addresses (probe 2026-09-22: 02a<->rno2a and ->08a all blocked,
# same-cluster controls reached), and workers talk to hosts directly. To use
# several clusters, run one deployment per cluster with workers on that cluster.
# Headroom on top of what sandboxes are promised. Memory is a HARD pod limit and
# nested sandboxes are charged to it, so the host must request its sandboxes'
# memory plus its own daemons; disk is the ephemeral-storage limit, which the
# kubelet enforces by evicting the whole pod.
HOST_MEMORY_OVERHEAD_GB = 16
HOST_IMAGE_CACHE_GB = 200
MAX_RETRIES = 10  # Iris defaults to 0: one hiccup would end a long-lived service.
BROKER_MEMORY_GB = 16
# glibc gives each thread its own malloc arena; with hundreds of request threads
# the broker's RSS reached 7.1 GB of its 8 GB limit (2026-09-29).
BROKER_ENV = {"MALLOC_ARENA_MAX": "2"}
# Liveness and pool knobs forwarded from the launching shell to broker and hosts.
TUNABLE_ENV = (
    "SILO_HEARTBEAT_SECONDS",
    "SILO_HEARTBEAT_TIMEOUT_SECONDS",
    "SILO_HOST_SUSPECT_AFTER_SECONDS",
    "SILO_HOST_DEAD_AFTER_SECONDS",
    "SILO_HOST_MAX_INFLIGHT_CREATES",
    "SILO_HOST_QUARANTINE_AFTER_FAILURES",
    "SILO_HOST_QUARANTINE_SECONDS",
    "SILO_HOST_QUARANTINE_MAX_SECONDS",
    "SILO_HOST_QUARANTINE_MAX_FRACTION",
    "SILO_BROKER_CONTROL_THREADS",
    "SILO_BROKER_HOST_IO_THREADS",
    "SILO_BROKER_CREATE_THREADS",
)


def _tunables() -> dict[str, str]:
    return {name: os.environ[name] for name in TUNABLE_ENV if os.environ.get(name)}


def _state_path(name: str) -> Path:
    return STATE_DIR / f"{name}.json"


def _load(name: str) -> dict[str, Any]:
    path = _state_path(name)
    if not path.exists():
        sys.exit(f"no deployment {name!r} (expected {path})")
    return json.loads(path.read_text())


def _save(name: str, state: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = _state_path(name)
    path.write_text(json.dumps(state, indent=2))
    path.chmod(0o600)


def _controller_address(cluster: str) -> str:
    """The cluster controller's in-cluster address, whose /proxy/ route reaches an
    endpoint by NAME -- how a worker re-finds a broker that restarted elsewhere."""
    config = load_config(str(resolve_cluster_config(cluster, dirs=IRIS_CLUSTER_CONFIG_DIRS)))
    if config.kubernetes_provider is None:
        sys.exit(f"cluster {cluster!r} has no kubernetes_provider; silo hosts need a CoreWeave (k8s) cluster")
    return config.kubernetes_provider.controller_address


def _cluster_of(state: dict[str, Any]) -> str:
    if "cluster" not in state:
        sys.exit(
            "deployment has no recorded cluster (it predates --cluster); run `up --hosts-only --hosts 0 "
            "--cluster <its cluster>` once to record it"
        )
    return state["cluster"]


def _proxy_name(endpoint: str) -> str:
    # Iris encodes "/" as "." in proxy paths (endpoint_service.py).
    return endpoint.strip("/").replace("/", ".")


def _connect():

    return connect_controller(cluster_name="marin")


def _job_user() -> str:
    # The namespace Iris files the jobs under (IRIS_USER, else the OS user) --
    # not the authenticated identity, which is an email and names no job path.
    return resolve_job_user()


def _submit(
    endpoint,
    *,
    name: str,
    command: list[str],
    env: dict[str, str],
    cpu: float,
    memory_gb: int,
    disk_gb: int,
    privileged: bool,
    cluster: str,
    max_retries: int = MAX_RETRIES,
    priority: str | None = None,
) -> None:

    code = run_iris_job(
        command=command,
        env_vars=env,
        controller_url=endpoint.url,
        credentials=endpoint.credentials,
        cpu=cpu,
        memory=f"{memory_gb}GB",
        disk=f"{disk_gb}GB",
        wait=False,
        job_name=name,
        max_retries=max_retries,
        # Only what the service imports; not the whole workspace (torch et al).
        sync_packages=["marin-silo", "marin-iris"],
        # Node class only. It does NOT stop Kueue evicting the job for a
        # higher-priority workload: that is the band (`priority`), and at the
        # default (interactive) a CPU job yields to same-band accelerator pods.
        preemptible=False,
        priority=priority,
        container_profile="CONTAINER_PROFILE_PRIVILEGED" if privileged else None,
        target_cluster=cluster,
    )
    if code != 0:
        sys.exit(f"submitting {name} failed with {code}")


def cmd_up(args: argparse.Namespace) -> None:
    path = _state_path(args.name)
    state = json.loads(path.read_text()) if path.exists() else {}
    state.setdefault("api_token", new_secret())
    state.setdefault("host_secret", new_secret())

    cluster = state.get("cluster") or args.cluster
    if cluster is None:
        sys.exit("a new deployment needs --cluster (e.g. cw-us-east-02a)")
    if args.cluster is not None and args.cluster != cluster:
        sys.exit(f"deployment {args.name!r} is on {cluster}; one deployment cannot span clusters")
    state["cluster"] = cluster
    if args.state_url:
        state["state_url"] = args.state_url
    with _connect() as endpoint:
        user = state.get("user") or _job_user()
        # The ENDPOINT name never changes: hosts and workers find the broker by
        # it. The JOB name may (--broker-job-name), so a replaced broker's job
        # keeps its post-mortem instead of being overwritten by a same-name relaunch.
        broker_job = args.broker_job_name or f"{args.name}-broker"
        broker_endpoint = state.get("broker_endpoint") or f"/{user}/{args.name}-broker"
        state.update(
            user=user,
            broker_endpoint=broker_endpoint,
            resolve_url=f"{_controller_address(cluster)}/proxy/{_proxy_name(broker_endpoint)}",
            host_jobs=state.get("host_jobs", []),
        )
        if not args.hosts_only:
            state["broker_job"] = f"/{user}/{broker_job}"
        _save(args.name, state)  # before submitting, so a partial up can be torn down

        secrets = {"SILO_API_TOKEN": state["api_token"], "SILO_HOST_SECRET": state["host_secret"]}
        if not args.hosts_only:
            broker_command = ["python", "-m", "silo.broker.main", "--endpoint-name", broker_endpoint]
            if state.get("state_url"):
                broker_command += ["--state-url", state["state_url"]]
            else:
                print("WARNING: no --state-url; this broker keeps snapshots in memory and a restart loses them")
            _submit(
                endpoint,
                name=broker_job,
                command=broker_command,
                env={**secrets, **BROKER_ENV, **_tunables()},
                cpu=2,
                memory_gb=args.broker_memory_gb,
                disk_gb=10,
                privileged=False,
                cluster=_cluster_of(state),
                priority=args.priority,
            )
        host_env = {"SILO_HOST_SECRET": state["host_secret"], "SILO_BROKER_NAME": broker_endpoint, **_tunables()}
        registry_auth = os.environ.get("SILO_REGISTRY_AUTH")
        if registry_auth:
            host_env["SILO_REGISTRY_AUTH"] = registry_auth
        start = len(state["host_jobs"])
        for index in range(start, start + args.hosts):
            job = f"{args.name}-host-{index}"
            _submit(
                endpoint,
                name=job,
                command=[
                    "python",
                    "-m",
                    "silo.host.main",
                    "--cpu-budget",
                    str(args.host_cpu),
                    "--memory-gb",
                    str(args.host_memory_gb),
                    "--disk-gb",
                    str(args.host_disk_gb),
                    "--oversubscribe",
                    str(args.oversubscribe),
                    "--default-runtime",
                    args.runtime,
                    "--host-id",
                    job,
                ],
                env=host_env,
                cpu=args.host_cpu,
                memory_gb=args.host_memory_gb + HOST_MEMORY_OVERHEAD_GB,
                disk_gb=args.host_disk_gb + HOST_IMAGE_CACHE_GB,
                privileged=True,
                cluster=_cluster_of(state),
                priority=args.priority,
            )
            state["host_jobs"].append(f"/{user}/{job}")
            _save(args.name, state)
    print(
        json.dumps(
            {k: state.get(k) for k in ("broker_job", "broker_endpoint", "resolve_url", "state_url", "host_jobs")},
            indent=2,
        )
    )


def cmd_down(args: argparse.Namespace) -> None:
    state = _load(args.name)
    with _connect() as endpoint:

        with IrisClient.remote(endpoint.url, workspace=None, credentials=endpoint.credentials) as client:
            for job in [*state.get("host_jobs", []), state["broker_job"]]:
                try:
                    client.cancel_job(JobName.from_wire(job))
                    print(f"cancelled {job}")
                except Exception as error:
                    print(f"cancel {job}: {error}")
    state["host_jobs"] = []
    # Every down retires the secrets, so the next up mints fresh ones. Keeping
    # them made down+up after a leak silently reuse the leaked token.
    state.pop("api_token", None)
    state.pop("host_secret", None)
    _save(args.name, state)
    print("secrets retired; the next `up` generates new ones")


def cmd_status(args: argparse.Namespace) -> None:
    state = _load(args.name)
    with _connect() as endpoint:

        with IrisClient.remote(endpoint.url, workspace=None, credentials=endpoint.credentials) as client:
            for job in client.list_jobs(prefix=f"/{state['user']}/{args.name}-"):
                print(f"{job.job_id}  {job.state.value}")


def cmd_acceptance(args: argparse.Namespace) -> None:
    """Submit silo.acceptance as a worker would run: unprivileged, token via env."""
    state = _load(args.name)
    with _connect() as endpoint:
        _submit(
            endpoint,
            name=f"{args.name}-acceptance-{args.suffix}",
            command=["python", "-m", "silo.acceptance", "--image", args.image, "--runtimes", args.runtimes],
            env={"SILO_API_TOKEN": state["api_token"], "SILO_BROKER_RESOLVE_URL": state["resolve_url"]},
            cpu=2,
            memory_gb=4,
            disk_gb=10,
            privileged=False,
            cluster=_cluster_of(state),
            max_retries=0,  # a verdict, not a service: never silently re-run it
        )
    print(f"/{state['user']}/{args.name}-acceptance-{args.suffix}")


def cmd_export_state(args: argparse.Namespace) -> None:
    """Submit silo.broker.export: copy the running broker's snapshots to a state document.

    Needed once, before replacing a broker that predates --state-url. It runs on
    the broker's cluster because the broker has no off-cluster address.
    """
    state = _load(args.name)
    out = args.out or state.get("state_url")
    if not out:
        sys.exit("export-state needs --out (or a state_url recorded by `up --state-url`)")
    job = f"{args.name}-export-state-{args.suffix}"
    with _connect() as endpoint:
        _submit(
            endpoint,
            name=job,
            command=["python", "-m", "silo.broker.export", "--out", out, *(["--force"] if args.force else [])],
            env={"SILO_API_TOKEN": state["api_token"], "SILO_BROKER_RESOLVE_URL": state["resolve_url"]},
            cpu=1,
            memory_gb=2,
            disk_gb=5,
            privileged=False,
            cluster=_cluster_of(state),
            max_retries=0,
        )
    print(f"/{state['user']}/{job} -> {out}")


def cmd_width(args: argparse.Namespace) -> None:
    """Submit silo.width: hundreds of concurrent trial-shaped sandboxes."""
    state = _load(args.name)
    job = f"{args.name}-width-{args.suffix}"
    with _connect() as endpoint:
        _submit(
            endpoint,
            name=job,
            command=[
                "python",
                "-m",
                "silo.width",
                "--image",
                args.image,
                "--concurrency",
                str(args.concurrency),
                "--work-seconds",
                str(args.work_seconds),
                "--min-hosts",
                str(args.min_hosts),
                "--trials",
                str(args.trials or 2 * args.concurrency),
                *(["--print-logs"] if args.print_logs else []),
            ],
            env={"SILO_API_TOKEN": state["api_token"], "SILO_BROKER_RESOLVE_URL": state["resolve_url"]},
            cpu=4,
            memory_gb=8,
            disk_gb=10,
            privileged=False,
            cluster=_cluster_of(state),
            max_retries=0,
        )
    print(f"/{state['user']}/{job}")


def cmd_worker_env(args: argparse.Namespace) -> None:
    """Emit, as JSON, the env a pipeline worker needs. Contains the API token.

    With ``--out`` it goes to a mode-0600 file and only the path is printed, so the
    token never lands in a terminal or an agent transcript.
    """
    state = _load(args.name)
    text = json.dumps({"SILO_API_TOKEN": state["api_token"], "SILO_BROKER_RESOLVE_URL": state["resolve_url"]})
    if args.out is None:
        print(text)
        return
    fd = os.open(args.out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(text + "\n")
    os.chmod(args.out, 0o600)  # O_CREAT's mode does not apply to an existing file
    print(f"wrote worker env to {args.out} (mode 0600)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="silo", help="deployment name; prefixes every job")
    sub = parser.add_subparsers(dest="command", required=True)

    up = sub.add_parser("up")
    up.add_argument("--cluster", default=None, help="required for a new deployment, e.g. cw-us-east-02a or cw-rno2a")
    up.add_argument("--hosts", type=int, default=1)
    up.add_argument("--host-cpu", type=int, default=96)
    up.add_argument("--host-memory-gb", type=int, default=384)
    up.add_argument("--host-disk-gb", type=int, default=600)
    up.add_argument(
        "--oversubscribe",
        type=float,
        default=1.0,
        help="sandbox cores per requested core; each sandbox keeps a hard cpu.max regardless",
    )
    up.add_argument("--runtime", choices=("runsc", "runc"), default="runsc")
    up.add_argument("--hosts-only", action="store_true", help="add hosts to an existing deployment")
    up.add_argument(
        "--broker-job-name",
        default=None,
        help="job name for a replacement broker (default NAME-broker); its endpoint name is unchanged",
    )
    up.add_argument(
        "--state-url",
        default=None,
        help="where the broker persists snapshots + host registry (recorded for later `up`s), "
        "e.g. s3://marin-us-east-02a/users/USER/silo/NAME/broker-state.json",
    )
    up.add_argument("--broker-memory-gb", type=int, default=BROKER_MEMORY_GB)
    up.add_argument(
        "--priority",
        default=None,
        help="Iris priority band for broker and hosts (default: inherit = interactive, which Kueue may "
        "preempt for same-band accelerator pods; `production` is admin-only and is not preempted by them)",
    )
    up.set_defaults(func=cmd_up)
    export = sub.add_parser("export-state")
    export.add_argument("--out", default=None, help="state document URL (default: the recorded state_url)")
    export.add_argument("--suffix", default="01")
    export.add_argument("--force", action="store_true")
    export.set_defaults(func=cmd_export_state)
    sub.add_parser("down").set_defaults(func=cmd_down)
    sub.add_parser("status").set_defaults(func=cmd_status)
    worker_env = sub.add_parser("worker-env")
    worker_env.add_argument("--out", type=Path, default=None, help="write to this file (mode 0600) instead of stdout")
    worker_env.set_defaults(func=cmd_worker_env)
    acc = sub.add_parser("acceptance")
    acc.add_argument("--image", required=True)
    acc.add_argument("--runtimes", default="runsc,runc")
    acc.add_argument("--suffix", default="01")
    acc.set_defaults(func=cmd_acceptance)
    width = sub.add_parser("width")
    width.add_argument("--image", required=True)
    width.add_argument("--concurrency", type=int, default=250)
    width.add_argument("--work-seconds", type=int, default=90)
    width.add_argument("--suffix", default="01")
    width.add_argument("--min-hosts", type=int, default=1)
    width.add_argument("--trials", type=int, default=None)
    width.add_argument("--print-logs", action="store_true")
    width.set_defaults(func=cmd_width)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
