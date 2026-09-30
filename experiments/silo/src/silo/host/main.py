# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Entrypoint for a silo host: a privileged Iris job running nested sandboxes.

    python -m silo.host.main --cpu-budget 96 --memory-gb 384 --disk-gb 600

Required environment:
    SILO_HOST_SECRET   shared with the broker (see silo.auth)
    SILO_BROKER_NAME   Iris endpoint name of the broker, e.g. /muchanem/silo-broker
Optional:
    SILO_BROKER_URL    skip resolution and use this address
    SILO_REGISTRY_AUTH docker config.json content for private registry pulls
    SILO_HOME          working root (default /tmp/silo)
    SILO_HEARTBEAT_SECONDS          heartbeat period (default 10)
    SILO_HEARTBEAT_TIMEOUT_SECONDS  heartbeat HTTP timeout (default 15)
        The broker reads the same two variables; its suspect window
        (SILO_HOST_SUSPECT_AFTER_SECONDS) must exceed 2x their sum.

The job must be submitted with --container-profile CONTAINER_PROFILE_PRIVILEGED:
nested containers need to create namespaces and write child cgroups.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import anyio
import anyio.to_thread
import uvicorn
from iris.client.client import iris_ctx

from silo.auth import HEADER_API_TOKEN, bearer
from silo.host.agent import HostAgent, HostConfig
from silo.host.runtime import RUNTIMES, NerdctlRuntime
from silo.host.server import build_app
from silo.http import HttpClient

logger = logging.getLogger("silo.host")

HEARTBEAT_SECONDS = 10.0
HEARTBEAT_TIMEOUT_SECONDS = 15.0
REAP_SECONDS = 30.0
_HERE = Path(__file__).resolve().parent


def _bootstrap(home: Path) -> None:
    subprocess.run(["bash", str(_HERE / "bootstrap.sh")], env={**os.environ, "SILO_HOME": str(home)}, check=True)


def _start_daemons(home: Path) -> tuple[str, dict[str, subprocess.Popen]]:
    """Start a private containerd and buildkitd, each owned by this process.

    Everything lives under ``home`` (an emptyDir on the node's NVMe), so nothing
    touches the node's own container runtime.
    """
    bin_dir = home / "bin"
    state = home / "containerd"
    for sub in ("root", "state"):
        (state / sub).mkdir(parents=True, exist_ok=True)
    sock = state / "containerd.sock"
    config = state / "config.toml"
    config.write_text(
        "version = 2\n"
        f'root = "{state / "root"}"\n'
        f'state = "{state / "state"}"\n'
        "[grpc]\n"
        f'  address = "{sock}"\n'
    )
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ.get('PATH', '')}"}
    daemons = {
        "containerd": _spawn_logged([str(bin_dir / "containerd"), "--config", str(config)], home / "containerd.log", env)
    }
    _wait_for_socket(sock, "containerd")

    buildkit = home / "buildkit"
    buildkit.mkdir(parents=True, exist_ok=True)
    buildkit_sock = buildkit / "buildkitd.sock"
    daemons["buildkitd"] = _spawn_logged(
        [
            str(bin_dir / "buildkitd"),
            "--addr",
            f"unix://{buildkit_sock}",
            "--root",
            str(buildkit / "root"),
            "--oci-worker=false",
            "--containerd-worker=true",
            "--containerd-worker-addr",
            str(sock),
            # Built snapshots land in the same namespace sandboxes run from.
            "--containerd-worker-namespace",
            "silo",
            # RUN steps need the network (pip, apt) and there is no CNI config on
            # a host, so say "host" rather than let buildkit guess. BUILD time
            # only: sandboxes still start with --network none.
            "--containerd-worker-net",
            "host",
        ],
        home / "buildkitd.log",
        env,
    )
    _wait_for_socket(buildkit_sock, "buildkitd")
    os.environ["BUILDKIT_HOST"] = f"unix://{buildkit_sock}"
    os.environ["PATH"] = env["PATH"]
    os.environ["CNI_PATH"] = str(home / "cni")
    return str(sock), daemons


def _spawn_logged(argv: list[str], log_path: Path, env: dict[str, str]) -> subprocess.Popen:
    # The child keeps its own copy of the descriptor; ours can close at once.
    with open(log_path, "ab") as log:
        return subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env)


def _wait_for_socket(path: Path, name: str, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() > deadline:
            raise RuntimeError(f"{name} did not create {path} within {timeout}s")
        time.sleep(0.2)


def _docker_config(home: Path) -> Path | None:
    raw = os.environ.pop("SILO_REGISTRY_AUTH", "")
    if not raw:
        return None
    json.loads(raw)  # fail fast on a malformed credential, never at first pull
    directory = home / "docker"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = directory / "config.json"
    path.write_text(raw)
    path.chmod(0o600)
    return directory


def _cpu_ids() -> tuple[int, ...]:
    """The cores cpusets are carved from.

    Iris sets CPU as a request with no limit, so the pod can see every core on
    the node. Spreading sandbox cpusets over all of them keeps load even; the
    cpu.max quota on each sandbox is what bounds its share.
    """
    available = sorted(os.sched_getaffinity(0))
    if len(available) < 1:
        raise RuntimeError("no cpus visible")
    return tuple(available)


def _broker_url() -> str:
    explicit = os.environ.get("SILO_BROKER_URL")
    if explicit:
        return explicit
    return iris_ctx().client.resolve_endpoint(os.environ["SILO_BROKER_NAME"])


def _env_seconds(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    value = float(raw) if raw else default
    if not value > 0:
        raise ValueError(f"{name}={raw!r} must be positive")
    return value


def heartbeat_settings() -> tuple[float, float]:
    """(period, timeout) in seconds, from SILO_HEARTBEAT_SECONDS / SILO_HEARTBEAT_TIMEOUT_SECONDS."""
    return (
        _env_seconds("SILO_HEARTBEAT_SECONDS", HEARTBEAT_SECONDS),
        _env_seconds("SILO_HEARTBEAT_TIMEOUT_SECONDS", HEARTBEAT_TIMEOUT_SECONDS),
    )


def heartbeat_body(agent: HostAgent, self_url: str, period: float, timeout: float) -> dict:
    return {
        "host_id": agent.config.host_id,
        "url": self_url,
        "capacity": agent.capacity(),
        "sandbox_ids": [r.id for r in agent.list_sandboxes()],
        # After a broker restart this is how it relearns where images are.
        "images": agent.images(),
        # So the broker can check its liveness windows against the real cycle.
        "heartbeat_seconds": period,
        "heartbeat_timeout_seconds": timeout,
    }


def _heartbeat_loop(
    agent: HostAgent, host_secret: str, self_url: str, stop: threading.Event, period: float, timeout: float
) -> None:
    broker: HttpClient | None = None
    failures = 0
    while not stop.is_set():
        started = time.monotonic()
        try:
            if broker is None:
                broker = HttpClient(_broker_url(), headers={HEADER_API_TOKEN: bearer(host_secret)}, timeout=timeout)
                logger.info("heartbeating to broker at %s", broker.base_url)
            broker.post("/hosts/heartbeat", heartbeat_body(agent, self_url, period, timeout))
            if failures:
                logger.warning("heartbeat ok again after %d failure(s)", failures)
            failures = 0
        except Exception as error:
            # Any error, not just transport ones: an exception escaping here would
            # end the thread and the host would go silent for good.
            failures += 1
            # Re-resolve next time: the broker may have restarted elsewhere.
            logger.warning("heartbeat failed (%s: %s); will re-resolve the broker", type(error).__name__, error)
            broker = None
        stop.wait(max(1.0, period - (time.monotonic() - started)))


def _reap_loop(agent: HostAgent, daemons: dict[str, subprocess.Popen], stop: threading.Event) -> None:
    while not stop.is_set():
        for name, daemon in daemons.items():
            if daemon.poll() is not None:
                # A host without its runtime cannot serve; exit so Iris retries
                # the job rather than advertising capacity that does not exist.
                logger.error("daemon %s exited with %s; host is going down", name, daemon.returncode)
                os._exit(3)
        try:
            agent.reap()
        except Exception:
            logger.exception("reap failed")
        stop.wait(REAP_SECONDS)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cpu-budget", type=int, required=True, help="cores this host may hand out")
    parser.add_argument("--memory-gb", type=int, required=True)
    parser.add_argument("--disk-gb", type=int, required=True)
    parser.add_argument("--oversubscribe", type=float, default=1.0)
    parser.add_argument("--default-runtime", choices=RUNTIMES, default="runsc")
    parser.add_argument("--host-id", default=None)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s", stream=sys.stdout)
    host_secret = os.environ["SILO_HOST_SECRET"]
    # Read before any thread starts, so a malformed value fails the host loudly
    # instead of silently killing its heartbeat thread.
    heartbeat_period, heartbeat_timeout = heartbeat_settings()
    logger.info("heartbeat period=%gs timeout=%gs", heartbeat_period, heartbeat_timeout)
    home = Path(os.environ.get("SILO_HOME", "/tmp/silo"))
    home.mkdir(parents=True, exist_ok=True)

    _bootstrap(home)
    containerd_sock, daemons = _start_daemons(home)
    runtime = NerdctlRuntime(
        address=containerd_sock,
        tools_dir=home / "tools",
        runsc_binary=home / "bin" / "runsc",
        docker_config_dir=_docker_config(home),
        nerdctl=str(home / "bin" / "nerdctl"),
        ctr=str(home / "bin" / "ctr"),
    )
    host_id = args.host_id or f"{socket.gethostname()}-{uuid.uuid4().hex[:6]}"
    config = HostConfig(
        host_id=host_id,
        cpu_budget=args.cpu_budget,
        memory_budget_bytes=args.memory_gb * 1024**3,
        disk_budget_bytes=args.disk_gb * 1024**3,
        cpu_ids=_cpu_ids(),
        work_dir=home / "work",
        cpu_oversubscribe=args.oversubscribe,
        default_runtime=args.default_runtime,
    )
    agent = HostAgent(config, runtime)

    # Hosts share the node's network namespace (hostNetwork), so a fixed port
    # could collide with another host on the same node. Take a kernel-assigned
    # one and advertise it.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", 0))
    advertise = os.environ.get("IRIS_ADVERTISE_HOST") or socket.gethostbyname(socket.gethostname())
    self_url = f"http://{advertise}:{sock.getsockname()[1]}"
    logger.info("host %s serving at %s capacity=%s", host_id, self_url, json.dumps(agent.capacity()))

    stop = threading.Event()
    threading.Thread(
        target=_heartbeat_loop,
        args=(agent, host_secret, self_url, stop, heartbeat_period, heartbeat_timeout),
        daemon=True,
    ).start()
    threading.Thread(target=_reap_loop, args=(agent, daemons, stop), daemon=True).start()

    server = uvicorn.Server(uvicorn.Config(build_app(agent, host_secret), log_level="warning", ws="none"))

    async def serve() -> None:
        # Each sandbox can hold a long exec plus a poll loop; the default
        # 40-thread pool would queue them. The limiter exists once the loop does.
        anyio.to_thread.current_default_thread_limiter().total_tokens = 512
        await server.serve(sockets=[sock])

    try:
        anyio.run(serve)
    finally:
        stop.set()
        for daemon in daemons.values():
            daemon.terminate()


if __name__ == "__main__":
    main()
