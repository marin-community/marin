# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Local TCP tunnels to remote services.

Provides a single ``open_tunnel`` context manager that dispatches by
target type — ``GcpSshForwardTarget`` (``gcloud compute ssh`` with an
``-L`` port forward) or ``K8sPortForwardTarget`` (``kubectl
port-forward``) — and yields a local ``http://127.0.0.1:<port>`` URL.

The GCP path goes over SSH rather than ``start-iap-tunnel`` so it shares
the same auth path the rest of the deploy tooling already uses (gcloud
SSH + OS Login + optional service-account impersonation). IAP TCP
tunnels require a separate ``iap.tunnelResourceAccessor`` role and a
``Testing if tunnel connection works`` pre-flight; SSH-over-IAP (with
``tunnel_through_iap=True``) reuses SSH credentials end-to-end.

Both underlying transports are subject to idle drops and transient
network blips, so a daemon watchdog respawns the child process for the
lifetime of the context. Callers that genuinely want one-shot semantics
(integration tests where a respawn would mask a problem) can pass
``watchdog=False``.

The module intentionally has no opinion on the service running over the
tunnel — it composes the right ``gcloud``/``kubectl`` argv, waits for the
local listener, and tears the child down on context exit. Callers like
finelog's CLI translate their own config schema into a ``TunnelTarget``
and use this directly; iris's k8s port-forward should migrate here too.
"""

from __future__ import annotations

import contextlib
import logging
import socket
import subprocess
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from rigging.timing import Deadline, Duration, ExponentialBackoff

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GcpSshForwardTarget:
    """SSH ``-L`` port forward to a GCE instance via ``gcloud compute ssh``.

    Reuses the same auth path the rest of the gcloud SSH tooling uses
    (OS Login + optional service-account impersonation), so any VM that
    already accepts ``gcloud compute ssh`` accepts this tunnel — no
    extra IAP TCP role needed.

    Set ``tunnel_through_iap=True`` for instances without a public IP;
    SSH itself rides over IAP and the port forward rides over SSH.
    """

    project: str
    zone: str
    instance: str
    port: int
    impersonate_service_account: str | None = None
    tunnel_through_iap: bool = False


@dataclass(frozen=True)
class K8sPortForwardTarget:
    """``kubectl port-forward`` to a Service in a cluster."""

    namespace: str
    service: str
    port: int
    kubeconfig: str | None = None
    context: str | None = None


TunnelTarget = GcpSshForwardTarget | K8sPortForwardTarget

SpawnFn = Callable[[list[str]], "subprocess.Popen[str]"]


def _build_argv(target: TunnelTarget, local_port: int) -> list[str]:
    if isinstance(target, GcpSshForwardTarget):
        argv = [
            "gcloud",
            "compute",
            "ssh",
            target.instance,
            f"--project={target.project}",
            f"--zone={target.zone}",
            f"--ssh-flag=-L{local_port}:localhost:{target.port}",
            "--ssh-flag=-N",
            "--ssh-flag=-T",
            "--ssh-flag=-oServerAliveInterval=30",
            "--ssh-flag=-oExitOnForwardFailure=yes",
        ]
        if target.tunnel_through_iap:
            argv.append("--tunnel-through-iap")
        if target.impersonate_service_account:
            argv.append(f"--impersonate-service-account={target.impersonate_service_account}")
        return argv
    if isinstance(target, K8sPortForwardTarget):
        argv = ["kubectl"]
        if target.kubeconfig:
            argv += ["--kubeconfig", target.kubeconfig]
        if target.context:
            argv += ["--context", target.context]
        argv += [
            "port-forward",
            f"svc/{target.service}",
            f"{local_port}:{target.port}",
            "-n",
            target.namespace,
        ]
        return argv
    raise TypeError(f"unsupported tunnel target: {type(target).__name__}")


def describe_target(target: TunnelTarget) -> str:
    """Stable short label for logs and error messages."""
    if isinstance(target, GcpSshForwardTarget):
        return f"ssh://{target.project}/{target.zone}/{target.instance}:{target.port}"
    if isinstance(target, K8sPortForwardTarget):
        return f"k8s://{target.namespace}/{target.service}:{target.port}"
    raise TypeError(f"unsupported tunnel target: {type(target).__name__}")


def _find_free_port() -> int:
    """Bind a kernel-assigned port and release it — the slot stays free briefly."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _default_spawn(argv: list[str]) -> subprocess.Popen[str]:
    return subprocess.Popen(
        argv,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )


def _pump_stderr_to_logger(proc: subprocess.Popen, label: str) -> threading.Thread:
    """Forward each line of the child's stderr through ``logger.info``.

    Routes ``gcloud``/``kubectl`` progress lines ("Listening on port [N]",
    permission failures, etc.) into the same log stream as our own
    ``logger.info("Tunnel ready: ...")``, so a single ``--log-level`` or
    handler filter governs *all* tunnel output.
    """

    def _pump() -> None:
        stream = proc.stderr
        if stream is None:
            return
        for raw in stream:
            line = raw.rstrip()
            if line:
                logger.info("[%s] %s", label, line)

    t = threading.Thread(target=_pump, name=f"tunnel-stderr-{label}", daemon=True)
    t.start()
    return t


def _terminate(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


@contextlib.contextmanager
def open_tunnel(
    target: TunnelTarget,
    *,
    local_port: int | None = None,
    timeout: float = 60.0,
    watchdog: bool = True,
    spawn: SpawnFn = _default_spawn,
) -> Iterator[str]:
    """Open a local TCP tunnel to ``target``; yield ``http://127.0.0.1:<port>``.

    Waits up to ``timeout`` seconds for the tunnel to start accepting
    connections; respawns the child during the warm-up if it exits
    early. With ``watchdog=True`` (default), a daemon thread keeps the
    tunnel self-healing for the lifetime of the context. Every spawned
    child's stderr is forwarded through ``logger.info`` so all output
    (our own state messages, ``gcloud``/``kubectl`` progress, errors)
    flows through the standard logging pipeline.

    ``spawn`` is injectable for tests; production callers should leave it
    at the default.
    """
    if local_port is None:
        local_port = _find_free_port()
    argv = _build_argv(target, local_port)
    label = describe_target(target)

    proc_lock = threading.Lock()
    proc_ref: list[subprocess.Popen | None] = [None]
    shutdown = threading.Event()

    def _restart_locked() -> None:
        proc = spawn(argv)
        proc_ref[0] = proc
        _pump_stderr_to_logger(proc, label)

    with proc_lock:
        _restart_locked()

    deadline = Deadline.from_now(Duration.from_seconds(timeout))
    backoff = ExponentialBackoff(initial=1.0, maximum=5.0, factor=2.0)
    ready = False
    while not deadline.expired():
        with proc_lock:
            current = proc_ref[0]
        if current is None or current.poll() is not None:
            logger.warning("Tunnel %s exited early (retrying)", label)
            with proc_lock:
                _restart_locked()
            remaining = max(0.0, deadline.remaining_seconds())
            shutdown.wait(timeout=min(backoff.next_interval(), remaining))
            continue
        try:
            with socket.create_connection(("127.0.0.1", local_port), timeout=1):
                ready = True
                break
        except OSError:
            shutdown.wait(timeout=0.5)

    if not ready:
        with proc_lock:
            current = proc_ref[0]
        if current is not None:
            _terminate(current)
        raise RuntimeError(f"tunnel {label} did not open local port {local_port} within {timeout:.0f}s")

    logger.info("Tunnel ready: 127.0.0.1:%d -> %s", local_port, label)

    def _watchdog_run() -> None:
        wd_backoff = ExponentialBackoff(initial=1.0, maximum=5.0, factor=2.0)
        while not shutdown.wait(timeout=1.0):
            with proc_lock:
                current = proc_ref[0]
            if current is None or current.poll() is None:
                continue
            logger.warning("Tunnel %s died; respawning", label)
            if shutdown.wait(timeout=min(wd_backoff.next_interval(), 5.0)):
                return
            with proc_lock:
                if shutdown.is_set():
                    return
                _restart_locked()

    wd_thread = None
    if watchdog:
        wd_thread = threading.Thread(
            target=_watchdog_run,
            name=f"tunnel-watchdog-{label}",
            daemon=True,
        )
        wd_thread.start()

    try:
        yield f"http://127.0.0.1:{local_port}"
    finally:
        shutdown.set()
        if wd_thread is not None:
            wd_thread.join(timeout=2)
        with proc_lock:
            current = proc_ref[0]
        if current is not None:
            _terminate(current)
