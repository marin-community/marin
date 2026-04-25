# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-task tunnel scope for off-cluster access.

Resolver plugins that talk to internal addresses can consult an active
:class:`ProxyStack` to open ssh tunnels lazily and reuse them for the
lifetime of the scope. Typical use::

    from rigging.proxy import proxy_stack

    with proxy_stack():
        client = LogClient.connect("iris://marin?endpoint=/system/log-server")
        client.write_batch(...)

Inside the block, the iris:// handler probes each cluster-internal
address it sees; unreachable ones are routed through a tunnel opened by
the cluster's :class:`ControllerProvider` and cached on the active
``ProxyStack``. On a cluster VM where addresses are directly reachable,
no tunnels open. Tunnels close when the block exits.

Tunnels are pinned to the ``proxy_stack`` block — addresses returned
from the resolver are bare ``(host, port)`` tuples, so we cannot
ref-count or TTL them. Treat ``proxy_stack`` as bracketing a logical
task (a CLI invocation, a batch run), not as a long-lived registry.
"""

import socket
import subprocess
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, ExitStack, contextmanager
from contextvars import ContextVar

# Long enough for a slow office network, short enough that an
# unreachable target doesn't block a CLI for noticeable wall time.
PROBE_TIMEOUT_SECONDS = 0.5


class ProxyStack:
    """Lifetime-scoped tunnel registry.

    Holds an :class:`~contextlib.ExitStack` and a remote→local cache.
    Use :meth:`proxy` to acquire a tunnel for an internal address:
    callers supply the opener (a tunnel context manager) and the stack
    memoizes by remote address so repeat calls within the scope reuse
    the same tunnel.
    """

    def __init__(self) -> None:
        self._stack = ExitStack()
        self._cache: dict[tuple[str, int], tuple[str, int]] = {}

    def proxy(
        self,
        addr: tuple[str, int],
        opener: Callable[[], AbstractContextManager[tuple[str, int]]],
    ) -> tuple[str, int]:
        """Return the local proxy for ``addr``, opening a tunnel if needed."""
        if addr in self._cache:
            return self._cache[addr]
        proxied = self._stack.enter_context(opener())
        self._cache[addr] = proxied
        return proxied

    def __enter__(self) -> "ProxyStack":
        self._stack.__enter__()
        return self

    def __exit__(self, *exc) -> bool | None:
        return self._stack.__exit__(*exc)


_active: ContextVar[ProxyStack | None] = ContextVar("rigging_proxy_stack", default=None)


@contextmanager
def proxy_stack() -> Iterator[ProxyStack]:
    """Enter a tunnel scope. Resolver plugins consult :func:`active_stack`
    to open lazy tunnels for unreachable addresses; all opened tunnels
    close when the block exits."""
    stack = ProxyStack()
    with stack:
        token = _active.set(stack)
        try:
            yield stack
        finally:
            _active.reset(token)


def active_stack() -> ProxyStack | None:
    """Return the active :class:`ProxyStack`, or ``None`` outside a
    :func:`proxy_stack` block."""
    return _active.get()


def is_reachable(host: str, port: int, timeout: float = PROBE_TIMEOUT_SECONDS) -> bool:
    """One TCP connect attempt with a short timeout."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@contextmanager
def ssh_proxy(
    ssh_target: str,
    remote_host: str,
    remote_port: int,
    *,
    local_port: int | None = None,
    extra_ssh_args: tuple[str, ...] = (),
    timeout: float = 30.0,
) -> Iterator[tuple[str, int]]:
    """Open a vanilla ``ssh -L`` tunnel through ``ssh_target``.

    ``ssh_target`` is anything ``ssh`` accepts: ``user@host``, an alias
    from ``~/.ssh/config``, etc. For gcloud-managed VMs use the iris
    provider's ``tunnel_to``, which handles project/zone/OS Login.
    """
    if local_port is None:
        local_port = _find_free_local_port()

    cmd = [
        "ssh",
        ssh_target,
        "-L",
        f"127.0.0.1:{local_port}:{remote_host}:{remote_port}",
        "-N",
        "-o",
        "BatchMode=yes",
        "-o",
        "ServerAliveInterval=60",
        "-o",
        "ServerAliveCountMax=3",
        *extra_ssh_args,
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True)
    try:
        if not _wait_for_local_port(local_port, timeout=timeout):
            stderr = proc.stderr.read().decode() if proc.stderr else ""
            raise RuntimeError(f"ssh -L failed to bind 127.0.0.1:{local_port}: {stderr}")
        yield ("127.0.0.1", local_port)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def _find_free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_local_port(port: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if is_reachable("127.0.0.1", port, timeout=0.2):
            return True
        time.sleep(0.1)
    return False
