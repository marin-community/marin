# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Open a controller tunnel via the iris CLI and yield its local URL.

Used by ``ducky query``, which appends ``/proxy/<endpoint>`` to reach the service.
Driving the iris CLI as a subprocess keeps ducky free of iris Python imports and robust
to iris internal API changes.
"""

from __future__ import annotations

import contextlib
import os
import re
import shlex
import subprocess
import threading
from collections.abc import Iterator

import click

# matches the local tunnel address the iris CLI prints (port varies)
_TUNNEL_URL_RE = re.compile(r"https?://127\.0\.0\.1:\d+")


@contextlib.contextmanager
def cluster_tunnel(cluster: str, *, timeout: float = 90.0) -> Iterator[str]:
    """Open a controller tunnel via ``iris cluster dashboard`` and yield its base URL.

    Spawns the iris CLI (override with ``$DUCKY_IRIS_CMD``), reads its output until the
    local tunnel URL appears, and tears the tunnel down on exit. The URL is the raw
    controller address (e.g. ``http://127.0.0.1:10000``); callers append any path.
    """
    iris_cmd = shlex.split(os.environ.get("DUCKY_IRIS_CMD", "iris"))
    proc = subprocess.Popen(
        [*iris_cmd, "--cluster", cluster, "cluster", "dashboard"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    # Reading proc.stdout line-by-line blocks if iris emits nothing, so a wall-clock
    # check inside the loop never fires — enforce the startup timeout by terminating
    # the process (which EOFs the read) if the URL hasn't appeared in time.
    startup_kill = threading.Timer(timeout, proc.terminate)
    startup_kill.start()
    base = None
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            match = _TUNNEL_URL_RE.search(line)
            if match:
                base = match.group(0)
                break
    finally:
        startup_kill.cancel()  # startup done (found or EOF); stop guarding it

    if base is None:
        proc.terminate()
        raise click.ClickException(f"Could not open a tunnel to cluster {cluster!r} via `iris cluster dashboard`.")
    try:
        yield base
    finally:
        proc.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=10)
