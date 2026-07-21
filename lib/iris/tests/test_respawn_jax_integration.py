# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end check of the respawn hook against a real JAX distributed gang.

This is the failure from marin issue #7430 in miniature: a multi-process JAX
world where one non-leader process dies from a crash signal. JAX's coordination
service fate-shares the gang — the leader aborts with LOG(FATAL)/SIGABRT once
the dead task misses its heartbeats — and each process's respawner restarts its
own child. The restarted children re-run ``jax.distributed.initialize`` (riding
XLA's registration retries through the window where the old coordination
service is still up) and the world re-forms on the same coordinator port.

Requires jax (CPU is fine) and real wall-clock for heartbeat detection, so it
is ``slow``-marked and skipped in the unit lanes. Run it from the repo root:

    uv run pytest lib/iris/tests/test_respawn_jax_integration.py -o addopts= -v
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("jax")

_NUM_PROCS = 2

_CHILD_SRC = textwrap.dedent(
    """
    import os
    import resource
    import signal
    import sys

    import jax

    proc_id = int(sys.argv[1])
    coordinator = sys.argv[2]
    out_dir = sys.argv[3]
    attempt = int(os.environ["IRIS_RESPAWN_ATTEMPT"])
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    jax.distributed.initialize(
        coordinator_address=coordinator,
        num_processes={num_procs},
        process_id=proc_id,
        local_device_ids=[0],
        cluster_detection_method="deactivate",
        initialization_timeout=120,
        heartbeat_timeout_seconds=5,
    )

    if attempt == 0:
        if proc_id == 1:
            # The issue-#7430 shape: a non-leader process crashes hard.
            os.kill(os.getpid(), signal.SIGSEGV)
        # Peers (including the leader hosting the coordination service) do no
        # work of their own; fate-sharing must abort them once the coordination
        # service notices the dead task. The parent timeout catches a missing abort.
        signal.pause()

    # A respawned attempt that re-initialized successfully proves the world
    # re-formed. Record it and exit clean.
    with open(os.path.join(out_dir, f"reformed_{{proc_id}}"), "w") as f:
        f.write(str(attempt))
    """
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.slow
@pytest.mark.timeout(300)
def test_gang_reforms_after_single_task_crash(tmp_path: Path) -> None:
    child_script = tmp_path / "child.py"
    child_script.write_text(_CHILD_SRC.format(num_procs=_NUM_PROCS))
    coordinator = f"127.0.0.1:{_free_port()}"

    respawners = []
    for proc_id in range(_NUM_PROCS):
        argv = [
            sys.executable,
            "-m",
            "iris.cluster.hooks.respawn_main",
            "--max-restarts",
            "3",
            "--",
            sys.executable,
            str(child_script),
            str(proc_id),
            coordinator,
            str(tmp_path),
        ]
        env = {**os.environ, "JAX_PLATFORMS": "cpu"}
        respawners.append(subprocess.Popen(argv, env=env))

    exit_codes = [p.wait(timeout=280) for p in respawners]

    assert exit_codes == [0, 0]
    for proc_id in range(_NUM_PROCS):
        marker = tmp_path / f"reformed_{proc_id}"
        assert marker.exists(), f"process {proc_id} never re-formed the world"
        assert int(marker.read_text()) >= 1
