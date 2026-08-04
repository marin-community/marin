# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import select
import signal
import subprocess
import sys

from rigging.tunnel import terminate_process_group


def test_terminate_process_group_signals_descendants() -> None:
    child = (
        "import signal,sys; "
        "signal.signal(signal.SIGTERM, lambda *_: (print('child-stopped', flush=True), sys.exit(0))); "
        "print('child-ready', flush=True); "
        "signal.pause()"
    )
    parent = "; ".join(
        (
            "import signal,subprocess,sys",
            f"subprocess.Popen([sys.executable, '-c', {child!r}])",
            "signal.pause()",
        )
    )
    process = subprocess.Popen(
        [sys.executable, "-c", parent],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    assert process.stdout is not None
    assert process.stdout.readline().strip() == "child-ready"

    terminate_process_group(process, grace_period=1)

    readable, _, _ = select.select([process.stdout], [], [], 2)
    assert readable
    assert process.stdout.readline().strip() == "child-stopped"


def test_terminate_process_group_escalates_when_process_ignores_sigterm() -> None:
    ignore_sigterm = "; ".join(
        (
            "import signal,time",
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)",
            "print('ready', flush=True)",
            "time.sleep(60)",
        )
    )
    process = subprocess.Popen(
        [sys.executable, "-c", ignore_sigterm],
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    assert process.stdout is not None
    assert process.stdout.readline().strip() == "ready"

    terminate_process_group(process, grace_period=0.01)

    assert process.returncode == -signal.SIGKILL
