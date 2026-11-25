# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import subprocess
import sys

import psutil
from rigging.timing import Duration, ExponentialBackoff

from experiments.evaluation.knotmeter.processes import stop_group


def test_cleanup_stops_a_descendant_that_ignores_sigterm():
    child_code = (
        "import os,signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        "print(os.getpid(),flush=True); time.sleep(90)"
    )
    parent_code = (
        "import subprocess,sys; "
        "child=subprocess.Popen([sys.executable,'-c',sys.argv[1]],stdout=subprocess.PIPE,text=True); "
        "print(child.stdout.readline(),end='',flush=True)"
    )
    with subprocess.Popen(
        [sys.executable, "-c", parent_code, child_code], stdout=subprocess.PIPE, text=True, start_new_session=True
    ) as parent:
        output, _ = parent.communicate(timeout=10)
        assert parent.returncode == 0
        child = psutil.Process(int(output.strip()))
        try:
            stop_group(parent)

            def stopped():
                return not child.is_running() or child.status() == psutil.STATUS_ZOMBIE

            ExponentialBackoff(initial=0.001, maximum=0.01).wait_until_or_raise(
                stopped, timeout=Duration.from_seconds(5), error_message="A descendant survived command cleanup"
            )
        finally:
            with contextlib.suppress(psutil.NoSuchProcess):
                child.kill()
