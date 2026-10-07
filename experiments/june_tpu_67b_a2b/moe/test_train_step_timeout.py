# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
import textwrap


def test_exit_if_hung_ends_a_blocked_process_with_status_1():
    # The block waits on an event nobody sets, like a step whose peers never join a collective.
    script = textwrap.dedent(
        """
        import threading

        from experiments.june_tpu_67b_a2b.moe.train import exit_if_hung

        with exit_if_hung(0.5):
            threading.Event().wait()
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
        capture_output=True,
        text=True,
        timeout=50,
    )
    assert result.returncode == 1, f"stdout={result.stdout}\nstderr={result.stderr}"
    assert "Timeout (" in result.stderr
