# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Run JAX test scripts with a fixed CPU device count."""

import os
import subprocess
import sys
import textwrap


def run_on_cpu_devices(script: str, *, device_count: int) -> None:
    """Run a script in a fresh interpreter so XLA can set the CPU device count."""
    env = os.environ.copy()
    env.pop("JAX_NUM_CPU_DEVICES", None)
    env["JAX_PLATFORMS"] = "cpu"
    env["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={device_count}"
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
