# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Device selection and CPU emulation for in-process JAX tests."""

import os
import subprocess
import sys
import textwrap
from collections.abc import Iterator
from contextlib import contextmanager

import jax
import pytest  # pyrefly: ignore[missing-import]  # Installed by Levanter's test group.
from jax.extend.backend import clear_backends


@contextmanager
def use_cpu_devices(device_count: int) -> Iterator[None]:
    """Run with exactly this many CPU devices, then restore JAX's configuration.

    Arrays and meshes must be created inside the context and stay within it.
    Backend configuration is process-wide, so this cannot run in parallel threads.
    """
    previous_count = jax.config.jax_num_cpu_devices
    previous_platforms = jax.config.jax_platforms
    clear_backends()
    try:
        jax.config.update("jax_platforms", "cpu")
        jax.config.update("jax_num_cpu_devices", device_count)
        yield
    finally:
        clear_backends()
        jax.config.update("jax_num_cpu_devices", previous_count)
        jax.config.update("jax_platforms", previous_platforms)


@contextmanager
def use_test_devices(device_count: int) -> Iterator[list[jax.Device]]:
    """Use native accelerators, or emulate the requested device count on CPU.

    Accelerator tests skip when too few local devices are available. CPU tests
    reset the backend when more devices are needed; create their arrays and meshes inside this context.
    """
    devices = jax.local_devices()
    if len(devices) >= device_count:
        yield devices[:device_count]
        return
    if jax.default_backend() != "cpu":
        pytest.skip(f"Requires {device_count} local devices; found {len(devices)}")

    with use_cpu_devices(device_count):
        yield jax.local_devices()


def run_on_cpu_devices(script: str, *, device_count: int) -> None:
    """Run a script in a fresh interpreter so XLA can set the CPU device count."""
    env = os.environ.copy()
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
