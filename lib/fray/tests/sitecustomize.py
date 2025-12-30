# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test-only Python startup hooks.

Ray vendors `psutil` and uses it in multiple subprocesses (dashboard, agent, etc).
In sandboxed macOS environments, certain sysctl calls used by psutil can raise
`PermissionError`/`SystemError`, which can crash those subprocesses and break
job submission. This module is loaded automatically by Python at startup when
it's on `PYTHONPATH` (via `site.py`).

This seems to be mostly necessary to run inside OpenAI Codex sandbox.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _patch_psutil() -> None:
    try:
        import psutil  # Ray vendors psutil under the top-level module name.
    except Exception:
        return

    orig_pids = getattr(psutil, "pids", None)
    if callable(orig_pids):

        def pids() -> list[int]:
            try:
                return orig_pids()
            except PermissionError:
                # Some psutil code assumes pids()[0] exists (lowest PID). Returning
                # [1] avoids IndexError and allows parent-walking code paths that
                # compare against "lowest PID" to keep working.
                return [1]

        psutil.pids = pids  # type: ignore[assignment]

    orig_cpu_count = getattr(psutil, "cpu_count", None)
    if callable(orig_cpu_count):

        def cpu_count(*, logical: bool = True) -> int | None:
            try:
                return orig_cpu_count(logical=logical)
            except Exception:
                # In restricted environments psutil can raise PermissionError or
                # surface it as a SystemError from its C-extension. For tests, a
                # best-effort value is sufficient.
                return 1

        psutil.cpu_count = cpu_count  # type: ignore[assignment]


def _patch_ray_address_discovery() -> None:
    bootstrap_path = None
    try:
        import os

        bootstrap_path = os.environ.get("FRAY_RAY_BOOTSTRAP_ADDRESS_PATH")
    except Exception:
        return

    if not bootstrap_path:
        return

    try:
        from pathlib import Path

        import ray._private.services as services
    except Exception:
        return

    orig = services.get_ray_address_from_environment

    def get_ray_address_from_environment(addr: str, temp_dir: str | None):
        try:
            text = Path(bootstrap_path).read_text().strip()
        except Exception:
            text = ""

        if (addr is None or addr == "auto") and text:
            return text
        return orig(addr, temp_dir)

    services.get_ray_address_from_environment = get_ray_address_from_environment  # type: ignore[assignment]


try:
    _patch_psutil()
    _patch_ray_address_discovery()
except Exception:
    # Avoid breaking interpreter startup for a test-only helper.
    logger.exception("Failed to apply test psutil patch; proceeding without it.")
