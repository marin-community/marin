# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare dashboard ports on demand; workers consume the local cache.

Preparation and offline workers must share the same precompiled trust policy,
which selects the cache namespace.
"""

import logging
import threading
from dataclasses import dataclass, field

import shellsim

logger = logging.getLogger(__name__)

TOOL_PACKAGES = ("numpy",)


@dataclass
class ToolRuntime:
    """Share package preparation across dashboard tool requests."""

    catalog: str
    _prepared: bool = field(default=False, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def prepare(self) -> None:
        """Populate the local cache once, retrying after a failed preparation."""
        with self._lock:
            if self._prepared:
                return
            logger.info("Preparing dashboard CPython and NumPy from %s", self.catalog)
            prepare_tool_runtime(self.catalog)
            self._prepared = True


def prepare_tool_runtime(catalog: str) -> None:
    shellsim.Context(
        pypi_catalog=catalog,
        trusted_precompiled=catalog == shellsim.DEFAULT_PYPI_CATALOG,
    ).Environment(pypi_packages=TOOL_PACKAGES)


def tool_environment(catalog: str, *, cpu: int, output: int) -> shellsim.Environment:
    return shellsim.Context(
        pypi_catalog=catalog,
        offline=True,
        trusted_precompiled=catalog == shellsim.DEFAULT_PYPI_CATALOG,
    ).Environment(
        pypi_packages=TOOL_PACKAGES,
        cpu=cpu,
        memory=1024 * 1024 * 1024,
        disk=512 * 1024 * 1024,
        output=output,
    )
