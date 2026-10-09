# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for the capability proposal source.

The live proposal and triage tests take ``capability_catalog``, which skips unless
``TASKFORGE_CAPABILITY_CATALOG`` names the capability catalog (``new_catalog.json``).
"""

import os
from pathlib import Path

import pytest

CATALOG_ENV = "TASKFORGE_CAPABILITY_CATALOG"


@pytest.fixture(scope="session")
def capability_catalog() -> Path:
    """The capability catalog; it is gitignored, so its path comes from the env."""
    path = os.environ.get(CATALOG_ENV)
    if not path:
        pytest.skip(f"live capability tests: set {CATALOG_ENV} (see lib/taskforge/README.md)")
    return Path(path).expanduser()
