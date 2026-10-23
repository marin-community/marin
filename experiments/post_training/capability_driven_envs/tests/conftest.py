# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for the capability live tests.

The live tests call GLM-5.3's high pool through
``live_endpoint``, which skips unless ``TASKFORGE_GLM_BASE_URL`` and ``GLM_HIGH_TOKEN`` are set, and
read the full capability catalog through ``capability_catalog``, which skips unless
``TASKFORGE_CAPABILITY_CATALOG`` names it. The full catalog is not checked in. Live tests write
evidence under ``evidence_root``: ``TASKFORGE_EVIDENCE_DIR``, else ``taskforge-evidence/`` in the
system temp directory.
"""

from __future__ import annotations

import importlib.util
import os
import tempfile
from pathlib import Path

import pytest

# These tests run in the marin-taskforge package environment. An environment without it, such as the
# marin unit job, must not collect them: a skip raised while pytest loads a conftest aborts the whole
# session rather than skipping, so the guard ignores the directory's modules instead. pytest applies
# collect_ignore_glob only to files it finds in a directory, not to a file passed by explicit path, as the marin
# unit job passes them, so every test module here also skips itself with pytest.importorskip("taskforge").
if importlib.util.find_spec("taskforge") is None:
    collect_ignore_glob = ["*.py"]
else:
    from taskforge.llm.glm_client import FixedEndpoint, GlmEndpoint, Pool

BASE_URL_ENV = "TASKFORGE_GLM_BASE_URL"
TOKEN_ENV = "GLM_HIGH_TOKEN"
CATALOG_ENV = "TASKFORGE_CAPABILITY_CATALOG"
EVIDENCE_DIR_ENV = "TASKFORGE_EVIDENCE_DIR"
DEFAULT_EVIDENCE_ROOT = Path(tempfile.gettempdir()) / "taskforge-evidence"


@pytest.fixture(scope="session")
def live_endpoint() -> FixedEndpoint:
    base_url = os.environ.get(BASE_URL_ENV)
    token = os.environ.get(TOKEN_ENV)
    if not base_url or not token:
        pytest.skip(f"live GLM test: set {BASE_URL_ENV} and {TOKEN_ENV} (see lib/taskforge/README.md)")
    return FixedEndpoint(GlmEndpoint(base_url.rstrip("/"), token, Pool.HIGH))


@pytest.fixture(scope="session")
def capability_catalog() -> Path:
    path = os.environ.get(CATALOG_ENV)
    if not path:
        pytest.skip(f"live capability test: set {CATALOG_ENV} to the capability catalog JSON file")
    return Path(path)


@pytest.fixture(scope="session")
def evidence_root() -> Path:
    return Path(os.environ.get(EVIDENCE_DIR_ENV) or DEFAULT_EVIDENCE_ROOT)
