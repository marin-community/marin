# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Importable helpers shared by Taskforge tests."""

import os
import uuid
from datetime import UTC, datetime

from taskforge.llm.glm_client import FixedEndpoint, GlmEndpoint, Pool

GLM_MODEL = "glm-5.3"


def unique_run_name(prefix: str) -> str:
    """Name a run uniquely across pytest workers and reruns."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    worker = os.environ.get("PYTEST_XDIST_WORKER", "main")
    return f"{prefix}-{stamp}-{worker}-{uuid.uuid4().hex[:8]}"


def fake_source(base_url: str) -> FixedEndpoint:
    return FixedEndpoint(GlmEndpoint(base_url, "test-token", Pool.HIGH))
