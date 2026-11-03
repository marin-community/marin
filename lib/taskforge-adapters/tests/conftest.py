# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for the Taskforge adapter tests.

``fake_glm`` and ``other_fake_glm`` are independent ``taskforge_adapters.testing.FakeGlm`` servers, the
second for an endpoint that moves. ``fast_provider`` is a provider config whose backoff is a
millisecond, so retries and holds are quick.

Live tests carry ``@pytest.mark.live_glm`` and request ``glm_source``, which skips them unless
``TASKFORGE_GLM_BASE_URL`` names the router's ``/v1`` URL and ``GLM_HIGH_TOKEN`` holds the high
pool's bearer token; ``live_provider`` is the provider config they call it with. Web tests take
``parallel_key``, which skips unless ``TASKFORGE_PARALLEL_KEY_FILE`` names the Parallel key file.
Live tests write raw evidence under ``evidence_root``, which is outside the checkout, and name run
directories with ``unique_run_name``.
"""

import os
import tempfile
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from taskforge.llm.policy import LLMProviderConfig

from taskforge_adapters.glm_client import FixedEndpoint
from taskforge_adapters.testing import (
    BASE_URL_ENV,
    FAST_PROVIDER,
    LIVE_PROVIDER,
    TOKEN_ENV,
    FakeGlm,
    live_glm_source,
    serving_fake_glm,
)

PARALLEL_KEY_FILE_ENV = "TASKFORGE_PARALLEL_KEY_FILE"
PARALLEL_KEY = "PARALLEL_KEY"
EVIDENCE_DIR_ENV = "TASKFORGE_EVIDENCE_DIR"
DEFAULT_EVIDENCE_ROOT = Path(tempfile.gettempdir()) / "taskforge-evidence"


@pytest.fixture(autouse=True)
def no_package_downloads(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest):
    if any(request.node.get_closest_marker(marker) is not None for marker in ("live_glm", "manual")):
        return

    def refuse_download(*args, **kwargs):
        raise AssertionError("Unit tests must not download ShellSim catalog or runtime assets")

    monkeypatch.setattr(urllib.request, "urlopen", refuse_download)


@dataclass(frozen=True)
class ParallelKey:
    """Kept out of ``repr`` so a failing test's traceback, which prints its fixtures, omits the key."""

    value: str = field(repr=False)


def read_key_line(path: Path, key: str) -> str:
    """Return the value of the ``<key>=...`` line in ``path``; the value is never echoed."""
    for line in path.read_text().splitlines():
        name, sep, value = line.strip().partition("=")
        if sep and name.strip() == key and value.strip():
            return value.strip()
    raise ValueError(f"{path} has no non-empty {key}= line")


@pytest.fixture(scope="session")
def glm_source() -> FixedEndpoint:
    """The live router's high pool, from ``TASKFORGE_GLM_BASE_URL`` and ``GLM_HIGH_TOKEN``."""
    source = live_glm_source()
    if source is None:
        pytest.skip(f"live GLM test: set {BASE_URL_ENV} and {TOKEN_ENV} (see lib/taskforge/README.md)")
    return source


@pytest.fixture(scope="session")
def live_provider() -> LLMProviderConfig:
    return LIVE_PROVIDER


@pytest.fixture(scope="session")
def parallel_key() -> ParallelKey:
    """The Parallel API key from the ``PARALLEL_KEY=...`` line of ``TASKFORGE_PARALLEL_KEY_FILE``."""
    key_file = os.environ.get(PARALLEL_KEY_FILE_ENV)
    if not key_file:
        pytest.skip(f"live web test: set {PARALLEL_KEY_FILE_ENV} to the Parallel key file")
    return ParallelKey(read_key_line(Path(key_file).expanduser(), PARALLEL_KEY))


@pytest.fixture(scope="session")
def evidence_root() -> Path:
    """Where live checks write raw evidence: ``TASKFORGE_EVIDENCE_DIR`` when set, else ``<tmp>/taskforge-evidence``."""
    return Path(os.environ.get(EVIDENCE_DIR_ENV) or DEFAULT_EVIDENCE_ROOT).expanduser()


@pytest.fixture
def fast_provider() -> LLMProviderConfig:
    return FAST_PROVIDER


@pytest.fixture
def fake_glm() -> Iterator[FakeGlm]:
    with serving_fake_glm() as fake:
        yield fake


@pytest.fixture
def other_fake_glm() -> Iterator[FakeGlm]:
    with serving_fake_glm() as fake:
        yield fake
