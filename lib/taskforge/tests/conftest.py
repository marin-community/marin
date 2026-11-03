# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for Taskforge tests.

``fake_glm`` is ``taskforge_adapters.testing.FakeGlm``, a scripted fake of the GLM router run as a
real local HTTP server. ``fast_provider`` is a provider config whose backoff is a millisecond, so
retries and holds are quick, and ``glm_client`` is a ``GlmClient`` on ``fake_glm`` with that
config. ``ledger`` is an in-memory ``Ledger`` that keeps every recorded entry in ``entries``.

Live tests carry ``@pytest.mark.live_glm`` and request ``glm_source``, which skips them unless
``TASKFORGE_GLM_BASE_URL`` names the router's ``/v1`` URL and ``GLM_HIGH_TOKEN`` holds the high
pool's bearer token; ``live_provider`` is the provider config they call it with.
Live tests write raw evidence under ``evidence_root``, which is outside the checkout, and name run
directories and workpiece ids with ``unique_run_name``, so parallel workers and reruns never collide.
"""

import os
import tempfile
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pytest
from taskforge_adapters.glm_client import FixedEndpoint, GlmClient
from taskforge_adapters.testing import (
    BASE_URL_ENV,
    FAST_PROVIDER,
    LIVE_PROVIDER,
    TOKEN_ENV,
    FakeGlm,
    fake_source,
    live_glm_source,
    serving_fake_glm,
)

from taskforge.ledger.records import LedgerEntry
from taskforge.llm.policy import LLMProviderConfig

EVIDENCE_DIR_ENV = "TASKFORGE_EVIDENCE_DIR"
DEFAULT_EVIDENCE_ROOT = Path(tempfile.gettempdir()) / "taskforge-evidence"


@pytest.fixture(autouse=True)
def no_package_downloads(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest):
    if any(request.node.get_closest_marker(marker) is not None for marker in ("live_glm", "manual")):
        return

    def refuse_download(*args, **kwargs):
        raise AssertionError("Unit tests must not download ShellSim catalog or runtime assets")

    monkeypatch.setattr(urllib.request, "urlopen", refuse_download)


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
def evidence_root() -> Path:
    """Where live checks write raw evidence: ``TASKFORGE_EVIDENCE_DIR`` when set, else ``<tmp>/taskforge-evidence``.

    The default is the system temp directory, so evidence survives across runs on one machine and never
    lands in the checkout.
    """
    return Path(os.environ.get(EVIDENCE_DIR_ENV) or DEFAULT_EVIDENCE_ROOT).expanduser()


class ListLedger:
    def __init__(self) -> None:
        self.entries: list[LedgerEntry] = []

    def record(self, entry: LedgerEntry) -> None:
        self.entries.append(entry)


@pytest.fixture
def ledger() -> ListLedger:
    return ListLedger()


@pytest.fixture
def fast_provider() -> LLMProviderConfig:
    return FAST_PROVIDER


@pytest.fixture
def fake_glm() -> Iterator[FakeGlm]:
    with serving_fake_glm() as fake:
        yield fake


@pytest.fixture
async def glm_client(fake_glm: FakeGlm, fast_provider: LLMProviderConfig):
    async with GlmClient(fake_source(fake_glm.base_url), fast_provider) as client:
        yield client
