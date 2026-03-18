# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for controller unit tests."""

import pytest

from iris.cluster.controller.provider import ProviderUnsupportedError


class FakeProvider:
    """Minimal TaskProvider for tests that only exercise transitions, not RPCs."""

    @property
    def is_direct_provider(self) -> bool:
        return False

    def sync(self, batches):
        return [(b, None, "no stub") for b in batches]

    def fetch_live_logs(self, worker_id, address, task_id, attempt_id, cursor, max_lines):
        raise ProviderUnsupportedError("fake")

    def fetch_process_logs(self, worker_id, address, request):
        raise ProviderUnsupportedError("fake")

    def get_process_status(self, worker_id, address, request):
        raise ProviderUnsupportedError("fake")

    def on_worker_failed(self, worker_id, address):
        pass

    def profile_task(self, address, request, timeout_ms):
        raise ProviderUnsupportedError("fake")

    def close(self):
        pass


@pytest.fixture
def fake_provider() -> FakeProvider:
    return FakeProvider()
