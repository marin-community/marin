# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise the temporary inventory-only DATA gate at its SDK boundary."""

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("daytona")

from daytona_api_client_async import SandboxState
from shellbox.backends.daytona.connection import DaytonaConnection
from shellbox.backends.daytona.usage import SandboxProfile

from experiments.post_training.task_curation.difficulty.daytona_admission import current_report


def test_data_gate_reads_all_inventory_and_reserves_twenty_percent_of_total(monkeypatch):
    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        def list(self, *, request_timeout):
            async def pages():
                for state, count in (
                    (SandboxState.STARTED, 397),
                    (SandboxState.STARTING, 1),
                    (SandboxState.CREATING, 1),
                    (SandboxState.STOPPED, 100),
                ):
                    for _ in range(count):
                        yield SimpleNamespace(state=state, cpu=4, memory=8, disk=10)

            return pages()

    connections = []

    def factory(connection):
        connections.append(connection)
        return Client

    monkeypatch.setattr(DaytonaConnection, "client_factory", factory)
    monkeypatch.setenv("DAYTONA_DATA_API_KEY", "data-fixture")
    monkeypatch.setenv("DAYTONA_API_KEY", "eval-must-not-be-used")
    monkeypatch.setenv("DAYTONA_TARGET", "us")
    report = asyncio.run(current_report(SandboxProfile(4, 8, 10), 0.2))
    assert report["sandboxes"] == 499
    assert report["regions"][0]["cpu"]["used"] == 1596
    assert report["max_concurrency"]["limit"] == 1
    assert connections[0].api_key == "data-fixture"
