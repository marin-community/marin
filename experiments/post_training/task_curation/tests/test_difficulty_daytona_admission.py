# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise Difficulty's shared DATA inventory command at its SDK boundary."""

import json
import runpy
import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("daytona")

from daytona_api_client_async import SandboxState
from shellbox.backends.daytona.connection import DaytonaConnection

from experiments.post_training.task_curation.difficulty.launch import DATA_ADMISSION_COMMAND


@pytest.mark.parametrize("started, expected_capacity", [(397, 1), (398, 0)])
def test_data_gate_reads_all_inventory_and_reserves_twenty_percent_of_total(
    monkeypatch, capsys, started, expected_capacity
):
    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            pass

        def list(self, *, request_timeout):
            async def pages():
                identifier = 0
                for state, count in (
                    (SandboxState.STARTED, started),
                    (SandboxState.STARTING, 1),
                    (SandboxState.CREATING, 1),
                    (SandboxState.STOPPED, 100),
                ):
                    for _ in range(count):
                        identifier += 1
                        yield SimpleNamespace(
                            id=str(identifier), organization_id="data-org", state=state, cpu=4, memory=4, disk=10
                        )

            return pages()

    connections = []

    def factory(connection):
        connections.append(connection)
        return Client

    monkeypatch.setattr(DaytonaConnection, "client_factory", factory)
    monkeypatch.setenv("DAYTONA_DATA_API_KEY", "data-fixture")
    monkeypatch.setenv("DAYTONA_API_KEY", "eval-must-not-be-used")
    monkeypatch.setenv("DAYTONA_TARGET", "us")
    monkeypatch.setenv("DAYTONA_API_URL", "https://fixture.invalid")
    monkeypatch.setattr(
        sys, "argv", [*DATA_ADMISSION_COMMAND[2:], "--cpu", "4", "--memory-gib", "8", "--disk-gib", "10", "--json"]
    )
    runpy.run_module(DATA_ADMISSION_COMMAND[2], run_name="__main__")
    report = json.loads(capsys.readouterr().out)
    assert report["sandboxes"]["total"] == started + 102
    assert report["resources"]["cpu"]["used"] == (started + 2) * 4
    assert report["max_concurrency"]["limit"] == expected_capacity
    assert connections[0].api_key == "data-fixture"
    assert connections[0].api_url == "https://fixture.invalid"
