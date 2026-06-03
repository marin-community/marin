# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase-0 parity smoke tests.

These run against both the Python and Rust servers over real HTTP. ``/health``
must work on both today. The RPC-level tests assert the long-term parity
contract; they pass on Python now and are marked ``rust_pending`` until the
relevant migration phase lands the RPC family in the Rust server.
"""

from __future__ import annotations

import httpx
import pytest
from finelog.client import LogClient

from tests.parity.conftest import Backend

# Spawning a server subprocess can exceed the global 10s per-test timeout under
# load, so give parity tests more room.
pytestmark = pytest.mark.timeout(40)


def test_health(finelog_url: str) -> None:
    resp = httpx.get(f"{finelog_url}/health", timeout=2.0)
    assert resp.status_code == 200
    assert resp.text.strip() == "ok"


def test_constant_query_round_trips(client: LogClient, server_backend: Backend) -> None:
    """A trivial Query RPC round-trips — exercises StatsService.Query end to end.

    No table registration needed: ``SELECT 1`` touches the SQL engine and the
    Arrow-IPC result path only. Green on both backends since Phase 3.
    """
    table = client.query("SELECT 1 AS n")
    assert table.num_rows == 1
    assert table.column_names == ["n"]
    assert table.column("n")[0].as_py() == 1
