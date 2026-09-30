# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The 2026-09-29 liveness fix over real HTTP: broker app, host app, SiloClient."""

import concurrent.futures
import threading
import time
import urllib.error
import urllib.request

import pytest
from silo.auth import HEADER_API_TOKEN, bearer
from silo.broker import export
from silo.broker.core import Broker, BrokerConfig
from silo.broker.server import PoolSizes, host_client_factory
from silo.broker.server import build_app as build_broker_app
from silo.broker.state import FileStateStore, restore
from silo.client import SiloClient
from silo.errors import SiloError
from silo.host.agent import HostAgent, HostConfig
from silo.host.server import build_app as build_host_app
from silo.http import HttpClient
from silo.pipeline_contract.daytona_snapshot import resource_not_found
from silo_testing import wait_until
from test_end_to_end import sandbox_params, snapshot_params, stack  # noqa: F401 - fixture

GB = 1024**3


def test_heartbeats_answer_promptly_while_creates_hang_on_a_sick_host(tmp_path, fake_runtime, serve):
    """The incident, end to end: creates to a host whose `nerdctl run` hangs.

    Before the fix they shared one pool with heartbeats, and once enough of them
    hung every heartbeat queued past the host's timeout. Now a heartbeat never
    enters a pool that host I/O can fill, and one host can pin at most
    max_inflight_creates_per_host create calls.
    """
    secret, token = "host-secret", "api-token"
    gate = threading.Event()
    real_start = fake_runtime.start

    def hanging_start(spec):
        gate.wait(30)
        real_start(spec)

    agent = HostAgent(HostConfig("h1", 32, 128 * GB, 600 * GB, tuple(range(32)), tmp_path / "work"), fake_runtime)
    host = serve(build_host_app(agent, secret))
    broker = Broker(
        host_secret=secret,
        host_client_factory=host_client_factory(secret),
        config=BrokerConfig(max_inflight_creates_per_host=3),
    )
    broker_server = serve(
        build_broker_app(
            broker,
            api_token=token,
            host_secret=secret,
            self_url=lambda: broker_server.url,
            pool_sizes=PoolSizes(control=2, host_io=2, create=2),
        )
    )
    broker.heartbeat("h1", host.url, agent.capacity(), [])
    client = SiloClient(broker_server.url, token)
    client.snapshot.create(snapshot_params("verifier", cpu=2, memory=1, disk=1), timeout=30)
    fake_runtime.start = hanging_start

    heartbeat = HttpClient(broker_server.url, headers={HEADER_API_TOKEN: bearer(secret)}, timeout=2)
    with concurrent.futures.ThreadPoolExecutor(12) as pool:
        creates = [pool.submit(client.create, sandbox_params("verifier"), 60) for _ in range(12)]
        wait_until(lambda: broker.capacity()["creates_inflight"] == 3, timeout=10)
        latencies = []
        for _ in range(5):
            started = time.monotonic()
            heartbeat.post("/hosts/heartbeat", {"host_id": "h1", "url": host.url, "capacity": agent.capacity()})
            latencies.append(time.monotonic() - started)
        assert max(latencies) < 1.0, latencies
        assert client.capacity()["creates_inflight"] == 3  # capped, though the host has 32 slots
        gate.set()
        results = [future.result(timeout=60) for future in creates]
    fake_runtime.start = real_start
    assert len({sandbox.id for sandbox in results}) == 12


def test_a_suspect_hosts_sandbox_is_a_retryable_503_over_the_wire(stack):  # noqa: F811
    record = stack.client.snapshot.create(snapshot_params("s503"), timeout=30)
    assert record.state == "active"
    sandbox = stack.client.create(sandbox_params("s503"))
    host = stack.broker._hosts["h1"]
    host.last_seen -= stack.broker.config.suspect_after_seconds + 5  # heartbeats stopped arriving
    with pytest.raises(SiloError) as info:
        stack.client.get(sandbox.id)
    assert info.value.status_code == 503
    assert not resource_not_found(info.value, "sandbox")  # the pipeline must not think it is gone
    request = urllib.request.Request(
        f"{stack.broker_server.url}/sandboxes/{sandbox.id}", headers={HEADER_API_TOKEN: bearer("api-token")}
    )
    with pytest.raises(urllib.error.HTTPError) as raw:
        urllib.request.urlopen(request, timeout=10)
    assert raw.value.code == 503 and raw.value.headers["Retry-After"]
    stack.broker.heartbeat("h1", stack.host.url, stack.agent.capacity(), [sandbox.id])
    assert stack.client.get(sandbox.id).id == sandbox.id


def test_export_then_restore_carries_snapshots_across_a_broker_replacement(stack, tmp_path, monkeypatch):  # noqa: F811
    stack.client.snapshot.create(snapshot_params("carry-a"), timeout=30)
    stack.client.snapshot.create(snapshot_params("carry-b", cpu=2, memory=1, disk=1), timeout=30)
    before = {name: stack.broker.get_snapshot(name) for name in ("carry-a", "carry-b")}
    out = tmp_path / "broker-state.json"
    monkeypatch.setenv("SILO_BROKER_URL", stack.broker_server.url)
    monkeypatch.setenv("SILO_API_TOKEN", "api-token")
    export.main(["--out", str(out)])
    with pytest.raises(SystemExit):
        export.main(["--out", str(out)])  # never clobbers an existing document by default

    replacement = Broker(host_secret="host-secret", host_client_factory=host_client_factory("host-secret"))
    writable, counts = restore(replacement, FileStateStore(out))
    assert writable and counts["snapshots"] == 2
    for name, record in before.items():
        restored = replacement.get_snapshot(name)
        assert (restored.id, restored.state, restored.resolved_ref, restored.profile) == (
            record.id,
            record.state,
            record.resolved_ref,
            record.profile,
        )
