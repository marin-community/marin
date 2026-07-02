# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end smoke for :class:`finelog.forwarder.LogForwarder`.

Boots two real embedded finelog servers — a source (loopback-admitting default) and a
target fronted by a jwt-only auth policy — and drives the forwarder through the native
wire contract. Covers the federation relay's guarantees: a batch ingested at the source
is forwarded to the target under the same key and readable there with auth; and a forward
that fails (an unauthenticated push) advances no watermark, so a later authed forwarder
ships the same batch exactly once (durable, at-least-once, no loss).

Skips when the native extension is not built.
"""

import json

import pytest
from finelog.client import LogClient
from finelog.embedded import is_available, require_embedded_server
from finelog.forwarder import CorruptForwarderStateError, LogForwarder
from finelog.rpc import logging_pb2
from rigging.auth import BearerTokenInjector, StaticTokenProvider

# A pinned HS256 delegation key + token (exp = 2100-01-01), matching the Rust verifier's
# interop test. A static token keeps this test free of a JWT-minting dependency; token
# refresh/minting is covered separately (iris `_DelegationTokenProvider`, Rust interop).
_KEY = "delegation-key-0123456789abcdefX"
_TOKEN = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
    ".eyJzdWIiOiJtYXJpbiIsInJvbGUiOiJmaW5lbG9nLXJlbGF5IiwianRpIjoiZml4ZWRqdGkwMDAxIiwiaWF0IjoxMDAwMDAwMDAwLCJleHAiOjQxMDI0NDQ4MDB9"
    ".kTVu3jf6JUbqdHe8WYswdHWzw7WBNT1NfyCxtMoiaPE"
)
_JWT_POLICY = json.dumps([{"type": "jwt", "keys": [{"cluster": "marin", "secret": _KEY}]}])


def _entries(n: int, base_ms: int = 1_000) -> list[logging_pb2.LogEntry]:
    return [
        logging_pb2.LogEntry(
            timestamp=logging_pb2.Timestamp(epoch_ms=base_ms + i),
            source="stdout",
            data=f"line {i}",
        )
        for i in range(n)
    ]


def _authed_client(address: str) -> LogClient:
    return LogClient.connect(address, interceptors=(BearerTokenInjector(StaticTokenProvider(_TOKEN), "authorization"),))


@pytest.fixture
def servers(tmp_path):
    if not is_available():
        pytest.skip("finelog native server extension (finelog_server) not available")
    embedded = require_embedded_server()
    source = embedded(log_dir=str(tmp_path / "source"))  # default policy admits loopback
    target = embedded(log_dir=str(tmp_path / "target"), auth_policy=_JWT_POLICY)  # jwt-only
    try:
        yield source, target
    finally:
        source.stop()
        target.stop()


def _read_key(reader: LogClient, key: str) -> list[logging_pb2.LogEntry]:
    resp = reader.fetch_logs(
        logging_pb2.FetchLogsRequest(
            source=key,
            match_scope=logging_pb2.MATCH_SCOPE_PREFIX,
            max_lines=100,
        )
    )
    return list(resp.entries)


def test_forwarder_forwards_with_auth(tmp_path, servers):
    source_server, target_server = servers
    state = tmp_path / "watermark.json"
    source = LogClient.connect(source_server.address)
    target = _authed_client(target_server.address)
    forwarder = LogForwarder(
        source=source,
        target=target,
        target_label=target_server.address,
        state_path=state,
    )

    # First tick seeds the watermark at the (empty) source's max cursor; forwards nothing.
    assert forwarder.forward_once() == 0
    assert "cursor" in json.loads(state.read_text())

    key = "/user/job-1/task-0:0"
    writer = LogClient.connect(source_server.address)
    writer.push_batch(key, _entries(5))
    writer.close()

    # Next tick forwards the newly-ingested batch under the same key.
    assert forwarder.forward_once() == 5

    reader = _authed_client(target_server.address)
    entries = _read_key(reader, key)
    assert {e.data for e in entries} == {f"line {i}" for i in range(5)}
    assert all(e.key == key for e in entries)

    reader.close()
    source.close()
    target.close()


def test_forwarder_retries_after_failed_push_without_loss(tmp_path, servers):
    source_server, target_server = servers
    state = tmp_path / "watermark.json"
    source = LogClient.connect(source_server.address)

    # Seed, then ingest a batch at the source.
    good_target = _authed_client(target_server.address)
    seeder = LogForwarder(source=source, target=good_target, target_label=target_server.address, state_path=state)
    assert seeder.forward_once() == 0  # seed

    key = "/user/job-2/task-0:0"
    writer = LogClient.connect(source_server.address)
    writer.push_batch(key, _entries(3))
    writer.close()

    watermark_before = json.loads(state.read_text())["cursor"]

    # A forwarder with no credential: the jwt-only target rejects the push, so the
    # watermark must not advance (the batch is not lost — it stays pending).
    unauthed_target = LogClient.connect(target_server.address)
    failing = LogForwarder(
        source=source,
        target=unauthed_target,
        target_label=target_server.address,
        state_path=state,
    )
    assert failing.forward_once() == 0
    assert json.loads(state.read_text())["cursor"] == watermark_before, "failed push must not advance"

    # A fresh authed forwarder over the same state file (a restart) ships the pending
    # batch — exactly once, no loss and no duplication from the failed attempt.
    recovered_target = _authed_client(target_server.address)
    recovered = LogForwarder(
        source=source,
        target=recovered_target,
        target_label=target_server.address,
        state_path=state,
    )
    assert recovered.forward_once() == 3

    reader = _authed_client(target_server.address)
    entries = _read_key(reader, key)
    assert {e.data for e in entries} == {f"line {i}" for i in range(3)}
    assert len(entries) == 3, "batch forwarded exactly once despite the earlier failure"

    reader.close()
    unauthed_target.close()
    good_target.close()
    recovered_target.close()
    source.close()


def test_forwarder_refuses_corrupt_state_instead_of_skipping(tmp_path, servers):
    source_server, target_server = servers
    state = tmp_path / "watermark.json"
    source = LogClient.connect(source_server.address)
    target = _authed_client(target_server.address)
    forwarder = LogForwarder(source=source, target=target, target_label=target_server.address, state_path=state)
    assert forwarder.forward_once() == 0  # seed at the empty source

    # Ingest a batch that is now pending forwarding.
    writer = LogClient.connect(source_server.address)
    writer.push_batch("/user/job-3/task-0:0", _entries(4))
    writer.close()

    # A corrupt (not absent) state file must NOT be read as a first run: seeding at the
    # source's current max here would skip the pending batch forever.
    state.write_text("{ truncated json")
    with pytest.raises(CorruptForwarderStateError):
        forwarder.forward_once()
    # The forwarder refused to advance — the corrupt file is left untouched, not reseeded.
    assert state.read_text() == "{ truncated json"

    source.close()
    target.close()


def test_forwarder_drains_backlog_beyond_one_batch(tmp_path, servers):
    # A backlog larger than batch_lines must fully drain in one pass, not ship only
    # batch_lines per poll interval — that cap would let a busy cluster outrun the
    # forwarder until source retention drops the un-forwarded rows (silent loss).
    source_server, target_server = servers
    state = tmp_path / "watermark.json"
    source = LogClient.connect(source_server.address)
    target = _authed_client(target_server.address)
    forwarder = LogForwarder(
        source=source,
        target=target,
        target_label=target_server.address,
        state_path=state,
        batch_lines=2,
    )
    assert forwarder.forward_once() == 0  # seed

    key = "/user/job-7/task-0:0"
    writer = LogClient.connect(source_server.address)
    writer.push_batch(key, _entries(5))
    writer.close()

    # batch_lines=2 over a 5-row backlog → 3 fetch cycles (2+2+1); one drain ships all 5.
    assert forwarder._drain() == 5
    reader = _authed_client(target_server.address)
    assert len(_read_key(reader, key)) == 5

    reader.close()
    source.close()
    target.close()
