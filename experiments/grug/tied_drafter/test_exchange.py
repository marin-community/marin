# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from experiments.grug.tied_drafter.exchange import (
    Exchange,
    decode_packet,
    encode_packet,
    exchange_session,
    serve_exchange,
)


def test_remote_draft_receives_snapshot_and_returns_matching_displacement():
    exchange = Exchange()
    snapshot = [np.arange(24, dtype=np.float32).reshape(2, 3, 4), np.array(7.0, dtype=np.float32)]
    metadata = {"generation": 13, "snapshot_step": 256, "horizon": 16}
    with serve_exchange(exchange, host="127.0.0.1", port=0, token="test-token") as port:
        address = f"http://127.0.0.1:{port}"
        with exchange_session("test-token") as client:
            client.post(address + "/ready", timeout=5).raise_for_status()
            assert exchange.ready.is_set()
            assert exchange.poll() is None
            exchange.offer(13, encode_packet(metadata, snapshot))
            response = client.get(address + "/snapshot", timeout=5)
            response.raise_for_status()
            actual_metadata, actual_arrays = decode_packet(response.content)
            assert actual_metadata == metadata
            for expected, actual in zip(snapshot, actual_arrays, strict=True):
                np.testing.assert_array_equal(expected, actual)
            displacement = [np.full_like(x, 0.125) for x in actual_arrays]
            client.post(
                address + "/proposal",
                data=encode_packet(metadata, displacement),
                headers={"X-Generation": "13"},
                timeout=5,
            ).raise_for_status()
            received_metadata, received_arrays = decode_packet(exchange.poll())
            assert received_metadata["snapshot_step"] == 256
            for expected, actual in zip(displacement, received_arrays, strict=True):
                np.testing.assert_array_equal(expected, actual)
            assert exchange.poll() is None
            exchange.offer(14, encode_packet({**metadata, "generation": 14}, snapshot))
            exchange.closed.set()
            assert client.get(address + "/snapshot", timeout=5).status_code == 410
            client.post(address + "/done", timeout=5).raise_for_status()
            assert exchange.finished.is_set()
        with exchange_session("wrong-token") as outsider:
            assert outsider.get(address + "/status", timeout=5).status_code == 403
