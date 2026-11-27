# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from levanter.inference import inference_pb2
from levanter.inference.router_replay import read_router_replay, write_router_replay


@pytest.mark.parametrize(
    ("capture_dtype", "values", "wire_dtype", "wire_bytes"),
    [
        (np.uint8, (0, 255), np.dtype("u1"), b"\x00\xff"),
        (np.int32, (256, 65535), np.dtype("<u2"), b"\x00\x01\xff\xff"),
    ],
)
def test_router_replay_round_trip(capture_dtype, values, wire_dtype, wire_bytes):
    experts = np.array(values, dtype=capture_dtype).reshape(1, 2, 1)
    valid = np.array([True, False])
    response = inference_pb2.GenerateResponse(prompt_token_ids=[10])
    choice = response.choices.add(token_ids=[20])
    write_router_replay(choice.router_replay, experts, valid)

    received = inference_pb2.GenerateResponse.FromString(response.SerializeToString())
    assert received.choices[0].router_replay.expert_ids == wire_bytes
    ids, mask = read_router_replay(received, 0, np.array([10, 20]))
    assert ids.dtype == wire_dtype
    np.testing.assert_array_equal(ids, experts)
    np.testing.assert_array_equal(mask, valid)

    received.choices[0].router_replay.expert_ids = wire_bytes[:-1]
    with pytest.raises(ValueError, match="Invalid expert routes"):
        read_router_replay(received, 0, np.array([10, 20]))
