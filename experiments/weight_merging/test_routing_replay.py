# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
import io
import json

import numpy as np
import pytest

from experiments.weight_merging.routing_replay import replay_request, streaming_trajectory, summarize_replay


def test_stream_replay_preserves_literal_tokens_and_original_phase_boundary():
    # Text can conceal control tokens; only literal IDs reconstruct the model input.
    chunks = [
        {"prompt_token_ids": [128000, 41], "choices": [{"index": 0, "token_ids": [], "delta": {"role": "assistant"}}]},
        {"choices": [{"index": 0, "token_ids": [128005, 42], "delta": {"content": "tool"}}]},
        {"choices": [{"index": 0, "token_ids": [128009], "finish_reason": "stop", "delta": {}}]},
        {"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 3}},
    ]
    wire = (
        b"".join(
            ("data: " + json.dumps({"id": "original", "model": "parent", **chunk}) + "\r\n\r\n").encode()
            for chunk in chunks
        )
        + b"data: [DONE]\r\n\r\n"
    )
    trajectory = streaming_trajectory(wire)
    assert replay_request(trajectory)["prompt"] == [128000, 41, 128005, 42]
    buffer = io.BytesIO()
    np.save(buffer, np.array([[[0, 1]], [[0, 2]], [[2, 3]], [[2, 3]]], dtype=np.uint8))
    payload = {
        "id": "replay",
        "choices": [
            {
                "prompt_token_ids": [128000, 41, 128005, 42],
                "token_ids": [77],
                "routed_experts": base64.b64encode(buffer.getvalue()).decode(),
            }
        ],
    }
    summary = summarize_replay(payload, trajectory, num_layers=1, num_experts=4, top_k=2)
    assert summary["phases"] == {
        "prompt": {"forwarded_tokens": 2, "selection_counts": [[2, 1, 1, 0]]},
        "generated": {"forwarded_tokens": 2, "selection_counts": [[0, 0, 2, 2]]},
    }
    # An interrupted stream must not silently become a shorter replay trajectory.
    with pytest.raises(ValueError, match="Incomplete streaming"):
        streaming_trajectory(wire.removesuffix(b"data: [DONE]\r\n\r\n"))
