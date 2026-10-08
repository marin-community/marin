# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import base64
import io

import numpy as np
import pytest

from experiments.weight_merging.routing_capture import summarize_routing


def test_prompt_and_generated_routes_use_forwarded_token_boundary():
    routes = np.array([[[0, 1], [2, 3]], [[0, 2], [1, 3]], [[2, 3], [0, 1]]], dtype=np.uint8)
    buffer = io.BytesIO()
    np.save(buffer, routes)
    payload = {
        "id": "capture",
        "model": "test",
        "prompt_token_ids": [10, 11],
        "choices": [
            {
                "token_ids": [12, 13],
                "routed_experts": base64.b64encode(buffer.getvalue()).decode(),
                "finish_reason": "stop",
            }
        ],
    }
    summary = summarize_routing(payload, num_layers=2, num_experts=4, top_k=2)
    assert summary["phases"] == {
        "prompt": {"forwarded_tokens": 2, "selection_counts": [[2, 1, 1, 0], [0, 1, 1, 2]]},
        "generated": {"forwarded_tokens": 1, "selection_counts": [[0, 0, 1, 1], [1, 1, 0, 0]]},
    }
    # A missing prompt row must not silently shift the generated-token boundary.
    payload["prompt_token_ids"].append(14)
    with pytest.raises(ValueError, match="Incomplete routing capture"):
        summarize_routing(payload, num_layers=2, num_experts=4, top_k=2)
