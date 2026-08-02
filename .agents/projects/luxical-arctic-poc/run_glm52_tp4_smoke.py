# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test whether the pinned GLM-5.2 server fits on one four-B200 node."""

import json
import logging

from glm_semantic_labels import completion

from experiments.rollout_data.glm52_vllm import Glm52LaunchConfig, ServerConfig, serve_glm52

TENSOR_PARALLEL_SIZE = 4
RAY_PORT = 6_379
HTTP_PORT = 8_000
MAX_MODEL_LEN = 16 * 1_024
MAX_NUM_SEQS = 4


def smoke(vllm_url: str) -> None:
    """Run one bounded JSON completion after server readiness."""
    result = completion(
        vllm_url,
        [{"role": "user", "content": 'Return only this JSON object: {"ready": true}'}],
        max_tokens=32,
        seed=42,
    )
    if result != {"ready": True}:
        raise ValueError(f"GLM smoke returned an unexpected value: {result}")
    logging.info("GLM52_TP4_SMOKE=%s", json.dumps(result, sort_keys=True))


def main() -> None:
    """Serve the four-device model and run one response smoke."""
    logging.basicConfig(level=logging.INFO)
    launch = Glm52LaunchConfig(
        vllm_endpoint="glm52-tp4-smoke",
        ray_endpoint="glm52-tp4-smoke-ray",
        server=ServerConfig(max_model_len=MAX_MODEL_LEN, max_num_seqs=MAX_NUM_SEQS),
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        client=smoke,
    )
    serve_glm52(launch, RAY_PORT, HTTP_PORT)


if __name__ == "__main__":
    main()
