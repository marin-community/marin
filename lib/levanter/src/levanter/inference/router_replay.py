# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from levanter.inference import inference_pb2


def read_router_replay(
    response: inference_pb2.GenerateResponse, choice_index: int, input_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return expert IDs and the validity mask for exact learner input tokens."""
    choice = response.choices[choice_index]
    if not choice.HasField("router_replay"):
        raise ValueError("Response has no captured routes for this choice")

    tokens = [*response.prompt_token_ids, *choice.token_ids]
    if input_ids.ndim != 1 or not np.array_equal(tokens, input_ids):
        raise ValueError("Captured token IDs differ from the learner input")

    replay = choice.router_replay
    layers, positions, slots = replay.num_layers, replay.num_positions, replay.experts_per_token
    if (
        not layers
        or not slots
        or positions != input_ids.size
        or len(replay.valid) != positions
        or len(replay.expert_ids) != layers * positions * slots
    ):
        raise ValueError("Invalid expert routes or token validity in generation response")

    experts = np.fromiter(replay.expert_ids, dtype=np.uint32).reshape(layers, positions, slots)
    return experts, np.fromiter(replay.valid, dtype=bool)
