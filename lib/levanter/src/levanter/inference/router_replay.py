# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from levanter.inference import inference_pb2


def write_router_replay(replay: inference_pb2.RouterReplay, experts: np.ndarray, valid: np.ndarray) -> None:
    """Store captured expert IDs in row-major order with their wire dtype."""
    replay.num_layers, replay.num_positions, replay.experts_per_token = experts.shape
    if experts.dtype == np.uint8:
        replay.expert_id_dtype = inference_pb2.EXPERT_ID_DTYPE_UINT8
    else:
        if experts.min(initial=0) < 0 or experts.max(initial=0) > np.iinfo(np.uint16).max:
            raise ValueError("Expert IDs must fit in uint16")
        replay.expert_id_dtype = inference_pb2.EXPERT_ID_DTYPE_UINT16_LE
        experts = experts.astype("<u2", copy=False)
    replay.expert_ids = experts.tobytes()
    replay.valid.extend(valid.tolist())


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
    if replay.expert_id_dtype == inference_pb2.EXPERT_ID_DTYPE_UINT8:
        dtype = np.dtype("u1")
    elif replay.expert_id_dtype == inference_pb2.EXPERT_ID_DTYPE_UINT16_LE:
        dtype = np.dtype("<u2")
    else:
        raise ValueError("Unknown expert ID dtype in generation response")
    expert_bytes = replay.expert_ids
    if (
        not layers
        or not slots
        or positions != input_ids.size
        or len(replay.valid) != positions
        or len(expert_bytes) != layers * positions * slots * dtype.itemsize
    ):
        raise ValueError("Invalid expert routes or token validity in generation response")

    experts = np.frombuffer(expert_bytes, dtype=dtype).reshape(layers, positions, slots)
    return experts, np.fromiter(replay.valid, dtype=bool)
