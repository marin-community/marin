# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import os
import logging

import numpy as np
from levanter.models.lm_model import LmHeadModel

from marin.rl.environments.inference_ctx.vllm import InferenceMode, vLLMInferenceContext, vLLMInferenceContextConfig

logger = logging.getLogger(__name__)


# Allow vLLM to serialize custom types needed for async inference
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"


def serialize_state_dict_for_rpc(state_dict: dict) -> dict:
    """Serialize numpy arrays to (bytes, dtype, shape) tuples for RPC transfer.

    vLLM's collective_rpc can corrupt numpy arrays during serialization.
    This converts them to a format that survives pickling.
    """
    serialized = {}
    for key, value in state_dict.items():
        if isinstance(value, np.ndarray):
            serialized[key] = (value.tobytes(), str(value.dtype), value.shape)
        else:
            # Already serializable (or will fail later with a clear error)
            serialized[key] = value
    return serialized


class AsyncvLLMInferenceContext(vLLMInferenceContext):
    """Inference context for async vLLM."""

    def __init__(self, inference_config: vLLMInferenceContextConfig):
        inference_config.mode = InferenceMode.ASYNC
        super().__init__(inference_config)

    def reload_model(self, model: LmHeadModel | None, state_dict: dict) -> None:
        # Serialize numpy arrays to (bytes, dtype, shape) tuples to survive RPC serialization.
        # vLLM's collective_rpc can corrupt numpy arrays during pickling.
        serialized_state_dict = serialize_state_dict_for_rpc(state_dict)
        self.llm.update_weights(serialized_state_dict, self.model_name)
        self.llm.reset_prefix_cache()  # Reset prefix cache because of new weights

    def shutdown(self):
        self.llm.shutdown()

    def start_server(self, model: LmHeadModel) -> None:
        pass
