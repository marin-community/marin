"""Align vLLM MoE routes with generated-token log probabilities."""

import numpy as np


def response_routes(captured: np.ndarray | None, num_response_tokens: int) -> np.ndarray | None:
    """Select the route rows that predicted each generated token.

    vLLM returns prompt routes on the first sample, then one route row for
    each previously generated token on later samples. For R generated tokens,
    the last R rows therefore correspond to the R prediction positions.
    """
    if captured is None:
        return None
    if captured.ndim != 3 or not np.issubdtype(captured.dtype, np.integer):
        raise ValueError("vLLM routed experts must be an integer [tokens, layers, topk] array")
    if captured.shape[1] == 0 or captured.shape[2] == 0:
        raise ValueError("vLLM routed experts must include layers and top-k choices")
    if num_response_tokens > captured.shape[0]:
        raise ValueError(f"vLLM returned {captured.shape[0]} routed rows for {num_response_tokens} generated tokens")
    if num_response_tokens == 0:
        return captured[:0].copy()
    # Release the prompt capture without expanding each expert ID into a Python integer.
    return captured[-num_response_tokens:].copy()
