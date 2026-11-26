"""Literal trace capture utilities.

Provides a recording proxy that sits in front of an OpenAI-compatible
(vLLM) endpoint, injects the parameters needed to recover literal token
IDs and logprobs, and records request/response pairs to a JSONL file.

This is the capture-side counterpart to ``harbor traces export --literal``:
installed agents that shell out to a CLI (opencode, claude-code, codex, …)
point their base URL at the proxy, which transparently enriches every
completion so the resulting trajectory.json carries literal token data.
"""

from harbor.literal.proxy import (
    CORRELATION_ID_HEADER,
    LITERAL_LOG_FILENAME,
    RecordProxy,
    extract_literal_from_response,
    inject_literal_params,
)
from harbor.literal.rollout_build import (
    build_rollout_details_for_trial,
    build_rollout_details_from_pairs,
    read_literal_pairs,
)

__all__ = [
    "CORRELATION_ID_HEADER",
    "LITERAL_LOG_FILENAME",
    "RecordProxy",
    "build_rollout_details_for_trial",
    "build_rollout_details_from_pairs",
    "extract_literal_from_response",
    "inject_literal_params",
    "read_literal_pairs",
]
