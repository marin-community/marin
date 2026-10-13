# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned candidates and conversation limits for the Snowball probe experiment."""

import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path


class ReasoningFormat(StrEnum):
    API_FIELDS = "api_fields"
    GRUG_INLINE = "grug_inline"


@dataclass(frozen=True)
class Candidate:
    key: str
    label: str
    checkpoint: str
    revision: str
    gpu_count: int
    tensor_parallel_size: int
    data_parallel_size: int
    reasoning_parser: str | None
    tool_call_parser: str
    reasoning_format: ReasoningFormat
    weights_revision: str | None = None

    def metadata(self) -> dict:
        return {**asdict(self), "model": self.key, "max_context": 65536, "thinking": True}


CANDIDATES = (
    Candidate(
        key="snowball-dr-doom",
        label="Snowball · Dr. Doom",
        checkpoint="open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21-dr-doom",
        revision="1dfdbb34b6bc72f7249605c0e2e88a854c9f343a",
        gpu_count=2,
        tensor_parallel_size=1,
        data_parallel_size=2,
        reasoning_parser=None,
        tool_call_parser="hermes",
        reasoning_format=ReasoningFormat.GRUG_INLINE,
    ),
    Candidate(
        key="qwen3.6-35b-a3b",
        label="Qwen 3.6 · 35B A3B",
        checkpoint="Qwen/Qwen3.6-35B-A3B",
        revision="995ad96eacd98c81ed38be0c5b274b04031597b0",
        gpu_count=2,
        tensor_parallel_size=2,
        data_parallel_size=1,
        reasoning_parser="qwen3",
        tool_call_parser="qwen3_coder",
        reasoning_format=ReasoningFormat.API_FIELDS,
    ),
)

WILDCHAT_DATASET = "allenai/WildChat-1M"
WILDCHAT_REVISION = "7d6490e462285cf85d91eabea0f9a954fbddcd1f"
TOKEN_CAP = 10_000_000_000
CONCURRENT_CHATS = 16
ORTET_RELAYS = {
    "cw-rno2a": "/muchanem/glm53-relay-rno2a",
    "cw-us-east-02a": "/muchanem/glm53-relay",
    "cw-us-east-08a": "/muchanem/glm53-relay-08a",
}
CATEGORIES = (
    "reasoning",
    "instruction_following",
    "factuality",
    "coding",
    "data_analysis",
    "tool_use",
    "planning",
    "state_tracking",
    "formatting",
    "repetition",
    "calibration",
)
BREAKAGE_CATEGORIES = (
    "incorrect_answer",
    "missed_instruction",
    "hallucination",
    "reasoning_loop",
    "truncation",
    "invalid_tool_call",
    "tool_misuse",
    "state_loss",
    "false_success",
    "format_violation",
    "unhelpful_refusal",
    "poor_calibration",
    "infrastructure",
    "other",
)


def load_candidates(path: Path | None) -> tuple[Candidate, ...]:
    if path is None:
        return CANDIDATES
    candidates = tuple(
        Candidate(**{**item, "reasoning_format": ReasoningFormat(item["reasoning_format"])})
        for item in json.loads(path.read_text())
    )
    if len(candidates) < 2 or len({candidate.key for candidate in candidates}) != len(candidates):
        raise ValueError("Choose at least two candidates with distinct keys")
    return candidates
