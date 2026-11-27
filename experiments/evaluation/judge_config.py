# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serializable Together serverless judge policy, also embedded in task packages."""

from dataclasses import dataclass
from enum import StrEnum


class JudgeProvider(StrEnum):
    TOGETHER_SERVERLESS = "together_serverless"


class ReasoningEffort(StrEnum):
    LOW = "low"
    HIGH = "high"
    MAX = "max"


@dataclass(frozen=True)
class JudgeConfig:
    """Judge transport, sampling, deadlines and versioned credential binding."""

    provider: JudgeProvider
    model: str
    base_url: str
    temperature: float
    top_p: float
    reasoning_effort: ReasoningEffort
    token_budgets: tuple[int, ...]
    request_timeout: float
    transport_budget: float
    transport_attempts: int
    bixbench_temperature: float
    bixbench_max_tokens: int
    bixbench_request_timeout: float
    bixbench_transport_attempts: int
    credential: str


MICRO210_JUDGE = JudgeConfig(
    provider=JudgeProvider.TOGETHER_SERVERLESS,
    model="zai-org/GLM-5.3-Flash",
    base_url="https://api.together.ai/v1",
    temperature=0.0,
    top_p=0.95,
    reasoning_effort=ReasoningEffort.LOW,
    token_budgets=(128, 512, 2048),
    request_timeout=300.0,
    transport_budget=1800.0,
    transport_attempts=6,
    bixbench_temperature=1.0,
    bixbench_max_tokens=1024,
    bixbench_request_timeout=600.0,
    bixbench_transport_attempts=3,
    credential="gcp-secret://projects/hai-gcp-models/secrets/TOGETHER_API_KEY/versions/1",
)
