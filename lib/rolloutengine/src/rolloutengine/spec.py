# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Serializable choices that run one task: the harness, the machines, the limits, and the trace mode."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field
from shellbox.machine import NetworkPolicy
from taskcompendium.models import TaskSpec


class TraceMode(StrEnum):
    """The evidence a rollout records for each model turn."""

    EXACT_TOKENS = "exact_tokens"
    """Served token IDs, loss masks, and log probabilities under the exact-token contract."""
    MESSAGES = "messages"
    """Unchecked, informational token IDs with an all-zero loss mask; never train on these rollouts."""


TRACE_METRIC = "trace"
"""``RolloutData.metrics`` key that marks every ``TraceMode.MESSAGES`` record with the mode's value."""


class MachineRuntimeSpec(BaseModel):
    """Deployment settings for one Shellbox machine."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    backend: str = Field(min_length=1)
    network: NetworkPolicy
    cpus: int | None = Field(gt=0)
    memory_mb: int | None = Field(gt=0)
    storage_mb: int | None = Field(gt=0)
    gpus: int = Field(ge=0)
    user: str | None
    startup_timeout: float | None = Field(gt=0)
    cleanup_timeout: float | None = Field(gt=0)


class RolloutLimits(BaseModel):
    """Turn, token, and time limits for one attempt.

    ``max_total_tokens`` stops model requests before one whose prompt alone would bring the
    cumulative prompt plus response tokens to the threshold; the last request sent may exceed it
    by its new observation and response tokens. ``None`` sets no threshold.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    max_turns: int = Field(gt=0)
    max_total_tokens: int | None = Field(gt=0)
    model_turn_timeout: float | None = Field(gt=0)
    tool_turn_timeout: float | None = Field(gt=0)
    total_turn_timeout: float | None = Field(gt=0)
    attempt_timeout: float | None = Field(gt=0)
    verifier_timeout: float | None = Field(gt=0)
    cleanup_timeout: float = Field(gt=0)


class RolloutSpec(BaseModel):
    """An unchanged task definition and the choices that run it.

    ``harness`` names a harness registered with the engine. A machine selection of ``None``
    supplies no machine.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    task: TaskSpec
    harness: str = Field(min_length=1)
    task_machine: MachineRuntimeSpec | None
    verifier_machine: MachineRuntimeSpec | None
    limits: RolloutLimits
    trace: TraceMode = TraceMode.EXACT_TOKENS
