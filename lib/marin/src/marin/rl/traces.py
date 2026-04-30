# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Trace contracts for RL/post-training environments."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from typing import Any


@dataclass(frozen=True)
class EpisodeResponseTrace:
    """Cold-path trace data for one sampled response."""

    response_text: str
    reward: float | None = None
    correctness_reward: float | None = None
    is_truncated: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeTrace:
    """Prompt-level trace data for one rollout group."""

    env_name: str
    env_example_id: str
    prompt: str | list[dict[str, str]]
    responses: tuple[EpisodeResponseTrace, ...]
    task_name: str = ""
    task_version: str = "v1"
    verifier_name: str | None = None
    verifier_version: str | None = None
    lesson_id: str = ""
    group_id: str = ""
    trace_id: str = ""
    trace_ref: str | None = None

    def with_runtime_metadata(
        self,
        *,
        lesson_id: str,
        group_id: str,
        trace_id: str,
        task_name: str,
        task_version: str,
        verifier_name: str | None,
        verifier_version: str | None,
    ) -> EpisodeTrace:
        """Return a copy with rollout-runtime metadata attached."""

        return replace(
            self,
            lesson_id=lesson_id,
            group_id=group_id,
            trace_id=trace_id,
            task_name=task_name,
            task_version=task_version,
            verifier_name=verifier_name,
            verifier_version=verifier_version,
        )


def make_group_id(run_id: str, lesson_id: str, weight_step: int, worker_id: str, group_index: int) -> str:
    """Build a stable group identifier for a rollout group."""

    seed = f"group:{run_id}:{lesson_id}:{weight_step}:{worker_id}:{group_index}"
    return f"group-{uuid.uuid5(uuid.NAMESPACE_URL, seed)}"


def make_trace_id(run_id: str, lesson_id: str, weight_step: int, worker_id: str, group_index: int) -> str:
    """Build a stable trace identifier for a rollout group."""

    seed = f"trace:{run_id}:{lesson_id}:{weight_step}:{worker_id}:{group_index}"
    return f"trace-{uuid.uuid5(uuid.NAMESPACE_URL, seed)}"
