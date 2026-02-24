# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Observation model for ATIF trajectories."""

from pydantic import BaseModel, Field

from harbor.models.trajectories.observation_result import ObservationResult


class Observation(BaseModel):
    """Environment feedback/result after actions or system events."""

    results: list[ObservationResult] = Field(
        default=...,
        description="Array of result objects from tool calls or actions",
    )

    model_config = {"extra": "forbid"}
