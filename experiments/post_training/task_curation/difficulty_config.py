# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared campaign and locked-worker configuration."""

from finestore.rl.difficulty import SamplingSettings, Solver, SolverIdentity
from pydantic import BaseModel, ConfigDict

from experiments.post_training.task_curation.difficulty_provider import ProviderLimits


class JudgeRun(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    endpoint: str
    api_model: str
    api_key_secret: str
    identity: SolverIdentity
    provider_limits: ProviderLimits


class CandidateRun(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    solver: Solver
    identity: SolverIdentity
    settings: SamplingSettings
    endpoint: str
    api_model: str
    api_key_secret: str | None
    provider_limits: ProviderLimits
    environment_options: dict
    judge: JudgeRun | None
    workers: int
    gpu_count: int
    attempts_per_task: int
    task_limit: int | None
