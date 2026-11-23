# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared campaign and locked-worker configuration."""

from finestore.rl.difficulty import SamplingSettings, Solver, SolverIdentity
from pydantic import BaseModel, ConfigDict

from experiments.post_training.task_curation.difficulty_provider import ProviderLimits

DATA_ACCOUNT = "data"
TOGETHER_API_KEY_ENV = "TOGETHER_API_KEY"
# Provider credential env names a task bundle may resolve from its granted
# guest environment. Each resolves from the GSM-backed secret bundle at wave
# start; a bundle naming any other provider credential fails the wave closed
# at dispatch.
TASK_PROVIDER_SECRETS: frozenset[str] = frozenset({TOGETHER_API_KEY_ENV})
# The frozen verifier env var a native bundle uses to name its own grader
# model, in LiteLLM provider form (for example together_ai/Qwen/Qwen3.5-9B).
DECLARED_JUDGE_ENV_VAR = "REWARDKIT_JUDGE"
DECLARED_JUDGE_PROVIDER_PREFIX = "together_ai/"
# LiteLLM's Together base-URL override; the declared judge's calls read it, so
# the wave points it at Harbor's in-guest relay to keep the credential host-side.
TOGETHER_AI_API_BASE_ENV = "TOGETHER_AI_API_BASE"
SNOWBALL_VLLM_RELEASE_TAG = "marin-vllm-gpu-20260924-01911be34fac"


class JudgeRun(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    endpoint: str
    api_model: str
    # A locally hosted tier judge needs no credential; the worker authenticates "EMPTY".
    api_key_secret: str | None
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
    # Daytona tasks provision through this mode: snapshot names a pre-created
    # RL-account snapshot; declarative builds the original image or Dockerfile
    # at sandbox creation and needs an account that permits declarative builds.
    daytona_provisioning: str = "snapshot"
    judge: JudgeRun | None
    workers: int
    gpu_count: int
    attempts_per_task: int
