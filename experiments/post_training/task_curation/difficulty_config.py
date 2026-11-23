# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared campaign and locked-worker configuration."""

from dataclasses import dataclass

from finestore.rl.difficulty import SamplingSettings, Solver, SolverIdentity
from pydantic import BaseModel, ConfigDict

from experiments.post_training.task_curation.difficulty_provider import ProviderLimits

DATA_ACCOUNT = "data"
SELF_HOSTED_SOLVERS = (Solver.SMALL, Solver.LARGE, Solver.SNOWBALL)
SOLVER_POOL_GPU_COUNT = 64
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


@dataclass(frozen=True)
class DeclaredJudgePin:
    """Tokenizer pins for a frozen bundle's own declared grader model."""

    revision: str
    template_sha256: str


# Judges a frozen native bundle may declare (REWARDKIT_JUDGE) that this lane
# serves on the shared Together route. Membership IS the qualification: each
# entry carries the tokenizer pins verified against the pinned revision and the
# fixture-judgement evidence from its qualification run
# (s3://marin-us-east-02a/marin/rl-data/atlas/difficulty-reconciler/validation/
# qwen-judge-qualification-20261123/evidence.json). An unlisted declared judge
# fails the wave closed instead of being silently re-judged by another model.
# This registry lives in the pure contract layer: Harbor's isolated trial
# process and the controller both import it, and neither has the marin package.
DECLARED_JUDGE_PINS = {
    "Qwen/Qwen3.5-9B": DeclaredJudgePin(
        revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
        template_sha256="a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
    ),
}
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
