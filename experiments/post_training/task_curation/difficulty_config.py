# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared campaign and locked-worker configuration."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

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

# The standing judge ruling (2026-11-24): a source's own LLM judge this lane cannot
# serve is re-declared to the served Together GLM-5.3-Flash route, and the protocol
# difference is recorded on the source declaration. The substitution is ruled, not
# qualified by a fixture-judgement campaign; the route still carries the tokenizer
# pins verified at the pinned revision so serving and receipts stay exact.
SUBSTITUTE_JUDGE_MODEL = "zai-org/GLM-5.3-Flash"
SUBSTITUTE_JUDGE_PIN = DeclaredJudgePin(
    revision="46a050bec13002ad29ee795c16e61cf4fa3c4a9c",
    template_sha256="2614a3c896d53e265f564c23701173c0049f985dacfbae79fc4bcd9f5ba083f4",
)


@dataclass(frozen=True)
class JudgeRouteRepair:
    """A versioned re-declaration of one source's unserved judge to a served route.

    ``declared`` is the frozen bundle's env-template value, ``resolved_model`` the
    model that template resolves to, and ``served_model`` the served route the
    standing ruling re-declares it to. ``template_var`` names the bundle's own
    template variable so serving can pin the re-declared judge through it.
    """

    source_id: str
    version: str
    declared: str
    resolved_model: str
    served_model: str
    template_var: str
    note: str


JUDGE_ROUTE_REPAIRS: tuple[JudgeRouteRepair, ...] = (
    JudgeRouteRepair(
        source_id="TaskTrove:laion__nemotron-gym-multichallenge-vanilla-v3",
        version="2026.11.24.1",
        declared="${JUDGE_MODEL:-openai/gpt-4o-mini}",
        resolved_model="openai/gpt-4o-mini",
        served_model=SUBSTITUTE_JUDGE_MODEL,
        template_var="JUDGE_MODEL",
        note=(
            "Judge route repair 2026.11.24.1 (standing ruling: source LLM judges this lane cannot "
            "serve are re-declared to the served Together GLM-5.3-Flash route): the frozen verifier "
            "declares its judge as the env template '${JUDGE_MODEL:-openai/gpt-4o-mini}', which "
            "resolves to the unserved openai/gpt-4o-mini route. The judge is re-declared to "
            "together_ai/zai-org/GLM-5.3-Flash and served at the protocol temperature 0.7. Protocol "
            "difference: grading runs on GLM-5.3-Flash instead of the source's declared judge; the "
            "frozen package bytes and the original declaration are retained."
        ),
    ),
)


JUDGE_ROUTE_REPAIRS_BY_RESOLVED_MODEL: Mapping[str, JudgeRouteRepair] = MappingProxyType(
    {repair.resolved_model: repair for repair in JUDGE_ROUTE_REPAIRS}
)
JUDGE_ROUTE_REPAIRS_BY_SOURCE: Mapping[str, JudgeRouteRepair] = MappingProxyType(
    {repair.source_id: repair for repair in JUDGE_ROUTE_REPAIRS}
)


def judge_route_repair(resolved_model: str) -> JudgeRouteRepair | None:
    """The standing-ruling repair serving one resolved unserved judge model.

    The substitution is keyed by the model the declared judge resolves to, so the
    source gate, the reconcile's judge repair and the runtime resolve one judge
    model to exactly one served route.
    """
    return JUDGE_ROUTE_REPAIRS_BY_RESOLVED_MODEL.get(resolved_model)


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
