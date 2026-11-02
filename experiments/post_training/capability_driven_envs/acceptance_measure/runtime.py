# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Turn a resolved arm (``treatment.Treatment``) into the objects the stack runs: one place, used by the live driver and
the dry run alike, so the dry run renders exactly what a launch would run.

The settings every arm shares and that the arm files do not vary (the infrastructure retry, the rollout limits, the
sampling of each role) are constants here and appear in ``manifest.json`` under ``shared``.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from rolloutengine.models import RolloutLimits
from taskforge.llm.policy import LLMPolicy
from taskforge.stage40_build_with_program.agentic_author import AgenticAuthorConfig, SameSessionBounds
from taskforge.stage40_build_with_program.services import AgentLimits
from taskforge.stage50_validate.adversary import AdversaryMode, AdversaryPolicy, SearchRole
from taskforge.stage50_validate.trials import RetryBackoff
from taskforge.stage60_review.decision import BandOutcome
from taskforge.stage60_review.rules import BandRules
from taskforge.stage_coordinator.coordinator import WorkpieceBudget
from taskforge.stage_coordinator.run import InfrastructureRetry

from experiments.post_training.capability_driven_envs.acceptance_measure.author import AuthorTreatment
from experiments.post_training.capability_driven_envs.acceptance_measure.review import ReviewTreatment
from experiments.post_training.capability_driven_envs.acceptance_measure.shadows import read_taxonomy
from experiments.post_training.capability_driven_envs.acceptance_measure.treatment import Treatment

PACKAGE = Path(__file__).resolve().parent
TAXONOMY = PACKAGE / "kit" / "taxonomy.json"
ROLES = ("author", "build", "controls", "adversary", "solver", "shadow_judge")

SAMPLING = LLMPolicy()
"""Every role samples as the model publishes (temperature 1.0, top_p 0.95, high reasoning), as the frozen run did."""
INFRASTRUCTURE_RETRY = InfrastructureRetry(
    max_attempts=4, backoff_initial=30.0, backoff_maximum=600.0, backoff_factor=2.0
)
BUILD_AGENT_LIMITS = AgentLimits(max_turns=None, max_tokens=None, max_calls=None)
TRIAL_LIMITS = RolloutLimits(
    max_turns=None,
    max_total_tokens=None,
    model_turn_timeout=1800,
    tool_turn_timeout=120,
    total_turn_timeout=None,
    attempt_timeout=3600,
    verifier_timeout=None,
    cleanup_timeout=60,
)


@dataclass(frozen=True)
class ArmRuntime:
    name: str
    models: dict[str, str]
    author_config: AgenticAuthorConfig
    author: AuthorTreatment
    review: ReviewTreatment
    budget: WorkpieceBudget
    adversaries: tuple[AdversaryPolicy, ...]
    solver_k: int
    max_retries: int
    retry_backoff: RetryBackoff
    command_timeout: float
    build_wall_timeout: float
    shadow_l3: bool
    shadow_l4: bool
    judge_prompt: str | None
    judge_max_claims: int


def arm_runtime(treatment: Treatment) -> ArmRuntime:
    t = treatment
    band = BandRules(low=t["review.band.low"], high=t["review.band.high"], accepted=frozenset({BandOutcome.IN_BAND}))
    adversaries = tuple(
        AdversaryPolicy(
            role=SearchRole(a["role"]),
            k=a["k"],
            submissions=a["submissions"],
            max_turns=a["max_turns"],
            sampling=SAMPLING,
            mode=AdversaryMode(a["mode"]),
            min_probes=a["min_probes"],
        )
        for a in t["validation.adversaries"]
    )
    return ArmRuntime(
        name=t.name,
        models={role: t[f"models.{role}"] for role in ROLES},
        author_config=AgenticAuthorConfig(
            turns=t["author.turns"],
            turn_max_tokens=t["author.turn_max_tokens"],
            session_token_cap=t["author.session_token_cap"],
            revision_same_session=SameSessionBounds(
                t["author.revision_same_session.max_prompt_tokens"], t["author.revision_same_session.max_turns_used"]
            ),
            max_continuations=t["author.max_continuations"],
        ),
        author=AuthorTreatment(
            guidance=t.texts_of("author.guidance"),
            reference_files=t.files_of("author.reference_files"),
            tools=frozenset(t["author.tools"]),
            enforce_battery=t["author.probe_battery.enforce_at_submit"],
            witness_replay=t["author.probe_battery.witness_replay"],
            battery_max_refusals=t["author.probe_battery.max_refusals"],
            unchanged_text=t.text("author.unchanged_resubmit.text") or "",
            unchanged_max_refusals=t["author.unchanged_resubmit.max_refusals"],
            repeated_error_text=t.text("author.repeated_error.text") or "",
            hint_after=t["author.repeated_error.hint_after"],
            hold_after=t["author.repeated_error.hold_after"],
        ),
        review=ReviewTreatment(
            band=band,
            solver_k=t["validation.solver_k"],
            saturated_accept=t["review.saturated_accept"],
            class_feedback=t.text("review.class_feedback"),
            taxonomy=read_taxonomy(TAXONOMY),
        ),
        budget=WorkpieceBudget(
            max_triage_repairs=t["budget.max_triage_repairs"],
            max_build_revisions=t["budget.max_build_revisions"],
            max_validation_retries=t["budget.max_validation_retries"],
            max_modify_rounds=t["budget.max_modify_rounds"],
        ),
        adversaries=adversaries,
        solver_k=t["validation.solver_k"],
        max_retries=t["validation.max_retries"],
        retry_backoff=RetryBackoff(**t["validation.retry_backoff"]),
        command_timeout=t["validation.command_timeout"],
        build_wall_timeout=t["build.wall_timeout"],
        shadow_l3=t["shadow.l3"],
        shadow_l4=t["shadow.l4"],
        judge_prompt=t.text("shadow.judge.prompt") if t["shadow.judge.enabled"] else None,
        judge_max_claims=t["shadow.judge.max_claims_per_component"],
    )


def shared_settings() -> dict:
    """The settings identical in every arm by construction, for ``manifest.json``."""
    return json.loads(
        json.dumps(
            {
                "sampling": asdict(SAMPLING),
                "infrastructure_retry": asdict(INFRASTRUCTURE_RETRY),
                "build_agent_limits": asdict(BUILD_AGENT_LIMITS),
                "trial_limits": TRIAL_LIMITS.model_dump(mode="json"),
                "taxonomy": str(TAXONOMY.relative_to(PACKAGE)),
            },
            default=str,
        )
    )
