# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Turn a resolved arm (``treatment.Treatment``) into the objects the stack runs: one place, used by the live driver and
the dry run alike, so the dry run renders exactly what a launch would run.

The settings every arm shares and that the arm files do not vary (the rollout limits and the sampling of each role)
are constants here and appear in ``manifest.json`` under ``shared``. The arm's policy settings and the driver's
``--policy`` overlays resolve into one ``Policy`` (``Treatment.policy``), which the coordinator, the author and the
adversary briefs read.
"""

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from rolloutengine.models import RolloutLimits
from taskforge.content_hash import digest
from taskforge.llm.policy import LLMPolicy
from taskforge.policy import Policy, Prompt
from taskforge.stage40_build_with_program.agentic_author import AgenticAuthorConfig, SameSessionBounds
from taskforge.stage40_build_with_program.services import AgentLimits
from taskforge.stage50_validate.adversary import AdversaryMode, AdversaryPolicy, AdversarySpec
from taskforge.stage50_validate.trials import RetryBackoff
from taskforge.stage60_review.decision import BandOutcome
from taskforge.stage60_review.rules import BandRules

from experiments.post_training.capability_driven_envs.acceptance_measure.author import AuthorTreatment
from experiments.post_training.capability_driven_envs.acceptance_measure.review import ReviewTreatment
from experiments.post_training.capability_driven_envs.acceptance_measure.shadows import read_taxonomy
from experiments.post_training.capability_driven_envs.acceptance_measure.treatment import Treatment

PACKAGE = Path(__file__).resolve().parent
TAXONOMY = PACKAGE / "kit" / "taxonomy.json"
ROLES = ("author", "build", "controls", "adversary", "solver", "shadow_judge")

SAMPLING = LLMPolicy()
"""Every role samples as the model publishes (temperature 1.0, top_p 0.95, high reasoning), as the frozen run did."""
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
    policy: Policy
    models: dict[str, str]
    author_config: AgenticAuthorConfig
    author: AuthorTreatment
    review: ReviewTreatment
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

    @property
    def band(self) -> BandRules:
        """The band review places a solve rate in, as ``StageCoordinator.band`` reads it from the same policy."""
        low, high = (self.policy.setting(key, float) for key in ("band.low", "band.high"))
        return BandRules(low, high, frozenset({BandOutcome.IN_BAND}))


def arm_runtime(treatment: Treatment, overlays: tuple[Mapping[str, object], ...] = ()) -> ArmRuntime:
    """The arm's runtime, with ``overlays`` (the driver's ``--policy``) applied over the arm's policy settings.

    An adversary's brief is the policy's ``adversary.<name>`` prompt. The arms run adversaries with unbounded
    submissions and turns, which the policy's integer ``adversary.*`` settings cannot hold, so the roster is the
    arm's ``validation.adversaries`` rather than ``adversary_roster``.
    """
    t = treatment
    policy = t.policy(*overlays)
    adversaries = tuple(
        AdversaryPolicy(
            spec=AdversarySpec(a["name"], policy.prompt(f"adversary.{a['name']}").template),
            k=a["k"],
            submissions=a["submissions"],
            max_turns=a["max_turns"],
            sampling=SAMPLING,
            mode=AdversaryMode(a["mode"]),
            min_probes=a["min_probes"],
            prompts=policy,
        )
        for a in t["validation.adversaries"]
    )
    return ArmRuntime(
        name=t.name,
        policy=policy,
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
            solver_k=policy.setting("solver.k", int),
            saturated_accept=t["review.saturated_accept"],
            class_feedback=t.text("review.class_feedback"),
            taxonomy=read_taxonomy(TAXONOMY),
        ),
        adversaries=adversaries,
        solver_k=policy.setting("solver.k", int),
        max_retries=policy.setting("solver.max_retries", int),
        retry_backoff=RetryBackoff(**t["validation.retry_backoff"]),
        command_timeout=t["validation.command_timeout"],
        build_wall_timeout=t["build.wall_timeout"],
        shadow_l3=t["shadow.l3"],
        shadow_l4=t["shadow.l4"],
        judge_prompt=t.text("shadow.judge.prompt") if t["shadow.judge.enabled"] else None,
        judge_max_claims=t["shadow.judge.max_claims_per_component"],
    )


HELD_RUNS = 4
"""Runs of an allocation's workpiece per child: a run that ends HELD is resumed from the event log after a backoff."""
HELD_BACKOFF = RetryBackoff(initial=30.0, maximum=600.0, factor=2.0, jitter=0.0)


def policy_digest(policy: Policy) -> str:
    """The digest of every policy entry, a prompt by its template's sha256."""
    return digest({key: v.sha256 if isinstance(v, Prompt) else v for key, v in sorted(policy.entries.items())})


def shared_settings() -> dict:
    """The settings identical in every arm by construction, for ``manifest.json``."""
    return json.loads(
        json.dumps(
            {
                "sampling": asdict(SAMPLING),
                "held_resume": {"runs": HELD_RUNS, "backoff": asdict(HELD_BACKOFF)},
                "build_agent_limits": asdict(BUILD_AGENT_LIMITS),
                "trial_limits": TRIAL_LIMITS.model_dump(mode="json"),
                "taxonomy": str(TAXONOMY.relative_to(PACKAGE)),
            },
            default=str,
        )
    )
