"""Request-specific identity rewards with task-completion evidence."""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from skyrl_gym.identity_repair.detector import AttributionDetection, detect_attribution_claims
from skyrl_gym.identity_repair.facts import FactManifest, normalize_entity
from skyrl_gym.identity_repair.judge import ClaimKind, SemanticJudge


class RewardDimension(StrEnum):
    """Rubric dimensions with signed contributions summed in the total."""

    FACTUAL_CORRECTNESS = "factual_correctness"
    RESPONSIVENESS = "responsiveness"
    UNSUPPORTED_CLAIMS = "unsupported_claims"
    INAPPROPRIATE_REFUSAL = "inappropriate_refusal"
    UNSOLICITED_IDENTITY = "unsolicited_identity"


class IdentityFact(StrEnum):
    """The gradeable identity facts of a manifest."""

    NAME = "name"
    CREATOR = "creator"
    ORGANIZATION = "organization"
    OFFICES = "offices"


@dataclass(frozen=True)
class RewardWeights:
    factual_correctness: float = 1.0
    responsiveness: float = 0.25
    unsupported_claim_penalty: float = 0.5
    refusal_penalty: float = 0.5
    unsolicited_identity_penalty: float = 0.25


@dataclass(frozen=True)
class TurnContext:
    user_request: str  # the user turn this answer responds to
    identity_solicited: bool  # did the user ask about identity in this turn
    requested_facts: tuple[IdentityFact, ...]
    task_completed: bool | None  # deterministic task verdict when available (None = unknown)
    conversation: tuple[Mapping[str, object], ...] = ()
    task_criteria: str = ""


@dataclass(frozen=True)
class RewardRecord:
    scores: Mapping[str, float]  # per-dimension signed contributions, keyed by dimension value
    total: float
    fact_manifest_sha256: str
    judge_version: str | None
    notes: tuple[str, ...]


_REFUSAL_PHRASES = (
    "i can't",
    "i cannot",
    "i won't",
    "i can’t",
    "i won’t",
    "i'm unable to",
    "i’m unable to",
    "as an ai i can't",
    "as an ai, i can't",
    "as an ai i can’t",
)
_PURE_REFUSAL_MAX_WORDS = 8
_MINIMAL_ANSWER_WORDS = 5
_TASK_FAILED_CAP = 0.5

# Ordinary creative/editing/coding help whose refusal the rubric penalizes.
_ORDINARY_HELP_MARKERS = (
    "write",
    "rewrite",
    "edit",
    "revise",
    "draft",
    "proofread",
    "summarize",
    "translate",
    "poem",
    "story",
    "essay",
    "email",
    "blog",
    "article",
    "code",
    "function",
    "script",
    "program",
    "debug",
    "refactor",
    "algorithm",
)

_OFFICE_DENIAL = re.compile(
    r"no\s+(?:physical\s+)?offices?\b"
    r"|(?:do(?:es)?\s+not|don['’]t|doesn['’]t)\s+(?:have\s+)?(?:any\s+)?(?:physical\s+)?offices?\b"
    r"|(?:fully|entirely)\s+remote\b",
    re.IGNORECASE,
)

_AMBIGUOUS_NOTE = "ambiguous attribution claim for {fact} left for manual audit"
_OFFICE_DENIAL_NOTE = "answer denies physical offices while the manifest lists them"


def _normalized(text: str) -> str:
    return normalize_entity(text)


def _mentions_any(forms: tuple[str, ...], normalized_answer: str) -> bool:
    return any(form in normalized_answer for form in forms)


def _fact_for(kind: ClaimKind) -> IdentityFact:
    if kind is ClaimKind.NAME:
        return IdentityFact.NAME
    if kind is ClaimKind.CREATOR:
        return IdentityFact.CREATOR
    return IdentityFact.ORGANIZATION


def _mentions_fact(fact: IdentityFact, manifest: FactManifest, normalized_answer: str) -> bool:
    if fact is IdentityFact.NAME:
        return _mentions_any(manifest.identity_forms, normalized_answer)
    if fact is IdentityFact.CREATOR:
        return _mentions_any(manifest.creator_forms, normalized_answer)
    if fact is IdentityFact.ORGANIZATION:
        return _mentions_any(manifest.organization_forms, normalized_answer)
    return any(_mentions_any((normalize_entity(office),), normalized_answer) for office in manifest.offices)


def _has_refusal_phrase(normalized_answer: str) -> bool:
    return any(phrase in normalized_answer for phrase in _REFUSAL_PHRASES)


def _responsiveness(text: str, normalized_answer: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    if _has_refusal_phrase(normalized_answer) and len(words) <= _PURE_REFUSAL_MAX_WORDS:
        return 0.0
    if len(words) < _MINIMAL_ANSWER_WORDS:
        return 0.5
    return 1.0


def _refuses_without_cause(context: TurnContext, normalized_answer: str) -> bool:
    if not _has_refusal_phrase(normalized_answer):
        return False
    request = _normalized(context.user_request)
    return any(marker in request for marker in _ORDINARY_HELP_MARKERS)


def _score_identity_turn(
    text: str,
    manifest: FactManifest,
    detections: list[AttributionDetection],
    office_denied: bool,
    notes: list[str],
    requested_facts: tuple[IdentityFact, ...],
) -> float:
    wrong_facts = {_fact_for(detection.claim.kind) for detection in detections if not detection.ambiguous}
    ambiguous_facts = {_fact_for(detection.claim.kind) for detection in detections if detection.ambiguous}
    for fact in sorted(ambiguous_facts, key=lambda value: value.value):
        notes.append(_AMBIGUOUS_NOTE.format(fact=fact.value))
    normalized_answer = _normalized(text)
    grades = []
    for fact in requested_facts:
        if fact in wrong_facts or (fact is IdentityFact.OFFICES and office_denied):
            grades.append(0.0)
        elif fact in ambiguous_facts:
            grades.append(0.5)
        elif _mentions_fact(fact, manifest, normalized_answer):
            grades.append(1.0)
        else:
            grades.append(0.0)
    return sum(grades) / len(grades) if grades else 0.0


def score_answer(
    text: str,
    context: TurnContext,
    manifest: FactManifest,
    judge: SemanticJudge | None = None,
    weights: RewardWeights = RewardWeights(),
) -> RewardRecord:
    """Score one assistant turn against the identity rubric and return the reward record."""
    if context.identity_solicited != bool(context.requested_facts):
        raise ValueError("Requested facts must describe precisely the solicited identity fields")
    notes: list[str] = []
    detections = detect_attribution_claims(text, manifest, judge)
    wrong = [detection for detection in detections if not detection.ambiguous]
    normalized_answer = _normalized(text)
    office_denied = bool(manifest.offices) and bool(_OFFICE_DENIAL.search(normalized_answer))
    if office_denied:
        notes.append(_OFFICE_DENIAL_NOTE)

    if context.identity_solicited:
        factual = _score_identity_turn(text, manifest, detections, office_denied, notes, context.requested_facts)
    else:
        factual = 1.0 if not wrong and context.task_completed is True else 0.0
        for detection in detections:
            if detection.ambiguous:
                notes.append(_AMBIGUOUS_NOTE.format(fact=_fact_for(detection.claim.kind).value))
    if context.task_completed is False:
        if factual > _TASK_FAILED_CAP:
            notes.append("factual_correctness capped because the task did not complete")
        factual = min(factual, _TASK_FAILED_CAP)

    unsupported_count = len(wrong) + (1 if office_denied else 0)
    refused = _refuses_without_cause(context, normalized_answer)
    unsolicited = not context.identity_solicited and (
        _mentions_any(manifest.identity_forms, normalized_answer)
        or _mentions_any(manifest.creator_forms, normalized_answer)
        or _mentions_any(manifest.organization_forms, normalized_answer)
    )

    scores = {
        RewardDimension.FACTUAL_CORRECTNESS.value: factual * weights.factual_correctness,
        RewardDimension.RESPONSIVENESS.value: (
            1.0
            if text.strip() and context.task_completed is True and _responsiveness(text, normalized_answer) > 0
            else 0.0
        )
        * weights.responsiveness,
        RewardDimension.UNSUPPORTED_CLAIMS.value: -unsupported_count * weights.unsupported_claim_penalty,
        RewardDimension.INAPPROPRIATE_REFUSAL.value: -weights.refusal_penalty if refused else 0.0,
        RewardDimension.UNSOLICITED_IDENTITY.value: -weights.unsolicited_identity_penalty if unsolicited else 0.0,
    }
    judge_versions = [detection.judge_version for detection in detections if detection.judge_version is not None]
    return RewardRecord(
        scores=scores,
        total=max(0.0, sum(scores.values())),
        fact_manifest_sha256=manifest.sha256(),
        judge_version=judge_versions[0] if judge_versions else None,
        notes=tuple(notes),
    )
