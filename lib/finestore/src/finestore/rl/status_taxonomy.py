# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Atlas source-status taxonomy: statuses, exclusion reasons and their evidence.

This module is the single source of truth for every source status and exclusion
reason. The catalog writer attaches typed claims to declarations, the Atlas
status page renders them, and the guide and docs render the same explanation
metadata from here. Nothing else may define status or reason labels.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class SourceStatus(StrEnum):
    """Catalog status of a source population.

    ``Deprecated`` means trainable but no longer supported or recommended:
    selection may use a Deprecated source when explicitly asked, never by
    default, and a Deprecated row must carry no exclusion reason.
    """

    PENDING = "Pending"
    AVAILABLE = "Available"
    EXCLUDED = "Excluded"
    DEPRECATED = "Deprecated"


class DataRestriction(StrEnum):
    """Typed restriction tier of a source population.

    The tier is orthogonal to license, intended use and catalog status: an
    ``Open`` source can be license-restricted and a ``PRIVATE_ALWAYS`` source
    can be ``Available``. Every source declares the tier explicitly in code on
    its ``SourceInfo``; the declaration, every derived artifact and every
    published record carry it, and a downgrade or a missing tag on a derived
    artifact is refused. A ``PRIVATE_ALWAYS`` source's task text, prompts,
    answers, resources, samples and transcripts never leave its private store:
    the Atlas shows counts, statuses, hashes, receipts and a visible private
    data marker instead of content, no matter what else changes.
    """

    OPEN = "open"
    PRIVATE_ALWAYS = "private_always"


class PendingStage(StrEnum):
    """Intake stage of a queued source that is not configured for the catalog yet."""

    INTAKE_QUEUED = "Intake queued"
    """No catalog measurement receipt is recorded for the source yet."""

    MEASUREMENTS_PENDING = "Measurements pending"
    """Some catalog measurement receipts exist; the rest are outstanding."""

    ADMISSION_PENDING = "Admission pending"
    """All catalog measurement receipts exist; the source awaits configuration
    and admission into the catalog."""

    LEXICAL_REMEDIATION_QUEUED = "Lexical remediation queued"
    """The source waits for a remediation agent to produce the clean or
    remediated private-release receipt its training use requires."""


class ExclusionEvidenceKind(StrEnum):
    """What kind of artifact backs an exclusion claim."""

    RECEIPT = "receipt"
    """A pinned receipt URI (optionally with its SHA-256)."""

    CHECK_OUTPUT = "check-output"
    """The failing check's own output, quoted as the message."""

    ISSUE_LINK = "issue-link"
    """A tracker issue that records the ruling, e.g. a license ban."""

    MISSING_ARTIFACT = "missing-artifact"
    """The expected-but-absent artifact the failing stage would produce."""

    NOTE = "note"
    """The source declaration's own recorded note text."""


class ExclusionReason(StrEnum):
    """Every reason a source can be Excluded, with one canonical explanation.

    The explanation for each member is the single ``REASON_EXPLANATIONS`` entry
    below; the status page, guide and docs all render from it.
    """

    LICENSE_BAN = "license-ban"
    LICENSE_MISSING = "license-missing"
    NOT_RL_TASK_DATA = "not-rl-task-data"
    ORACLE_FILTRATION_RETAINED_ZERO = "oracle-filtration-retained-zero"
    ACTOR_CENSUS_MISSING = "actor-census-missing"
    ACTOR_CENSUS_STALE = "actor-census-stale"
    RETAINED_POPULATION_EMPTY = "retained-population-empty"
    GRADING_CENSUS_INCOMPLETE = "grading-census-incomplete"
    GRADING_CENSUS_COUNTS_INCOMPLETE = "grading-census-counts-incomplete"
    GRADER_MISSING = "grader-missing"
    LEXICAL_RECEIPT_MISSING = "lexical-receipt-missing"
    LEXICAL_REMEDIATION_REQUIRED = "lexical-remediation-required"
    OVERSIGHT_MISSING = "oversight-missing"
    AGENTIC_TASK_FORMAT_MISSING = "agentic-task-format-missing"
    SNAPSHOT_IDENTITY_MISSING = "snapshot-identity-missing"
    EXECUTION_REQUIREMENT_MISSING = "execution-requirement-missing"
    EVAL_ONLY_BENCHMARK = "eval-only-benchmark"
    SUPERSEDED_BY_REPLACEMENT = "superseded-by-replacement"
    GRADER_DEFECT = "grader-defect"
    CONTAMINATION_RISK = "contamination-risk"
    DUPLICATE_OF_SOURCE = "duplicate-of-source"
    SNAPSHOT_BUDGET_EXCEEDED = "snapshot-budget-exceeded"
    UNSPECIFIED_LEGACY = "unspecified-legacy"
    ENVIRONMENT_ARTIFACT_MISSING = "environment-artifact-missing"
    NO_SUPPORTED_BACKEND = "no-supported-backend"
    GRADER_JUDGE_UNAVAILABLE = "grader-judge-unavailable"
    INVALID_REFERENCE = "invalid-reference"


class ReasonExplanation(BaseModel):
    """The canonical compact explanation of one exclusion reason."""

    model_config = ConfigDict(frozen=True)

    label: str
    """Short display label used by the status page and filter chips."""

    meaning: str
    """One line: what the reason asserts about the source."""

    clears: str
    """One line: which pipeline step and receipt clear the reason."""


REASON_EXPLANATIONS: dict[ExclusionReason, ReasonExplanation] = {
    ExclusionReason.LICENSE_BAN: ReasonExplanation(
        label="License ban",
        meaning="The pinned upstream or publisher license is banned for training use.",
        clears="A license receipt the ban registry accepts, with the registry updated off the ban list.",
    ),
    ExclusionReason.LICENSE_MISSING: ReasonExplanation(
        label="License missing",
        meaning="No license declaration exists in the pinned upstream primary evidence.",
        clears="A recorded permissive grant or license receipt established at the pin.",
    ),
    ExclusionReason.NOT_RL_TASK_DATA: ReasonExplanation(
        label="Not RL task data",
        meaning="The upstream provides no RL task data; it ships harness code or trajectories only.",
        clears="A republished upstream that contains usable RL task data.",
    ),
    ExclusionReason.ORACLE_FILTRATION_RETAINED_ZERO: ReasonExplanation(
        label="Oracle filtration retained zero",
        meaning="Oracle filtration over the full population admitted zero tasks.",
        clears="A republished conversion whose filtration admits more than zero tasks, with a publication-seal receipt.",
    ),
    ExclusionReason.ACTOR_CENSUS_MISSING: ReasonExplanation(
        label="Actor census missing",
        meaning="No actor census evidence exists for the source at all.",
        clears="The QUICK actor-census receipt bound at the current pin.",
    ),
    ExclusionReason.ACTOR_CENSUS_STALE: ReasonExplanation(
        label="Actor census stale",
        meaning="The actor census measured a different dataset pin or converter than the configured input.",
        clears="A re-measured census receipt at the current dataset pin and converter version.",
    ),
    ExclusionReason.RETAINED_POPULATION_EMPTY: ReasonExplanation(
        label="Retained population empty",
        meaning="The retained population at the current pin is empty.",
        clears="A nonempty retained population at the current pin.",
    ),
    ExclusionReason.GRADING_CENSUS_INCOMPLETE: ReasonExplanation(
        label="Grading census incomplete",
        meaning="The grading census does not cover the complete retained population.",
        clears="A grading census receipt covering every retained task.",
    ),
    ExclusionReason.GRADING_CENSUS_COUNTS_INCOMPLETE: ReasonExplanation(
        label="Grading census counts incomplete",
        meaning="The grading census's typed counts do not sum to the retained population.",
        clears="A grading census receipt whose required, declared, status and reason counts cover the population.",
    ),
    ExclusionReason.GRADER_MISSING: ReasonExplanation(
        label="Grader missing",
        meaning="Admitted tasks lack runnable graders.",
        clears="A grading census receipt with zero missing-grader tasks.",
    ),
    ExclusionReason.LEXICAL_RECEIPT_MISSING: ReasonExplanation(
        label="Lexical receipt missing",
        meaning="Evaluation use lacks a measured lexical receipt at the current dataset pin.",
        clears="A lexical receipt measured at the current pin for the declared use.",
    ),
    ExclusionReason.LEXICAL_REMEDIATION_REQUIRED: ReasonExplanation(
        label="Lexical remediation required",
        meaning="Training use lacks a clean or remediated private-release receipt at the current pin.",
        clears="A remediated private-release decontamination receipt bound to the current pin.",
    ),
    ExclusionReason.OVERSIGHT_MISSING: ReasonExplanation(
        label="Oversight missing",
        meaning="No oversight evidence exists at all: neither a QUICK conversion receipt "
        "nor a SAMPLE or FULL review receipt with recorded verification.",
        clears="A QUICK conversion receipt, or a SAMPLE or FULL review receipt with recorded verification.",
    ),
    ExclusionReason.AGENTIC_TASK_FORMAT_MISSING: ReasonExplanation(
        label="Agentic task format missing",
        meaning="An agentic source declares no accepted task format.",
        clears="A declared TaskCompendium or Harbor task format (or grandfathered registration).",
    ),
    ExclusionReason.SNAPSHOT_IDENTITY_MISSING: ReasonExplanation(
        label="Snapshot identity missing",
        meaning="No production snapshot identity measurement exists.",
        clears="The snapshot identity measurement receipt at the current pin.",
    ),
    ExclusionReason.EXECUTION_REQUIREMENT_MISSING: ReasonExplanation(
        label="Execution requirement missing",
        meaning="No execution-requirement measurement exists for the population.",
        clears="The execution-requirement measurement receipt.",
    ),
    ExclusionReason.EVAL_ONLY_BENCHMARK: ReasonExplanation(
        label="Eval only",
        meaning="The source is admitted for evaluation only; its training use is excluded by a recorded benchmark rule.",
        clears="A user ruling that admits training use, recorded with the benchmark evidence.",
    ),
    ExclusionReason.SUPERSEDED_BY_REPLACEMENT: ReasonExplanation(
        label="Superseded",
        meaning="A repaired or re-pinned cohort replaces this source's population.",
        clears="Retiring the replacement claim, or restoring this pin as the current cohort.",
    ),
    ExclusionReason.GRADER_DEFECT: ReasonExplanation(
        label="Grader defect",
        meaning="A recorded grader or oracle defect makes the population's reward untrustworthy.",
        clears="A republished grader that passes the recorded defect check.",
    ),
    ExclusionReason.CONTAMINATION_RISK: ReasonExplanation(
        label="Contamination risk",
        meaning="A known canary or contamination exposure is present in the source.",
        clears="A decontamination receipt that removes the recorded exposure.",
    ),
    ExclusionReason.DUPLICATE_OF_SOURCE: ReasonExplanation(
        label="Duplicate",
        meaning="The population duplicates another admitted source's population.",
        clears="A distinct-population ruling recorded against the other source.",
    ),
    ExclusionReason.SNAPSHOT_BUDGET_EXCEEDED: ReasonExplanation(
        label="Snapshot budget exceeded",
        meaning="Provisioning the cohort needs more machine-snapshot variants than the budget allows.",
        clears="A rebuilt payload whose provisioning fits the budget, with a snapshot-capacity-check receipt.",
    ),
    ExclusionReason.UNSPECIFIED_LEGACY: ReasonExplanation(
        label="Needs ruling",
        meaning="A legacy exclusion tag with no recoverable typed reason; transitional.",
        clears="A user ruling recorded as a typed reason; this reason must not persist.",
    ),
    ExclusionReason.ENVIRONMENT_ARTIFACT_MISSING: ReasonExplanation(
        label="Environment artifact missing",
        meaning="A declared environment artifact does not exist in any region.",
        clears="Rebuilding the artifact in-region and binding its receipt; "
        "exclusion applies only when it is unrebuildable.",
    ),
    ExclusionReason.NO_SUPPORTED_BACKEND: ReasonExplanation(
        label="No supported backend",
        meaning="No available backend can run the population's contracts, "
        "per the backend survey's per-source capabilities.",
        clears="A backend survey recording supported execution for at least one backend at the current pin.",
    ),
    ExclusionReason.GRADER_JUDGE_UNAVAILABLE: ReasonExplanation(
        label="Judge unavailable",
        meaning="The population's judge route is unqualified: no judge-model tier or judge API qualifies.",
        clears="Re-pinning to a JUDGE_TIERS tier through private remediation and running that version; "
        "exclusion applies only when no tier qualifies.",
    ),
    ExclusionReason.INVALID_REFERENCE: ReasonExplanation(
        label="Invalid reference",
        meaning="The population references an expected target that does not exist in its frozen sample.",
        clears="A faithfully repaired private remediation version whose references resolve.",
    ),
}


class ExclusionEvidence(BaseModel):
    """Typed evidence backing one exclusion claim."""

    model_config = ConfigDict(frozen=True)

    kind: ExclusionEvidenceKind
    message: str = Field(min_length=1)
    """The failing check's output, the note text, or a one-line summary of the receipt."""

    uri: str | None = None
    """Receipt or issue URI, when the kind carries one."""

    sha256: str | None = None
    """SHA-256 of the receipt payload, when pinned."""

    expected: str | None = None
    """For MISSING_ARTIFACT evidence: the artifact the failing stage expected."""


class ExclusionClaim(BaseModel):
    """One exclusion reason with its evidence for one source."""

    model_config = ConfigDict(frozen=True)

    reason: ExclusionReason
    evidence: ExclusionEvidence

    @property
    def explanation(self) -> ReasonExplanation:
        return REASON_EXPLANATIONS[self.reason]
