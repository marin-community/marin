# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The Atlas source-status taxonomy: statuses, stages, exclusion reasons and column values.

This module is the single source of truth for every source status, pending
stage, exclusion reason and Use/Method/Format column value, with the canonical
labels and explanations each renders from. The catalog writer attaches typed
claims to declarations, the Atlas status page renders them, and the guide and
docs render the same explanation metadata from here. Nothing else may define
status, stage, reason or column-value labels.
"""

from collections.abc import Sequence
from enum import StrEnum
from typing import Any

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


class SourceMethod(StrEnum):
    """Typed value of the Atlas Method column (ruling 70).

    A source's Method is the union of its dataset variants' preparation
    evidence: ``RL`` for the RL ingestion environment variant every source
    carries, and ``SFT-viable`` for a completed Datakit preparation receipt.
    The stored legacy tags ``rl`` and ``sft`` validate as these members and
    never re-enter new records. An unknown tag is refused.
    """

    RL = "RL"
    SFT_VIABLE = "SFT-viable"

    @classmethod
    def _missing_(cls, value: object) -> "SourceMethod | None":
        if value == "rl":
            return cls.RL
        if value == "sft":
            return cls.SFT_VIABLE
        return None


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
    """Intake stage of a queued source that is not configured for the catalog yet.

    Every Pending source carries exactly one stage, computed from pipeline
    evidence (which admission receipts and backend qualifications exist), never
    hand-entered text. ``STAGE_EXPLANATIONS`` holds each stage's canonical
    label, meaning and advancing evidence.
    """

    INTAKE_QUEUED = "Intake queued"
    """No measurement receipt covers the source at all yet."""

    CONSUMER_CHECK_PENDING = "Consumer check pending"
    """The source's consumer admission decision has not approved it yet."""

    MEASUREMENTS_PENDING = "Measurements pending"
    """Some catalog measurement receipts exist; the rest are outstanding."""

    ADMISSION_PENDING = "Admission pending"
    """All catalog measurement receipts exist; the source awaits configuration
    and admission into the catalog."""

    LEXICAL_REMEDIATION_QUEUED = "Lexical remediation queued"
    """The source waits for a remediation agent to produce the clean or
    remediated private-release receipt its training use requires."""

    SNAPSHOT_REPAIR_QUEUED = "Snapshot repair queued"
    """The actor census measured the cohort's snapshot provisioning over the
    shared budget; a snapshot-safe rebuild re-admits the source."""

    BACKEND_VERIFICATION_PENDING = "Backend verification pending"
    """A new admission awaits the live backend qualification: no production
    backend admits it and VERIFY is its only positive production outcome."""

    BACKEND_PROFILE_PENDING = "Backend profile pending"
    """The grader needs network while the surveyed grader profile denies it;
    a maintained per-source profile binding re-admits the source (r22 ruling)."""

    BACKEND_PREPARATION_PENDING = "Backend preparation pending"
    """The grader Docker build context is unresolved on every task, so no
    surveyed production backend can run the population; the image preparation
    lane re-admits the source with pinned grader images."""


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
    NOT_TASKSPEC_CONVERTED = "not-taskspec-converted"
    SNAPSHOT_IDENTITY_MISSING = "snapshot-identity-missing"
    EXECUTION_REQUIREMENT_MISSING = "execution-requirement-missing"
    GRADER_DEFECT = "grader-defect"
    CONTAMINATION_RISK = "contamination-risk"
    DUPLICATE_OF_SOURCE = "duplicate-of-source"
    UNSPECIFIED_LEGACY = "unspecified-legacy"
    ENVIRONMENT_ARTIFACT_MISSING = "environment-artifact-missing"
    NO_SUPPORTED_BACKEND = "no-supported-backend"
    GRADER_JUDGE_UNAVAILABLE = "grader-judge-unavailable"
    INVALID_REFERENCE = "invalid-reference"
    THIRD_PARTY_CREDENTIALS_REQUIRED = "third-party-credentials-required"


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
    ExclusionReason.NOT_TASKSPEC_CONVERTED: ReasonExplanation(
        label="Not TaskSpec converted",
        meaning="The source's admitted population carries no verified TaskSpec conversion receipt for its "
        "admitted revision: its declared shape is not an executable TaskSpec population, so the RL "
        "runtime cannot serve it.",
        clears="A verified TaskSpec conversion receipt covering the admitted revision — the conversion "
        "census bound at the admitted pin with the declared TaskSpec shape — produced by the required "
        "conversion stage.",
    ),
    ExclusionReason.AGENTIC_TASK_FORMAT_MISSING: ReasonExplanation(
        label="Agentic task format missing",
        meaning="An agentic source declares no accepted task format.",
        clears="A declared TaskSpec task format (or grandfathered registration).",
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
    ExclusionReason.THIRD_PARTY_CREDENTIALS_REQUIRED: ReasonExplanation(
        label="Third-party credentials required",
        meaning="Tasks call live third-party services whose credentials are not provisioned.",
        clears="An approved credential-provisioning design tracked in #12082 and a qualifier run "
        "with fixture credentials bound only to owned verifier/service jobs.",
    ),
}


STATUS_EXPLANATIONS: dict[SourceStatus, ReasonExplanation] = {
    SourceStatus.PENDING: ReasonExplanation(
        label="Pending",
        meaning="The source is not selectable yet; it waits on the intake stage shown beside it.",
        clears="The stage's own advancing evidence, recorded at the current pin.",
    ),
    SourceStatus.AVAILABLE: ReasonExplanation(
        label="Available",
        meaning="Every admission claim is cleared at the current pin; selection may admit the source.",
        clears="Nothing; the status holds while its receipts stay valid at the pin.",
    ),
    SourceStatus.EXCLUDED: ReasonExplanation(
        label="Excluded",
        meaning="A typed exclusion reason rules the source out of selection.",
        clears="The reason's own clearing evidence, recorded at the current pin.",
    ),
    SourceStatus.DEPRECATED: ReasonExplanation(
        label="Deprecated",
        meaning="Trainable but no longer supported or recommended; selection uses it only when explicitly asked.",
        clears="Nothing; deprecation is a recommendation, not an exclusion.",
    ),
}


STAGE_EXPLANATIONS: dict[PendingStage, ReasonExplanation] = {
    PendingStage.INTAKE_QUEUED: ReasonExplanation(
        label="Intake queued",
        meaning="The source is registered for intake and no standard measurement receipt covers it yet.",
        clears="The first standard measurement receipt (actor census, lexical, grading, sample, "
        "oracle or retained) recorded at the current pin.",
    ),
    PendingStage.CONSUMER_CHECK_PENDING: ReasonExplanation(
        label="Consumer check pending",
        meaning="The source's consumer admission decision has not approved it yet.",
        clears="A recorded consumer decision approving the source at the runtime commit.",
    ),
    PendingStage.MEASUREMENTS_PENDING: ReasonExplanation(
        label="Measurements pending",
        meaning="Some standard measurement receipts exist; the rest are outstanding.",
        clears="Every remaining standard measurement receipt at the current pin.",
    ),
    PendingStage.ADMISSION_PENDING: ReasonExplanation(
        label="Admission pending",
        meaning="Every standard measurement receipt exists; the source awaits configuration and admission.",
        clears="Configuration and admission into the catalog at the current pin.",
    ),
    PendingStage.LEXICAL_REMEDIATION_QUEUED: ReasonExplanation(
        label="Lexical remediation queued",
        meaning="Training use waits on a remediation agent's clean or remediated private-release receipt.",
        clears="The remediated private-release decontamination receipt bound to the current pin.",
    ),
    PendingStage.SNAPSHOT_REPAIR_QUEUED: ReasonExplanation(
        label="Snapshot repair queued",
        meaning="The cohort's measured snapshot provisioning exceeds the shared machine-snapshot budget; "
        "the constraint is capacity, not a property of the data.",
        clears="A snapshot-safe rebuild whose provisioning fits the budget — declarative or registry-image "
        "builds, shared images, LOCAL-tier routing, or snapshot cleanup — re-admitting the source.",
    ),
    PendingStage.BACKEND_VERIFICATION_PENDING: ReasonExplanation(
        label="Backend verification pending",
        meaning="No production backend admits the new source and VERIFY is its only positive production outcome.",
        clears="A live backend qualification admitting at least one production backend at the current pin.",
    ),
    PendingStage.BACKEND_PROFILE_PENDING: ReasonExplanation(
        label="Backend profile pending",
        meaning="The grader needs network while the surveyed grader profile denies it.",
        clears="A maintained per-source profile binding that allows the grader's network use (r22 ruling).",
    ),
    PendingStage.BACKEND_PREPARATION_PENDING: ReasonExplanation(
        label="Backend preparation pending",
        meaning="The grader Docker build context is unresolved on every task, so no surveyed "
        "production backend can run the population.",
        clears="The image preparation lane pinning the grader images and re-admitting the source.",
    ),
}


class ValueExplanation(BaseModel):
    """The canonical compact explanation of one Use, Method or Format value."""

    model_config = ConfigDict(frozen=True)

    label: str
    """Short display label used by the Atlas tables and filter chips."""

    meaning: str
    """One line: what the value asserts about the source."""


USE_EXPLANATIONS: dict[str, ValueExplanation] = {
    "train": ValueExplanation(
        label="Training",
        meaning="The registered converter's policy admits the population for training use.",
    ),
    "eval": ValueExplanation(
        label="Evaluation",
        meaning="The registered converter's policy admits the population for evaluation use only.",
    ),
}


class ProductionBackend(StrEnum):
    """The production deployment backends an Atlas source's fit is measured against.

    Dev-only backends (local Docker, Docker+runsc) and the test simulators are
    not members: admission counts production backends only, and the Backend
    Evidence page renders this enum. The member set must stay identical to the
    survey layer's production ``BACKEND_DEPLOYMENT`` entries; the catalog tests
    pin the two together.
    """

    DAYTONA = "daytona"
    IRIS = "iris"
    LOCAL = "local"


class BackendBlockerReason(StrEnum):
    """Why a production backend cannot host a role, beyond its typed demands.

    A typed demand the backend's survey cells refuse renders as the demand's
    own value; these members cover the blockers that are not demands. The
    reasons are auto-derived from the measured survey fit, never hand-entered.
    """

    REQUIRED_BACKEND = "required-backend"
    """The task pins a different backend than this one."""

    UNCLASSIFIED_CAPABILITY = "unclassified-capability"
    """A declared capability the survey vocabulary does not classify yet."""

    VALIDATION_ERROR = "validation-error"
    """The backend's checker refused the task's declared environment."""

    UNSUPPORTED_ENVIRONMENT_KIND = "unsupported-environment-kind"
    """The role's declared environment kind (for example a host image) is one the backend cannot run."""


BACKEND_EXPLANATIONS: dict[ProductionBackend, ValueExplanation] = {
    ProductionBackend.DAYTONA: ValueExplanation(
        label="Daytona",
        meaning="The production Daytona RL sandbox backend; admission counts its measured fit.",
    ),
    ProductionBackend.IRIS: ValueExplanation(
        label="Iris (gVisor)",
        meaning="The production Iris backend running gVisor-sandboxed workers.",
    ),
    ProductionBackend.LOCAL: ValueExplanation(
        label="LOCAL",
        meaning="The production LOCAL backend on cluster workers with the verified disk pool.",
    ),
}

BACKEND_BLOCKER_EXPLANATIONS: dict[BackendBlockerReason, ValueExplanation] = {
    BackendBlockerReason.REQUIRED_BACKEND: ValueExplanation(
        label="Required backend",
        meaning="The task pins a different production backend, so this one cannot host the role.",
    ),
    BackendBlockerReason.UNCLASSIFIED_CAPABILITY: ValueExplanation(
        label="Unclassified capability",
        meaning="A declared capability the survey vocabulary does not classify yet; the cells stay "
        "VERIFY until the vocabulary classifies it or a live probe resolves the role.",
    ),
    BackendBlockerReason.VALIDATION_ERROR: ValueExplanation(
        label="Validation error",
        meaning="The backend's checker refused the task's declared environment outright.",
    ),
    BackendBlockerReason.UNSUPPORTED_ENVIRONMENT_KIND: ValueExplanation(
        label="Unsupported environment kind",
        meaning="The role declares an environment kind (such as a host image or a Docker build) that this "
        "backend does not run, so the survey marks the role not applicable here.",
    ),
}


def backend_blocker_reason_slugs(requirement_values: Sequence[str]) -> frozenset[str]:
    """Every blocker-reason slug the served backend-reason table can render.

    The backend-fit derivation labels each blocker it emits with one of these
    slugs and fails closed on a blocker it cannot label, so the evidence pages
    never meet a blocker the taxonomy has no reason for.
    """
    return frozenset({reason.value for reason in BackendBlockerReason} | set(requirement_values))


def requirement_blocker_explanation(requirement: str) -> ValueExplanation:
    """The canonical explanation of one typed-demand blocker reason.

    A demand reason renders as the measured requirement's own value with this
    meaning; the requirement vocabulary itself is validated by the serving app
    against the survey layer's enum.
    """
    return ValueExplanation(
        label=requirement,
        meaning=f"The population's measured '{requirement}' demand is not available on this backend at the current pin.",
    )


METHOD_EXPLANATIONS: dict[SourceMethod, ValueExplanation] = {
    SourceMethod.RL: ValueExplanation(
        label="RL",
        meaning="The RL ingestion path created the source's RL environment variant.",
    ),
    SourceMethod.SFT_VIABLE: ValueExplanation(
        label="SFT-viable",
        meaning="A completed Datakit preparation receipt qualifies the source's variant for SFT (ruling 70).",
    ),
}


def taxonomy_tables(requirement_values: Sequence[str]) -> dict[str, Any]:
    """The canonical status, stage, reason and Use/Method explanation tables.

    The Atlas ``/api/taxonomy`` endpoint serves exactly this document and every
    page renders its labels and explanations from it; the release checker
    answers the same route from the candidate's code while the served revision
    predates the endpoint. ``requirement_values`` is the survey layer's typed
    demand vocabulary (shellbox ``Requirement``), which lives outside this
    module: the serving app and the checker pass the same enumeration so both
    render one document, and the backend-blocker table covers every typed
    demand beside the canonical non-demand reasons.
    """
    return {
        "status": {status.value: item.model_dump(mode="json") for status, item in STATUS_EXPLANATIONS.items()},
        "stage": {stage.value: item.model_dump(mode="json") for stage, item in STAGE_EXPLANATIONS.items()},
        "reason": {reason.value: item.model_dump(mode="json") for reason, item in REASON_EXPLANATIONS.items()},
        "use": {value: item.model_dump(mode="json") for value, item in USE_EXPLANATIONS.items()},
        "method": {method.value: item.model_dump(mode="json") for method, item in METHOD_EXPLANATIONS.items()},
        "backend": {backend.value: item.model_dump(mode="json") for backend, item in BACKEND_EXPLANATIONS.items()},
        "backend-reason": {
            **{reason.value: item.model_dump(mode="json") for reason, item in BACKEND_BLOCKER_EXPLANATIONS.items()},
            **{value: requirement_blocker_explanation(value).model_dump(mode="json") for value in requirement_values},
        },
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
