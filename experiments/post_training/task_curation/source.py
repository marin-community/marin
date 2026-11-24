# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Source declarations and their curation pipeline interfaces."""

from dataclasses import dataclass
from typing import Annotated, Protocol

from finestore.rl.curation import (
    DAYTONA_SAFE_SNAPSHOT_LIMIT,
    ActorCensusInput,
    ConsumerDecision,
    ConsumerNeed,
    ConsumerReason,
    ConsumerReceipt,
    ContentEvidenceReference,
    ConvertedExamples,
    DatasetRelease,
    DaytonaMeasurement,
    DaytonaUnsafeReason,
    GradingEnvironmentEvidence,
    IndependentQC,
    IntegrityContractSummary,
    LexicalContaminationEvidence,
    NoOracleExecutabilityPolicy,
    NoOracleStaticRetentionEvidence,
    OracleFiltrationEvidence,
    OracleValidation,
    OversightEvidence,
    OversightLevel,
    PipelineReference,
    PolicyRewardMeasurement,
    QualityResultFinalization,
    RecipeQAHistoricalNotes,
    RoleExecutionEvidence,
    SourceOrigin,
    SourceQualityResult,
    SourceUse,
    TaskFormat,
    WithinSourceDedupEvidence,
    consumer_blockers,
    consumer_decision_status,
    source_display_name,
    source_identifier,
)
from finestore.rl.curation import (
    QualifiedSourceDeclaration as SourceDeclaration,
)
from finestore.rl.licenses import LicenseReceipt, require_license
from finestore.rl.status_taxonomy import (
    DataRestriction,
    ExclusionClaim,
    ExclusionEvidence,
    ExclusionEvidenceKind,
    ExclusionReason,
    PendingStage,
    SourceStatus,
)
from marin.execution.lazy import ArtifactStep
from pydantic import Field
from taskcompendium.derived_contract import DerivedContractSummary
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.campaign import CampaignArtifact
from experiments.post_training.task_curation.config import PipelineOptions
from experiments.post_training.task_curation.consumer_scope import CONSUMER_RUNTIME_COMMIT
from experiments.post_training.task_curation.exclusion_registry import manual_exclusion_claims
from experiments.post_training.task_curation.full_content_results import FullContentResultBinding
from infra.marina.apps.rl_data_catalog.server.admission import (
    population_admission_blocker,
    require_lexical_qualification,
)
from infra.marina.apps.rl_data_catalog.server.licensing import require_unbanned, source_license_ban
from infra.marina.apps.rl_data_catalog.server.source_identity import canonical_source_id

POPULATION_BLOCKER_REASONS: dict[str, ExclusionReason] = {
    "admission requires a nonempty retained population": ExclusionReason.RETAINED_POPULATION_EMPTY,
    "actor census does not cover the retained population": ExclusionReason.ACTOR_CENSUS_STALE,
    "grading census does not cover the complete retained population": ExclusionReason.GRADING_CENSUS_INCOMPLETE,
    "grading census counts do not cover the complete retained population": (
        ExclusionReason.GRADING_CENSUS_COUNTS_INCOMPLETE
    ),
    "admitted source has tasks without runnable graders": ExclusionReason.GRADER_MISSING,
}

POPULATION_BLOCKER_PREFIX_REASONS: tuple[tuple[str, ExclusionReason], ...] = (
    (
        "declared environment artifacts exist in no storage region: ",
        ExclusionReason.ENVIRONMENT_ARTIFACT_MISSING,
    ),
)

_LEXICAL_EVAL_MESSAGE = "Evaluation requires a measured lexical receipt for the current dataset pin"
_DECON_MEASUREMENT_MISSING = "Decontamination measurement and private-release admission receipt are missing"
_DECON_BLOCKED_PREFIX = "Decontamination admission blocked: "
_TASKSPEC_CONVERSION_BLOCKER = "admission requires a verified TaskSpec conversion receipt for the admitted revision"


def population_blocker_reason(blocker: str) -> ExclusionReason:
    """Map a population admission blocker string to its exclusion reason."""
    if reason := POPULATION_BLOCKER_REASONS.get(blocker):
        return reason
    for prefix, reason in POPULATION_BLOCKER_PREFIX_REASONS:
        if blocker.startswith(prefix):
            return reason
    raise ValueError(f"Unmapped population admission blocker: {blocker}")


def lexical_blocker_reason(message: str) -> ExclusionReason:
    """Map a lexical qualification failure to its exclusion reason.

    The evaluation path requires a measured receipt; every training-path
    failure requires decontamination remediation at the current pin. The
    failure messages are produced by ``finestore.rl.curation`` and
    ``server.admission``; the tests pin them against those modules.
    """
    if message in (_LEXICAL_EVAL_MESSAGE, _DECON_MEASUREMENT_MISSING) or message.startswith(_DECON_BLOCKED_PREFIX):
        return (
            ExclusionReason.LEXICAL_RECEIPT_MISSING
            if message == _LEXICAL_EVAL_MESSAGE
            else ExclusionReason.LEXICAL_REMEDIATION_REQUIRED
        )
    if message in (
        "Atlas admission requires a private open-athena TaskCompendium release",
        "Atlas remediation requires the original private open-athena snapshot",
    ):
        return ExclusionReason.LEXICAL_REMEDIATION_REQUIRED
    raise ValueError(f"Unmapped lexical qualification failure: {message}")


class PinnedReference(Protocol):
    """Catalog identity exposed by a dataset's existing pinned input."""

    @property
    def name(self) -> str: ...

    @property
    def revision(self) -> str | None: ...

    @property
    def url(self) -> str: ...


class CatalogConfig(Protocol):
    """Catalog facts; each dataset supplies its own execution configuration."""

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def dataset(self) -> PinnedReference | None: ...

    @property
    def files(self) -> tuple[str, ...]: ...

    @property
    def intended_use(self) -> IntendedUse: ...


@dataclass(frozen=True)
class SourceReference:
    """A pinned dataset reference."""

    name: str
    revision: str | None
    url: str


@dataclass(frozen=True, kw_only=True)
class SourceInfo:
    """Identity and discovery information for one source population.

    ``count`` is the number of selected input rows at the dataset revision,
    before conversion or curation. Leave it unknown unless the pinned payload
    or a complete manifest establishes it. A sample size is not a source count.

    ``dataset`` identifies inventory entries without a configuration. Runnable
    sources expose their configured pinned input directly. Tags describe task type,
    interaction, benchmark status, and source-specific search terms.
    """

    id: str = ""
    origin: SourceOrigin
    upstream_dataset: str
    upstream_name: str
    build_path: str
    replacement_source_id: str | None = None
    """The typed link to the replacement source of a superseded population.

    Ruling 2026-11-23: a source superseded by a replacement version is
    ``Deprecated`` with this link, never Excluded. The replacement must exist
    in the Atlas in any status; the audit and the status page both resolve it.
    """
    data_restriction: DataRestriction = DataRestriction.OPEN
    """The source's typed restriction tier, set explicitly on every configured source.

    ``PRIVATE_ALWAYS`` keeps the source's content in its private store no matter
    what else changes; the declaration and every derived artifact inherit it and
    a downgrade is refused. Every source definition passes the tier explicitly;
    the ``Open`` default exists only so archived receipts and other historical
    records that embed a ``SourceInfo`` copy from before the tier still validate.
    """
    subset: str | None = None
    variant: str | None = None
    title: str | None = None
    current_release: DatasetRelease | None = None
    inherited_admission: DatasetRelease | None = None
    upstream_url: str | None = None
    quality_result: SourceQualityResult | None = None
    quality_result_finalization: QualityResultFinalization | None = None
    quality_result_waiver: ContentEvidenceReference | None = None
    full_content_result_binding: FullContentResultBinding | None = None
    family: str = ""
    tags: tuple[str, ...] = ()
    count: int | None = None
    notes: str = ""
    dataset: SourceReference | None = None
    license_receipt: LicenseReceipt | None = None
    independent_qc: IndependentQC | None = None
    daytona_snapshot_count: int | None = None
    """Named actor snapshots required by the measured provisioning path.

    Snapshot-backed provisioning counts distinct environment identities. An exact
    declarative execution binding requires zero named snapshots and archives the
    original image-identity census. ``None`` means the count was never taken.
    """

    daytona_count_basis: str | None = None
    """Identity rule and payload source behind ``daytona_snapshot_count``."""

    task_format: TaskFormat | None = None
    """The structural shape of this source's served payload rows.

    Required for every registered source: a Harbor task package row, a TaskSpec
    row with no task package, or a MarinSkyRL blend record. Every catalog
    declaration resolves to exactly one shape and the conversion census verifies
    the declaration against every row's detected shape.
    """

    daytona_execution_failures: int | None = None
    """Tasks the Daytona backend refuses on their declared execution requirements."""

    daytona_run_evidence: str | None = None
    """A campaign run that executed this source on the Daytona RL org, when one exists."""

    oracle_validation: OracleValidation | None = None
    """How the source's grader was qualified.

    ``executed`` runs the shipped oracle solutions; ``verifier_executed`` records
    a negative verdict on an untouched submission without certifying an oracle;
    ``skipped`` certifies neither. The Atlas shows this value beside the catalog.
    """

    policy_reward_measurements: tuple[PolicyRewardMeasurement, ...] = ()
    """Measured rewards specific policies realized on this source's populations.

    A measured-limitation receipt, not an exclusion: the source's catalog status and
    intended use are unchanged, and each measurement pins the policy identity, the
    population, the realized reward and the evidence receipt behind it. The Atlas
    shows these beside ``oracle_validation``.
    """

    oversight: OversightEvidence | None = None
    """The evidence-derived oversight level and its receipt pointer."""

    recipe_qa_history: RecipeQAHistoricalNotes | None = None
    """Historical recipe-QA prose; it can never assert an oversight level."""

    derived_contract: DerivedContractSummary | None = None
    """The late-stage derived contract shape and grader types of the measured population.

    Computed from the final TaskSpecs, never from declarations. ``None`` means the census has
    not been bound for the current pin; declarations must not fill it in by hand.
    """

    grading_environment: GradingEnvironmentEvidence | None = None
    integrity_contracts: IntegrityContractSummary | None = None
    actor_census_input: ActorCensusInput | None = None
    consumer_receipt: ConsumerReceipt | None = None
    consumer_requirements: tuple[ConsumerNeed, ...] = ()
    consumer_decision: ConsumerDecision | None = None
    converted_examples: ConvertedExamples | None = None
    oracle_filtration: Annotated[OracleFiltrationEvidence | None, Field(exclude_if=lambda value: value is None)] = None
    no_oracle_executability_policy: Annotated[
        NoOracleExecutabilityPolicy | None, Field(exclude_if=lambda value: value is None)
    ] = None
    no_oracle_static_retention: Annotated[
        NoOracleStaticRetentionEvidence | None, Field(exclude_if=lambda value: value is None)
    ] = None
    within_source_dedup: WithinSourceDedupEvidence | None = None
    lexical_contamination: LexicalContaminationEvidence | None = None
    daytona_machine_slots: int | None = None
    daytona_roles: dict[str, dict[str, RoleExecutionEvidence]] | None = None
    daytona_unsafe_reasons: tuple[DaytonaUnsafeReason, ...] = ()
    daytona_unsafe_details: dict[str, str] | None = None
    daytona_unsafe_counts: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.origin, SourceOrigin):
            raise ValueError("origin must be an upstream SourceOrigin")
        expected_id = source_identifier(self.origin, self.upstream_dataset, self.subset, self.variant)
        if self.id and canonical_source_id(self.id) != expected_id:
            raise ValueError("Source ID differs from generated identity")
        object.__setattr__(self, "id", expected_id)
        title = source_display_name(expected_id)
        if self.title is not None and self.title != title:
            # Archived receipt copies written before the identity-derived title
            # carry the legacy <upstream_name> · <subset> · <variant> form; they
            # validate as archived aliases and normalize on read. A hand-set
            # title that matches neither rule still fails.
            legacy = " · ".join(part for part in (self.upstream_name, self.subset, self.variant) if part)
            if self.title != legacy:
                raise ValueError("Source title differs from generated title")
        object.__setattr__(self, "title", title)


@dataclass(frozen=True)
class RlDataSource[ConfigT: CatalogConfig]:
    """One source population, its dataset-owned configuration and callable.

    Stable IDs preserve Atlas review history even when the callable changes.
    A catalog declaration requires a registered conversion pipeline. QUICK
    records conversion without review; it does not make tasks selectable.
    """

    info: SourceInfo
    config: ConfigT | None = None
    pipeline: "CurationPipeline[ConfigT] | None" = None

    def require_license(self) -> None:
        """Require stage-0 permission at the current pin before constructing any work."""
        dataset = self.dataset
        require_unbanned(self.info.id, dataset.name if dataset else None, self.info.license_receipt)
        require_license(
            self.info.license_receipt,
            self.info.id,
            dataset.name if dataset else None,
            dataset.revision if dataset else None,
            self.intended_use.value,
            ban=None,
        )

    @property
    def intended_use(self) -> SourceUse:
        """Resolve population policy from the registered converter."""
        if self.config is None:
            raise ValueError(f"No registered conversion Use: {self.info.id}")
        return SourceUse(self.config.intended_use.value)

    @property
    def name(self) -> str:
        return self.config.name if self.config is not None else self.info.id.partition(":")[2]

    @property
    def version(self) -> str:
        return self.config.version if self.config is not None else "1"

    @property
    def files(self) -> tuple[str, ...]:
        return self.config.files if self.config is not None else ()

    @property
    def dataset(self) -> PinnedReference | None:
        return self.config.dataset if self.config is not None else self.info.dataset

    @property
    def pipeline_reference(self) -> PipelineReference:
        if self.pipeline is None or self.config is None:
            raise ValueError(f"Source {self.info.id} has no registered conversion pipeline")
        dataset = self.dataset
        return PipelineReference(
            name=self.name,
            version=self.version,
            source=dataset.name if dataset else None,
            revision=dataset.revision if dataset else None,
            files=list(self.files),
        )

    @property
    def has_actor_census(self) -> bool:
        info = self.info
        return (
            info.daytona_snapshot_count is not None
            and info.daytona_machine_slots is not None
            and info.daytona_execution_failures is not None
            and bool((info.daytona_count_basis or "").strip())
        )

    @property
    def catalog_declaration(self) -> SourceDeclaration | None:
        """Hold metadata pending while QUICK produces the actor census."""
        filtered_to_zero = self.info.oracle_filtration is not None and self.info.oracle_filtration.admitted_count == 0
        if not self.has_actor_census and not filtered_to_zero:
            return None
        if self.info.task_format is None:
            raise ValueError(
                f"Source {self.info.id} requires a declared task format: "
                "every catalog declaration resolves to exactly one TaskFormat value"
            )
        return self.declaration

    @property
    def current_release(self) -> DatasetRelease:
        dataset = self.dataset
        if dataset is None or not dataset.revision:
            raise ValueError(f"Source {self.info.id} requires an exact current release pin")
        release = DatasetRelease(repo=dataset.name, revision=dataset.revision, url=dataset.url)
        declared = self.info.current_release
        if declared is not None and declared != release:
            raise ValueError("Current release must match the configured dataset pin")
        return release

    @property
    def admission_blockers(self) -> tuple[str, ...]:
        """Evidence missing before a source can enter admitted selection."""
        info = self.info
        blockers = []
        if self.approved_consumer_status is not None:
            assert info.consumer_decision is not None
            blockers.append(info.consumer_decision.reason.value)
        dataset = self.dataset
        if ban := source_license_ban(info.id, dataset.name if dataset else None, info.license_receipt):
            blockers.append(f"license ban: {ban.reason} ({ban.issue_url})")
        if info.oracle_filtration is not None and info.oracle_filtration.admitted_count == 0:
            blockers.append(
                "oracle filtration retained no tasks: " + ", ".join(sorted(info.oracle_filtration.exclusion_counts))
            )
        actor = info.actor_census_input
        if actor is not None and (
            dataset is None
            or (actor.dataset_id, actor.dataset_revision, actor.converter_version)
            != (dataset.name, dataset.revision, self.version)
        ):
            blockers.append("actor census does not cover the current configured input")
        if blocker := population_admission_blocker(
            actor,
            info.within_source_dedup.unique_count if info.within_source_dedup else 0,
            info.grading_environment,
        ):
            blockers.append(blocker)
        try:
            require_lexical_qualification(
                info.lexical_contamination,
                dataset.name if dataset else None,
                dataset.revision if dataset else None,
                self.intended_use,
            )
        except ValueError as error:
            blockers.append(str(error))
        if info.task_format is not TaskFormat.TASKSPEC:
            blockers.append(_TASKSPEC_CONVERSION_BLOCKER)
        if not self.has_actor_census:
            blockers.append("actor census evidence is missing")
        if (
            info.oversight is None
            or info.oversight.level is OversightLevel.NONE
            or not info.oversight.verification.strip()
        ):
            blockers.append("evidence-derived oversight with verification is missing")
        if "agentic" not in info.tags:
            return tuple(blockers)
        if info.task_format is None:
            blockers.append("accepted agentic task format is missing")
        if info.daytona_snapshot_count is None or not (info.daytona_count_basis or "").strip():
            blockers.append("production snapshot identity measurement is missing")
        if info.daytona_execution_failures is None:
            blockers.append("execution requirement measurement is missing")
        return tuple(blockers)

    @property
    def admission_claims(self) -> tuple[ExclusionClaim, ...]:
        """Typed exclusion claims mirroring ``admission_blockers`` one-to-one.

        Every blocker string this source can emit maps to exactly one
        ``ExclusionReason`` with typed evidence; ``population_blocker_reason``
        and the lexical message rules below define the mapping. This tree is
        deliberately parallel to ``admission_blockers`` while the taxonomy
        lands: the blocker strings are contract (notes, error messages and
        tests), and a follow-up can derive them from the claims' evidence
        messages once the status page is serving.
        """
        info = self.info
        claims: list[ExclusionClaim] = []
        dataset = self.dataset
        if ban := source_license_ban(info.id, dataset.name if dataset else None, info.license_receipt):
            claims.append(
                ExclusionClaim(
                    reason=ExclusionReason.LICENSE_BAN,
                    evidence=ExclusionEvidence(
                        kind=ExclusionEvidenceKind.ISSUE_LINK, message=ban.reason, uri=ban.issue_url
                    ),
                )
            )
        if info.oracle_filtration is not None and info.oracle_filtration.admitted_count == 0:
            claims.append(
                ExclusionClaim(
                    reason=ExclusionReason.ORACLE_FILTRATION_RETAINED_ZERO,
                    evidence=ExclusionEvidence(
                        kind=ExclusionEvidenceKind.RECEIPT,
                        message="oracle filtration retained no tasks: "
                        + ", ".join(sorted(info.oracle_filtration.exclusion_counts)),
                        uri=info.oracle_filtration.validation.uri,
                    ),
                )
            )
        actor = info.actor_census_input
        if actor is not None and (
            dataset is None
            or (actor.dataset_id, actor.dataset_revision, actor.converter_version)
            != (dataset.name, dataset.revision, self.version)
        ):
            claims.append(
                ExclusionClaim(
                    reason=ExclusionReason.ACTOR_CENSUS_STALE,
                    evidence=ExclusionEvidence(
                        kind=ExclusionEvidenceKind.CHECK_OUTPUT,
                        message="actor census does not cover the current configured input",
                        expected=f"census for ({dataset.name if dataset else None}, "
                        f"{dataset.revision if dataset else None}, {self.version})",
                    ),
                )
            )
        if blocker := population_admission_blocker(
            actor,
            info.within_source_dedup.unique_count if info.within_source_dedup else 0,
            info.grading_environment,
        ):
            claims.append(
                ExclusionClaim(
                    reason=population_blocker_reason(blocker),
                    evidence=ExclusionEvidence(kind=ExclusionEvidenceKind.CHECK_OUTPUT, message=blocker),
                )
            )
        try:
            require_lexical_qualification(
                info.lexical_contamination,
                dataset.name if dataset else None,
                dataset.revision if dataset else None,
                self.intended_use,
            )
        except ValueError as error:
            claims.append(
                ExclusionClaim(
                    reason=lexical_blocker_reason(str(error)),
                    evidence=ExclusionEvidence(kind=ExclusionEvidenceKind.CHECK_OUTPUT, message=str(error)),
                )
            )
        if info.task_format is not TaskFormat.TASKSPEC:
            claims.append(
                ExclusionClaim(
                    reason=ExclusionReason.NOT_TASKSPEC_CONVERTED,
                    evidence=ExclusionEvidence(
                        kind=ExclusionEvidenceKind.MISSING_ARTIFACT,
                        message=_TASKSPEC_CONVERSION_BLOCKER,
                        expected=(
                            "a verified TaskSpec conversion receipt covering the admitted revision: the "
                            "conversion census bound at the admitted pin for declared shape "
                            f"{info.task_format.value if info.task_format is not None else 'unspecified'}"
                        ),
                    ),
                )
            )
        if not self.has_actor_census:
            claims.append(
                ExclusionClaim(
                    reason=ExclusionReason.ACTOR_CENSUS_MISSING,
                    evidence=ExclusionEvidence(
                        kind=ExclusionEvidenceKind.MISSING_ARTIFACT,
                        message="actor census evidence is missing",
                        expected="actor census receipt at the current pin",
                    ),
                )
            )
        if (
            info.oversight is None
            or info.oversight.level is OversightLevel.NONE
            or not info.oversight.verification.strip()
        ):
            claims.append(
                ExclusionClaim(
                    reason=ExclusionReason.OVERSIGHT_MISSING,
                    evidence=ExclusionEvidence(
                        kind=ExclusionEvidenceKind.MISSING_ARTIFACT,
                        message="evidence-derived oversight with verification is missing",
                        expected="derived oversight with recorded verification (QUICK or stronger)",
                    ),
                )
            )
        if "agentic" not in info.tags:
            return tuple(claims)
        if info.task_format is None:
            claims.append(
                ExclusionClaim(
                    reason=ExclusionReason.AGENTIC_TASK_FORMAT_MISSING,
                    evidence=ExclusionEvidence(
                        kind=ExclusionEvidenceKind.MISSING_ARTIFACT,
                        message="accepted agentic task format is missing",
                        expected="a declared TaskFormat shape",
                    ),
                )
            )
        if info.daytona_snapshot_count is None or not (info.daytona_count_basis or "").strip():
            claims.append(
                ExclusionClaim(
                    reason=ExclusionReason.SNAPSHOT_IDENTITY_MISSING,
                    evidence=ExclusionEvidence(
                        kind=ExclusionEvidenceKind.MISSING_ARTIFACT,
                        message="production snapshot identity measurement is missing",
                        expected="production snapshot identity measurement at the current pin",
                    ),
                )
            )
        if info.daytona_execution_failures is None:
            claims.append(
                ExclusionClaim(
                    reason=ExclusionReason.EXECUTION_REQUIREMENT_MISSING,
                    evidence=ExclusionEvidence(
                        kind=ExclusionEvidenceKind.MISSING_ARTIFACT,
                        message="execution requirement measurement is missing",
                        expected="execution requirement measurement",
                    ),
                )
            )
        return tuple(claims)

    @property
    def exclusion_claims(self) -> tuple[ExclusionClaim, ...]:
        """All typed exclusion claims: admission-derived claims plus recorded manual rulings.

        A superseded source carries its replacement link instead of exclusion
        claims: it is Deprecated, and a Deprecated row carries no exclusion
        reason (ruling 2026-11-23).
        """
        if self.info.replacement_source_id:
            return ()
        return self.admission_claims + manual_exclusion_claims(self.info.id)

    @property
    def snapshot_repair_queued(self) -> bool:
        """The measured provisioning exceeds the snapshot budget.

        The Daytona snapshot budget is a capacity constraint, not a property of
        the data: while it binds, the source waits Pending for the snapshot-safe
        rebuild instead of being Excluded (ruling 2026-11-23). Other recorded
        claims stay visible beside the repair stage.
        """
        return self.info.daytona_unsafe_reasons == (DaytonaUnsafeReason.TOO_MANY_SNAPSHOTS,)

    @property
    def status(self) -> SourceStatus:
        """The catalog status for the Atlas taxonomy.

        A superseded source is Deprecated with its typed replacement link. The
        snapshot budget is a capacity constraint: a source whose measured
        provisioning exceeds it waits Pending for the snapshot-safe repair,
        whatever other claims stay visible beside the repair stage. Not
        selectable folds into Excluded. A deprecated
        tag yields Deprecated only when no exclusion reason applies; a
        Deprecated source must be trainable. An ``excluded`` tag without any
        claim is a registry bug and fails loudly. A source whose only exclusion
        claim is the missing lexical remediation receipt queues as Pending with
        the remediation stage when a remediation agent can produce it: no
        user-ruling claim applies and the population is nonempty. A legacy
        ``excluded`` tag without a recorded ruling does not override that queue
        (user ruling 2026-11-17: lexical-remediation sources are Pending, not
        Excluded); ruled exclusions and zero-task sources stay Excluded.
        """
        if self.info.replacement_source_id:
            return SourceStatus.DEPRECATED
        claims = self.exclusion_claims
        if "excluded" in self.info.tags and not claims:
            raise ValueError(
                f"Source {self.info.id} carries an excluded tag without a typed exclusion claim; "
                "record its reason in exclusion_registry"
            )
        if claims or "excluded" in self.info.tags:
            reasons = {claim.reason for claim in claims}
            if (
                not manual_exclusion_claims(self.info.id)
                and reasons == {ExclusionReason.LEXICAL_REMEDIATION_REQUIRED}
                and (self.info.within_source_dedup.unique_count or 0) > 0
            ):
                return SourceStatus.PENDING
            return SourceStatus.EXCLUDED
        if self.snapshot_repair_queued:
            return SourceStatus.PENDING
        if "deprecated" in self.info.tags:
            return SourceStatus.DEPRECATED
        if approved := self.approved_consumer_status:
            return approved
        return SourceStatus.AVAILABLE

    @property
    def stage(self) -> PendingStage | None:
        """The evidence-computed intake stage of a Pending declaration.

        A Pending source whose measured provisioning exceeds the snapshot
        budget queues for the snapshot-safe repair; a Pending source with
        claims queues for lexical remediation; a Pending source without claims
        waits on its consumer admission decision. Any other status carries no
        stage. Load-time backend downgrades assign the backend stages on top of
        this value.
        """
        if self.status is not SourceStatus.PENDING:
            return None
        if self.exclusion_claims:
            return PendingStage.LEXICAL_REMEDIATION_QUEUED
        if self.snapshot_repair_queued:
            return PendingStage.SNAPSHOT_REPAIR_QUEUED
        return PendingStage.CONSUMER_CHECK_PENDING

    @property
    def consumer_findings(self) -> tuple[ConsumerReason, ...]:
        return consumer_blockers(
            self.info.consumer_receipt,
            self.info.id,
            self.info.actor_census_input,
            self.info.within_source_dedup.unique_count if self.info.within_source_dedup else 0,
            self.info.consumer_requirements,
            runtime_commit=CONSUMER_RUNTIME_COMMIT,
        )

    @property
    def approved_consumer_status(self) -> SourceStatus | None:
        status = consumer_decision_status(
            self.info.consumer_decision,
            self.info.id,
            self.info.actor_census_input,
            self.consumer_findings,
            runtime_commit=CONSUMER_RUNTIME_COMMIT,
        )
        return SourceStatus(status) if status is not None else None

    @property
    def derived_turns(self) -> str:
        """The population's derived turn shape for the Atlas, empty while the census is pending."""
        summary = self.info.derived_contract
        if summary is None:
            return ""
        if summary.single_turn_count == summary.task_count:
            return "single-turn"
        if summary.single_turn_count == 0:
            return "multi-turn"
        return "mixed"

    @property
    def declaration(self) -> SourceDeclaration:
        """Project a source declaration into catalog metadata without executing it."""
        info, pipeline = self.info, self.pipeline
        if pipeline is None or self.config is None:
            raise ValueError(f"Source {info.id} has no registered conversion pipeline")
        dataset = self.dataset
        invocation = self.pipeline_reference
        dedup = info.within_source_dedup
        if dedup is None:
            raise ValueError(f"Source {info.id} requires a complete within-source deduplication census")
        tags = info.tags
        if info.oracle_filtration is not None and info.oracle_filtration.admitted_count == 0:
            safety = None
        else:
            if (
                info.daytona_snapshot_count is None
                or info.daytona_machine_slots is None
                or info.daytona_execution_failures is None
                or not info.daytona_count_basis
                or not info.daytona_count_basis.strip()
            ):
                raise ValueError(f"Source {info.id} requires an actor safety measurement before catalog declaration")
            release = self.current_release
            measurement = DaytonaMeasurement(
                dataset_id=release.repo,
                dataset_revision=release.revision,
                daytona_snapshot_count=info.daytona_snapshot_count,
                daytona_machine_slots=info.daytona_machine_slots,
                daytona_execution_failures=info.daytona_execution_failures,
                daytona_count_basis=info.daytona_count_basis,
                daytona_unsafe_reasons=list(info.daytona_unsafe_reasons),
                daytona_unsafe_details=info.daytona_unsafe_details or {},
                daytona_unsafe_counts=info.daytona_unsafe_counts or {},
                daytona_roles=info.daytona_roles,
            )
            safety = measurement.safety
        return SourceDeclaration(
            id=info.id,
            display_name=source_display_name(info.id),
            origin=info.origin,
            build_path=info.build_path,
            upstream_dataset=info.upstream_dataset,
            upstream_name=info.upstream_name,
            subset=info.subset,
            variant=info.variant,
            current_release=self.current_release,
            quality_result=info.quality_result,
            quality_result_finalization=info.quality_result_finalization,
            quality_result_waiver=info.quality_result_waiver,
            family=info.family,
            tags=list(tags),
            task_count=dedup.unique_count,
            within_source_dedup=dedup,
            notes=" ".join(
                (info.notes, *(f"Admission blocked: {reason}." for reason in self.admission_blockers))
            ).strip(),
            dataset_id=dataset.name if dataset else None,
            dataset_revision=dataset.revision if dataset else None,
            url=info.upstream_url or (dataset.url if dataset else None),
            task_format=info.task_format,
            type=next(
                (
                    label
                    for tag, label in (
                        ("eval", "eval"),
                        ("rlvr", "RLVR"),
                        ("agentic", "Agentic"),
                        ("alignment", "Alignment"),
                    )
                    if tag in tags
                ),
                "",
            ),
            turns=next(
                (label for tag, label in (("single-turn", "Single-turn"), ("multi-turn", "Multi-turn")) if tag in tags),
                "",
            ),
            is_benchmark="benchmark" in tags,
            intended_use=self.intended_use,
            data_restriction=info.data_restriction,
            kind="Generator" if "generator" in tags else "Dataset",
            status=self.status,
            replacement_source_id=info.replacement_source_id,
            stage=self.stage,
            exclusion_claims=list(self.exclusion_claims),
            pipeline=invocation,
            license_receipt=info.license_receipt,
            independent_qc=info.independent_qc,
            policy_reward_measurements=list(info.policy_reward_measurements),
            daytona_safety=safety,
            daytona_snapshot_count=info.daytona_snapshot_count,
            daytona_count_basis=info.daytona_count_basis,
            oracle_validation=info.oracle_validation,
            oversight=info.oversight.level if info.oversight else OversightLevel.NONE,
            oversight_evidence=info.oversight,
            derived_agentic=None if info.derived_contract is None else info.derived_contract.agentic_count > 0,
            derived_turns=self.derived_turns,
            derived_reply_only=(
                None
                if info.derived_contract is None
                else info.derived_contract.reply_only_count == info.derived_contract.task_count
            ),
            derived_workspace_graded=(
                None
                if info.derived_contract is None
                else info.derived_contract.workspace_graded_count == info.derived_contract.task_count
            ),
            grader_type=(info.derived_contract.grader_type.value if info.derived_contract is not None else ""),
            grader_type_counts=dict(info.derived_contract.grader_type_counts) if info.derived_contract else {},
            judge_rubric_counts=dict(info.derived_contract.judge_rubric_counts) if info.derived_contract else {},
            judge_reference_count=(info.derived_contract.judge_reference_count if info.derived_contract else 0),
            judge_no_reference_count=(info.derived_contract.judge_no_reference_count if info.derived_contract else 0),
            recipe_qa_history=info.recipe_qa_history,
            grading_environment=info.grading_environment,
            integrity_contracts=info.integrity_contracts,
            actor_census_input=info.actor_census_input,
            consumer_receipt=info.consumer_receipt,
            consumer_requirements=info.consumer_requirements,
            consumer_findings=self.consumer_findings,
            consumer_decision=info.consumer_decision,
            converted_examples=info.converted_examples,
            oracle_filtration=info.oracle_filtration,
            no_oracle_executability_policy=info.no_oracle_executability_policy,
            no_oracle_static_retention=info.no_oracle_static_retention,
            lexical_contamination=info.lexical_contamination,
            daytona_machine_slots=info.daytona_machine_slots,
            daytona_roles=info.daytona_roles,
            daytona_execution_failures=info.daytona_execution_failures,
            daytona_unsafe_reasons=list(info.daytona_unsafe_reasons),
            daytona_unsafe_details=info.daytona_unsafe_details or {},
            daytona_unsafe_counts=info.daytona_unsafe_counts or {},
            daytona_snapshot_threshold=(
                DAYTONA_SAFE_SNAPSHOT_LIMIT if info.daytona_snapshot_count is not None else None
            ),
        )


class CurationPipeline[ConfigT: CatalogConfig](Protocol):
    def __call__(self, source: RlDataSource[ConfigT], options: PipelineOptions) -> ArtifactStep[CampaignArtifact]: ...
