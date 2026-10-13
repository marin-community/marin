# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Data exchanged by curriculum generation, review, and task mapping."""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SubjectGuidepost(StrictModel):
    """One broad coverage target in the subject inventory."""

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)


class SubjectArea(StrictModel):
    """One subject root and its coverage guideposts."""

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    guideposts: list[SubjectGuidepost] = Field(min_length=1)


class SubjectInventory(StrictModel):
    """Versioned subject roots used to generate a curriculum catalog."""

    version: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    areas: list[SubjectArea] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_identifiers(self) -> SubjectInventory:
        area_ids = [area.id for area in self.areas]
        if len(area_ids) != len(set(area_ids)):
            raise ValueError("subject area IDs must be unique")
        guidepost_ids = [guidepost.id for area in self.areas for guidepost in area.guideposts]
        if len(guidepost_ids) != len(set(guidepost_ids)):
            raise ValueError("subject guidepost IDs must be unique")
        return self


class ProbeKind(StrEnum):
    ENTRY = "entry"
    REPRESENTATIVE = "representative"


EXPECTED_PROBE_KINDS = (ProbeKind.ENTRY, ProbeKind.REPRESENTATIVE)
PILOT_READY_SCORE = 85
REGENERATE_SCORE = 70
CURRICULUM_IDENTIFIER_PATTERN = r"^[a-z0-9][a-z0-9._-]*$"


class SampleTask(StrictModel):
    kind: ProbeKind
    instruction: str = Field(min_length=1)


class CurriculumNodeKind(StrEnum):
    CAPABILITY = "capability"
    GROUP = "group"


class SamplingFacet(StrictModel):
    id: str = Field(pattern=CURRICULUM_IDENTIFIER_PATTERN)
    description: str = Field(min_length=1)


class CurriculumNodeBase(StrictModel):
    id: str = Field(pattern=CURRICULUM_IDENTIFIER_PATTERN)
    parent_id: str | None
    name: str = Field(min_length=1)
    includes: list[str] = Field(min_length=1)
    excludes: list[str] = Field(min_length=1)


class CapabilitySection(CurriculumNodeBase):
    kind: Literal[CurriculumNodeKind.CAPABILITY]
    outcome: str = Field(min_length=1)
    prerequisites: list[str]
    sampling_facets: list[SamplingFacet] = Field(default_factory=list)
    sample_tasks: list[SampleTask]

    @model_validator(mode="after")
    def validate_sampling_facets(self) -> CapabilitySection:
        facet_ids = [facet.id for facet in self.sampling_facets]
        if len(facet_ids) != len(set(facet_ids)):
            raise ValueError(f"section {self.id} sampling facet IDs must be unique")
        return self

    def scope_text(self) -> str:
        return self.outcome


class GroupSection(CurriculumNodeBase):
    kind: Literal[CurriculumNodeKind.GROUP]
    scope: str = Field(min_length=1)

    def scope_text(self) -> str:
        return self.scope


CurriculumNode = Annotated[CapabilitySection | GroupSection, Field(discriminator="kind")]


def _reject_cycles(edges: dict[str, Iterable[str]], label: str) -> None:
    visited: set[str] = set()
    visiting: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise ValueError(f"{label} cycle includes {node}")
        if node in visited:
            return
        visiting.add(node)
        for dependency in edges[node]:
            visit(dependency)
        visiting.remove(node)
        visited.add(node)

    for node in edges:
        visit(node)


class Curriculum(StrictModel):
    version: str
    subject_id: str
    subject_name: str
    sections: list[CurriculumNode] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_graph(self) -> Curriculum:
        ids = [section.id for section in self.sections]
        if len(ids) != len(set(ids)):
            raise ValueError("section IDs must be unique")

        known = set(ids)
        capabilities = {section.id for section in self.sections if isinstance(section, CapabilitySection)}
        if not capabilities:
            raise ValueError("curriculum must contain at least one capability section")
        parent_edges: dict[str, list[str]] = {}
        prerequisite_edges: dict[str, list[str]] = {}
        for section in self.sections:
            parent_edges[section.id] = [] if section.parent_id is None else [section.parent_id]
            prerequisites = section.prerequisites if isinstance(section, CapabilitySection) else []
            prerequisite_edges[section.id] = prerequisites
            references = set(parent_edges[section.id]) | set(prerequisites)
            unknown = sorted(references - known)
            if unknown:
                raise ValueError(f"section {section.id} has unknown references: {unknown}")
            if section.id in references:
                raise ValueError(f"section {section.id} refers to itself")
            invalid_prerequisites = sorted(set(prerequisites) - capabilities)
            if invalid_prerequisites:
                raise ValueError(f"section {section.id} has non-capability prerequisites: {invalid_prerequisites}")

        parents = {section.parent_id for section in self.sections if section.parent_id is not None}
        empty_groups = sorted(
            section.id for section in self.sections if isinstance(section, GroupSection) and section.id not in parents
        )
        if empty_groups:
            raise ValueError(f"group sections must contain at least one child: {empty_groups}")

        _reject_cycles(parent_edges, "parent")
        _reject_cycles(prerequisite_edges, "prerequisite")
        return self

    def depths(self) -> dict[str, int]:
        by_id = {section.id: section for section in self.sections}
        depths: dict[str, int] = {}

        def depth(section_id: str) -> int:
            if section_id not in depths:
                parent_id = by_id[section_id].parent_id
                depths[section_id] = 1 if parent_id is None else depth(parent_id) + 1
            return depths[section_id]

        for section_id in by_id:
            depth(section_id)
        return depths

    def check_generation_contract(self, maximum_depth: int) -> None:
        actual_depth = max(self.depths().values())
        if actual_depth > maximum_depth:
            raise ValueError(f"maximum depth {maximum_depth}, generated depth {actual_depth}")
        wrong_probes = [
            section.id
            for section in self.sections
            if isinstance(section, CapabilitySection)
            if tuple(task.kind for task in section.sample_tasks) != EXPECTED_PROBE_KINDS
        ]
        if wrong_probes:
            raise ValueError(f"sections must have one entry and one representative probe in order: {wrong_probes}")
        embedded_prerequisites = [
            section.id for section in self.sections if isinstance(section, CapabilitySection) and section.prerequisites
        ]
        if embedded_prerequisites:
            raise ValueError(
                f"generated curricula leave learning prerequisites to the catalog-level pass: {embedded_prerequisites}"
            )

    def capability_sections(self) -> list[CapabilitySection]:
        return [section for section in self.sections if isinstance(section, CapabilitySection)]


class RoutingFacet(StrEnum):
    """Semantic-key projection used to route tasks into a curriculum graph."""

    SUBJECT_DOMAIN = "subject_domain"
    TASK_MECHANIC = "task_mechanic"


class CatalogCurriculum(StrictModel):
    """One subject curriculum and its declared graph-routing projection."""

    routing_facet: RoutingFacet
    curriculum: Curriculum


class CurriculumCatalog(StrictModel):
    catalog_version: str
    curricula: list[CatalogCurriculum] = Field(min_length=1)
    learning_progression: LearningProgression | None = None

    @model_validator(mode="after")
    def validate_catalog(self) -> CurriculumCatalog:
        subject_ids = [entry.curriculum.subject_id for entry in self.curricula]
        if len(subject_ids) != len(set(subject_ids)):
            raise ValueError("subject IDs must be unique across curricula")
        section_ids = [section.id for entry in self.curricula for section in entry.curriculum.sections]
        if len(section_ids) != len(set(section_ids)):
            raise ValueError("section IDs must be unique across curricula")
        if self.learning_progression is not None:
            embedded_prerequisites = [
                section.id
                for entry in self.curricula
                for section in entry.curriculum.capability_sections()
                if section.prerequisites
            ]
            if embedded_prerequisites:
                raise ValueError(
                    "catalogs with a learning progression cannot embed capability prerequisites: "
                    f"{embedded_prerequisites}"
                )
            self.learning_progression.validate_against_catalog(self)
        return self


class LearningProgressionWitness(StrictModel):
    """One concise prerequisite-to-entry task-family transfer example."""

    prerequisite_family: str = Field(min_length=3, max_length=160)
    dependent_entry_family: str = Field(min_length=3, max_length=160)
    shared_foundation: str = Field(min_length=3, max_length=200)
    new_operation: str = Field(min_length=3, max_length=160)


class LearningPrerequisiteEdge(StrictModel):
    """A structural hypothesis that one capability enables learning another."""

    prerequisite_id: str = Field(pattern=CURRICULUM_IDENTIFIER_PATTERN)
    dependent_id: str = Field(pattern=CURRICULUM_IDENTIFIER_PATTERN)
    enabled_scope: str = Field(min_length=3, max_length=240)
    transfer_basis: str = Field(min_length=3, max_length=320)
    artifact_substitution_test: str = Field(min_length=3, max_length=320)
    witnesses: list[LearningProgressionWitness] = Field(min_length=2, max_length=2)

    @model_validator(mode="after")
    def validate_witnesses(self) -> LearningPrerequisiteEdge:
        task_pairs = {(witness.prerequisite_family, witness.dependent_entry_family) for witness in self.witnesses}
        if len(task_pairs) != len(self.witnesses):
            raise ValueError("learning-prerequisite witnesses must use distinct task-family pairs")
        return self


class LearningProgression(StrictModel):
    """Catalog-level learning edges produced after capability generation."""

    catalog_version: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    scope_subject_ids: list[str] = Field(min_length=1)
    edges: list[LearningPrerequisiteEdge]

    @model_validator(mode="after")
    def validate_graph(self) -> LearningProgression:
        if len(self.scope_subject_ids) != len(set(self.scope_subject_ids)):
            raise ValueError("learning-progression subject IDs must be unique")
        pairs = [(edge.prerequisite_id, edge.dependent_id) for edge in self.edges]
        if len(pairs) != len(set(pairs)):
            raise ValueError("learning-prerequisite edges must be unique")
        if any(prerequisite == dependent for prerequisite, dependent in pairs):
            raise ValueError("learning-prerequisite edges cannot refer to themselves")
        nodes = {item for pair in pairs for item in pair}
        dependencies = {node: [] for node in nodes}
        for prerequisite, dependent in pairs:
            dependencies[dependent].append(prerequisite)
        _reject_cycles(dependencies, "learning-prerequisite")
        return self

    def validate_against_catalog(self, catalog: CurriculumCatalog) -> None:
        """Validate subject and capability references against one catalog."""

        if self.catalog_version != catalog.catalog_version:
            raise ValueError("learning progression and catalog versions differ")
        subject_capabilities = {
            entry.curriculum.subject_id: {section.id for section in entry.curriculum.capability_sections()}
            for entry in catalog.curricula
        }
        unknown_subjects = sorted(set(self.scope_subject_ids) - set(subject_capabilities))
        if unknown_subjects:
            raise ValueError(f"learning progression has unknown subjects: {unknown_subjects}")
        capability_subject = {
            capability_id: subject_id
            for subject_id, capability_ids in subject_capabilities.items()
            for capability_id in capability_ids
        }
        references = {
            capability_id for edge in self.edges for capability_id in (edge.prerequisite_id, edge.dependent_id)
        }
        unknown_capabilities = sorted(references - set(capability_subject))
        if unknown_capabilities:
            raise ValueError(f"learning progression has unknown capabilities: {unknown_capabilities}")
        out_of_scope_dependents = sorted(
            edge.dependent_id
            for edge in self.edges
            if capability_subject[edge.dependent_id] not in self.scope_subject_ids
        )
        if out_of_scope_dependents:
            raise ValueError(f"learning progression has out-of-scope dependents: {out_of_scope_dependents}")


CurriculumCatalog.model_rebuild()


class LearningEdgeVerdict(StrEnum):
    ACCEPT = "accept"
    REJECT = "reject"


class LearningEdgeReview(StrictModel):
    prerequisite_id: str = Field(pattern=CURRICULUM_IDENTIFIER_PATTERN)
    dependent_id: str = Field(pattern=CURRICULUM_IDENTIFIER_PATTERN)
    verdict: LearningEdgeVerdict
    rationale: str = Field(min_length=1)


class LearningProgressionRecommendation(StrEnum):
    ACCEPT = "accept"
    REVISE = "revise"


class LearningProgressionReview(StrictModel):
    """Independent edge review and complete witness-backed omissions."""

    catalog_version: str = Field(min_length=1)
    progression_prompt_version: str = Field(min_length=1)
    review_prompt_version: str = Field(min_length=1)
    scope_subject_ids: list[str] = Field(min_length=1)
    edge_reviews: list[LearningEdgeReview]
    missing_edges: list[LearningPrerequisiteEdge]
    findings: list[str]
    recommendation: LearningProgressionRecommendation

    def validate_against_progression(
        self,
        progression: LearningProgression,
        catalog: CurriculumCatalog,
    ) -> None:
        """Validate identity, exact edge accounting, and proposed omissions."""

        if self.catalog_version != progression.catalog_version:
            raise ValueError("learning-progression review and proposal catalog versions differ")
        if self.progression_prompt_version != progression.prompt_version:
            raise ValueError("learning-progression review names the wrong proposal prompt")
        if len(self.scope_subject_ids) != len(set(self.scope_subject_ids)):
            raise ValueError("learning-progression review subject IDs must be unique")
        if set(self.scope_subject_ids) != set(progression.scope_subject_ids):
            raise ValueError("learning-progression review and proposal scopes differ")
        expected = {(edge.prerequisite_id, edge.dependent_id) for edge in progression.edges}
        actual = [(edge.prerequisite_id, edge.dependent_id) for edge in self.edge_reviews]
        if len(actual) != len(set(actual)) or set(actual) != expected:
            raise ValueError("learning-progression review must judge every proposed edge exactly once")
        missing_pairs = [(edge.prerequisite_id, edge.dependent_id) for edge in self.missing_edges]
        if len(missing_pairs) != len(set(missing_pairs)) or set(missing_pairs) & expected:
            raise ValueError("missing learning-prerequisite edges must be unique and absent from the proposal")
        missing_progression = LearningProgression(
            catalog_version=self.catalog_version,
            prompt_version=self.review_prompt_version,
            scope_subject_ids=self.scope_subject_ids,
            edges=self.missing_edges,
        )
        missing_progression.validate_against_catalog(catalog)
        accepted_pairs = {
            (review.prerequisite_id, review.dependent_id)
            for review in self.edge_reviews
            if review.verdict == LearningEdgeVerdict.ACCEPT
        }
        proposed_by_pair = {(edge.prerequisite_id, edge.dependent_id): edge for edge in progression.edges}
        combined_progression = LearningProgression(
            catalog_version=self.catalog_version,
            prompt_version=self.review_prompt_version,
            scope_subject_ids=self.scope_subject_ids,
            edges=[proposed_by_pair[pair] for pair in accepted_pairs] + self.missing_edges,
        )
        combined_progression.validate_against_catalog(catalog)
        has_revision = any(edge.verdict == LearningEdgeVerdict.REJECT for edge in self.edge_reviews) or bool(
            self.missing_edges
        )
        expected_recommendation = (
            LearningProgressionRecommendation.REVISE if has_revision else LearningProgressionRecommendation.ACCEPT
        )
        if self.recommendation != expected_recommendation:
            raise ValueError("learning-progression recommendation does not match its edge findings")


class DifficultyIntent(StrEnum):
    """Sampling role used to audit blind-set diversity and hidden from the fit judge."""

    ENTRY = "entry"
    REPRESENTATIVE = "representative"
    BOUNDARY = "boundary"


class BlindTask(StrictModel):
    id: str = Field(min_length=1)
    instruction: str = Field(min_length=1)
    guidepost_basis: list[str] = Field(min_length=1)
    operation_family: str = Field(min_length=1)
    difficulty_intent: DifficultyIntent


class BlindTaskSet(StrictModel):
    subject_id: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    tasks: list[BlindTask] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_task_ids(self) -> BlindTaskSet:
        task_ids = [task.id for task in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("blind task IDs must be unique")
        return self


class BlindFitStatus(StrEnum):
    EXACT = "exact"
    AMBIGUOUS = "ambiguous"
    GAP = "gap"
    INVALID = "invalid"


class BlindFitJudgment(StrictModel):
    task_id: str = Field(min_length=1)
    status: BlindFitStatus
    acceptable_capability_ids: list[str]
    decisive_operation: str = Field(min_length=1)
    explanation: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_targets(self) -> BlindFitJudgment:
        target_count = len(self.acceptable_capability_ids)
        if self.status == BlindFitStatus.EXACT and target_count != 1:
            raise ValueError("exact fit judgments require one capability")
        if self.status == BlindFitStatus.AMBIGUOUS and target_count < 2:
            raise ValueError("ambiguous fit judgments require at least two capabilities")
        if self.status in {BlindFitStatus.GAP, BlindFitStatus.INVALID} and target_count:
            raise ValueError("gap and invalid fit judgments cannot name capabilities")
        if target_count != len(set(self.acceptable_capability_ids)):
            raise ValueError("acceptable capability IDs must be unique")
        return self


class BlindFitCounts(StrictModel):
    exact: int = Field(ge=0)
    ambiguous: int = Field(ge=0)
    gap: int = Field(ge=0)
    invalid: int = Field(ge=0)


class BlindFitReview(StrictModel):
    subject_id: str = Field(min_length=1)
    curriculum_version: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    judgments: list[BlindFitJudgment] = Field(min_length=1)
    counts: BlindFitCounts
    fit_numerator: int = Field(ge=0)
    fit_denominator: int = Field(gt=0)
    systematic_gaps: list[str]

    @model_validator(mode="after")
    def validate_totals(self) -> BlindFitReview:
        task_ids = [judgment.task_id for judgment in self.judgments]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("fit judgment task IDs must be unique")
        expected = BlindFitCounts(
            exact=sum(judgment.status == BlindFitStatus.EXACT for judgment in self.judgments),
            ambiguous=sum(judgment.status == BlindFitStatus.AMBIGUOUS for judgment in self.judgments),
            gap=sum(judgment.status == BlindFitStatus.GAP for judgment in self.judgments),
            invalid=sum(judgment.status == BlindFitStatus.INVALID for judgment in self.judgments),
        )
        if self.counts != expected:
            raise ValueError("fit counts do not match judgments")
        if self.fit_numerator != expected.exact + expected.ambiguous:
            raise ValueError("fit numerator must equal exact plus ambiguous")
        if self.fit_denominator != len(self.judgments) - expected.invalid:
            raise ValueError("fit denominator must exclude invalid tasks")
        if len(self.systematic_gaps) != len(set(self.systematic_gaps)):
            raise ValueError("systematic gaps must be unique")
        return self


class GapDispositionStatus(StrEnum):
    BLOCKING = "blocking"
    REJECTED = "rejected"


class SystematicGapDisposition(StrictModel):
    gap: str = Field(min_length=1)
    status: GapDispositionStatus
    rationale: str = Field(min_length=1)


class EvidenceStatus(StrEnum):
    SUPPORT = "support"
    EXCLUDED = "excluded"
    MALFORMED = "malformed"
    UNDERDETERMINED = "underdetermined"


class GuidepostAccounting(StrictModel):
    guidepost_id: str = Field(min_length=1)
    section_ids: list[str]
    rationale: str = Field(min_length=1)


class EvidenceAccounting(StrictModel):
    item_id: str = Field(min_length=1)
    status: EvidenceStatus
    section_ids: list[str]
    rationale: str = Field(min_length=1)


class HolisticReviewStatus(StrEnum):
    PILOT_READY = "pilot_ready"
    REVISE = "revise"
    REGENERATE = "regenerate"


class EvidenceConfidence(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class HolisticDimensionScores(StrictModel):
    coverage: int = Field(ge=0, le=25)
    mutual_self_confidence: int = Field(ge=0, le=25)
    local_progression: int = Field(ge=0, le=25)
    observable_boundaries: int = Field(ge=0, le=15)
    probe_quality_and_parsimony: int = Field(ge=0, le=10)

    def total(self) -> int:
        return (
            self.coverage
            + self.mutual_self_confidence
            + self.local_progression
            + self.observable_boundaries
            + self.probe_quality_and_parsimony
        )


class HolisticReview(StrictModel):
    subject_id: str = Field(min_length=1)
    curriculum_version: str = Field(min_length=1)
    score: int = Field(ge=0, le=100)
    dimension_scores: HolisticDimensionScores
    status: HolisticReviewStatus
    confidence: EvidenceConfidence
    blockers: list[str]
    highest_risk_sections: list[str]
    guidepost_accounting: list[GuidepostAccounting]
    discovery_accounting: list[EvidenceAccounting]
    evaluation_accounting: list[EvidenceAccounting]
    findings: list[str]
    recommended_changes: list[str]
    proposed_rubric_changes: list[str]

    @model_validator(mode="after")
    def validate_score_and_gate(self) -> HolisticReview:
        if self.score != self.dimension_scores.total():
            raise ValueError("holistic score must equal the dimension-score sum")
        is_pilot_ready = self.score >= PILOT_READY_SCORE and not self.blockers
        if is_pilot_ready:
            expected_status = HolisticReviewStatus.PILOT_READY
        elif self.score < REGENERATE_SCORE:
            expected_status = HolisticReviewStatus.REGENERATE
        else:
            expected_status = HolisticReviewStatus.REVISE
        if self.status != expected_status:
            raise ValueError("holistic status does not match the score and blocker gate")
        return self
