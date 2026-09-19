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


class ProbeKind(StrEnum):
    ENTRY = "entry"
    REPRESENTATIVE = "representative"


EXPECTED_PROBE_KINDS = (ProbeKind.ENTRY, ProbeKind.REPRESENTATIVE)
MIN_COSINE_SIMILARITY = -1.0
MAX_COSINE_SIMILARITY = 1.0
PILOT_READY_SCORE = 85
REGENERATE_SCORE = 70


class SampleTask(StrictModel):
    kind: ProbeKind
    instruction: str = Field(min_length=1)


class CurriculumNodeKind(StrEnum):
    CAPABILITY = "capability"
    GROUP = "group"


class SamplingFacet(StrictModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    description: str = Field(min_length=1)


class CurriculumNodeBase(StrictModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
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

    def capability_sections(self) -> list[CapabilitySection]:
        """Return trainable sections that may receive task assignments."""
        return [section for section in self.sections if isinstance(section, CapabilitySection)]


class RoutingFacet(StrEnum):
    SUBJECT_DOMAIN = "subject_domain"
    TASK_MECHANIC = "task_mechanic"


class AnchorKind(StrEnum):
    GRAPH = "graph"
    SECTION = "section"


class CatalogCurriculum(StrictModel):
    routing_facet: RoutingFacet
    curriculum: Curriculum


class CurriculumCatalog(StrictModel):
    catalog_version: str
    curricula: list[CatalogCurriculum] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_catalog(self) -> CurriculumCatalog:
        subject_ids = [entry.curriculum.subject_id for entry in self.curricula]
        if len(subject_ids) != len(set(subject_ids)):
            raise ValueError("subject IDs must be unique across curricula")
        section_ids = [section.id for entry in self.curricula for section in entry.curriculum.sections]
        if len(section_ids) != len(set(section_ids)):
            raise ValueError("section IDs must be unique across curricula")
        return self


class AssignmentAnchor(StrictModel):
    kind: AnchorKind
    subject_id: str
    section_id: str | None
    text: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_target(self) -> AssignmentAnchor:
        if self.kind == AnchorKind.GRAPH and self.section_id is not None:
            raise ValueError("graph anchors cannot name a section")
        if self.kind == AnchorKind.SECTION and self.section_id is None:
            raise ValueError("section anchors must name a section")
        return self


class SemanticKey(StrictModel):
    subject_domain: str = Field(
        min_length=1,
        description="Operational knowledge domain needed to solve the task, excluding narrative subject matter",
    )
    task_mechanic: str = Field(
        min_length=1,
        description="Domain-independent transformation or reasoning action required by the task",
    )
    summary: str = Field(min_length=1, description="Requested result without source or curriculum labels")
    hardest_operation: str = Field(min_length=1)
    required_operations: list[str] = Field(min_length=1)
    answer_form: str = Field(min_length=1)

    def membership_text(self, facet: RoutingFacet) -> str:
        if facet == RoutingFacet.SUBJECT_DOMAIN:
            return f"Subject domain: {self.subject_domain}."
        return f"Task mechanic: {self.task_mechanic}."

    def operation_text(self) -> str:
        return (
            f"Task: {self.summary}. "
            f"Hardest operation: {self.hardest_operation}. "
            f"Required operations: {'; '.join(self.required_operations)}. "
            f"Answer form: {self.answer_form}."
        )


class TaskAnnotation(StrictModel):
    task_id: str
    task_hash: str
    model: str
    prompt_version: str
    key: SemanticKey


class MappingCandidate(StrictModel):
    section_id: str
    similarity: float = Field(ge=MIN_COSINE_SIMILARITY, le=MAX_COSINE_SIMILARITY)


class GraphMapping(StrictModel):
    subject_id: str
    routing_facet: RoutingFacet
    membership_similarity: float = Field(ge=MIN_COSINE_SIMILARITY, le=MAX_COSINE_SIMILARITY)
    candidates: list[MappingCandidate]


class TaskMapping(StrictModel):
    task_id: str
    task_hash: str
    annotation_model: str
    annotation_prompt_version: str
    catalog_version: str
    embedding_model: str
    graphs: list[GraphMapping]


class DifficultyIntent(StrEnum):
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
    progression_and_epsilon_continuity: int = Field(ge=0, le=25)
    observable_boundaries: int = Field(ge=0, le=15)
    probe_quality_and_parsimony: int = Field(ge=0, le=10)

    def total(self) -> int:
        return (
            self.coverage
            + self.mutual_self_confidence
            + self.progression_and_epsilon_continuity
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
