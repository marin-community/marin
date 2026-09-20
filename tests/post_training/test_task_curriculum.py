# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from experiments.post_training.task_curriculum.models import (
    BlindFitReview,
    BlindTaskSet,
    CapabilitySection,
    CatalogCurriculum,
    Curriculum,
    CurriculumCatalog,
    GapDispositionStatus,
    GroupSection,
    HolisticReview,
    HolisticReviewStatus,
    RoutingFacet,
    SampleTask,
    SystematicGapDisposition,
)
from experiments.post_training.task_curriculum.task_mapping.cache import EmbeddingCache, cached_embeddings
from experiments.post_training.task_curriculum.task_mapping.embedding import (
    MappingInputs,
    graph_anchors,
    map_task_vectors,
    section_anchors,
)
from experiments.post_training.task_curriculum.task_mapping.models import (
    AnchorKind,
    AssignmentAnchor,
    SemanticKey,
    TaskAnnotation,
)
from experiments.post_training.task_curriculum.validation import (
    MIN_BLIND_TASK_COUNT,
    SubjectEvidenceIds,
    SubjectRunArtifacts,
    validate_subject_promotion,
    validate_subject_run,
)


def _section(section_id: str, parent_id: str | None, task_texts: tuple[str, str]) -> CapabilitySection:
    return CapabilitySection(
        kind="capability",
        id=section_id,
        parent_id=parent_id,
        name=section_id,
        outcome=f"Complete {section_id} work",
        includes=[f"inside {section_id}"],
        excludes=[f"outside {section_id}"],
        prerequisites=[],
        sample_tasks=[
            SampleTask(kind="entry", instruction=task_texts[0]),
            SampleTask(kind="representative", instruction=task_texts[1]),
        ],
    )


def _group(section_id: str) -> GroupSection:
    return GroupSection(
        kind="group",
        id=section_id,
        parent_id=None,
        name=section_id,
        scope=f"Organize {section_id} capabilities",
        includes=[f"inside {section_id}"],
        excludes=[f"outside {section_id}"],
    )


def _curriculum(subject_id: str = "C00", prefix: str = "") -> Curriculum:
    return Curriculum(
        version="pilot-1",
        subject_id=subject_id,
        subject_name="Pilot",
        sections=[
            _group(f"{prefix}files"),
            _section(
                f"{prefix}files.search",
                f"{prefix}files",
                ("search file contents", "filter matching lines"),
            ),
            _group(f"{prefix}processes"),
            _section(
                f"{prefix}processes.logs",
                f"{prefix}processes",
                ("read service logs", "filter log entries"),
            ),
        ],
    )


def _catalog() -> CurriculumCatalog:
    return CurriculumCatalog(
        catalog_version="catalog-1",
        curricula=[
            CatalogCurriculum(routing_facet=RoutingFacet.SUBJECT_DOMAIN, curriculum=_curriculum()),
            CatalogCurriculum(
                routing_facet=RoutingFacet.TASK_MECHANIC,
                curriculum=_curriculum("C01", "practice."),
            ),
        ],
    )


def _annotation(task_id: str) -> TaskAnnotation:
    return TaskAnnotation(
        task_id=task_id,
        task_hash=f"hash-{task_id}",
        model="annotation-model",
        prompt_version="key-v1",
        key=SemanticKey(
            subject_domain="systems",
            task_mechanic="inspect and select an operating-system object",
            summary="inspect an operating-system object",
            hardest_operation="identify the relevant object",
            required_operations=["inspect", "select"],
            answer_form="text",
        ),
    )


def _subject_run() -> tuple[SubjectRunArtifacts, SubjectEvidenceIds]:
    curriculum = Curriculum(
        version="pilot-1",
        subject_id="C00",
        subject_name="Pilot",
        sections=[_section("c00.example", None, ("exercise an entry", "exercise a representative"))],
    )
    blind_tasks = BlindTaskSet.model_validate(
        {
            "subject_id": "C00",
            "prompt_version": "blind-v1",
            "tasks": [
                {
                    "id": f"blind-{index}",
                    "instruction": f"Exercise capability with input {index}.",
                    "guidepost_basis": ["C00.1"],
                    "operation_family": "exercise capability",
                    "difficulty_intent": "entry",
                }
                for index in range(MIN_BLIND_TASK_COUNT)
            ],
        }
    )
    judgments = [
        {
            "task_id": task.id,
            "status": "exact",
            "acceptable_capability_ids": ["c00.example"],
            "decisive_operation": "exercise capability",
            "explanation": "The complete task matches the capability.",
        }
        for task in blind_tasks.tasks
    ]
    fit_review = BlindFitReview.model_validate(
        {
            "subject_id": "C00",
            "curriculum_version": "pilot-1",
            "prompt_version": "fit-v1",
            "judgments": judgments,
            "counts": {"exact": MIN_BLIND_TASK_COUNT, "ambiguous": 0, "gap": 0, "invalid": 0},
            "fit_numerator": MIN_BLIND_TASK_COUNT,
            "fit_denominator": MIN_BLIND_TASK_COUNT,
            "systematic_gaps": [],
        }
    )
    holistic_review = HolisticReview.model_validate(
        {
            "subject_id": "C00",
            "curriculum_version": "pilot-1",
            "score": 85,
            "dimension_scores": {
                "coverage": 22,
                "mutual_self_confidence": 22,
                "progression_and_epsilon_continuity": 21,
                "observable_boundaries": 12,
                "probe_quality_and_parsimony": 8,
            },
            "status": "pilot_ready",
            "confidence": "medium",
            "blockers": [],
            "highest_risk_sections": ["c00.example"],
            "guidepost_accounting": [
                {
                    "guidepost_id": "C00.1",
                    "section_ids": ["c00.example"],
                    "rationale": "The capability covers the guidepost.",
                }
            ],
            "discovery_accounting": [],
            "evaluation_accounting": [],
            "findings": [],
            "recommended_changes": [],
            "proposed_rubric_changes": [],
        }
    )

    return (
        SubjectRunArtifacts(
            curriculum=curriculum,
            blind_tasks=blind_tasks,
            fit_review=fit_review,
            holistic_review=holistic_review,
            gap_dispositions=(),
        ),
        SubjectEvidenceIds(
            guideposts=frozenset({"C00.1"}),
            discovery_items=frozenset(),
            evaluation_items=frozenset(),
        ),
    )


def test_subject_promotion_rejects_non_ready_review() -> None:
    artifacts, evidence_ids = _subject_run()
    not_ready_review = artifacts.holistic_review.model_copy(
        update={
            "score": 84,
            "dimension_scores": artifacts.holistic_review.dimension_scores.model_copy(update={"coverage": 21}),
            "status": HolisticReviewStatus.REVISE,
        }
    )
    not_ready_artifacts = replace(artifacts, holistic_review=not_ready_review)

    validate_subject_run(not_ready_artifacts, evidence_ids)
    with pytest.raises(ValueError):
        validate_subject_promotion(not_ready_artifacts)


def test_subject_run_validation_rejects_stale_fit_version() -> None:
    artifacts, evidence_ids = _subject_run()
    stale_fit = artifacts.fit_review.model_copy(update={"curriculum_version": "pilot-0"})

    with pytest.raises(ValueError):
        validate_subject_run(replace(artifacts, fit_review=stale_fit), evidence_ids)


def test_subject_run_validation_rejects_unknown_capability_reference() -> None:
    artifacts, evidence_ids = _subject_run()
    invalid_fit = artifacts.fit_review.model_copy(deep=True)
    invalid_fit.judgments[0].acceptable_capability_ids = ["c00.unknown"]

    with pytest.raises(ValueError):
        validate_subject_run(replace(artifacts, fit_review=invalid_fit), evidence_ids)


def test_subject_promotion_rejects_confirmed_systematic_gap() -> None:
    artifacts, evidence_ids = _subject_run()
    fit_with_gap = artifacts.fit_review.model_copy(update={"systematic_gaps": ["missing operation"]})
    blocking_gap = SystematicGapDisposition(
        gap="missing operation",
        status=GapDispositionStatus.BLOCKING,
        rationale="Two blind tasks require the same uncovered operation.",
    )
    artifacts_with_gap = replace(
        artifacts,
        fit_review=fit_with_gap,
        gap_dispositions=(blocking_gap,),
    )

    validate_subject_run(artifacts_with_gap, evidence_ids)
    with pytest.raises(ValueError):
        validate_subject_promotion(artifacts_with_gap)


def test_subject_promotion_rejects_gap_in_uncovered_guidepost() -> None:
    artifacts, _ = _subject_run()
    uncovered_fit_data = artifacts.fit_review.model_dump()
    uncovered_fit_data["judgments"][0].update(status="gap", acceptable_capability_ids=[])
    uncovered_fit_data.update(
        counts={"exact": 23, "ambiguous": 0, "gap": 1, "invalid": 0},
        fit_numerator=23,
    )
    uncovered_fit = BlindFitReview.model_validate(uncovered_fit_data)
    uncovered_review = artifacts.holistic_review.model_copy(deep=True)
    uncovered_review.guidepost_accounting[0].section_ids = []

    with pytest.raises(ValueError):
        validate_subject_promotion(replace(artifacts, fit_review=uncovered_fit, holistic_review=uncovered_review))


def test_section_anchors_include_only_capabilities() -> None:
    catalog = _catalog()
    section_anchor_rows = section_anchors(catalog)
    section_ids = [anchor.section_id for anchor in section_anchor_rows]
    assert set(section_ids) == {
        "files.search",
        "processes.logs",
        "practice.files.search",
        "practice.processes.logs",
    }


def test_section_anchors_reject_groups() -> None:
    catalog = _catalog()
    with pytest.raises(ValueError):
        section_anchors(
            catalog,
            [
                AssignmentAnchor(
                    kind=AnchorKind.SECTION,
                    subject_id="C00",
                    section_id="files",
                    text="A group must not receive task assignments.",
                )
            ],
        )


def test_mapping_ranks_sections_independently_inside_each_graph() -> None:
    catalog = _catalog()
    graph_anchor_rows = graph_anchors(catalog)
    section_anchor_rows = section_anchors(catalog)
    section_ids = [anchor.section_id for anchor in section_anchor_rows]
    annotations = [_annotation("file-task"), _annotation("log-task"), _annotation("process-task")]
    graph_vectors = np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=np.float32)
    membership_vectors = {
        RoutingFacet.SUBJECT_DOMAIN: np.asarray([[1.0, 0.0]] * 3, dtype=np.float32),
        RoutingFacet.TASK_MECHANIC: np.asarray([[0.0, 1.0]] * 3, dtype=np.float32),
    }
    section_vectors = np.zeros((len(section_ids), 2), dtype=np.float32)
    target_vectors = {
        "files.search": [1.0, 0.0],
        "processes.logs": [0.0, 1.0],
        "practice.files.search": [1.0, 0.0],
        "practice.processes.logs": [0.0, 1.0],
    }
    for index, section_id in enumerate(section_ids):
        section_vectors[index] = target_vectors[section_id]
    task_vectors = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=np.float32)

    mappings = map_task_vectors(
        MappingInputs(
            annotations=annotations,
            membership_vectors=membership_vectors,
            operation_vectors=task_vectors,
            catalog=catalog,
            graph_anchor_rows=graph_anchor_rows,
            graph_anchor_vectors=graph_vectors,
            section_anchor_rows=section_anchor_rows,
            section_anchor_vectors=section_vectors,
            embedding_model="embed-v1",
            top_k=2,
            row_batch_size=1,
        )
    )

    for subject_id, prefix in (("C00", ""), ("C01", "practice.")):
        graph_rows = [next(graph for graph in mapping.graphs if graph.subject_id == subject_id) for mapping in mappings]
        assert [graph.candidates[0].section_id for graph in graph_rows] == [
            f"{prefix}files.search",
            f"{prefix}processes.logs",
            f"{prefix}processes.logs",
        ]
        assert all(len(graph.candidates) == 2 for graph in graph_rows)


def test_embedding_cache_reuses_vectors_by_text_and_model(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def embed(texts: Sequence[str]) -> np.ndarray:
        calls.append(list(texts))
        return np.asarray([[float(len(text)), 1.0] for text in texts], dtype=np.float32)

    cache_path = tmp_path / "nested" / "embeddings.sqlite"
    with EmbeddingCache(cache_path) as cache:
        first = cached_embeddings(cache, ["alpha", "beta", "alpha"], "embed-v1", embed)
        second = cached_embeddings(cache, ["beta", "alpha"], "embed-v1", embed)
        third = cached_embeddings(cache, ["alpha"], "embed-v2", embed)

    np.testing.assert_array_equal(first[[0, 1]], second[::-1])
    np.testing.assert_array_equal(first[0], first[2])
    np.testing.assert_array_equal(third, np.asarray([[5.0, 1.0]], dtype=np.float32))
    assert calls == [["alpha", "beta"], ["alpha"]]
