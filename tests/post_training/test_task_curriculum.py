# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from experiments.post_training.task_curriculum.cache import EmbeddingCache, cached_embeddings
from experiments.post_training.task_curriculum.mapping import (
    MappingInputs,
    graph_anchors,
    map_task_vectors,
    section_anchors,
)
from experiments.post_training.task_curriculum.models import (
    AnchorKind,
    AssignmentAnchor,
    CapabilitySection,
    CatalogCurriculum,
    Curriculum,
    CurriculumCatalog,
    GroupSection,
    RoutingFacet,
    SampleTask,
    SamplingFacet,
    SemanticKey,
    TaskAnnotation,
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


def test_curriculum_contract_rejects_incomplete_probe_pair() -> None:
    curriculum = Curriculum(
        version="pilot-1",
        subject_id="C00",
        subject_name="Pilot",
        sections=[
            CapabilitySection(
                kind="capability",
                id="files",
                parent_id=None,
                name="files",
                outcome="Complete file work",
                includes=["file work"],
                excludes=["process work"],
                prerequisites=[],
                sample_tasks=[SampleTask(kind="representative", instruction="rename a file")],
            )
        ],
    )

    with pytest.raises(ValueError):
        curriculum.check_generation_contract(maximum_depth=2)


def test_curriculum_rejects_group_prerequisite() -> None:
    with pytest.raises(ValueError, match="non-capability prerequisites"):
        Curriculum(
            version="pilot-1",
            subject_id="C00",
            subject_name="Pilot",
            sections=[
                _group("files"),
                CapabilitySection(
                    **(
                        _section("files.search", "files", ("find a file", "filter matching lines")).model_dump()
                        | {"prerequisites": ["files"]}
                    ),
                ),
            ],
        )


def test_sampling_facets_are_unique_within_capability() -> None:
    section = _section("translate", None, ("translate a greeting", "translate a short letter"))
    duplicate = SamplingFacet(id="language_direction", description="Source and target language pair")
    section_data = section.model_dump()
    section_data["sampling_facets"] = [duplicate, duplicate]

    with pytest.raises(ValueError, match="sampling facet IDs must be unique"):
        CapabilitySection.model_validate(section_data)


def test_curriculum_contract_rejects_depth_above_limit() -> None:
    with pytest.raises(ValueError):
        _curriculum().check_generation_contract(maximum_depth=1)


def test_mapping_ranks_sections_independently_inside_each_graph() -> None:
    catalog = _catalog()
    graph_anchor_rows = graph_anchors(catalog)
    section_anchor_rows = section_anchors(catalog)
    section_ids = [anchor.section_id for anchor in section_anchor_rows]
    assert set(section_ids) == {
        "files.search",
        "processes.logs",
        "practice.files.search",
        "practice.processes.logs",
    }
    with pytest.raises(ValueError, match="unknown section"):
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
