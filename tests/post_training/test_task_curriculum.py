# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from experiments.post_training.task_curriculum.cache import EmbeddingCache, cached_embeddings
from experiments.post_training.task_curriculum.mapping import curriculum_anchors, map_task_vectors
from experiments.post_training.task_curriculum.models import (
    Curriculum,
    CurriculumSection,
    SampleTask,
    SemanticKey,
    TaskAnnotation,
)


def _section(section_id: str, parent_id: str | None, task_texts: tuple[str, str]) -> CurriculumSection:
    return CurriculumSection(
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


def _curriculum() -> Curriculum:
    return Curriculum(
        version="pilot-1",
        subject_id="C00",
        subject_name="Pilot",
        sections=[
            _section("files", None, ("find a file", "rename a file")),
            _section("files.search", "files", ("search file contents", "filter matching lines")),
            _section("processes", None, ("inspect a process", "stop a process")),
            _section("processes.logs", "processes", ("read service logs", "filter log entries")),
        ],
    )


def _annotation(task_id: str) -> TaskAnnotation:
    return TaskAnnotation(
        task_id=task_id,
        task_hash=f"hash-{task_id}",
        model="annotation-model",
        prompt_version="key-v1",
        key=SemanticKey(
            subject="systems",
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
            CurriculumSection(
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


def test_curriculum_contract_rejects_depth_above_limit() -> None:
    with pytest.raises(ValueError, match="maximum depth 1, generated depth 2"):
        _curriculum().check_generation_contract(maximum_depth=1)


def test_mapping_uses_best_anchor_per_section_across_row_batches() -> None:
    curriculum = _curriculum()
    anchor_ids, _ = curriculum_anchors([curriculum])
    annotations = [_annotation("file-task"), _annotation("log-task"), _annotation("process-task")]
    anchor_vectors = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.8, 0.2, 0.0],
            [0.7, 0.3, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1],
            [0.0, 0.8, 0.2],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    task_vectors = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    mappings = map_task_vectors(
        annotations,
        task_vectors,
        [curriculum],
        anchor_ids,
        anchor_vectors,
        embedding_model="embed-v1",
        top_k=2,
        row_batch_size=1,
    )

    assert [mapping.candidates[0].section_id for mapping in mappings] == ["files", "processes.logs", "processes"]
    assert all(len(mapping.candidates) == 2 for mapping in mappings)
    assert {mapping.embedding_model for mapping in mappings} == {"embed-v1"}
    assert {tuple(mapping.curriculum_versions) for mapping in mappings} == {("C00",)}


def test_mapping_ranks_sections_across_curricula() -> None:
    first = Curriculum(
        version="first-1",
        subject_id="C01",
        subject_name="First",
        sections=[_section("first.files", None, ("find a file", "rename files"))],
    )
    second = Curriculum(
        version="second-1",
        subject_id="C02",
        subject_name="Second",
        sections=[_section("second.processes", None, ("inspect a process", "stop processes"))],
    )
    anchor_ids, _ = curriculum_anchors([first, second])

    mappings = map_task_vectors(
        [_annotation("process-task")],
        np.asarray([[0.0, 1.0]], dtype=np.float32),
        [first, second],
        anchor_ids,
        np.asarray([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=np.float32),
        embedding_model="embed-v1",
        top_k=2,
        row_batch_size=1,
    )

    assert [candidate.section_id for candidate in mappings[0].candidates] == ["second.processes", "first.files"]
    assert mappings[0].curriculum_versions == {"C01": "first-1", "C02": "second-1"}


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
