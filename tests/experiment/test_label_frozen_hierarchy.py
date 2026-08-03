# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import sys
from dataclasses import asdict
from pathlib import Path

import pytest

PROJECT = Path(__file__).parents[2] / ".agents/projects/luxical-arctic-poc"
sys.path.insert(0, str(PROJECT))

import label_frozen_hierarchy as frozen  # noqa: E402
import label_frozen_hierarchy_training as training  # noqa: E402
from glm_semantic_labels import SampleDocument, write_jsonl  # noqa: E402
from rigging.filesystem import StoragePath  # noqa: E402


def document(index: int, source: str = "source") -> SampleDocument:
    return SampleDocument(index, f"sha-{index}", source, "standard", index, f"text-{index}")


def test_documents_excluding_remove_all_specified_rows_and_reindex(monkeypatch: pytest.MonkeyPatch) -> None:
    pilot = [document(0), document(1)]
    candidates = [document(9), document(0), document(8), document(1), document(7)]
    requested_sizes = []

    def select_sample(_manifest, sample_size):
        requested_sizes.append(sample_size)
        return candidates

    monkeypatch.setattr(frozen, "select_sample", select_sample)

    selected = frozen.documents_excluding({"sources": {}}, pilot, sample_size=3)

    assert requested_sizes == [5]
    assert [row.eval_rank for row in selected] == [9, 8, 7]
    assert [row.sample_index for row in selected] == [0, 1, 2]


def test_documents_excluding_reject_missing_row(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(frozen, "select_sample", lambda _manifest, _sample_size: [document(2), document(3)])

    with pytest.raises(ValueError, match="missing 1 excluded rows"):
        frozen.documents_excluding({"sources": {}}, [document(0)], sample_size=1)


def test_excluded_sample_documents_combine_disjoint_complete_samples(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl.gz"
    second = tmp_path / "second.jsonl.gz"
    write_jsonl(StoragePath(str(first)), ({**asdict(document(1)), "sample_index": 0},))
    write_jsonl(StoragePath(str(second)), ({**asdict(document(2)), "sample_index": 0},))

    excluded = frozen.excluded_sample_documents([document(0)], [str(first), str(second)])

    assert [row.eval_rank for row in excluded] == [0, 1, 2]


def test_excluded_sample_documents_reject_overlapping_samples(tmp_path: Path) -> None:
    sample = tmp_path / "sample.jsonl.gz"
    write_jsonl(StoragePath(str(sample)), ({**asdict(document(0)), "sample_index": 0},))

    with pytest.raises(ValueError, match="duplicate evaluation identities"):
        frozen.excluded_sample_documents([document(0)], [str(sample)])


def test_main_passes_all_excluded_samples_to_client(monkeypatch: pytest.MonkeyPatch) -> None:
    launches = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "label_frozen_hierarchy.py",
            "--pilot-run-id",
            "pilot",
            "--variant",
            "compact",
            "--evaluation-run-id",
            "release",
            "--excluded-sample-url",
            "first.jsonl.gz",
            "--excluded-sample-url",
            "second.jsonl.gz",
        ],
    )
    monkeypatch.setattr(frozen, "serve_glm52", lambda launch, *_ports: launches.append(launch))

    frozen.main()

    assert launches[0].client.keywords["excluded_sample_urls"] == ["first.jsonl.gz", "second.jsonl.gz"]


def test_projection_training_documents_exclude_pilot_and_fixed_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidates = [document(index) for index in range(4)]
    monkeypatch.setattr(frozen, "select_sample", lambda _manifest, _sample_size: candidates)

    selected = training.projection_training_documents(
        {"sources": {}},
        pilot_documents=[document(0)],
        evaluation_documents=[document(1)],
        training_size=2,
    )

    assert [row.eval_rank for row in selected] == [2, 3]
    assert [row.sample_index for row in selected] == [0, 1]


def test_projection_training_sample_round_trip_is_exact(tmp_path: Path) -> None:
    documents = [document(0), document(1)]

    stored = training.write_projection_training_documents(training.StoragePath(str(tmp_path)), documents)

    assert stored == documents
