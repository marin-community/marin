# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.datakit.normalize import generate_id
from marin.execution.artifact import write_artifact
from marin.execution.lazy import ArtifactStep, StepContext

from experiments.grug.fast_track.quality import EmbeddingHeadScorer, LabelSplit, QualityScoringBatch, label_split
from experiments.grug.fast_track.quality_features import HARRIER_FEATURE_IDENTITY
from experiments.grug.fast_track.quality_labels import (
    GLM_ORACLE,
    FittedRidgeQualityHead,
    FrozenQualityLabels,
    QualityLabelSpec,
    build_ridge_quality_head,
    freeze_quality_labels,
    read_labelled_embeddings,
)


@pytest.fixture
def label_inputs(tmp_path):
    originals, joined = [], []
    for index in range(101):
        text = f"Document {index}: a full normalized text, not the stored label excerpt."
        quality = index % 5 + 1
        original = {
            "source": "sample",
            "id": generate_id(text),
            "quality": quality,
            "score_normalized": (quality - 1) / 4,
            "valid": quality != 1,
            "content_type": "prose",
            "label_batch": "glm52_rubric_v2",
            "text": text[:12],
        }
        originals.append(original)
        if index < 100:
            embedding = np.zeros(1024, dtype=np.int8)
            embedding[:2] = [quality * 20, 100 - quality * 20]
            joined.append(
                {
                    "id": original["id"],
                    "text": text,
                    "embedding": embedding.tolist(),
                    **{f"glm52_{key}": value for key, value in original.items() if key not in ("id", "text")},
                }
            )
    labels_path = tmp_path / "original.parquet"
    joined_root = tmp_path / "joined"
    outputs = joined_root / "outputs" / "source" / "leaf"
    outputs.mkdir(parents=True)
    joined_path = outputs / "part-000.parquet"
    pq.write_table(pa.Table.from_pylist(originals), labels_path)
    pq.write_table(pa.Table.from_pylist(joined), joined_path)
    return (
        QualityLabelSpec(str(labels_path), str(joined_root), GLM_ORACLE, HARRIER_FEATURE_IDENTITY),
        originals,
        joined_path,
    )


def test_frozen_labels_fit_roundtrip_and_exclude_missing_and_audit_documents(tmp_path, label_inputs):
    spec, originals, joined_path = label_inputs
    frozen = freeze_quality_labels(spec, output_path=str(tmp_path / "frozen"))
    rows = read_labelled_embeddings(frozen)
    assert len(rows) == 100
    assert sum(row.label == 0 for row in rows) == 20
    assert frozen.label_exclusion.normalized_document_ids == {row["id"] for row in originals}
    assert frozen.missing_by_source == {"sample": 1}
    assert frozen.missing_by_content_type == {"prose": 1}
    joined = pq.read_table(joined_path).to_pylist()
    assert frozen.label_exclusion.duplicate_groups == {
        hashlib.sha256(row["text"].encode()).hexdigest() for row in joined
    }
    np.testing.assert_allclose(np.linalg.norm([row.embedding for row in rows], axis=1), 1.0, atol=1e-7)

    labels_path = str(tmp_path / "labels-artifact")
    write_artifact(frozen, labels_path)
    step = build_ridge_quality_head(
        ArtifactStep.adopt("labels-test", "2026.10.04", labels_path, kind=FrozenQualityLabels),
        regularization=0.001,
        split_seed=0,
        version="2026.10.04",
    )
    context = StepContext.for_run(output_path=str(tmp_path / "head"), prefix=str(tmp_path), deps=step.deps)
    fitted = step.run(step.build_config(context))
    head_path = str(tmp_path / "head-artifact")
    write_artifact(fitted, head_path)
    restored = FittedRidgeQualityHead.raw_load(head_path)
    scorer = EmbeddingHeadScorer(restored.head)
    batch = QualityScoringBatch(
        texts=["unused"] * len(rows),
        document_ids=[row.document_id for row in rows],
        embeddings=np.asarray([row.embedding for row in rows]),
    )
    predictions = scorer.scores(batch)
    assert np.mean((predictions - np.asarray([row.label for row in rows])) ** 2) < 0.005
    assert fitted.training_documents == sum(label_split(row.duplicate_group, 0) == LabelSplit.TRAIN for row in rows)
    assert fitted.development.documents == sum(
        label_split(row.duplicate_group, 0) == LabelSplit.DEVELOPMENT for row in rows
    )
    assert restored.label_exclusion == frozen.label_exclusion
    manifest = json.loads((tmp_path / "frozen" / "labels.json").read_text())
    assert (
        manifest["input_sha256"][spec.labels_path]
        == hashlib.sha256((tmp_path / "original.parquet").read_bytes()).hexdigest()
    )

    # A changed frozen file must fail before a new model can consume it.
    pq.write_table(pa.Table.from_pylist([{"changed": True}]), frozen.table_path)
    with pytest.raises(ValueError, match="changed after preparation"):
        read_labelled_embeddings(frozen)


def test_label_freeze_keeps_first_embedding_and_reports_each_duplicate(tmp_path, label_inputs):
    spec, originals, joined_path = label_inputs
    first_rows = pq.read_table(joined_path).to_pylist()
    retained = next(row for row in first_rows if row["id"] == originals[0]["id"])
    duplicate_same = {**retained}
    duplicate_different = {**retained, "embedding": retained["embedding"].copy()}
    duplicate_different["embedding"][0] += 2
    joined_root = tmp_path / "reordered-joined"
    outputs = joined_root / "outputs" / "sample"
    outputs.mkdir(parents=True)
    duplicate_path = outputs / "z-later.parquet"
    pq.write_table(pa.Table.from_pylist([duplicate_same, duplicate_different]), duplicate_path)
    retained_path = outputs / "a-first.parquet"
    pq.write_table(pa.Table.from_pylist(first_rows), retained_path)
    spec = QualityLabelSpec(spec.labels_path, str(joined_root), spec.oracle_identity, spec.feature_identity)

    frozen = freeze_quality_labels(spec, output_path=str(tmp_path / "frozen-duplicates"))

    saved = next(row for row in pq.read_table(frozen.table_path).to_pylist() if row["id"] == retained["id"])
    assert saved["embedding"] == retained["embedding"]
    assert frozen.label_exclusion.normalized_document_ids == {row["id"] for row in originals}
    report = pq.read_table(frozen.duplicate_report_path).to_pylist()
    assert len(report) == 2
    assert [row["embedding_identical"] for row in report] == [True, False]
    assert all(row["retained_path"] == str(retained_path) and row["retained_row"] == 0 for row in report)
    assert [row["duplicate_path"] for row in report] == [str(duplicate_path)] * 2
    assert [row["duplicate_row"] for row in report] == [0, 1]
    assert report[0]["embedding_max_abs_delta_int8"] == 0
    assert report[0]["embedding_l2_delta_int8"] == 0
    assert report[0]["embedding_cosine_similarity_int8"] == 1
    assert report[1]["embedding_max_abs_delta_int8"] == 2
    assert report[1]["embedding_l2_delta_int8"] == 2

    manifest = json.loads((tmp_path / "frozen-duplicates" / "labels.json").read_text())
    summary = manifest["duplicate_embeddings"]
    assert summary["policy"] == "first_sorted_path_row"
    assert summary["occurrences"] == 2
    assert summary["affected_documents"] == 1
    assert summary["identical_occurrences"] == 1
    assert summary["differing_occurrences"] == 1
    assert summary["max_abs_delta_int8"] == 2
    assert summary["max_l2_delta_int8"] == 2
    assert summary["minimum_cosine_similarity_int8"] == report[1]["embedding_cosine_similarity_int8"]
    assert summary["report_path"] == frozen.duplicate_report_path
    assert (
        manifest["duplicate_report_sha256"]
        == hashlib.sha256((tmp_path / "frozen-duplicates" / "duplicate_embeddings.parquet").read_bytes()).hexdigest()
    )


def test_label_freeze_reports_zero_vector_cosine_without_nonfinite_values(tmp_path, label_inputs):
    spec, originals, joined_path = label_inputs
    rows = pq.read_table(joined_path).to_pylist()
    zero_id = originals[0]["id"]
    rows[0]["embedding"] = [0] * 1024
    pq.write_table(pa.Table.from_pylist(rows), joined_path)
    duplicate_path = joined_path.parent / "part-001.parquet"
    duplicate = {**rows[0], "embedding": [0] * 1024}
    one_zero = {**rows[0], "embedding": [0] * 1024}
    one_zero["embedding"][0] = 1
    pq.write_table(pa.Table.from_pylist([duplicate, one_zero]), duplicate_path)

    frozen = freeze_quality_labels(spec, output_path=str(tmp_path / "frozen-zero"))

    report = pq.read_table(frozen.duplicate_report_path).to_pylist()
    assert [row["embedding_cosine_similarity_int8"] for row in report] == [1, 0]
    assert all(np.isfinite(row["embedding_cosine_similarity_int8"]) for row in report)
    assert report[0]["id"] == zero_id


@pytest.mark.parametrize("corruption", ["excerpt-text", "wrong-label", "duplicate-text", "duplicate-target"])
def test_label_freeze_rejects_wrong_join_before_fitting(tmp_path, label_inputs, corruption):
    spec, _, joined_path = label_inputs
    rows = pq.read_table(joined_path).to_pylist()
    if corruption == "excerpt-text":
        rows[0]["text"] = rows[0]["text"][:12]
    elif corruption == "wrong-label":
        rows[0]["glm52_quality"] = 5
        rows[0]["glm52_score_normalized"] = 1.0
    elif corruption == "duplicate-text":
        duplicate = {**rows[0], "text": rows[0]["text"] + " changed"}
        rows.append(duplicate)
    else:
        duplicate = {**rows[1], "glm52_quality": 3, "glm52_score_normalized": 0.5}
        rows.append(duplicate)
    pq.write_table(pa.Table.from_pylist(rows), joined_path)
    with pytest.raises(ValueError):
        freeze_quality_labels(spec, output_path=str(tmp_path / "bad-frozen"))
    assert not (tmp_path / "bad-frozen" / "labels.json").exists()
