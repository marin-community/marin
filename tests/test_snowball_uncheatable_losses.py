# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import hashlib
import json
import math

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from levanter.analysis.document_losses import Document, DocumentSourceConfig, iter_documents

from experiments.evaluation.prepare_uncheatable_losses import CATEGORIES, DATASET_ID, prepare_manifest
from experiments.evaluation.snowball_uncheatable_losses import (
    CheckpointSpec,
    DatasetSpec,
    ScoringBackend,
    Subset,
    SubsetTotals,
    completed_subset,
    score_subset,
    validate_checkpoint_locality,
)


class FixedScorer:
    eval_batch_size = 2

    def score_token_totals(self, texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        totals = {"a": (2.0, 1), "éé": (3.0, 3), "": (0.0, 0)}
        sums, counts = zip(*(totals[text] for text in texts), strict=True)
        return np.asarray(sums), np.asarray(counts)


def test_subset_metrics_distinguish_token_byte_and_document_weighting():
    totals = SubsetTotals()
    first = totals.add(Document("a", "subset", "a"), 2.0, 1)
    second = totals.add(Document("b", "subset", "éé"), 3.0, 3)
    empty = totals.add(Document("empty", "subset", ""), 0.0, 0)
    summary = totals.summary()
    assert first.loss == 2.0 and second.loss == 1.0
    assert empty.loss is None and empty.bits_per_byte is None
    assert summary["documents"] == 3 and summary["scored_documents"] == 2
    assert summary["total_bytes"] == 5 and summary["scored_bytes"] == 5
    assert summary["scored_tokens"] == 4
    assert summary["token_weighted_loss"] == 1.25
    assert summary["mean_document_loss"] == 1.5
    assert summary["byte_weighted_bits_per_byte"] == pytest.approx(1 / math.log(2))
    assert summary["mean_document_bits_per_byte"] == pytest.approx((2 + 3 / 4) / (2 * math.log(2)))


def test_completed_subset_resume_verifies_identity_and_output(tmp_path):
    source = tmp_path / "input.jsonl"
    source.write_text(
        "".join(json.dumps({"id": index, "text": text}) + "\n" for index, text in enumerate(["a", "éé", ""]))
    )
    subset = Subset("fixture", str(source), 3)
    output = str(tmp_path / "output")
    expected = score_subset(FixedScorer(), subset, output, "manifest-and-model")
    # Removing the source demonstrates a completed subset is reused without
    # silently rescoring or depending on mutable input discovery.
    source.unlink()
    assert score_subset(FixedScorer(), subset, output, "manifest-and-model") == expected
    rows = [json.loads(line) for line in (tmp_path / "output/fixture/document-losses.jsonl").read_text().splitlines()]
    assert [row["doc_id"] for row in rows] == ["0", "1", "2"]
    assert [row["total_nll"] for row in rows] == [2.0, 3.0, 0.0]
    assert [row["scored_tokens"] for row in rows] == [1, 3, 0]
    assert [row["num_bytes"] for row in rows] == [1, 4, 0]
    assert rows[1]["text_sha256"] == hashlib.sha256("éé".encode()).hexdigest()
    with pytest.raises(ValueError, match="different manifest"):
        completed_subset(subset, output, "different-checkpoint")
    (tmp_path / "output/fixture/document-losses.jsonl").write_text("corrupted\n")
    with pytest.raises(ValueError, match="checksum"):
        completed_subset(subset, output, "manifest-and-model")


def test_incomplete_subset_does_not_get_a_completion_marker(tmp_path):
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps({"id": "a", "text": "a"}) + "\n")
    output = str(tmp_path / "output")
    with pytest.raises(ValueError, match="manifest expects 2"):
        score_subset(FixedScorer(), Subset("fixture", str(source), 2), output, "identity")
    assert not (tmp_path / "output/fixture/summary.json").exists()
    source.write_text(source.read_text() + json.dumps({"id": "b", "text": "éé"}) + "\n")
    summary = score_subset(FixedScorer(), Subset("fixture", str(source), 2), output, "identity")
    assert summary["documents"] == 2
    assert len((tmp_path / "output/fixture/document-losses.jsonl").read_text().splitlines()) == 2


def test_july_release_preserves_benchmark_content_and_source_row_ids(tmp_path):
    rows = [
        {
            "content": f"benchmark text {index}",
            "untruncated_content": "Different source text that must not be scored",
            "category": category,
            "date": "2026-07-01",
            "url": "https://example.com/shared-url",
        }
        for index in range(500)
        for category in CATEGORIES
    ]
    source = tmp_path / "release.parquet"
    pq.write_table(pa.Table.from_pylist(rows), source)
    manifest = prepare_manifest(str(source), str(tmp_path / "normalized"))
    exported = []
    for subset in manifest["subsets"]:
        documents = list(iter_documents(DocumentSourceConfig(input_path=subset["input_path"])))
        assert len(documents) == subset["expected_documents"] == 500
        assert all(doc.corpus_id == subset["name"] for doc in documents)
        exported.extend(documents)
    expected = {f"{DATASET_ID}#test:{index}": row["content"] for index, row in enumerate(rows)}
    assert {doc.doc_id: doc.text for doc in exported} == expected


def test_nonprimary_process_scores_without_publishing_outputs(tmp_path, monkeypatch):
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps({"id": "a", "text": "a"}) + "\n")
    monkeypatch.setattr("experiments.evaluation.snowball_uncheatable_losses.jax.process_index", lambda: 1)
    output = tmp_path / "output"
    summary = score_subset(FixedScorer(), Subset("fixture", str(source), 1), str(output), "identity")
    assert summary["documents"] == 1 and summary["loss_sum"] == 2.0
    assert not output.exists()


def test_checkpoint_locality_rejects_cross_region_weight_reads():
    checkpoint = CheckpointSpec(
        "hot-3000",
        "gs://marin-us-central2/grug/run/checkpoints/step-3000",
        ScoringBackend.NATIVE_TPU,
        "hot",
        3000,
        executor_info_path="gs://marin-us-central2/grug/run/.executor_info",
    )
    dataset = DatasetSpec("july", "gs://marin-us-central2/evaluation/manifest.json", "pinned", 7500, 15)
    validate_checkpoint_locality(checkpoint, [dataset], "gs://marin-us-central2/evaluation/output")
    with pytest.raises(ValueError, match="remain under"):
        validate_checkpoint_locality(
            dataclasses.replace(checkpoint, path="gs://marin-us-east5/grug/run/checkpoints/step-3000"),
            [dataset],
            "gs://marin-us-central2/evaluation/output",
        )
    with pytest.raises(ValueError, match="remain under"):
        validate_checkpoint_locality(
            dataclasses.replace(checkpoint, backend=ScoringBackend.HF_GPU),
            [dataset],
            "s3://marin-us-east-02a/evaluation/output",
        )
