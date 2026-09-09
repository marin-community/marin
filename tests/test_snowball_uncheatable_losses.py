# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import math

import numpy as np
import pytest
from levanter.analysis.document_losses import Document

from experiments.evaluation.snowball_uncheatable_losses import Subset, SubsetTotals, completed_subset, score_subset


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
    assert all(set(row) == {"doc_id", "corpus_id", "loss", "bits_per_byte"} for row in rows)
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
