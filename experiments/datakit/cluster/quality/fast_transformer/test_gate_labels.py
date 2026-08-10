# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior of the label-set gate.

Both failures this gate exists for produced label sets that looked healthy by their
headline numbers, so the tests assert on the specific shapes rather than on any
aggregate: quality falling with length (a hard cut the grader read as damage), and
the longest documents quietly missing (prompts rejected for overflowing the
context). The second is the reason the gate reads the input set at all — drop every
long document and the survivors are still perfectly well behaved.
"""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.datakit.cluster.quality.fast_transformer.gate_labels import CAP_CHARS, gate


def _write(path, rows):
    pq.write_table(pa.Table.from_pylist(rows), path)
    return str(path)


def _corpus(n=1200, seed=0):
    """A label set whose quality is independent of length, above the stub floor."""
    rng = np.random.default_rng(seed)
    lengths = rng.integers(600, 3 * CAP_CHARS, size=n)
    quality = rng.integers(1, 6, size=n)
    return [
        {
            "id": f"d{i}",
            "text": "word " * (int(lengths[i]) // 5),
            "quality": int(quality[i]),
            "valid": True,
            "content_type": "prose",
            "source": f"s{i % 7}",
        }
        for i in range(n)
    ]


def _checks(tmp_path, labels, inputs):
    labels_path = _write(tmp_path / "labels.parquet", labels)
    input_path = _write(tmp_path / "input.parquet", [{"id": r["id"], "text": r["text"]} for r in inputs])
    return {c.name: c for c in gate(labels_path=labels_path, label_set_path=input_path)}


def test_a_healthy_label_set_passes(tmp_path):
    rows = _corpus()
    checks = _checks(tmp_path, rows, rows)
    failed = [name for name, c in checks.items() if not c.passed]
    assert not failed, f"healthy set should pass everything, failed: {failed}"


def test_quality_falling_with_length_is_caught(tmp_path):
    """The truncation-poisoning signature: the longer the document, the worse the score.

    This is what a hard cut produces — the grader sees text ending mid-token, calls
    it damaged, and assigns the floor. It struck the longest documents hardest.
    """
    rows = _corpus()
    order = np.argsort([-len(r["text"]) for r in rows])
    for rank, i in enumerate(order):
        rows[i]["quality"] = 1 if rank < len(rows) // 2 else 5
    checks = _checks(tmp_path, rows, rows)
    assert not checks["length does not drive quality"].passed


def test_missing_long_documents_are_caught_against_the_input(tmp_path):
    """Selective loss of the longest documents, invisible from the survivors alone.

    The labeled set here is internally consistent — quality does not track length,
    nothing is marked invalid — and is still broken, because the documents the
    server rejected were exactly the long ones.
    """
    rows = _corpus()
    keep = [r for r in rows if len(r["text"]) < CAP_CHARS]
    checks = _checks(tmp_path, keep, rows)
    assert not checks["long documents retained"].passed
    # The survivors on their own look fine, which is the point.
    assert checks["length does not drive quality"].passed


def test_a_grader_that_never_spends_its_top_score_is_caught(tmp_path):
    """A scale whose top is unreachable cannot rank the end that data selection uses."""
    rows = _corpus()
    for r in rows:
        r["quality"] = min(r["quality"], 4)
    checks = _checks(tmp_path, rows, rows)
    assert not checks["top-of-scale used"].passed


def test_dropped_rows_are_caught_as_coverage(tmp_path):
    rows = _corpus()
    checks = _checks(tmp_path, rows[: len(rows) // 2], rows)
    assert not checks["coverage"].passed


@pytest.mark.parametrize("name", ["coverage", "top-of-scale used", "invalid rate", "long documents retained"])
def test_every_named_gate_is_present(tmp_path, name):
    """Each gate must actually run; a check that silently disappears protects nothing."""
    rows = _corpus()
    assert name in _checks(tmp_path, rows, rows)
