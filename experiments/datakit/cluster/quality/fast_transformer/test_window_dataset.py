# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the scaled window-dataset assembly."""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from experiments.datakit.cluster.quality.fast_transformer.scaled_exp import grouped_val_split
from experiments.datakit.cluster.quality.fast_transformer.window_dataset import (
    WINDOW_COLUMNS,
    assemble_training_windows,
    drop_cut_artifact_grades,
    load_window_labels,
    subsample_mask,
)


def _window_row(doc_id, window, text, quality=3.0, valid=True, why="fine"):
    return {
        "id": doc_id,
        "source": "src/a",
        "window": window,
        "text": text,
        "quality": quality,
        "score_normalized": (quality - 1) / 4,
        "valid": valid,
        "why": why,
    }


def test_load_window_labels_collapses_exact_duplicates_and_drops_ambiguous_ids(tmp_path):
    """A key duplicated with identical text is one grade recorded twice; a key
    duplicated with differing text is one id naming two documents, whose
    id-keyed embedding join is ambiguous — every window of that id must go."""
    rows = [
        _window_row("keep", "begin", "kept text"),
        _window_row("twice", "middle", "same text", quality=2.0),
        _window_row("twice", "middle", "same text", quality=4.0),
        _window_row("ambiguous", "begin", "doc A text"),
        _window_row("ambiguous", "begin", "doc B text"),
        _window_row("ambiguous", "end", "doc A tail"),
    ]
    path = tmp_path / "windows.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)

    out = load_window_labels(str(path))
    assert sorted(zip(out["id"], out["window"], strict=True)) == [("keep", "begin"), ("twice", "middle")]
    # keep-first, like the 88k loader's id dedup
    assert out["quality"][out["id"].index("twice")] == 2.0
    assert set(out.keys()) == set(WINDOW_COLUMNS)


def test_assemble_excludes_every_window_of_a_holdout_doc():
    """The scale-up added middle/end windows for legacy docs; the ones naming
    holdout ids must not leak into training."""
    legacy = {
        "id": ["hold", "train"],
        "glm52_source": ["src/a", "src/a"],
        "glm52_score_normalized": [0.75, 0.5],
        "embedding": [[1] * 4, [2] * 4],
    }
    windows = {
        "id": ["hold", "hold", "train", "new"],
        "source": ["src/a"] * 4,
        "window": ["middle", "end", "middle", "begin"],
        "text": ["hm", "he", "tm", "nb"],
        "quality": [3.0] * 4,
        "score_normalized": [0.5] * 4,
    }
    scaleup = {"id": ["new"], "embedding": [[3] * 4]}

    examples, stats = assemble_training_windows(windows, legacy, ["hold begin", "train begin"], scaleup, {"hold"})
    assert "hold" not in examples.ids
    assert sorted(zip(examples.ids, examples.positions.tolist(), strict=True)) == [
        ("new", "begin"),
        ("train", "begin"),
        ("train", "middle"),
    ]
    # 1 legacy begin + 2 topup/new windows of the holdout doc
    assert stats.holdout_excluded == 3


def test_assemble_aligns_each_window_with_its_documents_embedding():
    """Middle/end top-ups of legacy docs read the 88k join's embedding; new
    docs read the scale-up join's; a window with no stored document is dropped."""
    legacy = {
        "id": ["old"],
        "glm52_source": ["src/a"],
        "glm52_score_normalized": [0.25],
        "embedding": [[10] * 4],
    }
    windows = {
        "id": ["old", "old", "new", "orphan"],
        "source": ["src/a"] * 4,
        "window": ["middle", "begin", "begin", "begin"],
        "text": ["om", "ob-regrade", "nb", "xx"],
        "quality": [3.0] * 4,
        "score_normalized": [0.5, 0.5, 1.0, 0.5],
    }
    scaleup = {"id": ["new"], "embedding": [[20] * 4]}

    examples, stats = assemble_training_windows(windows, legacy, ["old begin cut"], scaleup, set())
    by_key = {
        (i, p): (t, e, s)
        for i, p, t, e, s in zip(
            examples.ids, examples.positions, examples.texts, examples.embeddings, examples.targets, strict=True
        )
    }
    assert by_key[("old", "begin")] == ("old begin cut", [10] * 4, np.float32(0.25))
    assert by_key[("old", "middle")] == ("om", [10] * 4, np.float32(0.5))
    assert by_key[("new", "begin")] == ("nb", [20] * 4, np.float32(1.0))
    # the 88k grade is the begin verdict for legacy docs; the scale-up row is a re-grade
    assert stats.begin_regrades_skipped == 1
    assert stats.missing_embedding == 1
    assert len(examples.ids) == 3


def test_cut_artifact_filter_drops_only_invalid_grades_that_blame_the_cut():
    """A quality-1 verdict for the harness cutting mid-expression labels the
    harness, not the document; a genuine-junk invalid and any valid grade
    (even one mentioning the cut) must survive."""
    rows = {
        "id": ["cut", "junk", "downgraded", "clean"],
        "window": ["begin"] * 4,
        "valid": [False, False, True, True],
        "why": [
            "Code is truncated mid-expression at 'x.size(0' — incomplete.",
            "SEO spam with keyword-stuffed nonsense.",
            "Solid tutorial, though the example is truncated at the end.",
            "Well-structured reference page.",
        ],
    }
    out = drop_cut_artifact_grades(rows)
    assert out["id"] == ["junk", "downgraded", "clean"]


def test_grouped_val_split_never_straddles_a_doc():
    """Sibling windows share the document embedding, so a doc split across
    fit/val would let the model meet val documents during fitting."""
    ids = [f"doc{i // 3}" for i in range(300)]  # 100 docs x 3 windows
    fit_idx, val_idx = grouped_val_split(ids, val_frac=0.1, seed=0)
    fit_docs = {ids[i] for i in fit_idx}
    val_docs = {ids[i] for i in val_idx}
    assert not fit_docs & val_docs
    assert len(fit_idx) + len(val_idx) == 300
    assert len(val_docs) == 10


def test_subsample_mask_keeps_a_docs_rows_aligned_across_tables():
    ids_a = ["a", "b", "c", "d", "e", "f"]
    ids_b = ["c", "a", "c", "f"]  # same docs, different table/order/multiplicity
    mask_a = subsample_mask(ids_a, 2)
    mask_b = subsample_mask(ids_b, 2)
    kept_a = {i for i, k in zip(ids_a, mask_a, strict=True) if k}
    kept_b = {i for i, k in zip(ids_b, mask_b, strict=True) if k}
    assert kept_b == kept_a & set(ids_b)
    assert subsample_mask(ids_a, 1).all()
