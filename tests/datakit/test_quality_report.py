# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The quality report's spot-check reads text from the normalized shard by position."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from marin.datakit.normalize import NormalizedData

from experiments.datakit.cluster.quality.fast_transformer.artifact import BUCKET_EDGES, QualityScores
from experiments.datakit.reports.quality import _spot_check_docs

SHARD = "part-00000-of-00001.parquet"


def make_inputs(root: Path, normalized_ids: list[str], scored_ids: list[str]):
    text_dir, scored_dir = root / "normalized", root / "quality"
    text_dir.mkdir()
    scored_dir.mkdir()
    pq.write_table(pa.table({"id": normalized_ids, "text": [f"text of {i}" for i in normalized_ids]}), text_dir / SHARD)
    pq.write_table(
        pa.table(
            {
                "source": ["src"] * len(scored_ids),
                "id": scored_ids,
                "score": [0.1 * (i + 1) for i in range(len(scored_ids))],
                "quality_bucket": [0, 1, 2][: len(scored_ids)],
            }
        ),
        scored_dir / SHARD,
    )
    scores = QualityScores(
        main_output_dir=str(scored_dir),
        model_dir="model",
        calib_file="calib.json",
        bucket_edges=list(BUCKET_EDGES),
        counters={},
    )
    normalized = NormalizedData(main_output_dir=str(text_dir), dup_output_dir="", counters={})
    return {"src": scores}, {"src": normalized}


def test_spot_check_pairs_each_scored_row_with_its_normalized_text(tmp_path):
    scores, normalized = make_inputs(tmp_path, ["b", "a", "c"], ["b", "a", "c"])

    docs = _spot_check_docs(scores, normalized)

    assert [(d["id"], d["ft_bucket"], d["text"]) for d in docs] == [
        ("b", 0, "text of b"),
        ("a", 1, "text of a"),
        ("c", 2, "text of c"),
    ]


def test_spot_check_refuses_a_shard_out_of_normalized_order(tmp_path):
    scores, normalized = make_inputs(tmp_path, ["a", "b"], ["b", "a"])

    with pytest.raises(ValueError, match="not in its normalized shard's row order"):
        _spot_check_docs(scores, normalized)
