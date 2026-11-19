# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score normalized parquet batches without changing their ID order."""

from collections.abc import Iterator

import pyarrow as pa
from pydantic import BaseModel

from marin.datakit.low_value.model import OperatingPoint
from marin.datakit.low_value.scoring import DocumentScorer, score_views, text_views
from marin.datakit.source_key import DatakitArtifactPath
from marin.processing.embeddings.read_chunks import rows_per_chunk

SCORE_SCHEMA = pa.schema(
    [
        pa.field("source", pa.string()),
        pa.field("id", pa.string()),
        pa.field("tokens", pa.int64()),
        pa.field("low_value_score", pa.float64()),
        pa.field("is_low_value", pa.bool_()),
    ]
)


class LowValueScores(BaseModel):
    """A co-partitioned side leaf; scores do not overwrite graded quality buckets."""

    output_dir: DatakitArtifactPath
    source: str
    model_sha256: str
    calibration_sha256: str
    target_false_drop: float
    documents_scored_this_run: int


def score_batches(
    batches: Iterator[pa.RecordBatch],
    *,
    source: str,
    scorer: DocumentScorer,
    operating_point: OperatingPoint,
) -> Iterator[pa.RecordBatch]:
    """Emit one row per normalized document in its original order."""
    for batch in batches:
        batch_size = min(128, rows_per_chunk(batch.num_rows, batch.nbytes))
        for start in range(0, len(batch), batch_size):
            part = batch.slice(start, batch_size)
            documents = [
                text_views(text, scorer.tokenizer, scorer.model.window_tokens)
                for text in part.column("text").to_pylist()
            ]
            scores = score_views(documents, scorer.model)
            yield pa.RecordBatch.from_arrays(
                [
                    pa.array([source] * len(part), type=pa.string()),
                    part.column("id"),
                    pa.array([document.tokens for document in documents], type=pa.int64()),
                    pa.array(scores, type=pa.float64()),
                    pa.array(operating_point.drop(scores), type=pa.bool_()),
                ],
                schema=SCORE_SCHEMA,
            )
