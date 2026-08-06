# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Turn an oracle-scored PDF-OCR sample into a scorer-training label parquet.

The oracle (``openai/gpt-5.6-luna`` under a FineWeb-Edu rubric) scores three
512-token windows per document -- begin / middle / end -- so one label describes
one *segment*, not one document, and training examples are segments.

The sample stores the exact text that was scored in ``edu_segment_v2_{begin,
middle,end}`` alongside the score, so this module reads the windows rather than
re-cutting them. That matters: a locally re-derived window that disagrees with
the oracle's by even a few tokens trains the model on text the grader never saw,
and there is no way to notice from the metrics.

Every document emits all three rows, but only rows with ``use_for_training`` are
fit on. Below ``MIN_TOKENS_FOR_ALL_SEGMENTS`` the three windows would overlap, so
the sampling job scored the begin window once and copied its text and score into
all three columns; such a document marks only its begin row, since the other two
are literal duplicates that would inflate any split not grouped by document. The
unmarked rows still carry text and a score, so a trained scorer can be applied to
all three windows of a document for inspection.

Emits the column names ``fast_transformer/train.py`` and ``calibrate.py`` expect:
``score_normalized`` in [0, 1] for the regression target and ``quality`` in 1..5
for the calibration fit, alongside ``id``/``segment`` so downstream splits can
group by document.
"""

import argparse
import functools
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from rigging.filesystem import StoragePath
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging

logger = logging.getLogger(__name__)

SAMPLE_DIR = "s3://marin-us-east-02a/marin/data/pdf_quality/cc_focus_2026_22_sample100k"

SEGMENTS = ("begin", "middle", "end")
SEGMENT_COLUMNS = {segment: f"edu_segment_v2_{segment}" for segment in SEGMENTS}
SCORE_COLUMNS = {segment: f"edu_score_v2_{segment}" for segment in SEGMENTS}
READ_COLUMNS = ["id", "source", "doc_tokens", "needs_ocr", *SEGMENT_COLUMNS.values(), *SCORE_COLUMNS.values()]

# Window size the oracle scored, and the point below which three windows stop
# being disjoint. Both are properties of the sampling job, not choices made here.
SEGMENT_TOKENS = 512
MIN_TOKENS_FOR_ALL_SEGMENTS = 3 * SEGMENT_TOKENS

# The rubric emits 0..5; this corpus only ever reaches 4, so 4 is full marks and
# the normalized target spans the observed range instead of leaving a dead zone.
MAX_SCORE = 4
READ_THREADS = 16


def segment_rows(table: pa.Table) -> pa.Table:
    """Explode one sample shard into three label rows per document.

    Drops documents whose oracle call failed (a negative or null score) in any
    window, so a partially graded document never contributes a real label under
    one segment and a sentinel under another.
    """
    graded = functools.reduce(
        pc.and_,
        (pc.greater_equal(pc.fill_null(table[column], -1), 0) for column in SCORE_COLUMNS.values()),
    )
    table = table.filter(graded)

    distinct = pc.greater_equal(table["doc_tokens"], MIN_TOKENS_FOR_ALL_SEGMENTS)
    always = pa.chunked_array([pa.array([True] * table.num_rows, pa.bool_())])
    parts = []
    for segment in SEGMENTS:
        score = table[SCORE_COLUMNS[segment]].cast(pa.int16())
        parts.append(
            pa.table(
                {
                    "id": table["id"],
                    "source": table["source"],
                    "segment": pa.array([segment] * table.num_rows, pa.string()),
                    "text": table[SEGMENT_COLUMNS[segment]].cast(pa.string()),
                    "quality": pc.add(score, 1).cast(pa.int16()),  # calibrate.py wants oracle levels 1..5
                    "score_normalized": pc.divide(score.cast(pa.float32()), np.float32(MAX_SCORE)),
                    "doc_tokens": table["doc_tokens"].cast(pa.int32()),
                    # Extraction route. The corpus mixes VLM-OCR'd scans with born-digital
                    # text, and the two carry very different surface cues, so any metric
                    # over the pooled corpus can hide a scorer that only works on one.
                    "needs_ocr": pc.fill_null(table["needs_ocr"], False),
                    "use_for_training": always if segment == "begin" else distinct,
                }
            )
        )
    return pa.concat_tables(parts)


def load_sample(sample_dir: str) -> pa.Table:
    """Read every shard's label columns and explode them into segment rows."""
    shards = StoragePath(f"{sample_dir.rstrip('/')}/*.parquet").glob()
    if not shards:
        raise FileNotFoundError(f"no *.parquet under {sample_dir}")
    logger.info("reading %d shards from %s", len(shards), sample_dir)

    def read(shard: StoragePath) -> pa.Table:
        with shard.open("rb") as stream:
            return segment_rows(pq.read_table(stream, columns=READ_COLUMNS))

    with ThreadPoolExecutor(max_workers=READ_THREADS) as pool:
        tables = list(pool.map(read, shards))
    return pa.concat_tables(tables)


def write_labels(table: pa.Table, output_path: str) -> None:
    with StoragePath(output_path).open("wb") as stream:
        pq.write_table(table, stream)
    trainable = table.filter(table["use_for_training"])
    levels = np.array(trainable["quality"].to_pylist())
    logger.info(
        "wrote %d rows (%d trainable) over %d docs -> %s; trainable oracle level counts %s",
        table.num_rows,
        trainable.num_rows,
        table.num_rows // len(SEGMENTS),
        output_path,
        {level: int((levels == level).sum()) for level in sorted(set(levels.tolist()))},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", default=SAMPLE_DIR)
    parser.add_argument("--out", required=True, help="label parquet path (local or s3://)")
    args = parser.parse_args()
    configure_logging(logging.INFO)
    configure_coreweave_s3()  # no-op inside a CoreWeave pod; wires CW_KEY_* on a dev box

    table = load_sample(args.sample_dir)
    short = pc.sum(pc.less(table["doc_tokens"], MIN_TOKENS_FOR_ALL_SEGMENTS)).as_py() // len(SEGMENTS)
    logger.info(
        "%d/%d docs are begin-only for training (< %d tokens)",
        short,
        table.num_rows // len(SEGMENTS),
        MIN_TOKENS_FOR_ALL_SEGMENTS,
    )
    write_labels(table, args.out)


if __name__ == "__main__":
    main()
