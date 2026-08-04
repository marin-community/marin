# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Turn the oracle-scored 10k PDF-OCR sample into a scorer-training label parquet.

The oracle (``openai/gpt-5.6-luna`` under a FineWeb-Edu rubric) scored three
512-token segments per document -- begin / middle / end -- cut with the **gpt2**
tokenizer. One label therefore describes one *segment*, not one document, so
training examples are segments and this module reproduces the oracle's slices
exactly: tokenize the full text with gpt2, slice, decode back to text.

Every document emits all three segments, but only rows with ``use_for_training``
are fit on. Under ``MIN_TOKENS_FOR_ALL_SEGMENTS`` tokens the three windows overlap
(and at <= 512 tokens they are byte-identical, which is why the labeling job scored
such documents once and copied the score into all three columns), so a short
document marks only its begin row: training on the other two would feed the model
near-duplicate rows and inflate any split that does not group by document. The
unmarked rows still carry text and an oracle score, so a trained scorer can be
applied to all three windows of a document for inspection.

Emits the column names ``fast_transformer/train.py`` and ``calibrate.py`` expect:
``score_normalized`` in [0, 1] for the regression target and ``quality`` in 1..5
for the calibration fit, alongside ``id``/``segment`` so downstream splits can
group by document.
"""

import argparse
import logging
import pathlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem import StoragePath
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

SAMPLE_DIR = "/tmp/cc_focus_2026_22_pdf_ocr_all_sample10k"
SOURCE = "cc_focus_2026_22_pdf_ocr"

# Segmentation, fixed by how the labels were produced (scratchpad quality_label.py).
ORACLE_TOKENIZER = "gpt2"
SEGMENT_TOKENS = 512
SEGMENTS = ("begin", "middle", "end")
SCORE_COLUMNS = {segment: f"edu_score_v2_{segment}" for segment in SEGMENTS}

# Below three disjoint windows the segments overlap, so only `begin` is kept.
MIN_TOKENS_FOR_ALL_SEGMENTS = 3 * SEGMENT_TOKENS

# The rubric emits 0..5; this sample only ever reaches 4, so 4 is full marks and
# the normalized target spans the observed range instead of leaving a dead zone.
MAX_SCORE = 4
TOKENIZE_BATCH = 16


def _segment_slices(token_ids: list[int]) -> dict[str, list[int]]:
    """The oracle's begin/middle/end token slices for one document."""
    n = len(token_ids)
    middle_start = max(0, (n - SEGMENT_TOKENS) // 2)
    return {
        "begin": token_ids[:SEGMENT_TOKENS],
        "middle": token_ids[middle_start : middle_start + SEGMENT_TOKENS],
        "end": token_ids[-SEGMENT_TOKENS:],
    }


def build_rows(ids: list[str], texts: list[str], scores: dict[str, list[int]]) -> list[dict]:
    """One row per (document, kept segment) with the decoded segment text."""
    tokenizer = AutoTokenizer.from_pretrained(ORACLE_TOKENIZER)
    rows: list[dict] = []
    truncated_docs = 0
    for start in range(0, len(ids), TOKENIZE_BATCH):
        batch = slice(start, start + TOKENIZE_BATCH)
        encoded = tokenizer(texts[batch], add_special_tokens=False)["input_ids"]
        for offset, token_ids in enumerate(encoded):
            index = start + offset
            distinct = len(token_ids) >= MIN_TOKENS_FOR_ALL_SEGMENTS
            truncated_docs += not distinct
            slices = _segment_slices(token_ids)
            for segment in SEGMENTS:
                score = scores[segment][index]
                rows.append(
                    {
                        "id": ids[index],
                        "source": SOURCE,
                        "segment": segment,
                        "text": tokenizer.decode(slices[segment]),
                        "quality": score + 1,  # calibrate.py wants oracle levels 1..5
                        "score_normalized": score / MAX_SCORE,
                        "doc_tokens": len(token_ids),
                        "use_for_training": distinct or segment == "begin",
                    }
                )
        if start and start % (TOKENIZE_BATCH * 50) == 0:
            logger.info("tokenized %d/%d docs (%d rows)", start, len(ids), len(rows))
    logger.info(
        "%d/%d docs marked begin-only for training (< %d tokens)",
        truncated_docs,
        len(ids),
        MIN_TOKENS_FOR_ALL_SEGMENTS,
    )
    return rows


def load_sample(sample_dir: str) -> tuple[list[str], list[str], dict[str, list[int]]]:
    """Read ids, text, and the v2 oracle scores from the sample parquets."""
    files = sorted(pathlib.Path(sample_dir).glob("sample-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no sample-*.parquet under {sample_dir}")
    columns = ["id", "text", *SCORE_COLUMNS.values()]
    table = pa.concat_tables([pq.read_table(f, columns=columns) for f in files])
    scores = {segment: table[column].to_pylist() for segment, column in SCORE_COLUMNS.items()}
    logger.info("loaded %d docs from %d files", table.num_rows, len(files))
    return table["id"].to_pylist(), table["text"].to_pylist(), scores


def write_labels(rows: list[dict], output_path: str) -> None:
    table = pa.Table.from_pylist(rows)
    with StoragePath(output_path).open("wb") as stream:
        pq.write_table(table, stream)
    trainable = [row for row in rows if row["use_for_training"]]
    levels = np.array([row["quality"] for row in trainable])
    logger.info(
        "wrote %d rows (%d trainable) -> %s; trainable oracle level counts %s",
        len(rows),
        len(trainable),
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

    ids, texts, scores = load_sample(args.sample_dir)
    valid = [
        i
        for i, _ in enumerate(ids)
        if all(scores[segment][i] is not None and scores[segment][i] >= 0 for segment in SEGMENTS)
    ]
    if len(valid) != len(ids):
        logger.info("dropping %d docs with missing/failed oracle scores", len(ids) - len(valid))
        ids = [ids[i] for i in valid]
        texts = [texts[i] for i in valid]
        scores = {segment: [scores[segment][i] for i in valid] for segment in SEGMENTS}

    write_labels(build_rows(ids, texts, scores), args.out)


if __name__ == "__main__":
    main()
