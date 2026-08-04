# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Merge fast-transformer holdout scores into the local sample parquets.

``train_pdf_scorer.py --scores-out`` writes one row per (document, segment) for
the documents the scorer never trained on. This folds them back into the sample
as ``ft_score_{begin,middle,end}`` float columns on the oracle's own 0..4 scale,
so the browser can show them beside ``edu_score_v2_*``.

Documents outside the holdout keep a null score. That is deliberate: the browser
restricts itself to scored documents, and a score on a document the model trained
on would read as prediction when it is partly memorisation.

Each parquet is rewritten via a temporary file and an atomic replace, following
the same pattern as the labeling job's merge step. Do not run it while another
writer is touching the same files.
"""

import argparse
import logging
import pathlib

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem import StoragePath
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging

logger = logging.getLogger(__name__)

SAMPLE_DIR = "/tmp/cc_focus_2026_22_pdf_ocr_all_sample10k"
SEGMENTS = ("begin", "middle", "end")
COLUMN_PREFIX = "ft_score"


def load_scores(scores_path: str) -> dict[tuple[str, str], float]:
    """(id, segment) -> model score. Later rows win, as in the labeling merge."""
    with StoragePath(scores_path).open("rb") as stream:
        table = pq.read_table(stream, columns=["id", "segment", "ft_score"])
    scores = {
        (doc_id, segment): score
        for doc_id, segment, score in zip(
            table["id"].to_pylist(), table["segment"].to_pylist(), table["ft_score"].to_pylist(), strict=True
        )
    }
    logger.info("loaded %d segment scores from %s", len(scores), scores_path)
    return scores


def merge(sample_dir: str, scores: dict[tuple[str, str], float]) -> None:
    paths = sorted(pathlib.Path(sample_dir).glob("sample-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no sample-*.parquet under {sample_dir}")
    matched = 0
    for path in paths:
        table = pq.read_table(path)
        ids = table["id"].to_pylist()
        for segment in SEGMENTS:
            name = f"{COLUMN_PREFIX}_{segment}"
            values = [scores.get((doc_id, segment)) for doc_id in ids]
            matched += sum(v is not None for v in values)
            column = pa.array(values, pa.float32())
            if name in table.column_names:
                table = table.set_column(table.column_names.index(name), name, column)
            else:
                table = table.append_column(name, column)
        temporary = path.with_suffix(".parquet.tmp")
        pq.write_table(table, temporary)
        temporary.replace(path)
        logger.info("rewrote %s (%d rows)", path.name, table.num_rows)
    logger.info("filled %d segment scores across %d files", matched, len(paths))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", default=SAMPLE_DIR)
    parser.add_argument("--scores", required=True, help="holdout scores parquet from train_pdf_scorer.py")
    args = parser.parse_args()
    configure_logging(logging.INFO)
    configure_coreweave_s3()

    merge(args.sample_dir, load_scores(args.scores))


if __name__ == "__main__":
    main()
