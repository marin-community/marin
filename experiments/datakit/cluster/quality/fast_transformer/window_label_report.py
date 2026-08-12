# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read a window-label run by window position, and by whether the window was cut.

The scale-up's begin windows were cut at the window boundary with no marker and
came back 36.5% invalid, the rationales blaming the cut rather than the text — a
label set that trains a scorer to call long documents junk. That failure is only
visible when the invalid rate is split by *position* and by whether the window
ends before its document does, so this report is the check to run on a labeling
run's first chunks before committing a fleet to the rest of it.

Reports, over one or more label parquets (chunks included, via globs): rows and
invalid rate per position, the same split for begin windows that end mid-document
against those that carry a whole document, the share of invalid rationales that
blame the cut, per-type invalid rate and mean quality, and sample rationales.
"""

import argparse
import json
import logging
import random
from collections import Counter

import pyarrow.parquet as pq
from rigging.filesystem import StoragePath
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging

from experiments.datakit.cluster.quality.fast_transformer.window_dataset import CUT_WHY_PATTERN

logger = logging.getLogger(__name__)

COLUMNS = ["id", "window", "token_end", "doc_tokens", "content_type", "valid", "quality", "why"]
BEGIN = "begin"
SAMPLE_WHYS = 12


def read_labels(patterns: list[str]) -> list[dict]:
    paths: list[str] = []
    for pattern in patterns:
        if "*" in pattern:
            paths.extend(sorted(str(m) for m in StoragePath(pattern).glob()))
        else:
            paths.append(pattern)
    rows: list[dict] = []
    for path in paths:
        with StoragePath(path).open("rb") as fh:
            rows.extend(pq.ParquetFile(fh).read(columns=COLUMNS).to_pylist())
    logger.info("window_label_report: %d rows over %d files", len(rows), len(paths))
    return rows


def group_stats(rows: list[dict]) -> dict:
    """Count, invalid rate, mean quality, and cut-blaming share of one group."""
    if not rows:
        return {"windows": 0}
    invalid = [r for r in rows if not r["valid"]]
    blaming = [r for r in invalid if CUT_WHY_PATTERN.search(r["why"] or "")]
    return {
        "windows": len(rows),
        "invalid": len(invalid),
        "invalid_rate": len(invalid) / len(rows),
        "mean_quality": sum(r["quality"] for r in rows) / len(rows),
        "invalid_blaming_the_cut": len(blaming),
        "invalid_blaming_the_cut_rate": len(blaming) / len(invalid) if invalid else 0.0,
    }


def report(rows: list[dict], seed: int) -> dict:
    """The by-position, by-cut, and by-type breakdown of one labeling run."""
    cut_begin = [r for r in rows if r["window"] == BEGIN and r["token_end"] < r["doc_tokens"]]
    whole_begin = [r for r in rows if r["window"] == BEGIN and r["token_end"] >= r["doc_tokens"]]
    sampler = random.Random(seed)
    invalid_cut_begin = [r for r in cut_begin if not r["valid"]]
    return {
        "overall": group_stats(rows),
        "by_position": {
            position: group_stats([r for r in rows if r["window"] == position])
            for position in sorted({r["window"] for r in rows})
        },
        "begin_cut_mid_document": group_stats(cut_begin),
        "begin_whole_document": group_stats(whole_begin),
        "by_content_type": {
            content_type: group_stats([r for r in rows if r["content_type"] == content_type])
            for content_type in sorted({r["content_type"] for r in rows})
        },
        "quality_counts": {str(q): n for q, n in sorted(Counter(r["quality"] for r in rows).items())},
        "sample_invalid_cut_begin_whys": [
            r["why"] for r in sampler.sample(invalid_cut_begin, min(SAMPLE_WHYS, len(invalid_cut_begin)))
        ],
        "sample_cut_begin_whys": [r["why"] for r in sampler.sample(cut_begin, min(SAMPLE_WHYS, len(cut_begin)))],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, nargs="+", help="label parquet path(s) or glob(s)")
    parser.add_argument("--out", default=None, help="optional JSON path for the report")
    parser.add_argument("--seed", type=int, default=0, help="seed for the sampled rationales")
    args = parser.parse_args()
    configure_logging(logging.INFO)
    configure_coreweave_s3()

    result = report(read_labels(args.labels), args.seed)
    logger.info("window_label_report: %s", json.dumps(result, indent=2))
    if args.out:
        with StoragePath(args.out).open("w") as fh:
            json.dump(result, fh, indent=2)


if __name__ == "__main__":
    main()
