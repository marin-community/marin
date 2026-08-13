# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Print the lowest- and highest-scored documents from a scored leaf, with their text.

A score distribution that is much tighter than the corpus is either a real property of
the data or a scoring path that has collapsed. The way to tell them apart is to read the
documents at both ends: if the bottom and top of the range are visibly different in
quality, the scale is working and the corpus is simply homogeneous; if they read alike,
the scores carry no signal.

The text comes from the normalized stage, which is where ``id`` still maps to a document.
A manifest ``source_key`` *is* the normalized output dir, and normalize, tokenize, embed
and the scores are all co-partitioned with the same basenames, so shard ``k`` of the
scores and shard ``k`` of the normalized side hold the same documents.

    uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \\
        --cpu 8 --memory 32g --disk 32g --enable-extra-resources \\
        -- python -m experiments.datakit.scripts.read_extreme_scored_documents \\
           --manifest s3://.../manifest_depth2 --shards-per-source 2
"""

import argparse
import json
import logging
import re
from io import BytesIO

import fsspec
import polars as pl
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging

from experiments.datakit.cluster.quality.fast_transformer.score_corpus import DEFAULT_MANIFEST, read_manifest

logger = logging.getLogger(__name__)

PREFIX = "marin-us-east-02a/marin"
TEXT_CHARS = 900


def _read(fs, url: str, columns: list[str]) -> pl.DataFrame:
    return pl.read_parquet(BytesIO(fs.cat(url.removeprefix("s3://"))), columns=columns)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--sources", nargs="*", default=[], help="substrings selecting source_keys (default: a spread)")
    ap.add_argument("--shards-per-source", type=int, default=2)
    ap.add_argument("--per-end", type=int, default=4, help="documents to print from each end")
    args = ap.parse_args()
    configure_logging(logging.INFO)
    configure_coreweave_s3()
    fs = fsspec.filesystem("s3")

    rows = read_manifest(args.manifest).to_pylist()
    rows = [r for r in rows if r.get("embed_rows")]
    wanted = args.sources or ["penfever-traces", "safety_pt"]
    picked: list[dict] = []
    for want in wanted:
        hits = [r for r in rows if want in r["source_key"]]
        # Spread across distinct leaves rather than taking one leaf's first shards.
        seen: set[str] = set()
        for row in sorted(hits, key=lambda r: -r["embed_rows"]):
            if row["source_key"] in seen:
                continue
            seen.add(row["source_key"])
            picked.append(row)
            if len(seen) >= args.shards_per_source:
                break

    report = []
    for row in picked:
        scores = _read(fs, row["output_path"], ["id", "score"])
        normalized = f"s3://{PREFIX}/{row['source_key'].strip('/')}"
        basename = row["output_path"].rsplit("/", 1)[-1]
        try:
            text = _read(fs, f"{normalized}/{basename}", ["id", "text"])
        except Exception as exc:
            logger.warning("no normalized text for %s/%s: %s", normalized, basename, exc)
            continue
        joined = scores.join(text, on="id", how="inner")
        ordered = joined.sort("score")
        ends = [("LOW", ordered.head(args.per_end)), ("HIGH", ordered.tail(args.per_end))]
        logger.info(
            "SOURCE %s shard %s: %d scored, %d with text, score min=%.4f max=%.4f mean=%.4f std=%.4f",
            row["source_key"],
            row["shard_index"],
            scores.height,
            joined.height,
            float(scores["score"].min()),
            float(scores["score"].max()),
            float(scores["score"].mean()),
            float(scores["score"].std() or 0.0),
        )
        for end, frame in ends:
            for record in frame.to_dicts():
                body = re.sub(r"\s+", " ", record["text"])[:TEXT_CHARS]
                logger.info(
                    "DOC %s %s score=%.4f id=%s :: %s", end, row["source_key"], record["score"], record["id"], body
                )
                report.append(
                    {
                        "source_key": row["source_key"],
                        "end": end,
                        "score": record["score"],
                        "id": record["id"],
                        "chars": len(record["text"]),
                        "text": body,
                    }
                )
    logger.info("EXTREMES %s", json.dumps({"documents": len(report)}))


if __name__ == "__main__":
    main()
