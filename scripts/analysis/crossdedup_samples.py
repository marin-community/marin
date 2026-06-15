# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sample cross-dedup `3`<->`4plus` pairs across Jaccard bands and extract both texts.

Reads the probe's verified pairs, samples a few per 0.05 Jaccard band, then pulls
the `3` doc text (from the probe query subsample) and the matched `4plus` doc text
(by scanning the scanned `4plus` shards). Writes one small JSON with both full
texts so the pairs can be eyeballed — is a J=0.35 "match" a real duplicate or just
two different math pages sharing boilerplate?

Runs IN-REGION (us-east5) so the `4plus` shard reads stay local; only the small
JSON is downloaded.

    uv run iris --controller-url=http://localhost:10000 --cluster=marin job run --no-wait \
        --cpu 4 --memory 32GB --disk 20GB --priority interactive --extra cpu \
        --enable-extra-resources --preemptible --region us-east5 \
        --job-name crossdedup-samples \
        -- python scripts/analysis/crossdedup_samples.py --per-band 4
"""

import argparse
import json
import logging
import random
from collections import defaultdict

import fsspec
import pyarrow.parquet as pq
from marin.utils import fsspec_glob

logger = logging.getLogger(__name__)

PROBE = "gs://marin-us-east5/scratch/ahmed/midtrain_dedup/crossdedup_probe"
FOURP_CORPUS = "gs://marin-us-east5/normalized/nemotron_cc_math_v1/4plus_b05688a8/outputs/main"


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-band", type=int, default=4, help="Pairs to sample per 0.05 Jaccard band.")
    parser.add_argument("--corpus-shards", type=int, default=24, help="`4plus` shards the probe scanned.")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    pairs: list[tuple[str, str, float]] = []
    for path in fsspec_glob(f"{PROBE}/verified/*.parquet"):
        with fsspec.open(path, "rb") as f:
            t = pq.read_table(f, columns=["val_id", "other_id", "jaccard"])
        pairs += list(
            zip(
                t.column("val_id").to_pylist(),
                t.column("other_id").to_pylist(),
                t.column("jaccard").to_pylist(),
                strict=True,
            )
        )
    logger.info("loaded %d verified pairs", len(pairs))

    bands: dict[float, list] = defaultdict(list)
    for three_id, fourp_id, jac in pairs:
        bands[round((jac // 0.05) * 0.05, 2)].append((three_id, fourp_id, jac))
    rng = random.Random(args.seed)
    sampled: list[tuple[str, str, float]] = []
    for band in sorted(bands):
        sampled += rng.sample(bands[band], min(args.per_band, len(bands[band])))
    logger.info("sampled %d pairs across %d bands", len(sampled), len(bands))

    target_three = {t for t, _, _ in sampled}
    target_fourp = {f for _, f, _ in sampled}

    three_text: dict[str, str] = {}
    for path in fsspec_glob(f"{PROBE}/query/*.parquet"):
        with fsspec.open(path, "rb") as f:
            t = pq.read_table(f, columns=["id", "text"])
        for doc_id, text in zip(t.column("id").to_pylist(), t.column("text").to_pylist(), strict=True):
            if doc_id in target_three:
                three_text[doc_id] = text

    fourp_text: dict[str, str] = {}
    for shard in sorted(fsspec_glob(f"{FOURP_CORPUS}/*.parquet"))[: args.corpus_shards]:
        with fsspec.open(shard, "rb") as f:
            t = pq.read_table(f, columns=["id", "text"])
        for doc_id, text in zip(t.column("id").to_pylist(), t.column("text").to_pylist(), strict=True):
            if doc_id in target_fourp:
                fourp_text[doc_id] = text
        logger.info("scanned %s; found %d/%d 4plus docs", shard.rsplit("/", 1)[1], len(fourp_text), len(target_fourp))
        if len(fourp_text) == len(target_fourp):
            break

    out = [
        {
            "jaccard": round(jac, 4),
            "three_id": three_id,
            "fourplus_id": fourp_id,
            "three_text": three_text.get(three_id),
            "fourplus_text": fourp_text.get(fourp_id),
        }
        for three_id, fourp_id, jac in sorted(sampled, key=lambda r: r[2])
    ]
    with fsspec.open(f"{PROBE}/samples.json", "w") as f:
        json.dump(out, f, indent=2)
    logger.info(
        "wrote %d pairs -> %s/samples.json (3_text %d, 4plus_text %d)",
        len(out),
        PROBE,
        sum(1 for o in out if o["three_text"]),
        sum(1 for o in out if o["fourplus_text"]),
    )


if __name__ == "__main__":
    main()
