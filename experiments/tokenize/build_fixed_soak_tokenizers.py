# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retrain the confounded soak SuperBPE tokenizers on the existing corpus, and push them as new arms.

Commit 11bd2f4e9c fixed a bug in ``train_tokenizers._sample_stage2_corpus``: SuperBPE stage 2
sampled an un-shuffled byte prefix of the training corpus, and ``read_corpus`` concatenates
domains in a fixed order (``english_web`` first, ~half the corpus), so the currently-deployed
``soak-superbpe-64k``/``-128k``/``-64k-digits``/``-128k-digits`` arms never saw code,
multilingual, or math text at stage 2. This script retrains those four with the fix
(:data:`~experiments.tokenize.train_tokenizers.FIXED_SOAK_SPECS`) and pushes each under a new
``-fixed`` name (:data:`~experiments.tokenize.bakeoff_tokenizers.SOAK_FIXED_ARMS`) so a re-run
can select the fixed tokenizer via ``BAKEOFF_ARM`` without touching the already-soaked arms.

Unlike :mod:`experiments.tokenize.build_soak_tokenizers`, this reads the existing corpus at
``$MARIN_PREFIX/raw/soak_tokenizer_corpus/<version>`` rather than rebuilding it: the fix is in
stage-2 sampling, not corpus composition, so the same corpus applies unchanged.

Run on a cluster CPU box (needs ``$MARIN_PREFIX``, region-local S3 write, and HF access):

    uv run iris --cluster=cw-rno2a job run --no-wait --cpu 32 --memory 200GB --extra cpu \\
      --enable-extra-resources --job-name build-fixed-soak-tokenizers \\
      -e MARIN_PREFIX s3://marin-us-east-02a/marin \\
      -- python -m experiments.tokenize.build_fixed_soak_tokenizers
"""

from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from experiments.tokenize.push_trained_tokenizers import push_one
from experiments.tokenize.train_tokenizers import FIXED_SOAK_SPECS, read_corpus, train_one

logger = logging.getLogger(__name__)

# Must match build_soak_tokenizers.py's `_CORPUS_VERSION` -- the corpus the confounded soak
# tokenizers were trained from. Only stage-2 sampling changed, not corpus composition, so it is
# still the right corpus for the fixed retrain.
_CORPUS_VERSION = "2026.07.04"
_OUT_BASE = "/tmp/fixed_soak_tokenizers"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--arms",
        help="comma-separated fixed-spec names to (re)build; default all of FIXED_SOAK_SPECS",
    )
    args = ap.parse_args()

    specs = FIXED_SOAK_SPECS
    if args.arms:
        wanted = {name.strip() for name in args.arms.split(",")}
        unknown = wanted - {s.name for s in FIXED_SOAK_SPECS}
        if unknown:
            raise SystemExit(f"unknown --arms {sorted(unknown)}; known: {[s.name for s in FIXED_SOAK_SPECS]}")
        specs = tuple(s for s in FIXED_SOAK_SPECS if s.name in wanted)

    prefix = os.environ["MARIN_PREFIX"].rstrip("/")
    corpus_dir = f"{prefix}/raw/soak_tokenizer_corpus/{_CORPUS_VERSION}"

    logger.info("reading existing corpus <- %s", corpus_dir)
    texts = read_corpus(corpus_dir)
    total_mb = sum(len(t.encode("utf-8")) for t in texts) / 1e6
    logger.info("loaded corpus: %d docs, %.1f MB", len(texts), total_mb)

    logger.info("training %d fixed soak tokenizers: %s", len(specs), [s.name for s in specs])

    # Fork workers share the parent's already-loaded corpus copy-on-write, so parallelism does not
    # duplicate the corpus per worker; each SuperBPE stage-2 is single-threaded numpy.
    with ProcessPoolExecutor(max_workers=len(specs)) as pool:
        futures = {pool.submit(train_one, spec, texts, f"{_OUT_BASE}/{spec.name}"): spec.name for spec in specs}
        rows = [future.result() for future in as_completed(futures)]

    for row in rows:
        pushed = push_one(row["tokenizer_dir"], row["name"])
        logger.info("pushed %s: vocab=%d ref=%s", row["name"], row["vocab_size"], pushed["ref"])

    logger.info("done: %d fixed soak tokenizers trained + pushed", len(rows))


if __name__ == "__main__":
    main()
