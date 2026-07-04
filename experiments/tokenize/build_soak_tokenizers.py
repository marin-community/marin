# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the representative corpus, train the 6 soak SuperBPE tokenizers, and push them.

One reproducible driver for the 24h soak's tokenizers (the ``soak-*`` :data:`TRAIN_SPECS`):

1. stream the representative multi-domain corpus (:mod:`experiments.tokenize.corpus`) to a fixed
   ``$MARIN_PREFIX/raw/soak_tokenizer_corpus/<version>`` path,
2. train each ``soak-*`` spec on it (:func:`train_one`), in parallel processes,
3. push each to ``mirror://tokenizers/trained/<name>/`` (:func:`push_one`) so ``BAKEOFF_ARM``
   resolves it through ``levanter.load_tokenizer``.

Run on a cluster CPU box (needs ``$MARIN_PREFIX``, region-local S3 write, and HF access):

    uv run iris --cluster=cw-rno2a job run --no-wait --cpu 32 --memory 200GB --extra cpu \
      --enable-extra-resources --job-name build-soak-tokenizers \
      -- python -m experiments.tokenize.build_soak_tokenizers
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from experiments.tokenize.corpus import CorpusBuildConfig, build_tokenizer_training_corpus
from experiments.tokenize.push_trained_tokenizers import push_one
from experiments.tokenize.train_tokenizers import TRAIN_SPECS, read_corpus, train_one

logger = logging.getLogger(__name__)

# Bump alongside corpus.py's version when the corpus composition changes.
_CORPUS_VERSION = "2026.07.04"
_OUT_BASE = "/tmp/soak_tokenizers"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    prefix = os.environ["MARIN_PREFIX"].rstrip("/")
    corpus_dir = f"{prefix}/raw/soak_tokenizer_corpus/{_CORPUS_VERSION}"

    logger.info("building corpus -> %s", corpus_dir)
    build_tokenizer_training_corpus(CorpusBuildConfig(output_path=corpus_dir))

    texts = read_corpus(corpus_dir)
    total_mb = sum(len(t.encode("utf-8")) for t in texts) / 1e6
    logger.info("loaded corpus: %d docs, %.1f MB", len(texts), total_mb)

    specs = [s for s in TRAIN_SPECS if s.name.startswith("soak-")]
    logger.info("training %d soak tokenizers: %s", len(specs), [s.name for s in specs])

    # Fork workers share the parent's already-loaded corpus copy-on-write, so parallelism does not
    # duplicate the corpus per worker; each SuperBPE stage-2 is single-threaded numpy.
    with ProcessPoolExecutor(max_workers=len(specs)) as pool:
        futures = {pool.submit(train_one, spec, texts, f"{_OUT_BASE}/{spec.name}"): spec.name for spec in specs}
        rows = [future.result() for future in as_completed(futures)]

    for row in rows:
        pushed = push_one(row["tokenizer_dir"], row["name"])
        logger.info("pushed %s: vocab=%d ref=%s", row["name"], row["vocab_size"], pushed["ref"])

    logger.info("done: %d soak tokenizers trained + pushed", len(rows))


if __name__ == "__main__":
    main()
