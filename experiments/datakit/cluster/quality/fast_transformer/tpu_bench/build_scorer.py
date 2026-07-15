# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build a *config-faithful* fast-transformer scorer for the TPU throughput benchmark.

The deployed ``.eqx`` lives in the CoreWeave object store (no creds from a GCP box), so
this reconstructs an equivalent scorer in a GCS model dir: the deployed architecture +
tokenizer, a vocab remap built from a **real** corpus slice (``min_count=2``, exactly as
training does), and a monotonic calibration fit to this model's own raw-score
distribution. Weights are random -- throughput depends only on config + vocab size +
token-length distribution, all of which this preserves -- so the benchmark numbers are
faithful. Quality is NOT faithful (that comes from the recorded Spearman 0.69 vs 0.44),
and the calibration only guarantees non-degenerate buckets, not oracle agreement.

Run once (CPU is fine) before the scoring pipelines:

    python -m ...tpu_bench.build_scorer \
        --corpus 'gs://marin-eu-west4/.../data-*.parquet' \
        --out-dir gs://marin-eu-west4/user/rav/quality/ft-tpu-bench
"""

import argparse
import json
import logging

import jax
import numpy as np
import pyarrow.parquet as pq
from rigging.filesystem import StoragePath, open_url

from experiments.datakit.cluster.quality.fast_transformer.data import build_remap, encode_texts
from experiments.datakit.cluster.quality.fast_transformer.model import (
    FastTransformer,
    FastTransformerConfig,
    count_params,
)
from experiments.datakit.cluster.quality.fast_transformer.scorer import load_pooled_scorer, score_bme
from experiments.datakit.cluster.quality.fast_transformer.train import DEPLOY_CONFIG, MAX_TOKENS, TOKENIZER, _save_scorer

logger = logging.getLogger(__name__)


def read_texts(corpus_glob: str, n_docs: int, text_col: str) -> list[str]:
    """Read up to ``n_docs`` document texts from the corpus glob (in listing order)."""
    files = sorted(str(m) for m in StoragePath(corpus_glob).glob())
    if not files:
        raise ValueError(f"no files matched {corpus_glob}")
    texts: list[str] = []
    for f in files:
        with StoragePath(f).open("rb") as fh:
            table = pq.read_table(fh, columns=[text_col])
        texts.extend(t or "" for t in table.column(text_col).to_pylist())
        logger.info("read %d docs (%s)", len(texts), f)
        if len(texts) >= n_docs:
            break
    return texts[:n_docs]


def fit_calibration(scorer, texts: list[str]) -> dict:
    """Percentile calibration: map this model's raw bme scores monotonically onto [0,1] so
    the fixed bucket cutpoints land on a spread distribution (not oracle-calibrated)."""
    raw = score_bme(scorer, texts)
    qs = np.linspace(0.0, 1.0, 11)
    xk = np.quantile(raw, qs)
    xk = np.maximum.accumulate(xk)
    xk[0] -= 1e-6
    xk[-1] += 1e-6
    return {"xk": [float(x) for x in xk], "yk": [float(q) for q in qs]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", required=True, help="glob of text parquet files for vocab + calibration")
    p.add_argument("--out-dir", required=True, help="GCS model dir to write scorer + calib_bme.json")
    p.add_argument("--text-col", default="text")
    p.add_argument("--vocab-docs", type=int, default=50000, help="docs used to build the vocab remap")
    p.add_argument("--calib-docs", type=int, default=3000, help="docs used to fit the calibration")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)

    texts = read_texts(args.corpus, args.vocab_docs, args.text_col)
    logger.info("building vocab from %d docs with tokenizer %s", len(texts), TOKENIZER)
    raw_ids = encode_texts(TOKENIZER, texts, MAX_TOKENS)
    remap = build_remap(raw_ids, min_count=2)
    vocab = len(remap) + 2
    config = FastTransformerConfig(
        vocab_size=vocab, max_tokens=MAX_TOKENS, dropout=0.0, final_pool="mean", **DEPLOY_CONFIG
    )
    model = FastTransformer(config, key=jax.random.PRNGKey(0))
    logger.info(
        "model: vocab=%d params=%.2fM flops/token=%.0f", vocab, count_params(model) / 1e6, config.flops_per_token()
    )

    _save_scorer(model, remap, TOKENIZER, config, args.out_dir, name="pooled_junkgate2")

    scorer = load_pooled_scorer(args.out_dir)
    calib = fit_calibration(scorer, texts[: args.calib_docs])
    with open_url(f"{args.out_dir.rstrip('/')}/calib_bme.json", "w") as fh:
        fh.write(json.dumps(calib))
    logger.info("wrote scorer + calib_bme.json to %s (vocab=%d)", args.out_dir, vocab)


if __name__ == "__main__":
    main()
