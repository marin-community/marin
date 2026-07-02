# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Does pretraining on Nemotron-CC quality buckets beat training from scratch?

The pooled fast-transformer plateaus on the 5.6k-doc oracle set (regularization
moves val but not holdout Spearman) -- a data-limited ceiling. This driver tests
the obvious lever: pretrain the representation on a large, free, quality-bucketed
Nemotron-CC slice ([`nemotron_sample`](nemotron_sample.py) output), then
fine-tune on the oracle labels.

Result (negative): the Nemotron buckets encode Nvidia's quality rubric, which is
misaligned with our oracle's "value as pretraining data" -- pretrain zero-shot
transfers at rho 0.28 and fine-tuning lands *below* the from-scratch control. The
three-model comparison itself lives in [`transfer`](transfer.py).

Submit on a v6e slice (see ../sweep.py for the flag set)::

    python -m experiments.datakit.cluster.quality.fast_transformer.pretrain.nemotron_bucket \\
      --nemotron gs://.../fast_transformer/nemotron-60k.parquet \\
      --train ...train-n7000....parquet --eval ...eval-n1000....parquet \\
      --out gs://.../fast_transformer/nemotron-bucket
"""

import argparse
import json
import logging

import jax
from rigging.filesystem import open_url
from rigging.log_setup import configure_logging

from experiments.datakit.cluster.quality.fast_transformer.data import NUM_RESERVED, build_remap, encode_corpus, pack
from experiments.datakit.cluster.quality.fast_transformer.model import FastTransformerConfig
from experiments.datakit.cluster.quality.fast_transformer.pretrain.transfer import pretrain_then_finetune
from experiments.datakit.cluster.quality.fast_transformer.train import TrainHParams

logger = logging.getLogger(__name__)

BASELINE = {"auc": 0.846, "spearman_rho": 0.641}  # fasttext, for reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nemotron", required=True, help="Nemotron pretraining parquet (oracle schema)")
    parser.add_argument("--train", required=True, help="Oracle train parquet")
    parser.add_argument("--eval", required=True, help="Oracle holdout parquet")
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer", default="marin-community/marin-tokenizer")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--pretrain-epochs", type=int, default=15)
    parser.add_argument("--pretrain-lr", type=float, default=5e-4)
    parser.add_argument("--finetune-epochs", type=int, default=40)
    parser.add_argument("--finetune-lr", type=float, default=2e-4)
    args = parser.parse_args()

    configure_logging(logging.INFO)
    logger.info("jax backend=%s devices=%s", jax.default_backend(), jax.devices())

    # Tokenize all corpora and build ONE shared vocab (union of nemotron + oracle
    # train) so the pretrained embeddings transfer to fine-tuning.
    nemo_raw, nemo_scores, nemo_src = encode_corpus(args.tokenizer, args.nemotron, args.max_tokens)
    tr_raw, tr_scores, tr_src = encode_corpus(args.tokenizer, args.train, args.max_tokens)
    ev_raw, ev_scores, ev_src = encode_corpus(args.tokenizer, args.eval, args.max_tokens)
    logger.info("encoded: nemotron=%d train=%d eval=%d", len(nemo_raw), len(tr_raw), len(ev_raw))

    remap = build_remap(nemo_raw + tr_raw, args.min_count)
    vocab_size = len(remap) + NUM_RESERVED
    nemo = pack(nemo_raw, remap, nemo_scores, nemo_src, args.max_tokens)
    gold = pack(tr_raw, remap, tr_scores, tr_src, args.max_tokens)
    holdout = pack(ev_raw, remap, ev_scores, ev_src, args.max_tokens)

    config = FastTransformerConfig(
        vocab_size=vocab_size,
        max_tokens=args.max_tokens,
        pool_window=64,
        pool_kind="meanmaxmin",
        embed_dim=256,
        hidden_dim=512,
        num_layers=4,
        num_heads=8,
    )
    pretrain_hp = TrainHParams(lr=args.pretrain_lr, epochs=args.pretrain_epochs)
    finetune_hp = TrainHParams(lr=args.finetune_lr, epochs=args.finetune_epochs)
    outcome = pretrain_then_finetune(
        config, nemo, gold, holdout, tokenizer=args.tokenizer, pretrain_hp=pretrain_hp, finetune_hp=finetune_hp
    )

    summary = {
        "baseline_fasttext": BASELINE,
        "vocab_size": vocab_size,
        "params": outcome["params"],
        "flops_per_token": outcome["flops_per_token"],
        "n_nemotron": nemo.n,
        "results": {k: outcome[k] for k in ("scratch", "pretrain_only", "pretrain_ft")},
    }
    logger.info("NEMOTRON-BUCKET SUMMARY:\n%s", json.dumps(summary, indent=2))
    with open_url(args.out.rstrip("/") + "/summary.json", "wb") as fh:
        fh.write(json.dumps(summary, indent=2).encode())
    logger.info("wrote %s/summary.json", args.out)


if __name__ == "__main__":
    main()
