# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Phase 2: does Nemotron-quality pretraining beat training from scratch?

The fast-transformer plateaus on the 5.6k-doc oracle set (regularization moves
val but not holdout Spearman) -- a data-limited ceiling. This driver tests the
obvious lever: pretrain the representation on a large, free, quality-labelled
Nemotron-CC slice (``nemotron_sample`` output), then fine-tune on the oracle
labels and evaluate on the oracle holdout.

It runs three models on a shared (union) vocabulary so the comparison is clean:

  1. **scratch**   -- the from-scratch oracle baseline (control).
  2. **pretrain**  -- trained only on Nemotron buckets; evaluated zero-shot on
     the oracle holdout (transfer signal).
  3. **finetune**  -- pretrained, then fine-tuned on the oracle labels.

Submit on a v6e slice (see sweep.py for the flag set)::

    python -m experiments.datakit.cluster.quality.fast_transformer.pretrain_finetune \\
      --nemotron gs://.../fast_transformer/nemotron-60k.parquet \\
      --train ...train-n7000....parquet --eval ...eval-n1000....parquet \\
      --out gs://.../fast_transformer/phase2
"""

import argparse
import json
import logging
from dataclasses import asdict

import jax
from rigging.filesystem import open_url
from rigging.log_setup import configure_logging

from experiments.datakit.cluster.quality.fast_transformer.data import (
    NUM_RESERVED,
    PackedData,
    build_remap,
    encode_corpus,
    pack,
)
from experiments.datakit.cluster.quality.fast_transformer.model import FastTransformerConfig
from experiments.datakit.cluster.quality.fast_transformer.train import (
    TrainHParams,
    _metrics,
    _predict,
    fit,
)

logger = logging.getLogger(__name__)

BASELINE = {"auc": 0.846, "spearman_rho": 0.641}  # fasttext, for reference


def _evaluate(model, holdout, label: str) -> dict:
    pred = _predict(model, holdout.ids)
    m = _metrics(pred, holdout.scores)
    logger.info("%s holdout: AUC=%.4f spearman=%.4f acc=%.4f F1=%.4f", label, m.auc, m.spearman_rho, m.accuracy, m.f1)
    return asdict(m)


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
    # train) so pretrained embeddings transfer to fine-tuning.
    nemo_raw, nemo_scores, nemo_src = encode_corpus(args.tokenizer, args.nemotron, args.max_tokens)
    tr_raw, tr_scores, tr_src = encode_corpus(args.tokenizer, args.train, args.max_tokens)
    ev_raw, ev_scores, ev_src = encode_corpus(args.tokenizer, args.eval, args.max_tokens)
    logger.info("encoded: nemotron=%d train=%d eval=%d", len(nemo_raw), len(tr_raw), len(ev_raw))

    remap = build_remap(nemo_raw + tr_raw, args.min_count)
    vocab_size = len(remap) + NUM_RESERVED
    nemo = pack(nemo_raw, remap, nemo_scores, nemo_src, args.max_tokens)
    train = pack(tr_raw, remap, tr_scores, tr_src, args.max_tokens)
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
    pretrain_data = PackedData(
        train=nemo, eval=holdout, vocab_size=vocab_size, tokenizer_name=args.tokenizer, max_tokens=args.max_tokens
    )
    oracle_data = PackedData(
        train=train, eval=holdout, vocab_size=vocab_size, tokenizer_name=args.tokenizer, max_tokens=args.max_tokens
    )

    # 1. Control: from scratch on the oracle labels (shared vocab).
    scratch = fit(config, oracle_data, TrainHParams())
    scratch_m = _evaluate(scratch.model, holdout, "scratch")

    # 2. Pretrain on Nemotron buckets; evaluate zero-shot on the oracle holdout.
    pre = fit(config, pretrain_data, TrainHParams(lr=args.pretrain_lr, epochs=args.pretrain_epochs))
    pretrain_m = _evaluate(pre.model, holdout, "pretrain(zero-shot)")

    # 3. Fine-tune the pretrained model on the oracle labels.
    ft = fit(config, oracle_data, TrainHParams(lr=args.finetune_lr, epochs=args.finetune_epochs), init_model=pre.model)
    finetune_m = _evaluate(ft.model, holdout, "finetune")

    summary = {
        "baseline_fasttext": BASELINE,
        "vocab_size": vocab_size,
        "params": scratch.params,
        "flops_per_token": scratch.flops_per_token,
        "n_nemotron": nemo.n,
        "scratch": scratch_m,
        "pretrain_zeroshot": pretrain_m,
        "finetune": finetune_m,
    }
    logger.info("PHASE2 SUMMARY:\n%s", json.dumps(summary, indent=2))
    with open_url(args.out.rstrip("/") + "/summary.json", "wb") as fh:
        fh.write(json.dumps(summary, indent=2).encode())
    logger.info("wrote %s/summary.json", args.out)


if __name__ == "__main__":
    main()
