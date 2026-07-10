# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train the chosen fast-transformer config and evaluate it like the baseline.

Trains one configuration (the sweep winner), then reports held-out oracle
metrics directly comparable to ``ops/eval_holdout.py`` for the fasttext model:

  * AUC + Spearman of predicted quality vs the oracle (threshold-free).
  * accuracy / precision / recall / F1 at the fixed 0.5 threshold.
  * the same at a **val-calibrated** threshold (the threshold maximizing F1 on
    the internal val split). The regression head's output is compressed, so 0.5
    under-fires; calibrating the operating point makes the acc/F1 comparison to
    fasttext's naturally-calibrated softmax fair.
  * per-source breakdown and a per-doc predictions TSV.

Submit on a v6e slice (see sweep.py for the full flag set)::

    python -m experiments.datakit.cluster.quality.fast_transformer.eval_best \\
      --train ... --eval ... --out gs://.../fast_transformer/best \\
      --pool-kind meanmaxmin --pool-window 64 --hidden-dim 512 --num-layers 4
"""

import argparse
import json
import logging
from collections import defaultdict

import numpy as np
from rigging.filesystem import open_url
from rigging.log_setup import configure_logging

from experiments.datakit.cluster.quality.fast_transformer.data import load_packed
from experiments.datakit.cluster.quality.fast_transformer.model import FastTransformerConfig
from experiments.datakit.cluster.quality.fast_transformer.train import (
    DEFAULT_THRESHOLD,
    TrainHParams,
    _binary_metrics,
    _predict,
    fit,
)
from experiments.datakit.cluster.quality.v0.ops.eval_holdout import auc, spearman_rho

logger = logging.getLogger(__name__)


def _metrics_at(scores: np.ndarray, targets: np.ndarray, label_threshold: float, decision_threshold: float) -> dict:
    y_true = [1 if s >= label_threshold else 0 for s in targets.tolist()]
    y_pred = [1 if p >= decision_threshold else 0 for p in scores.tolist()]
    acc, prec, rec, f1 = _binary_metrics(y_true, y_pred)
    return {
        "n": len(y_true),
        "n_pos": int(sum(y_true)),
        "auc": auc(y_true, scores.tolist()),
        "spearman_rho": spearman_rho(scores.tolist(), targets.tolist()),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "decision_threshold": decision_threshold,
    }


def _best_f1_threshold(scores: np.ndarray, targets: np.ndarray, label_threshold: float) -> float:
    """Threshold on predictions that maximizes F1 of the positive class on val."""
    y_true = [1 if s >= label_threshold else 0 for s in targets.tolist()]
    best_t, best_f1 = 0.5, -1.0
    for t in np.quantile(scores, np.linspace(0.02, 0.98, 49)).tolist():
        y_pred = [1 if p >= t else 0 for p in scores.tolist()]
        _, _, _, f1 = _binary_metrics(y_true, y_pred)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer", default="marin-community/marin-tokenizer")
    parser.add_argument("--cache-dir", default="/tmp/ft-quality-cache")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--min-count", type=int, default=2)
    parser.add_argument("--pool-kind", default="meanmaxmin")
    parser.add_argument("--pool-window", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--final-pool", default="mean")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args()

    configure_logging(logging.INFO)
    data = load_packed(
        train_path=args.train,
        eval_path=args.eval,
        tokenizer_name=args.tokenizer,
        max_tokens=args.max_tokens,
        min_count=args.min_count,
        cache_dir=args.cache_dir,
    )
    config = FastTransformerConfig(
        vocab_size=data.vocab_size,
        max_tokens=args.max_tokens,
        pool_window=args.pool_window,
        pool_kind=args.pool_kind,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        final_pool=args.final_pool,
        dropout=args.dropout,
    )
    hp = TrainHParams(weight_decay=args.weight_decay, epochs=args.epochs)
    fitted = fit(config, data, hp)

    val_pred = _predict(fitted.model, fitted.val_ids)
    holdout_pred = _predict(fitted.model, data.eval.ids)
    holdout_scores = data.eval.scores
    cal_t = _best_f1_threshold(val_pred, fitted.val_scores, DEFAULT_THRESHOLD)

    summary = {
        "config": {k: getattr(config, k) for k in config.__dataclass_fields__},
        "params": fitted.params,
        "flops_per_token": fitted.flops_per_token,
        "best_epoch": fitted.best_epoch,
        "calibrated_threshold": cal_t,
        "holdout_at_0.5": _metrics_at(holdout_pred, holdout_scores, DEFAULT_THRESHOLD, 0.5),
        "holdout_calibrated": _metrics_at(holdout_pred, holdout_scores, DEFAULT_THRESHOLD, cal_t),
    }

    # Per-source breakdown (sources with >= 5 docs), at the calibrated threshold.
    by_source: dict[str, list[int]] = defaultdict(list)
    for i, src in enumerate(data.eval.sources):
        by_source[src].append(i)
    per_source = {}
    for src, idx in by_source.items():
        if len(idx) < 5:
            continue
        per_source[src] = _metrics_at(holdout_pred[idx], holdout_scores[idx], DEFAULT_THRESHOLD, cal_t)
    summary["per_source"] = per_source

    logger.info("SUMMARY:\n%s", json.dumps({k: v for k, v in summary.items() if k != "per_source"}, indent=2))

    out = args.out.rstrip("/")
    with open_url(out + "/summary.json", "wb") as fh:
        fh.write(json.dumps(summary, indent=2).encode())
    lines = ["row\tsource\toracle_score\toracle_label\tp_high"]
    for i in range(len(holdout_pred)):
        lab = 1 if holdout_scores[i] >= DEFAULT_THRESHOLD else 0
        lines.append(f"{i}\t{data.eval.sources[i]}\t{holdout_scores[i]:.4f}\t{lab}\t{holdout_pred[i]:.6f}")
    with open_url(out + "/predictions.tsv", "wb") as fh:
        fh.write(("\n".join(lines) + "\n").encode())
    logger.info("wrote summary + predictions to %s", out)


if __name__ == "__main__":
    main()
