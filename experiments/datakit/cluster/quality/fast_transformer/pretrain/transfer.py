# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared scratch -> pretrain -> finetune comparison for supervised pretraining.

Both supervised-pretraining experiments ([`nemotron_bucket`](nemotron_bucket.py)
and [`source_prior`](source_prior.py)) ask the same question: does pretraining
the pooled :class:`~..model.FastTransformer` on a large, cheaply labelled corpus
beat training from scratch on the 5.6k oracle labels? They differ only in how the
weak corpus is built and labelled, so the three-model comparison and holdout
evaluation live here.
"""

import logging
from dataclasses import asdict

from experiments.datakit.cluster.quality.fast_transformer.data import PackedData, PackedSplit
from experiments.datakit.cluster.quality.fast_transformer.model import FastTransformerConfig
from experiments.datakit.cluster.quality.fast_transformer.train import (
    TrainHParams,
    _metrics,
    _predict,
    fit,
)

logger = logging.getLogger(__name__)


def evaluate_holdout(model, holdout: PackedSplit, label: str) -> dict:
    """Predict on the oracle holdout and log/return the comparison metrics."""
    m = _metrics(_predict(model, holdout.ids), holdout.scores)
    logger.info("%-14s AUC=%.4f spearman=%.4f acc=%.4f F1=%.4f", label, m.auc, m.spearman_rho, m.accuracy, m.f1)
    return asdict(m)


def pretrain_then_finetune(
    config: FastTransformerConfig,
    weak: PackedSplit,
    gold: PackedSplit,
    holdout: PackedSplit,
    *,
    tokenizer: str,
    pretrain_hp: TrainHParams,
    finetune_hp: TrainHParams,
) -> dict:
    """Train three pooled models on a shared vocab and evaluate each on the holdout.

    ``weak``, ``gold`` and ``holdout`` must already be packed with the same vocab
    (``config.vocab_size``) so the pretrained embeddings transfer. Returns a dict
    with the three runs plus ``params`` / ``flops_per_token``:

      - ``scratch``: from-scratch on the gold labels (control).
      - ``pretrain_only``: trained only on the weak corpus; evaluated zero-shot.
      - ``pretrain_ft``: pretrained, then fine-tuned on the gold labels.
    """
    gold_data = PackedData(gold, holdout, config.vocab_size, tokenizer, config.max_tokens)
    weak_data = PackedData(weak, holdout, config.vocab_size, tokenizer, config.max_tokens)

    scratch = fit(config, gold_data, finetune_hp)
    pretrained = fit(config, weak_data, pretrain_hp)
    finetuned = fit(config, gold_data, finetune_hp, init_model=pretrained.model)

    return {
        "scratch": {**evaluate_holdout(scratch.model, holdout, "scratch"), "best_epoch": scratch.best_epoch},
        "pretrain_only": {
            **evaluate_holdout(pretrained.model, holdout, "pretrain_only"),
            "best_epoch": pretrained.best_epoch,
        },
        "pretrain_ft": {**evaluate_holdout(finetuned.model, holdout, "pretrain_ft"), "best_epoch": finetuned.best_epoch},
        "params": scratch.params,
        "flops_per_token": scratch.flops_per_token,
    }
