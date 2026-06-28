# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train and evaluate one :class:`FastTransformer` on the oracle-scored data.

Trains a regression head (MSE on the continuous normalized quality score) with
an internal train/val split for early model selection, then reports the held-out
oracle metrics using the *same* functions as the fasttext baseline
(:mod:`experiments.datakit.cluster.quality.v0.ops.eval_holdout`) so the numbers
are directly comparable: AUC and Spearman of predicted quality vs the Claude
oracle, plus accuracy / precision / recall / F1 at threshold 0.5.
"""

import logging
import time
from dataclasses import asdict, dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from experiments.datakit.cluster.quality.fast_transformer.data import PackedData
from experiments.datakit.cluster.quality.fast_transformer.model import (
    FastTransformer,
    FastTransformerConfig,
    count_params,
)
from experiments.datakit.cluster.quality.v0.ops.eval_holdout import auc, spearman_rho

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.5


@dataclass(frozen=True)
class TrainHParams:
    # Large batches keep the step count (and thus per-step XLA dispatch overhead)
    # low -- the model is tiny, so dispatch latency, not compute, dominates.
    lr: float = 5e-4
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    epochs: int = 60
    batch_size: int = 512
    warmup_frac: float = 0.15
    val_frac: float = 0.1
    eval_every: int = 2
    seed: int = 0


@dataclass
class EvalMetrics:
    n: int
    auc: float
    spearman_rho: float
    accuracy: float
    precision: float
    recall: float
    f1: float


def _binary_metrics(y_true: list[int], y_pred: list[int]) -> tuple[float, float, float, float]:
    tp = fp = fn = tn = 0
    for t, p in zip(y_true, y_pred, strict=True):
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 1 and p == 0:
            fn += 1
        else:
            tn += 1
    n = tp + fp + fn + tn
    acc = (tp + tn) / n if n else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return acc, prec, rec, f1


def _metrics(scores: np.ndarray, targets: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> EvalMetrics:
    y_true = [1 if s >= threshold else 0 for s in targets.tolist()]
    y_pred = [1 if p >= 0.5 else 0 for p in scores.tolist()]
    acc, prec, rec, f1 = _binary_metrics(y_true, y_pred)
    return EvalMetrics(
        n=len(y_true),
        auc=auc(y_true, scores.tolist()),
        spearman_rho=spearman_rho(scores.tolist(), targets.tolist()),
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
    )


@eqx.filter_jit
def _predict_batch(model: FastTransformer, ids):
    """Sigmoid quality score for a fixed-shape batch (jitted, inference mode)."""
    return jax.nn.sigmoid(model(ids, key=None, inference=True))


def _predict(model: FastTransformer, ids: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Sigmoid quality score for every row in ``ids``.

    Chunks are padded to a constant ``batch_size`` so ``_predict_batch`` compiles
    once per model structure and is reused across all epochs and configs (the
    per-epoch val eval would otherwise dominate the sweep with recompiles).
    """
    out: list[np.ndarray] = []
    n = ids.shape[0]
    for start in range(0, n, batch_size):
        chunk = ids[start : start + batch_size]
        pad = batch_size - chunk.shape[0]
        if pad:
            chunk = np.concatenate([chunk, np.zeros((pad, ids.shape[1]), dtype=ids.dtype)], axis=0)
        preds = np.asarray(_predict_batch(model, jnp.asarray(chunk)))
        out.append(preds[: batch_size - pad] if pad else preds)
    return np.concatenate(out)


@dataclass
class RunResult:
    config: dict
    hparams: dict
    params: int
    flops_per_token: float
    val: EvalMetrics
    holdout: EvalMetrics
    best_epoch: int
    train_seconds: float


def train_one(config: FastTransformerConfig, data: PackedData, hp: TrainHParams) -> RunResult:
    key = jax.random.PRNGKey(hp.seed)
    model_key, key = jax.random.split(key)
    model = FastTransformer(config, key=model_key)
    n_params = count_params(model)
    flops = config.flops_per_token()
    logger.info(
        "model: params=%.2fM flops/token=%.0fK pool=%s w=%d L=%d d=%d",
        n_params / 1e6,
        flops / 1e3,
        config.pool_kind,
        config.pool_window,
        config.num_layers,
        config.hidden_dim,
    )

    # Internal train/val split for model selection; never touches the holdout.
    tr = data.train
    rng = np.random.default_rng(hp.seed)
    perm = rng.permutation(tr.n)
    n_val = max(1, int(tr.n * hp.val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    tr_ids, tr_scores = tr.ids[train_idx], tr.scores[train_idx]
    val_ids, val_scores = tr.ids[val_idx], tr.scores[val_idx]

    steps_per_epoch = max(1, len(train_idx) // hp.batch_size)
    total_steps = steps_per_epoch * hp.epochs
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=hp.lr * 0.05,
        peak_value=hp.lr,
        warmup_steps=max(1, int(total_steps * hp.warmup_frac)),
        decay_steps=total_steps,
        end_value=hp.lr * 0.05,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(hp.grad_clip),
        optax.adamw(schedule, weight_decay=hp.weight_decay),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def step(model, opt_state, ids, targets, step_key):
        def loss_fn(m):
            logits = m(ids, key=step_key, inference=False)
            preds = jax.nn.sigmoid(logits)
            return jnp.mean((preds - targets) ** 2)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    best_val_rho = -2.0
    best_model = model
    best_epoch = 0
    t0 = time.time()
    for epoch in range(hp.epochs):
        ep_perm = rng.permutation(len(train_idx))
        for s in range(steps_per_epoch):
            batch = ep_perm[s * hp.batch_size : (s + 1) * hp.batch_size]
            key, step_key = jax.random.split(key)
            model, opt_state, loss = step(
                model,
                opt_state,
                jnp.asarray(tr_ids[batch]),
                jnp.asarray(tr_scores[batch]),
                step_key,
            )
        if epoch % hp.eval_every != 0 and epoch != hp.epochs - 1:
            continue
        val_pred = _predict(model, val_ids)
        val_rho = spearman_rho(val_pred.tolist(), val_scores.tolist())
        if np.isfinite(val_rho) and val_rho > best_val_rho:
            best_val_rho = val_rho
            best_model = model
            best_epoch = epoch
        logger.info(
            "epoch %d: train_loss=%.4f val_rho=%.4f (best=%.4f @ %d)",
            epoch,
            float(loss),
            val_rho,
            best_val_rho,
            best_epoch,
        )
    train_seconds = time.time() - t0

    val_pred = _predict(best_model, val_ids)
    holdout_pred = _predict(best_model, data.eval.ids)
    val_metrics = _metrics(val_pred, val_scores)
    holdout_metrics = _metrics(holdout_pred, data.eval.scores)
    logger.info(
        "HOLDOUT: AUC=%.4f spearman=%.4f acc=%.4f F1=%.4f (params=%.2fM flops/tok=%.0fK)",
        holdout_metrics.auc,
        holdout_metrics.spearman_rho,
        holdout_metrics.accuracy,
        holdout_metrics.f1,
        n_params / 1e6,
        flops / 1e3,
    )
    return RunResult(
        config={k: v for k, v in asdict(config).items()},
        hparams=asdict(hp),
        params=n_params,
        flops_per_token=flops,
        val=val_metrics,
        holdout=holdout_metrics,
        best_epoch=best_epoch,
        train_seconds=train_seconds,
    )
