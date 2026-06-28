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
from jax.sharding import Mesh, NamedSharding, PartitionSpec

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
    epochs: int = 40  # hard cap; early stopping usually ends well before this
    batch_size: int = 512
    warmup_frac: float = 0.15
    val_frac: float = 0.1
    eval_every: int = 1
    patience: int = 2  # eval rounds without val-Spearman improvement before stopping
    remat: bool = False  # gradient-checkpoint the forward (needed for long-context token models)
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


def _forward(model, ids, key):
    """Training-mode logits; pulled out so it can be gradient-checkpointed."""
    return model(ids, key=key, inference=False)


@eqx.filter_jit
def _predict_batch(model: FastTransformer, ids):
    """Sigmoid quality score for a fixed-shape batch (jitted, inference mode)."""
    return jax.nn.sigmoid(model(ids, key=None, inference=True))


# Tokens per inference batch; bounds the [B, T, E] embedding activation so long
# context (T up to 16k) auto-shrinks the batch instead of OOMing one v6e chip.
_PREDICT_TOKEN_BUDGET = 262_144


def _predict(model: FastTransformer, ids: np.ndarray, batch_size: int | None = None) -> np.ndarray:
    """Sigmoid quality score for every row in ``ids``.

    Chunks are padded to a constant ``batch_size`` so ``_predict_batch`` compiles
    once per sequence length and is reused across all epochs and configs (the
    per-epoch val eval would otherwise dominate the sweep with recompiles). When
    ``batch_size`` is not given it is sized from the sequence length to keep the
    activation footprint bounded.
    """
    if batch_size is None:
        batch_size = max(8, _PREDICT_TOKEN_BUDGET // ids.shape[1])
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


@dataclass
class FitResult:
    model: FastTransformer
    val_ids: np.ndarray
    val_scores: np.ndarray
    best_epoch: int
    train_seconds: float
    params: int
    flops_per_token: float


def data_parallel_shardings():
    """(num_devices, replicated, batch-sharded) shardings over all chips.

    Data parallelism: the model + optimizer state are replicated on every chip and
    the batch is split across them, so all of a v6e slice's chips are used instead
    of one. With one device this is a no-op.
    """
    devices = jax.devices()
    mesh = Mesh(np.asarray(devices), ("dp",))
    return len(devices), NamedSharding(mesh, PartitionSpec()), NamedSharding(mesh, PartitionSpec("dp"))


def train_regressor(
    model,
    tr_ids: np.ndarray,
    tr_scores: np.ndarray,
    val_ids: np.ndarray,
    val_scores: np.ndarray,
    hp: TrainHParams,
):
    """Train any ``(ids, key, inference) -> logits`` model on the MSE-on-sigmoid
    regression objective.

    Selects the checkpoint with the best internal-val Spearman and stops early
    after ``hp.patience`` eval rounds without improvement (best epoch is typically
    < 15, so running the full epoch cap wastes most of the trial). Data-parallel
    across all chips; ``hp.remat`` gradient-checkpoints the forward for long
    context. Works for both :class:`FastTransformer` and the pretrained encoder
    classifier. Returns ``(best_model, best_epoch, train_seconds)``.
    """
    key = jax.random.PRNGKey(hp.seed)
    ndev, replicated, batch_shard = data_parallel_shardings()
    batch_size = max(ndev, (hp.batch_size // ndev) * ndev)  # divisible across chips
    steps_per_epoch = max(1, len(tr_ids) // batch_size)
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
    model = jax.device_put(model, replicated)
    opt_state = jax.device_put(optimizer.init(eqx.filter(model, eqx.is_inexact_array)), replicated)
    forward = eqx.filter_checkpoint(_forward) if hp.remat else _forward

    @eqx.filter_jit
    def step(model, opt_state, ids, targets, step_key):
        def loss_fn(m):
            preds = jax.nn.sigmoid(forward(m, ids, step_key))
            return jnp.mean((preds - targets) ** 2)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
        return eqx.apply_updates(model, updates), opt_state, loss

    rng = np.random.default_rng(hp.seed)
    best_val_rho, best_model, best_epoch, no_improve = -2.0, model, 0, 0
    t0 = time.time()
    logger.info("train_regressor: %d chips, global batch=%d, %d steps/epoch", ndev, batch_size, steps_per_epoch)
    for epoch in range(hp.epochs):
        ep_perm = rng.permutation(len(tr_ids))
        for s in range(steps_per_epoch):
            batch = ep_perm[s * batch_size : (s + 1) * batch_size]
            key, step_key = jax.random.split(key)
            ids = jax.device_put(jnp.asarray(tr_ids[batch]), batch_shard)
            targets = jax.device_put(jnp.asarray(tr_scores[batch]), batch_shard)
            model, opt_state, loss = step(model, opt_state, ids, targets, step_key)
        if epoch % hp.eval_every != 0 and epoch != hp.epochs - 1:
            continue
        # Reuse the (memory-sized) training batch for eval -- token-level encoders
        # at long context can't fit the default inference batch's O(T^2) attention.
        val_rho = spearman_rho(_predict(model, val_ids, batch_size=batch_size).tolist(), val_scores.tolist())
        improved = bool(np.isfinite(val_rho) and val_rho > best_val_rho)
        if improved:
            best_val_rho, best_model, best_epoch, no_improve = val_rho, model, epoch, 0
        else:
            no_improve += 1
        logger.info(
            "epoch %d: train_loss=%.4f val_rho=%.4f (best=%.4f @ %d, stale=%d)",
            epoch,
            float(loss),
            val_rho,
            best_val_rho,
            best_epoch,
            no_improve,
        )
        if no_improve >= hp.patience:
            logger.info("early stop at epoch %d (val Spearman stale for %d evals)", epoch, no_improve)
            break
    return best_model, best_epoch, time.time() - t0


def fit(
    config: FastTransformerConfig,
    data: PackedData,
    hp: TrainHParams,
    *,
    init_model: FastTransformer | None = None,
) -> FitResult:
    """Train one model, selecting the checkpoint with the best internal-val Spearman.

    ``init_model`` continues training from existing weights (e.g. fine-tuning a
    pretrained model) instead of fresh init.
    """
    model_key = jax.random.PRNGKey(hp.seed)
    model = init_model if init_model is not None else FastTransformer(config, key=model_key)
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
    val_ids, val_scores = tr.ids[val_idx], tr.scores[val_idx]

    best_model, best_epoch, train_seconds = train_regressor(
        model, tr.ids[train_idx], tr.scores[train_idx], val_ids, val_scores, hp
    )
    return FitResult(
        model=best_model,
        val_ids=val_ids,
        val_scores=val_scores,
        best_epoch=best_epoch,
        train_seconds=train_seconds,
        params=n_params,
        flops_per_token=flops,
    )


def train_one(config: FastTransformerConfig, data: PackedData, hp: TrainHParams) -> RunResult:
    fitted = fit(config, data, hp)
    val_pred = _predict(fitted.model, fitted.val_ids)
    holdout_pred = _predict(fitted.model, data.eval.ids)
    val_metrics = _metrics(val_pred, fitted.val_scores)
    holdout_metrics = _metrics(holdout_pred, data.eval.scores)
    logger.info(
        "HOLDOUT: AUC=%.4f spearman=%.4f acc=%.4f F1=%.4f (params=%.2fM flops/tok=%.0fK)",
        holdout_metrics.auc,
        holdout_metrics.spearman_rho,
        holdout_metrics.accuracy,
        holdout_metrics.f1,
        fitted.params / 1e6,
        fitted.flops_per_token / 1e3,
    )
    return RunResult(
        config={k: v for k, v in asdict(config).items()},
        hparams=asdict(hp),
        params=fitted.params,
        flops_per_token=fitted.flops_per_token,
        val=val_metrics,
        holdout=holdout_metrics,
        best_epoch=fitted.best_epoch,
        train_seconds=fitted.train_seconds,
    )
