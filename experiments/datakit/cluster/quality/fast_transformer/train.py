# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train and evaluate one :class:`FastTransformer` on the oracle-scored data.

Trains a regression head (MSE on the continuous normalized quality score) with
an internal train/val split for early model selection, then reports the held-out
oracle metrics (:mod:`experiments.datakit.cluster.quality.fast_transformer.metrics`):
AUC and Spearman of predicted quality vs the Claude oracle, plus accuracy /
precision / recall / F1 at threshold 0.5.
"""

import argparse
import json
import logging
import os
import tempfile
import time
from dataclasses import asdict, dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pyarrow.parquet as pq
from rigging.filesystem import StoragePath
from rigging.log_setup import configure_logging

from experiments.datakit.cluster.quality.fast_transformer.data import PackedData, build_remap, encode_texts, pack
from experiments.datakit.cluster.quality.fast_transformer.inference import data_parallel_shardings, predict
from experiments.datakit.cluster.quality.fast_transformer.metrics import auc, spearman_rho
from experiments.datakit.cluster.quality.fast_transformer.model import (
    FastTransformer,
    FastTransformerConfig,
    count_params,
)
from experiments.datakit.cluster.quality.fast_transformer.scorer import MODEL_STEM, artifact_names

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 0.5

# The deployed scorer's config + tokenizer (selected by the earlier architecture sweep).
TOKENIZER = "intfloat/multilingual-e5-small"
MAX_TOKENS = 512
DEPLOY_CONFIG = {
    "embed_dim": 256,
    "hidden_dim": 256,
    "num_layers": 2,
    "num_heads": 4,
    "pool_window": 64,
    "pool_kind": "meanmaxmin",
}
DEFAULT_LABELS = "s3://marin-us-east-02a/marin/datakit/quality_labels_20260709.parquet"


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


@dataclass
class FitResult:
    model: FastTransformer
    val_ids: np.ndarray
    val_scores: np.ndarray
    best_epoch: int
    train_seconds: float
    params: int
    flops_per_token: float


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
        val_rho = spearman_rho(predict(model, val_ids, batch_size=batch_size).tolist(), val_scores.tolist())
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


def _save_scorer(model, remap: dict, tokenizer: str, config: FastTransformerConfig, out_dir: str, name: str) -> None:
    """Serialise the model + vocab remap + meta in the format `scorer.py` loads."""
    out_dir = out_dir.rstrip("/")
    eqx_name, remap_name, meta_name = artifact_names(name)
    fd, local = tempfile.mkstemp(suffix=".eqx")
    os.close(fd)
    eqx.tree_serialise_leaves(local, model)  # eqx serialise needs a local path
    with open(local, "rb") as src, StoragePath(f"{out_dir}/{eqx_name}").open("wb") as dst:
        dst.write(src.read())
    with StoragePath(f"{out_dir}/{remap_name}").open("w") as fh:
        json.dump(remap, fh)
    # Serialise the FULL config (not a hand-picked subset) so the loader rebuilds the exact
    # architecture -- otherwise a non-default final_pool / mlp_ratio silently falls back to
    # the dataclass default and scores with the wrong (or shape-mismatched) model.
    meta = {"tokenizer": tokenizer, "max_tokens": config.max_tokens, "config": asdict(config)}
    with StoragePath(f"{out_dir}/{meta_name}").open("w") as fh:
        json.dump(meta, fh)
    logger.info("saved scorer -> %s/%s (+ %s, %s)", out_dir, eqx_name, remap_name, meta_name)


def train_from_labels(
    *,
    labels_path: str,
    out_dir: str,
    name: str = MODEL_STEM,
    tokenizer: str = TOKENIZER,
    max_tokens: int = MAX_TOKENS,
    eval_frac: float = 1 / 7,
    hp: TrainHParams | None = None,
) -> FitResult:
    """Train the deployed pooled scorer from the merged oracle-label parquet
    (``source``/``text``/``score_normalized``) and save it in the scorer format."""
    hp = hp or TrainHParams()
    with StoragePath(labels_path).open("rb") as fh:
        table = pq.read_table(fh, columns=["text", "score_normalized"])
    texts = [t or "" for t in table.column("text").to_pylist()]
    scores = np.array(table.column("score_normalized").to_pylist(), dtype=np.float32)

    perm = np.random.default_rng(hp.seed).permutation(len(texts))
    n_eval = max(1, int(len(texts) * eval_frac))
    eval_idx, train_idx = perm[:n_eval], perm[n_eval:]

    def _split(idx):
        return [texts[i] for i in idx], scores[idx]

    tr_texts, tr_scores = _split(train_idx)
    ev_texts, ev_scores = _split(eval_idx)
    tr_raw = encode_texts(tokenizer, tr_texts, max_tokens)
    ev_raw = encode_texts(tokenizer, ev_texts, max_tokens)
    remap = build_remap(tr_raw, min_count=2)
    vocab = len(remap) + 2
    data = PackedData(
        train=pack(tr_raw, remap, tr_scores, max_tokens),
        eval=pack(ev_raw, remap, ev_scores, max_tokens),
        vocab_size=vocab,
        tokenizer_name=tokenizer,
        max_tokens=max_tokens,
    )
    config = FastTransformerConfig(
        vocab_size=vocab, max_tokens=max_tokens, dropout=0.1, final_pool="mean", **DEPLOY_CONFIG
    )
    logger.info("training on %d labels (%d train / %d eval); vocab=%d", len(texts), len(train_idx), len(eval_idx), vocab)
    fitted = fit(config, data, hp)
    holdout = _metrics(predict(fitted.model, data.eval.ids), data.eval.scores)
    logger.info("HOLDOUT AUC=%.4f spearman=%.4f (best_epoch=%d)", holdout.auc, holdout.spearman_rho, fitted.best_epoch)
    _save_scorer(fitted.model, remap, tokenizer, config, out_dir, name)
    return fitted


def main() -> None:
    p = argparse.ArgumentParser(description="Train the pooled fast-transformer quality scorer on the oracle labels.")
    p.add_argument("--labels", default=DEFAULT_LABELS, help="merged oracle-label parquet")
    p.add_argument("--out-dir", required=True, help="dir to write <name>.eqx + _remap.json + _meta.json")
    p.add_argument("--name", default=MODEL_STEM)
    args = p.parse_args()
    configure_logging(logging.INFO)
    train_from_labels(labels_path=args.labels, out_dir=args.out_dir, name=args.name)


if __name__ == "__main__":
    main()
