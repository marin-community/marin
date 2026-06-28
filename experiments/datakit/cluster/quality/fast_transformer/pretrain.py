# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pretrain the encoder with next-token prediction, then fine-tune for quality.

All architecture/capacity/context sweeps show the same wall: the model memorizes
the 5.6k-doc oracle set (train loss -> 0 by ~epoch 8, holdout Spearman ~0.70).
This driver tests the data lever the right way -- self-supervised pretraining on
*free* text:

  1. NTP-pretrain a :class:`~.encoder.TokenEncoder` on a large unlabeled corpus
     (the Nemotron sample's text; labels ignored). This learns the embedding +
     transformer from billions of tokens instead of thousands of documents.
  2. Fine-tune an :class:`~.encoder.EncoderClassifier` built on the pretrained
     encoder against the oracle labels.
  3. Train a from-scratch classifier as the control.

Unlike the failed Nemotron-bucket *supervised* pretraining (which transferred a
misaligned quality notion, rho 0.28), NTP is self-supervised: it learns general
language structure, which is exactly what 5.6k docs cannot teach.

Submit on a v6e slice (long-context flag set in sweep.py)::

    python -m experiments.datakit.cluster.quality.fast_transformer.pretrain \\
      --corpus gs://.../fast_transformer/nemotron-60k.parquet \\
      --train ...train-n7000....parquet --eval ...eval-n1000....parquet \\
      --out gs://.../fast_transformer/pretrain-ntp
"""

import argparse
import json
import logging
import time
from dataclasses import asdict

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from rigging.filesystem import open_url
from rigging.log_setup import configure_logging

from experiments.datakit.cluster.quality.fast_transformer.data import (
    NUM_RESERVED,
    UNK_ID,
    build_remap,
    encode_corpus,
    pack,
)
from experiments.datakit.cluster.quality.fast_transformer.encoder import EncoderClassifier, EncoderConfig, TokenEncoder
from experiments.datakit.cluster.quality.fast_transformer.model import count_params
from experiments.datakit.cluster.quality.fast_transformer.train import (
    TrainHParams,
    _metrics,
    _predict,
    data_parallel_shardings,
    train_regressor,
)

logger = logging.getLogger(__name__)

BASELINE = {"auc": 0.846, "spearman_rho": 0.641}  # fasttext, for reference


def _pack_lm_blocks(raw_ids: list[list[int]], remap: dict[int, int], max_tokens: int, max_blocks: int) -> np.ndarray:
    """Concatenate the remapped token stream and chunk into [N, max_tokens] LM blocks."""
    stream: list[int] = []
    for row in raw_ids:
        stream.extend(remap.get(t, UNK_ID) for t in row)
    n_blocks = min(len(stream) // max_tokens, max_blocks)
    if n_blocks == 0:
        raise ValueError(f"corpus too small: {len(stream)} tokens < block size {max_tokens}")
    arr = np.asarray(stream[: n_blocks * max_tokens], dtype=np.int32).reshape(n_blocks, max_tokens)
    return arr


def pretrain_ntp(
    encoder: TokenEncoder,
    blocks: np.ndarray,
    *,
    lr: float,
    epochs: int,
    batch_size: int,
    patience: int,
    weight_decay: float = 0.01,
    val_frac: float = 0.02,
    seed: int = 0,
) -> TokenEncoder:
    """Train ``encoder`` with next-token prediction; early-stop on val LM loss."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(blocks))
    n_val = max(1, int(len(blocks) * val_frac))
    val_blocks, tr_blocks = blocks[perm[:n_val]], blocks[perm[n_val:]]
    ndev, replicated, batch_shard = data_parallel_shardings()
    batch_size = max(ndev, (batch_size // ndev) * ndev)  # divisible across chips
    steps_per_epoch = max(1, len(tr_blocks) // batch_size)
    total_steps = steps_per_epoch * epochs
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=lr * 0.05,
        peak_value=lr,
        warmup_steps=max(1, total_steps // 20),
        decay_steps=total_steps,
        end_value=lr * 0.05,
    )
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule, weight_decay=weight_decay))
    encoder = jax.device_put(encoder, replicated)
    opt_state = jax.device_put(optimizer.init(eqx.filter(encoder, eqx.is_inexact_array)), replicated)

    def _lm_loss(enc, ids):
        # Gradient-checkpointed: the [B, T, vocab] logits are recomputed in backward
        # rather than stored, so the NTP softmax stops being the memory bottleneck.
        logits = enc.lm_logits(enc.encode(ids, key=None, inference=False))[:, :-1]
        return optax.softmax_cross_entropy_with_integer_labels(logits, ids[:, 1:]).mean()

    lm_loss = eqx.filter_checkpoint(_lm_loss)

    @eqx.filter_jit
    def step(encoder, opt_state, ids):
        loss, grads = eqx.filter_value_and_grad(lm_loss)(encoder, ids)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(encoder, eqx.is_inexact_array))
        return eqx.apply_updates(encoder, updates), opt_state, loss

    @eqx.filter_jit
    def batch_val_loss(encoder, ids):
        logits = encoder.lm_logits(encoder.encode(ids, key=None, inference=True))[:, :-1]
        return optax.softmax_cross_entropy_with_integer_labels(logits, ids[:, 1:]).mean()

    def val_loss(encoder) -> float:
        losses = [
            float(batch_val_loss(encoder, jnp.asarray(val_blocks[i : i + batch_size])))
            for i in range(0, len(val_blocks), batch_size)
        ]
        return float(np.mean(losses))

    best_loss, best_encoder, no_improve = float("inf"), encoder, 0
    t0 = time.time()
    logger.info(
        "NTP pretrain: %d chips, %d train / %d val blocks, global batch=%d, %d steps/epoch",
        ndev,
        len(tr_blocks),
        len(val_blocks),
        batch_size,
        steps_per_epoch,
    )
    for epoch in range(epochs):
        ep_perm = rng.permutation(len(tr_blocks))
        for s in range(steps_per_epoch):
            batch = ep_perm[s * batch_size : (s + 1) * batch_size]
            ids = jax.device_put(jnp.asarray(tr_blocks[batch]), batch_shard)
            encoder, opt_state, loss = step(encoder, opt_state, ids)
        vl = val_loss(encoder)
        improved = vl < best_loss
        if improved:
            best_loss, best_encoder, no_improve = vl, encoder, 0
        else:
            no_improve += 1
        logger.info(
            "NTP epoch %d: train_loss=%.4f val_loss=%.4f (best=%.4f, stale=%d)",
            epoch,
            float(loss),
            vl,
            best_loss,
            no_improve,
        )
        if no_improve >= patience:
            logger.info("NTP early stop at epoch %d", epoch)
            break
    logger.info("NTP pretrain done in %.0fs, best val_loss=%.4f", time.time() - t0, best_loss)
    return best_encoder


def _evaluate(model, holdout_ids, holdout_scores, label: str, batch_size: int) -> dict:
    m = _metrics(_predict(model, holdout_ids, batch_size=batch_size), holdout_scores)
    logger.info("%s holdout: AUC=%.4f spearman=%.4f acc=%.4f F1=%.4f", label, m.auc, m.spearman_rho, m.accuracy, m.f1)
    return asdict(m)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, help="Unlabeled text parquet for NTP pretraining")
    parser.add_argument("--train", required=True, help="Oracle train parquet")
    parser.add_argument("--eval", required=True, help="Oracle holdout parquet")
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer", default="marin-community/marin-tokenizer")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--min-count", type=int, default=3, help="Vocab prune (drop rare tokens)")
    parser.add_argument("--max-vocab", type=int, default=32000, help="Cap vocab (top-K) to bound the NTP softmax")
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--max-blocks", type=int, default=40000)
    parser.add_argument("--pretrain-lr", type=float, default=3e-4)
    parser.add_argument("--pretrain-epochs", type=int, default=20)
    parser.add_argument("--pretrain-patience", type=int, default=2)
    parser.add_argument("--finetune-lr", type=float, default=3e-4)
    parser.add_argument("--dropouts", type=float, nargs="+", default=[0.1, 0.3], help="Fine-tune dropout grid")
    args = parser.parse_args()

    configure_logging(logging.INFO)
    logger.info("jax backend=%s devices=%s", jax.default_backend(), jax.devices())

    corpus_raw, _, _ = encode_corpus(args.tokenizer, args.corpus, args.max_tokens)
    tr_raw, tr_scores, _ = encode_corpus(args.tokenizer, args.train, args.max_tokens)
    ev_raw, ev_scores, ev_src = encode_corpus(args.tokenizer, args.eval, args.max_tokens)
    remap = build_remap(corpus_raw + tr_raw, args.min_count, args.max_vocab)
    vocab_size = len(remap) + NUM_RESERVED
    logger.info("encoded: corpus=%d train=%d eval=%d vocab=%d", len(corpus_raw), len(tr_raw), len(ev_raw), vocab_size)

    blocks = _pack_lm_blocks(corpus_raw, remap, args.max_tokens, args.max_blocks)
    train = pack(tr_raw, remap, tr_scores, [""] * len(tr_raw), args.max_tokens)
    holdout = pack(ev_raw, remap, ev_scores, ev_src, args.max_tokens)

    # With remat the O(T^2) attention and [B,T,vocab] softmax are recomputed in
    # backward, so per-chip batch is bounded by the forward materialization; size
    # per chip, then x num_chips since training is data-parallel across the slice.
    t = args.max_tokens
    ndev = jax.device_count()
    per_chip_pretrain = max(2, min(800_000_000 // (t * vocab_size), 120_000_000 // (t * t)))
    per_chip_ft = max(8, 120_000_000 // (t * t))
    pretrain_batch = per_chip_pretrain * ndev
    ft_batch = per_chip_ft * ndev
    logger.info(
        "LM blocks=%s oracle train=%s holdout=%s | %d chips, pretrain_batch=%d ft_batch=%d",
        blocks.shape,
        train.ids.shape,
        holdout.ids.shape,
        ndev,
        pretrain_batch,
        ft_batch,
    )

    config = EncoderConfig(vocab_size=vocab_size, max_tokens=t, dim=args.dim, num_layers=args.layers, num_heads=8)
    logger.info("encoder: flops/token=%.0fK", config.flops_per_token() / 1e3)

    # Internal train/val split of the oracle labels for fine-tune model selection.
    ft_hp = TrainHParams(lr=args.finetune_lr, batch_size=ft_batch, remat=True)
    rng = np.random.default_rng(ft_hp.seed)
    perm = rng.permutation(train.n)
    n_val = max(1, int(train.n * ft_hp.val_frac))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    split = (train.ids[tr_idx], train.scores[tr_idx], train.ids[val_idx], train.scores[val_idx])

    results: dict[str, dict] = {}

    # From-scratch control (no pretraining), token-level long context.
    ctrl = EncoderClassifier(config, key=jax.random.PRNGKey(1), dropout=0.1)
    logger.info("classifier params=%.2fM", count_params(ctrl) / 1e6)
    ctrl, _, _ = train_regressor(ctrl, *split, ft_hp)
    results["scratch"] = _evaluate(ctrl, holdout.ids, holdout.scores, "scratch", ft_batch)

    # NTP-pretrain once, then fine-tune across the dropout grid from the same encoder.
    enc = TokenEncoder(config, key=jax.random.PRNGKey(2))
    enc = pretrain_ntp(
        enc,
        blocks,
        lr=args.pretrain_lr,
        epochs=args.pretrain_epochs,
        batch_size=pretrain_batch,
        patience=args.pretrain_patience,
    )
    for d in args.dropouts:
        clf = EncoderClassifier(config, key=jax.random.PRNGKey(3), encoder=enc, dropout=d)
        clf, ep, _ = train_regressor(clf, *split, ft_hp)
        m = _evaluate(clf, holdout.ids, holdout.scores, f"pretrain+ft drop={d}", ft_batch)
        results[f"pretrain_ft_drop{d}"] = {**m, "best_epoch": ep}

    summary = {
        "baseline_fasttext": BASELINE,
        "vocab_size": vocab_size,
        "max_tokens": t,
        "n_lm_blocks": int(blocks.shape[0]),
        "encoder_params": count_params(ctrl),
        "flops_per_token": config.flops_per_token(),
        "pretrain_batch": pretrain_batch,
        "ft_batch": ft_batch,
        "results": results,
    }
    logger.info("PRETRAIN SUMMARY:\n%s", json.dumps(summary, indent=2))
    with open_url(args.out.rstrip("/") + "/summary.json", "wb") as fh:
        fh.write(json.dumps(summary, indent=2).encode())
    logger.info("wrote %s/summary.json", args.out)


if __name__ == "__main__":
    main()
