# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Weak-supervised pretrain the pooled quality model, then fine-tune on gold.

Every architecture/capacity/context sweep hits the same wall: the pooled
:class:`~.model.FastTransformer` memorizes the 5.6k-doc oracle set and plateaus
at holdout AUC ~0.875. The lever is more labels, and the data already supplies a
free, well-aligned one: a document's *source* predicts its oracle score at AUC
0.852 on the gold set (source identity explains 41% of score variance), because
the oracle scores "value as pretraining data" and the datakit sources span the
full quality spectrum (arxiv/pubmed/peS2o high, parser-junk/IRC low).

So instead of next-token prediction (which forced the weaker token-level encoder
and an indirect objective), this pretrains the *winning pooled architecture* on
the *exact task*:

  1. Sample a large corpus across every datakit source and label each doc with
     its source's gold-calibrated mean score (the source-prior soft target).
  2. Weak-supervised pretrain the pooled model on those soft targets -- the same
     MSE-on-sigmoid regression as fine-tuning, just at ~40x the data.
  3. Fine-tune on the 5.6k gold labels (light, low-LR, early-stopped) starting
     from the pretrained weights.
  4. Controls: a from-scratch pooled model (reproduces ~0.875) and the
     pretrained model evaluated directly (the weak-teacher ceiling, ~0.85).

Unlike the failed Nemotron-bucket pretraining (which borrowed Nvidia's 3-level
quality notion and transferred at rho 0.28), the source-prior is calibrated
against our own oracle gold and spans 104 fine-grained levels -- alignment is
measured, not assumed.

Submit on a v6e slice (region-local to the gold/source data)::

    python -m experiments.datakit.cluster.quality.fast_transformer.weak_pretrain \\
      --train ...train-n7000....parquet --eval ...eval-n1000....parquet \\
      --out gs://.../fast_transformer/weak-pretrain-v1 --weak-size 200000
"""

import argparse
import hashlib
import io
import json
import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict

import jax
import numpy as np
from rigging.filesystem import marin_temp_bucket, open_url, url_to_fs
from rigging.log_setup import configure_logging

from experiments.datakit.cluster.quality.fast_transformer.data import (
    NUM_RESERVED,
    PackedData,
    PackedSplit,
    build_remap,
    encode_corpus,
    encode_texts,
    pack,
)
from experiments.datakit.cluster.quality.fast_transformer.model import FastTransformerConfig, count_params
from experiments.datakit.cluster.quality.fast_transformer.train import (
    TrainHParams,
    _metrics,
    _predict,
    fit,
)
from experiments.datakit.cluster.quality.v0.sample import (
    _active_sources,
    _per_source_seed,
    _sample_one_source,
    compute_quotas,
)

logger = logging.getLogger(__name__)

BASELINE = {"auc": 0.846, "spearman_rho": 0.641}  # fasttext, for reference
POOLED_PLATEAU = {"auc": 0.875, "spearman_rho": 0.703}  # best from-scratch pooled sweep


def sample_weak_corpus(
    total_size: int, floor_per_source: int, seed: int, num_workers: int
) -> tuple[list[str], list[str]]:
    """Sample ``total_size`` docs across every active datakit source.

    Reuses the gold-set sampler's sqrt-quota allocation so the corpus mirrors the
    gold's per-source composition. Returns ``(texts, sources)`` held in memory;
    the source label is the only "annotation" needed for the source-prior target.
    """
    sources = _active_sources()
    quotas = compute_quotas(sources, total_size=total_size, floor_per_source=floor_per_source)
    texts: list[str] = []
    srcs: list[str] = []
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futs = {pool.submit(_sample_one_source, q, _per_source_seed(q.name, seed)): q for q in quotas}
        for fut in as_completed(futs):
            q = futs[fut]
            try:
                rows = fut.result()
            except Exception:
                logger.exception("source %s: failed to sample", q.name)
                continue
            for row in rows:
                texts.append(str(row["text"]))
                srcs.append(str(row["source"]))
    logger.info("sampled %d weak docs across %d sources", len(texts), len(quotas))
    return texts, srcs


def _gold_source_prior(
    train_path: str, max_tokens: int, tokenizer: str
) -> tuple[dict[str, float], float, list[list[int]], np.ndarray, list[str]]:
    """Tokenize the gold train split and compute its per-source score prior.

    Returns ``(prior, grand_mean, raw_ids, scores, sources)`` so the caller reuses
    the tokenized gold train without re-reading the parquet.
    """
    raw_ids, scores, sources = encode_corpus(tokenizer, train_path, max_tokens)
    by_src: dict[str, list[float]] = defaultdict(list)
    for s, sc in zip(sources, scores.tolist(), strict=True):
        by_src[s].append(sc)
    prior = {k: float(np.mean(v)) for k, v in by_src.items()}
    grand = float(np.mean(scores))
    logger.info("source-prior over %d gold-train sources, grand mean=%.3f", len(prior), grand)
    return prior, grand, raw_ids, scores, sources


def _load_or_build_data(args) -> tuple[PackedSplit, PackedSplit, PackedSplit, int]:
    """Build (weak, gold-train, gold-holdout) packed splits with a shared vocab.

    Tokenization (the slow part) is cached to the region-local temp bucket keyed by
    everything that affects it; reruns with the same settings skip it entirely.
    """
    spec = [
        args.tokenizer,
        args.train,
        args.eval,
        args.max_tokens,
        args.min_count,
        args.max_vocab,
        args.weak_size,
        args.weak_floor,
        args.weak_seed,
    ]
    key = hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
    cache_path = marin_temp_bucket(30, f"fast_transformer/weak-cache/{key}.npz", source_prefix=args.train)
    fs, resolved = url_to_fs(cache_path)
    if fs.exists(resolved):
        logger.info("loading weak-pretrain cache %s", cache_path)
        with fs.open(resolved, "rb") as fh:
            z = np.load(io.BytesIO(fh.read()), allow_pickle=True)
        weak = PackedSplit(z["wk_ids"], z["wk_len"], z["wk_score"], list(z["wk_src"]))
        gold = PackedSplit(z["tr_ids"], z["tr_len"], z["tr_score"], list(z["tr_src"]))
        holdout = PackedSplit(z["ev_ids"], z["ev_len"], z["ev_score"], list(z["ev_src"]))
        return weak, gold, holdout, int(z["vocab_size"])

    prior, grand, tr_raw, tr_scores, tr_src = _gold_source_prior(args.train, args.max_tokens, args.tokenizer)
    ev_raw, ev_scores, ev_src = encode_corpus(args.tokenizer, args.eval, args.max_tokens)
    weak_texts, weak_src = sample_weak_corpus(args.weak_size, args.weak_floor, args.weak_seed, args.sample_workers)
    weak_raw = encode_texts(args.tokenizer, weak_texts, args.max_tokens)
    weak_scores = np.asarray([prior.get(s, grand) for s in weak_src], dtype=np.float32)

    # Shared vocab from weak corpus + gold train: the embedding table that overfits
    # on 5.6k docs is now built (and pretrained) over the far larger weak corpus.
    remap = build_remap(weak_raw + tr_raw, args.min_count, args.max_vocab)
    vocab_size = len(remap) + NUM_RESERVED
    logger.info(
        "encoded weak=%d gold-train=%d gold-eval=%d vocab=%d", len(weak_raw), len(tr_raw), len(ev_raw), vocab_size
    )

    weak = pack(weak_raw, remap, weak_scores, weak_src, args.max_tokens)
    gold = pack(tr_raw, remap, tr_scores, tr_src, args.max_tokens)
    holdout = pack(ev_raw, remap, ev_scores, ev_src, args.max_tokens)

    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        wk_ids=weak.ids,
        wk_len=weak.lengths,
        wk_score=weak.scores,
        wk_src=np.asarray(weak.sources, dtype=object),
        tr_ids=gold.ids,
        tr_len=gold.lengths,
        tr_score=gold.scores,
        tr_src=np.asarray(gold.sources, dtype=object),
        ev_ids=holdout.ids,
        ev_len=holdout.lengths,
        ev_score=holdout.scores,
        ev_src=np.asarray(holdout.sources, dtype=object),
        vocab_size=vocab_size,
    )
    with fs.open(resolved, "wb") as fh:
        fh.write(buf.getvalue())
    logger.info("wrote weak-pretrain cache %s", cache_path)
    return weak, gold, holdout, vocab_size


def _evaluate(model, holdout: PackedSplit, label: str, batch_size: int) -> dict:
    m = _metrics(_predict(model, holdout.ids, batch_size=batch_size), holdout.scores)
    logger.info("%-26s AUC=%.4f spearman=%.4f acc=%.4f F1=%.4f", label, m.auc, m.spearman_rho, m.accuracy, m.f1)
    return asdict(m)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True, help="Oracle gold train parquet")
    parser.add_argument("--eval", required=True, help="Oracle gold holdout parquet")
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer", default="marin-community/marin-tokenizer")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--min-count", type=int, default=3, help="Vocab prune (drop tokens seen < this)")
    parser.add_argument("--max-vocab", type=int, default=None, help="Optional top-K vocab cap (default: full vocab)")
    # Weak corpus
    parser.add_argument("--weak-size", type=int, default=200_000, help="Total weak docs sampled across sources")
    parser.add_argument("--weak-floor", type=int, default=50, help="Min weak docs per source")
    parser.add_argument("--weak-seed", type=int, default=7)
    parser.add_argument("--sample-workers", type=int, default=1)
    # Pooled architecture (the sweep anchor / ~0.875 winner)
    parser.add_argument("--pool-kind", default="meanmaxmin")
    parser.add_argument("--pool-window", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    # Optimization
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--pretrain-lr", type=float, default=5e-4)
    parser.add_argument("--pretrain-epochs", type=int, default=25)
    parser.add_argument("--pretrain-patience", type=int, default=3)
    parser.add_argument("--finetune-lr", type=float, default=2e-4)
    parser.add_argument("--dropout", type=float, default=0.1, help="Shared dropout for pretrain + finetune")
    args = parser.parse_args()

    configure_logging(logging.INFO)
    logger.info("jax backend=%s devices=%s", jax.default_backend(), jax.devices())

    weak, gold, holdout, vocab_size = _load_or_build_data(args)
    config = FastTransformerConfig(
        vocab_size=vocab_size,
        max_tokens=args.max_tokens,
        pool_window=args.pool_window,
        pool_kind=args.pool_kind,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    )
    logger.info(
        "weak=%s gold=%s holdout=%s vocab=%d flops/tok=%.0fK",
        weak.ids.shape,
        gold.ids.shape,
        holdout.ids.shape,
        vocab_size,
        config.flops_per_token() / 1e3,
    )

    results: dict[str, dict] = {}

    # Control: from-scratch on gold (should reproduce the ~0.875 plateau).
    scratch_hp = TrainHParams(lr=args.finetune_lr, batch_size=args.batch_size)
    scratch = fit(config, PackedData(gold, holdout, vocab_size, args.tokenizer, args.max_tokens), scratch_hp)
    results["scratch"] = {
        **_evaluate(scratch.model, holdout, "scratch", args.batch_size),
        "best_epoch": scratch.best_epoch,
    }

    # Weak-supervised pretrain on the source-prior soft targets (gold holdout only
    # for logging; model selection is on the internal weak-val split inside fit()).
    pre_hp = TrainHParams(
        lr=args.pretrain_lr, epochs=args.pretrain_epochs, batch_size=args.batch_size, patience=args.pretrain_patience
    )
    pretrained = fit(config, PackedData(weak, holdout, vocab_size, args.tokenizer, args.max_tokens), pre_hp)
    results["pretrain_only"] = {
        **_evaluate(pretrained.model, holdout, "pretrain_only", args.batch_size),
        "best_epoch": pretrained.best_epoch,
    }

    # Fine-tune on gold from the pretrained weights (low LR, early-stopped).
    ft_hp = TrainHParams(lr=args.finetune_lr, batch_size=args.batch_size)
    ft = fit(
        config,
        PackedData(gold, holdout, vocab_size, args.tokenizer, args.max_tokens),
        ft_hp,
        init_model=pretrained.model,
    )
    results["pretrain_ft"] = {
        **_evaluate(ft.model, holdout, "pretrain_ft", args.batch_size),
        "best_epoch": ft.best_epoch,
    }

    summary = {
        "baseline_fasttext": BASELINE,
        "pooled_plateau": POOLED_PLATEAU,
        "vocab_size": vocab_size,
        "max_tokens": args.max_tokens,
        "n_weak": int(weak.ids.shape[0]),
        "n_gold_train": int(gold.ids.shape[0]),
        "params": count_params(scratch.model),
        "flops_per_token": config.flops_per_token(),
        "config": {k: v for k, v in asdict(config).items()},
        "results": results,
    }
    logger.info("WEAK-PRETRAIN SUMMARY:\n%s", json.dumps(summary, indent=2))
    with open_url(args.out.rstrip("/") + "/summary.json", "wb") as fh:
        fh.write(json.dumps(summary, indent=2).encode())
    logger.info("wrote %s/summary.json", args.out)


if __name__ == "__main__":
    main()
