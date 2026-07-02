# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Distill the FineWeb-Edu educational-value score into the pooled quality model.

Our Sonnet oracle is effectively an educational-value classifier (arxiv / pubmed
/ textbooks score highest), which makes the FineWeb-Edu classifier a near-twin:
both rate "educational value" 0-5, FineWeb-Edu via Llama-3-70B annotations on
~460k web pages (arXiv 2406.17557). Its per-document ``score`` ships free with
the dataset, so this is a no-spend teacher that -- unlike the source-prior -- is
per-document (no collapse to source identity) and unlike the Nemotron buckets is
aligned with our rubric.

Two questions, answered by [`transfer.pretrain_then_finetune`](transfer.py):

  - ``pretrain_only`` (FineWeb-Edu-trained model, evaluated zero-shot on our
    oracle holdout) measures how close FineWeb-Edu's notion is to our oracle.
  - ``pretrain_ft`` (then fine-tuned on the 5.6k gold) tests whether the
    aligned, downstream-validated teacher beats the ~0.875 from-scratch plateau.

FineWeb-Edu is English web only; the gold spans the full Marin mixture, so the
fine-tune adapts the web-edu representation to the rest of the mixture.

Submit on a v6e slice (region-local to the eu-west4 FineWeb-Edu mirror)::

    python -m experiments.datakit.cluster.quality.fast_transformer.pretrain.fineweb_edu \\
      --fineweb gs://marin-eu-west4/raw/fineweb-edu-87f0914/sample/10BT \\
      --train ...train-n7000....parquet --eval ...eval-n1000....parquet \\
      --out gs://.../fast_transformer/fineweb-edu-v1 --fineweb-size 200000
"""

import argparse
import hashlib
import io
import json
import logging

import jax
import numpy as np
import pyarrow.parquet as pq
from rigging.filesystem import marin_temp_bucket, open_url, url_to_fs
from rigging.log_setup import configure_logging

from experiments.datakit.cluster.quality.fast_transformer.data import (
    NUM_RESERVED,
    PackedSplit,
    build_remap,
    encode_corpus,
    encode_texts,
    pack,
)
from experiments.datakit.cluster.quality.fast_transformer.model import FastTransformerConfig
from experiments.datakit.cluster.quality.fast_transformer.pretrain.transfer import pretrain_then_finetune
from experiments.datakit.cluster.quality.fast_transformer.train import TrainHParams

logger = logging.getLogger(__name__)

BASELINE = {"auc": 0.846, "spearman_rho": 0.641}  # fasttext, for reference
POOLED_PLATEAU = {"auc": 0.875, "spearman_rho": 0.703}  # best from-scratch pooled sweep

# FineWeb-Edu's regression score is nominally 0-5 (occasionally just outside);
# clamp and divide so the teacher target lands in [0, 1] like the oracle labels.
_FWE_SCORE_MAX = 5.0


def read_fineweb_edu(path: str, n: int) -> tuple[list[str], np.ndarray]:
    """Read ``n`` (text, normalized edu score) pairs from a FineWeb-Edu parquet dir.

    Streams row groups across the dir's parquets until ``n`` English, non-empty
    docs are collected. The target is ``clip(score, 0, 5) / 5``.
    """
    fs, resolved = url_to_fs(path)
    files = sorted(f for f in fs.ls(resolved, detail=False) if f.endswith(".parquet"))
    if not files:
        raise ValueError(f"no parquets under {path}")
    texts: list[str] = []
    scores: list[float] = []
    for shard in files:
        with fs.open(shard, "rb") as fh:
            pfile = pq.ParquetFile(fh)
            for batch in pfile.iter_batches(batch_size=4096, columns=["text", "score", "language"]):
                langs = batch.column("language").to_pylist()
                txts = batch.column("text").to_pylist()
                scs = batch.column("score").to_pylist()
                for lang, text, score in zip(langs, txts, scs, strict=True):
                    if lang != "en" or not text or score is None:
                        continue
                    texts.append(str(text))
                    scores.append(min(max(float(score), 0.0), _FWE_SCORE_MAX) / _FWE_SCORE_MAX)
                    if len(texts) >= n:
                        logger.info("read %d FineWeb-Edu docs from %s", len(texts), path)
                        return texts, np.asarray(scores, dtype=np.float32)
    logger.info("read %d FineWeb-Edu docs from %s (exhausted)", len(texts), path)
    return texts, np.asarray(scores, dtype=np.float32)


def _load_or_build_data(args) -> tuple[PackedSplit, PackedSplit, PackedSplit, int]:
    """Build (fineweb teacher, gold train, gold holdout) packed splits, shared vocab.

    Cached to the region-local temp bucket keyed by everything that affects it.
    """
    spec = [
        args.tokenizer,
        args.fineweb,
        args.train,
        args.eval,
        args.max_tokens,
        args.min_count,
        args.max_vocab,
        args.fineweb_size,
    ]
    key = hashlib.sha1(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:16]
    cache_path = marin_temp_bucket(30, f"fast_transformer/fineweb-cache/{key}.npz", source_prefix=args.train)
    fs, resolved = url_to_fs(cache_path)
    if fs.exists(resolved):
        logger.info("loading fineweb cache %s", cache_path)
        with fs.open(resolved, "rb") as fh:
            z = np.load(io.BytesIO(fh.read()), allow_pickle=True)
        fineweb = PackedSplit(z["fw_ids"], z["fw_len"], z["fw_score"], list(z["fw_src"]))
        gold = PackedSplit(z["tr_ids"], z["tr_len"], z["tr_score"], list(z["tr_src"]))
        holdout = PackedSplit(z["ev_ids"], z["ev_len"], z["ev_score"], list(z["ev_src"]))
        return fineweb, gold, holdout, int(z["vocab_size"])

    fw_texts, fw_scores = read_fineweb_edu(args.fineweb, args.fineweb_size)
    fw_raw = encode_texts(args.tokenizer, fw_texts, args.max_tokens)
    tr_raw, tr_scores, tr_src = encode_corpus(args.tokenizer, args.train, args.max_tokens)
    ev_raw, ev_scores, ev_src = encode_corpus(args.tokenizer, args.eval, args.max_tokens)

    remap = build_remap(fw_raw + tr_raw, args.min_count, args.max_vocab)
    vocab_size = len(remap) + NUM_RESERVED
    logger.info(
        "encoded fineweb=%d gold-train=%d gold-eval=%d vocab=%d", len(fw_raw), len(tr_raw), len(ev_raw), vocab_size
    )

    fineweb = pack(fw_raw, remap, fw_scores, ["fineweb-edu"] * len(fw_raw), args.max_tokens)
    gold = pack(tr_raw, remap, tr_scores, tr_src, args.max_tokens)
    holdout = pack(ev_raw, remap, ev_scores, ev_src, args.max_tokens)

    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        fw_ids=fineweb.ids,
        fw_len=fineweb.lengths,
        fw_score=fineweb.scores,
        fw_src=np.asarray(fineweb.sources, dtype=object),
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
    logger.info("wrote fineweb cache %s", cache_path)
    return fineweb, gold, holdout, vocab_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fineweb", required=True, help="FineWeb-Edu parquet dir (text + score)")
    parser.add_argument("--train", required=True, help="Oracle gold train parquet")
    parser.add_argument("--eval", required=True, help="Oracle gold holdout parquet")
    parser.add_argument("--out", required=True)
    parser.add_argument("--tokenizer", default="marin-community/marin-tokenizer")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--max-vocab", type=int, default=None)
    parser.add_argument("--fineweb-size", type=int, default=200_000, help="FineWeb-Edu teacher docs")
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
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    configure_logging(logging.INFO)
    logger.info("jax backend=%s devices=%s", jax.default_backend(), jax.devices())

    fineweb, gold, holdout, vocab_size = _load_or_build_data(args)
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
        "fineweb=%s gold=%s holdout=%s vocab=%d flops/tok=%.0fK",
        fineweb.ids.shape,
        gold.ids.shape,
        holdout.ids.shape,
        vocab_size,
        config.flops_per_token() / 1e3,
    )

    pretrain_hp = TrainHParams(
        lr=args.pretrain_lr, epochs=args.pretrain_epochs, batch_size=args.batch_size, patience=args.pretrain_patience
    )
    finetune_hp = TrainHParams(lr=args.finetune_lr, batch_size=args.batch_size)
    outcome = pretrain_then_finetune(
        config, fineweb, gold, holdout, tokenizer=args.tokenizer, pretrain_hp=pretrain_hp, finetune_hp=finetune_hp
    )

    summary = {
        "baseline_fasttext": BASELINE,
        "pooled_plateau": POOLED_PLATEAU,
        "vocab_size": vocab_size,
        "max_tokens": args.max_tokens,
        "n_fineweb": int(fineweb.ids.shape[0]),
        "n_gold_train": int(gold.ids.shape[0]),
        "params": outcome["params"],
        "flops_per_token": outcome["flops_per_token"],
        "results": {k: outcome[k] for k in ("scratch", "pretrain_only", "pretrain_ft")},
    }
    logger.info("FINEWEB-EDU SUMMARY:\n%s", json.dumps(summary, indent=2))
    with open_url(args.out.rstrip("/") + "/summary.json", "wb") as fh:
        fh.write(json.dumps(summary, indent=2).encode())
    logger.info("wrote %s/summary.json", args.out)


if __name__ == "__main__":
    main()
