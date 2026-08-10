# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Embedding-table experiments for the pooled fast-transformer scorer.

Variants of :func:`~experiments.datakit.cluster.quality.fast_transformer.train.train_from_labels`
that change only how the embedding table is built, holding the deployed
architecture, hyperparameters, split seed, and holdout fraction fixed so every
arm shares the same held-out documents:

* ``--full-vocab`` keeps the whole tokenizer vocabulary (an identity remap)
  instead of the min_count=2 remap.
* ``--init-embed {e5,gemma}`` warm-starts the embedding table from a donor
  model's trained word embeddings, PCA-projected to ``embed_dim`` and rescaled
  to the cold-start init std. Requires ``--full-vocab`` so donor row *i* is the
  embedding of raw token *i*.
* ``--tokenizer`` swaps the tokenizer (e.g. the Gemma-3 tokenizer).
* ``--bigram-buckets`` adds the hashed-bigram side table
  (:class:`~experiments.datakit.cluster.quality.fast_transformer.model.FastTransformer`).

The saved artifact is scored by the unchanged ``scorer.py`` path: the identity
remap and the bigram hash parameters both travel with the model.
"""

import argparse
import logging

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from rigging.filesystem import StoragePath
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging
from safetensors import safe_open

from experiments.datakit.cluster.quality.fast_transformer.data import (
    NUM_RESERVED,
    PackedData,
    build_remap,
    encode_texts,
    load_tokenizer,
    pack,
)
from experiments.datakit.cluster.quality.fast_transformer.inference import predict
from experiments.datakit.cluster.quality.fast_transformer.model import FastTransformer, FastTransformerConfig
from experiments.datakit.cluster.quality.fast_transformer.train import (
    DEPLOY_CONFIG,
    MAX_TOKENS,
    TOKENIZER,
    TrainHParams,
    _metrics,
    _save_scorer,
    fit,
)

logger = logging.getLogger(__name__)

# Must stay identical to train.py so every arm (and gate_model.py) shares the
# same held-out documents.
TRAIN_SEED = 0
EVAL_FRAC = 1 / 7

# Donor repos for --init-embed. Gemma tries the gated Google repo first (HF_TOKEN)
# and falls back to the unsloth mirror, whose tokenizer is byte-identical.
DONOR_REPOS = {
    "e5": ["intfloat/multilingual-e5-small"],
    "gemma": ["google/gemma-3-270m", "unsloth/gemma-3-270m-it"],
}
DONOR_TENSOR_KEYS = {
    "e5": "embeddings.word_embeddings.weight",
    "gemma": "model.embed_tokens.weight",
}
EMBED_INIT_STD = 0.02  # FastTransformer's cold-start embedding init scale


def full_vocab_remap(tokenizer_name: str) -> dict[int, int]:
    """Identity remap over the entire tokenizer vocabulary (raw id -> raw id + reserved)."""
    tok = load_tokenizer(tokenizer_name)
    size = max(tok.vocab_size, len(tok))
    return {t: t + NUM_RESERVED for t in range(size)}


def donor_embedding_table(donor: str) -> np.ndarray:
    """The donor model's word-embedding table as float32 [rows, donor_dim]."""
    key = DONOR_TENSOR_KEYS[donor]
    last_error: Exception | None = None
    for repo in DONOR_REPOS[donor]:
        try:
            path = hf_hub_download(repo, "model.safetensors")
        except Exception as e:  # gated repo without a token; try the mirror
            logger.warning("donor %s unavailable (%s); trying the next repo", repo, e)
            last_error = e
            continue
        with safe_open(path, framework="flax") as f:
            if key not in f.keys():
                raise ValueError(f"{repo} has no tensor {key!r}; found e.g. {sorted(f.keys())[:10]}")
            table = np.asarray(f.get_tensor(key).astype(jnp.float32))
        logger.info("donor embeddings: %s %s %s", repo, key, table.shape)
        return table
    raise RuntimeError(f"no donor repo reachable for {donor}") from last_error


def pca_project(table: np.ndarray, dim: int, target_std: float = EMBED_INIT_STD) -> np.ndarray:
    """PCA-project donor rows to ``dim`` and rescale to the cold-start init std.

    The rescale matters: donor tables have per-dim scales far from the 0.02 the
    rest of the network was initialized against, and an embedding 10x too large
    saturates the pooled statistics before training can adapt.
    """
    x = table.astype(np.float64)
    x -= x.mean(axis=0)
    cov = x.T @ x / len(x)
    _, eigvecs = np.linalg.eigh(cov)
    proj = x @ eigvecs[:, ::-1][:, :dim]
    return (proj * (target_std / proj.std())).astype(np.float32)


def warm_start(model: FastTransformer, donor_rows: np.ndarray, vocab: int) -> FastTransformer:
    """Overwrite embedding rows for raw tokens with donor rows (PAD/UNK stay random)."""
    table = np.asarray(model.embed).copy()
    n = min(len(donor_rows), vocab - NUM_RESERVED)
    table[NUM_RESERVED : NUM_RESERVED + n] = donor_rows[:n]
    logger.info("warm start: filled %d of %d embedding rows from the donor", n, vocab)
    return eqx.tree_at(lambda m: m.embed, model, jnp.asarray(table))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", required=True, help="oracle-label parquet (text/score_normalized)")
    p.add_argument("--out-dir", required=True, help="dir to write <name>.eqx + _remap.json + _meta.json")
    p.add_argument("--name", required=True, help="artifact stem (one model per out-dir)")
    p.add_argument("--tokenizer", default=TOKENIZER)
    p.add_argument("--full-vocab", action="store_true", help="identity remap over the full tokenizer vocab")
    p.add_argument("--init-embed", choices=["none", "e5", "gemma"], default="none")
    p.add_argument("--bigram-buckets", type=int, default=0, help="hashed-bigram side table buckets (0 = off)")
    p.add_argument("--bigram-seed", type=int, default=0)
    args = p.parse_args()
    configure_logging(logging.INFO)
    configure_coreweave_s3()
    if args.init_embed != "none" and not args.full_vocab:
        raise ValueError("--init-embed requires --full-vocab (donor row i must be raw token i)")

    hp = TrainHParams(seed=TRAIN_SEED)
    with StoragePath(args.labels).open("rb") as fh:
        table = pq.read_table(fh, columns=["text", "score_normalized"])
    texts = [t or "" for t in table.column("text").to_pylist()]
    scores = np.array(table.column("score_normalized").to_pylist(), dtype=np.float32)

    perm = np.random.default_rng(hp.seed).permutation(len(texts))
    n_eval = max(1, int(len(texts) * EVAL_FRAC))
    eval_idx, train_idx = perm[:n_eval], perm[n_eval:]
    tr_texts, tr_scores = [texts[i] for i in train_idx], scores[train_idx]
    ev_texts, ev_scores = [texts[i] for i in eval_idx], scores[eval_idx]
    tr_raw = encode_texts(args.tokenizer, tr_texts, MAX_TOKENS)
    ev_raw = encode_texts(args.tokenizer, ev_texts, MAX_TOKENS)

    remap = full_vocab_remap(args.tokenizer) if args.full_vocab else build_remap(tr_raw, min_count=2)
    vocab = len(remap) + NUM_RESERVED
    data = PackedData(
        train=pack(tr_raw, remap, tr_scores, MAX_TOKENS),
        eval=pack(ev_raw, remap, ev_scores, MAX_TOKENS),
        vocab_size=vocab,
        tokenizer_name=args.tokenizer,
        max_tokens=MAX_TOKENS,
    )
    config = FastTransformerConfig(
        vocab_size=vocab,
        max_tokens=MAX_TOKENS,
        dropout=0.1,
        final_pool="mean",
        bigram_buckets=args.bigram_buckets,
        bigram_seed=args.bigram_seed,
        **DEPLOY_CONFIG,
    )

    init_model = None
    if args.init_embed != "none":
        donor = donor_embedding_table(args.init_embed)
        projected = pca_project(donor, config.embed_dim)
        init_model = warm_start(FastTransformer(config, key=jr.PRNGKey(hp.seed)), projected, vocab)

    logger.info(
        "arm %s: %d labels (%d train / %d eval) tokenizer=%s vocab=%d init=%s bigram_buckets=%d",
        args.name,
        len(texts),
        len(train_idx),
        len(eval_idx),
        args.tokenizer,
        vocab,
        args.init_embed,
        args.bigram_buckets,
    )
    fitted = fit(config, data, hp, init_model=init_model)
    holdout = _metrics(predict(fitted.model, data.eval.ids), data.eval.scores)
    logger.info(
        "HOLDOUT AUC=%.4f spearman=%.4f (best_epoch=%d, params=%.1fM)",
        holdout.auc,
        holdout.spearman_rho,
        fitted.best_epoch,
        fitted.params / 1e6,
    )
    _save_scorer(fitted.model, remap, args.tokenizer, config, args.out_dir, args.name)


if __name__ == "__main__":
    main()
