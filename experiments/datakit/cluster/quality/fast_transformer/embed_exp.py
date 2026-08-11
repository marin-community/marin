# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Document-embedding experiments for the pooled fast-transformer scorer.

Asks whether a strong pretrained document embedding (the 1024-d harrier vector
already computed for every stored document) carries quality signal the token
model misses, using the ``glm52_labels_88k`` labels joined against the 50M
embedded sample. Three arms plus a control, all sharing one split:

* ``--arm probe``: ridge and small-MLP regressions on the embedding alone. A
  diagnostic, not a candidate scorer — if the probe cannot rank within a type,
  the embedding does not carry that signal and the fusion arms have no ceiling
  to reach.
* ``--arm control``: the token model (noremap-e5-cold config: full vocab, e5
  tokenizer, cold start) retrained on exactly the joined subset, so the fusion
  arms are compared against a control that saw the same documents.
* ``--arm head``: the token model plus the embedding projected to hidden_dim
  and read by a zero-init head-side skip (the concat head, decomposed). Starts
  bit-identical to the control.
* ``--arm token``: ``head`` plus the projected embedding appended as an extra
  always-valid super-token behind a zero-init gate, so attention can condition
  on it. Not bit-identical at start: an attendable zero-value token still
  redistributes softmax mass (~0.6 max pre-sigmoid logit shift at init).

Split discipline: the holdout is the *id set* of the rows ``train.py`` holds
out of the original 87,948-row label parquet (seed 0, 1/7). The join against
the 50M sample keeps ~92% of labels; rows whose id is in that holdout set are
evaluated, all surviving others are trained on. No embedding-side information
crosses the split: embeddings were computed per document by a frozen encoder,
and the probe/model only ever sees train-row embeddings during fitting.

Reported per arm on the surviving holdout: overall Spearman vs the oracle,
within-content-type and within-source Spearman, and per-source prediction
standard deviation (share of sources below the stage report's FLAT_STD flags a
scorer that goes blind inside a source).

Artifacts of the fusion arms serialize ``doc_embed_dim`` in their config, so
the deployed scorer path fails loudly (missing ``doc_embed``) rather than
silently scoring without the embedding.
"""

import argparse
import logging

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pyarrow.parquet as pq
from rigging.filesystem import StoragePath
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging
from scipy import stats

from experiments.datakit.cluster.quality.fast_transformer.data import NUM_RESERVED, encode_texts, pack
from experiments.datakit.cluster.quality.fast_transformer.inference import predict
from experiments.datakit.cluster.quality.fast_transformer.model import FastTransformer, FastTransformerConfig
from experiments.datakit.cluster.quality.fast_transformer.train import (
    DEPLOY_CONFIG,
    MAX_TOKENS,
    TOKENIZER,
    TrainHParams,
    _save_scorer,
    train_regressor,
)
from experiments.datakit.cluster.quality.fast_transformer.train_exp import EVAL_FRAC, TRAIN_SEED, full_vocab_remap

logger = logging.getLogger(__name__)

DEFAULT_JOINED = (
    "s3://marin-us-east-02a/marin/user/muchanem/quality_v2/glm52_labels_88k-x-harrier-oss-v1-0.6b-50m-text-v1"
)
DEFAULT_LABELS = "s3://marin-us-east-02a/marin/user/rav/quality_v2/glm52_labels_88k.parquet"
EMBED_DIM = 1024
# Group-support floors for the per-group Spearman tables. Types are large;
# sources run ~150-300 holdout rows each, so the source floor is lower.
MIN_TYPE_LABELS = 300
MIN_SOURCE_LABELS = 80
# Below this within-source prediction std the scorer is not discriminating there
# (same threshold the stage report and compare_by_domain use).
FLAT_STD = 0.03
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0)
MLP_HIDDEN = 256
ARMS = ("probe", "control", "head", "token")

JOINED_COLUMNS = [
    "id",
    "text",
    "embedding",
    "glm52_source",
    "glm52_content_type",
    "glm52_quality",
    "glm52_score_normalized",
]


def holdout_id_set(labels_path: str) -> set[str]:
    """Ids of the rows every arm of the campaign holds out (seed 0, 1/7 of rows)."""
    with StoragePath(labels_path).open("rb") as fh:
        ids = pq.read_table(fh, columns=["id"]).column("id").to_pylist()
    perm = np.random.default_rng(TRAIN_SEED).permutation(len(ids))
    return {ids[i] for i in perm[: max(1, int(len(ids) * EVAL_FRAC))]}


def _walk_parquet(root: str, max_depth: int = 5) -> list[str]:
    """Every ``*.parquet`` under ``root``, via single-level globs only (a recursive
    glob HeadObjects the prefix, which the CW store answers with a 400). The join
    mirrors each source's own layout, so shard depth varies from 1 to 3."""
    shards: list[str] = []
    dirs = [root.rstrip("/")]
    for _ in range(max_depth):
        next_dirs: list[str] = []
        for d in dirs:
            for entry in sorted(str(m) for m in StoragePath(f"{d}/*").glob()):
                if entry.endswith(".parquet"):
                    shards.append(entry)
                else:
                    # Descending into a non-directory just globs to nothing, so no
                    # name heuristic (source dirs like `numinamath-1.5` carry dots).
                    next_dirs.append(entry)
        dirs = next_dirs
        if not dirs:
            break
    return shards


def load_joined(joined_dir: str) -> dict[str, list]:
    """All joined label rows, deduplicated by id."""
    root = joined_dir.rstrip("/")
    shards = _walk_parquet(f"{root}/outputs")
    if not shards:
        raise ValueError(f"no parquet shards under {root}/outputs/")
    out: dict[str, list] = {c: [] for c in JOINED_COLUMNS}
    seen: set[str] = set()
    dupes = 0
    for shard in shards:
        with StoragePath(shard).open("rb") as fh:
            table = pq.ParquetFile(fh).read(columns=JOINED_COLUMNS)
        rows = {c: table.column(c).to_pylist() for c in JOINED_COLUMNS}
        for i, doc_id in enumerate(rows["id"]):
            if doc_id in seen:
                dupes += 1
                continue
            seen.add(doc_id)
            for c in JOINED_COLUMNS:
                out[c].append(rows[c][i])
    logger.info("joined labels: %d rows from %d shards (%d duplicate ids dropped)", len(out["id"]), len(shards), dupes)
    return out


def embedding_matrix(raw: list) -> np.ndarray:
    """int8 rows -> float32, L2-normalized (recovers direction; drops the
    quantization scale, which carries no per-document information)."""
    x = np.asarray(raw, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, 1e-6)


def grouped_spearman(preds: np.ndarray, quality: np.ndarray, groups: np.ndarray, min_n: int) -> dict[str, float]:
    out = {}
    for name in sorted(set(groups.tolist())):
        mask = groups == name
        if int(mask.sum()) >= min_n:
            out[name] = float(stats.spearmanr(preds[mask], quality[mask]).statistic)
    return out


def report_arm(name: str, preds: np.ndarray, quality: np.ndarray, types: np.ndarray, sources: np.ndarray) -> None:
    """The holdout table one arm contributes; grep for ``EMBED_EXP`` to harvest."""
    overall = float(stats.spearmanr(preds, quality).statistic)
    by_type = grouped_spearman(preds, quality, types, MIN_TYPE_LABELS)
    by_source = grouped_spearman(preds, quality, sources, MIN_SOURCE_LABELS)
    stds = [float(preds[sources == s].std()) for s in sorted(set(sources.tolist()))]
    flat = sum(1 for s in stds if s < FLAT_STD)
    logger.info("EMBED_EXP %s overall_rho=%+.4f", name, overall)
    for type_name in sorted(by_type):
        logger.info("EMBED_EXP %s type %-14s %+.3f", name, type_name, by_type[type_name])
    logger.info(
        "EMBED_EXP %s source_rho mean=%+.3f median=%+.3f min=%+.3f (%d sources)",
        name,
        float(np.mean(list(by_source.values()))),
        float(np.median(list(by_source.values()))),
        float(np.min(list(by_source.values()))),
        len(by_source),
    )
    logger.info(
        "EMBED_EXP %s per-source pred std mean=%.4f min=%.4f flat(<%.2f)=%d/%d",
        name,
        float(np.mean(stds)),
        float(np.min(stds)),
        FLAT_STD,
        flat,
        len(stds),
    )


def fit_ridge(x: np.ndarray, y: np.ndarray, alphas=RIDGE_ALPHAS, val_frac: float = 0.1, seed: int = TRAIN_SEED):
    """Closed-form ridge with the alpha chosen on an internal validation split."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(x))
    n_val = max(1, int(len(x) * val_frac))
    va, tr = perm[:n_val], perm[n_val:]
    xtr = np.concatenate([x[tr], np.ones((len(tr), 1), dtype=np.float32)], axis=1)
    xva = np.concatenate([x[va], np.ones((len(va), 1), dtype=np.float32)], axis=1)
    gram = xtr.T @ xtr
    xty = xtr.T @ y[tr]
    fits = []
    for alpha in alphas:
        w = np.linalg.solve(gram + alpha * np.eye(gram.shape[0], dtype=np.float32), xty)
        rho = float(stats.spearmanr(xva @ w, y[va]).statistic)
        fits.append((rho, alpha, w))
    rho, alpha, w = max(fits, key=lambda f: f[0])
    logger.info("ridge: alpha=%s val_rho=%+.4f", alpha, rho)
    return w


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--joined-dir", default=DEFAULT_JOINED, help="labels-x-embeddings join output root")
    p.add_argument("--labels", default=DEFAULT_LABELS, help="original label parquet defining the holdout id set")
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--out-dir", default=None, help="artifact dir (fusion/control arms)")
    p.add_argument("--name", default=None, help="artifact stem (fusion/control arms)")
    args = p.parse_args()
    configure_logging(logging.INFO)
    configure_coreweave_s3()

    holdout_ids = holdout_id_set(args.labels)
    joined = load_joined(args.joined_dir)
    is_eval = np.array([doc_id in holdout_ids for doc_id in joined["id"]])
    quality = np.array(joined["glm52_quality"], dtype=float)
    target = np.array(joined["glm52_score_normalized"], dtype=np.float32)
    types = np.array(joined["glm52_content_type"])
    sources = np.array(joined["glm52_source"])
    emb = embedding_matrix(joined["embedding"])
    logger.info(
        "split: %d joined rows -> %d train / %d holdout (of %d original holdout ids)",
        len(is_eval),
        int((~is_eval).sum()),
        int(is_eval.sum()),
        len(holdout_ids),
    )

    tr, ev = ~is_eval, is_eval
    if args.arm == "probe":
        w = fit_ridge(emb[tr], target[tr])
        ridge_preds = np.concatenate([emb[ev], np.ones((int(ev.sum()), 1), dtype=np.float32)], axis=1) @ w
        report_arm("probe_ridge", ridge_preds, quality[ev], types[ev], sources[ev])
        mlp_preds = _mlp_probe_preds(emb, target, tr, ev)
        report_arm("probe_mlp", mlp_preds, quality[ev], types[ev], sources[ev])
        return

    if args.arm in ("head", "token") and (not args.out_dir or not args.name):
        raise ValueError("fusion arms need --out-dir and --name")

    texts = [t or "" for t in joined["text"]]
    hp = TrainHParams(seed=TRAIN_SEED)
    remap = full_vocab_remap(TOKENIZER)
    vocab = len(remap) + NUM_RESERVED
    tr_idx, ev_idx = np.flatnonzero(tr), np.flatnonzero(ev)
    tr_raw = encode_texts(TOKENIZER, [texts[i] for i in tr_idx], MAX_TOKENS)
    ev_raw = encode_texts(TOKENIZER, [texts[i] for i in ev_idx], MAX_TOKENS)
    tr_pack = pack(tr_raw, remap, target[tr_idx], MAX_TOKENS)
    ev_pack = pack(ev_raw, remap, target[ev_idx], MAX_TOKENS)

    config = FastTransformerConfig(
        vocab_size=vocab,
        max_tokens=MAX_TOKENS,
        dropout=0.1,
        final_pool="mean",
        doc_embed_dim=0 if args.arm == "control" else EMBED_DIM,
        doc_embed_super_token=args.arm == "token",
        **DEPLOY_CONFIG,
    )
    model = FastTransformer(config, key=jr.PRNGKey(hp.seed))

    # Internal train/val split for model selection, mirroring train.fit.
    rng = np.random.default_rng(hp.seed)
    perm = rng.permutation(tr_pack.n)
    n_val = max(1, int(tr_pack.n * hp.val_frac))
    val_i, fit_i = perm[:n_val], perm[n_val:]
    use_emb = args.arm != "control"
    tr_emb = emb[tr_idx] if use_emb else None
    best, best_epoch, seconds = train_regressor(
        model,
        tr_pack.ids[fit_i],
        tr_pack.scores[fit_i],
        tr_pack.ids[val_i],
        tr_pack.scores[val_i],
        hp,
        tr_doc_embed=None if tr_emb is None else tr_emb[fit_i],
        val_doc_embed=None if tr_emb is None else tr_emb[val_i],
    )
    preds = predict(best, ev_pack.ids, doc_embed=emb[ev_idx] if use_emb else None)
    logger.info("arm %s: best_epoch=%d train_seconds=%.0f", args.arm, best_epoch, seconds)
    report_arm(args.arm, preds, quality[ev], types[ev], sources[ev])
    if args.out_dir and args.name:
        _save_scorer(best, remap, TOKENIZER, config, args.out_dir, args.name)


def _mlp_probe_preds(emb: np.ndarray, target: np.ndarray, tr: np.ndarray, ev: np.ndarray) -> np.ndarray:
    """Out-of-holdout predictions of a small JAX MLP trained on the embedding alone."""
    hp = TrainHParams(seed=TRAIN_SEED)

    class Mlp(eqx.Module):
        w1: jax.Array
        b1: jax.Array
        w2: jax.Array
        b2: jax.Array

        def __init__(self, key):
            k1, k2 = jr.split(key)
            self.w1 = jr.normal(k1, (EMBED_DIM, MLP_HIDDEN)) * (2.0 / (EMBED_DIM + MLP_HIDDEN)) ** 0.5
            self.b1 = jnp.zeros(MLP_HIDDEN)
            self.w2 = jr.normal(k2, (MLP_HIDDEN, 1)) * (2.0 / (MLP_HIDDEN + 1)) ** 0.5
            self.b2 = jnp.zeros(1)

        def __call__(self, x):
            return (jax.nn.gelu(x @ self.w1 + self.b1) @ self.w2 + self.b2)[:, 0]

    x_tr, y_tr = emb[tr], target[tr]
    rng = np.random.default_rng(hp.seed)
    perm = rng.permutation(len(x_tr))
    n_val = max(1, int(len(x_tr) * 0.1))
    va, fi = perm[:n_val], perm[n_val:]
    model = Mlp(jr.PRNGKey(hp.seed))
    optimizer = optax.adamw(1e-3, weight_decay=0.01)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def step(model, opt_state, xb, yb):
        def loss_fn(m):
            return jnp.mean((jax.nn.sigmoid(m(xb)) - yb) ** 2)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_inexact_array))
        return eqx.apply_updates(model, updates), opt_state, loss

    best_rho, best_model, stale = -2.0, model, 0
    for epoch in range(40):
        ep = rng.permutation(len(fi))
        for s in range(0, len(ep), 1024):
            batch = fi[ep[s : s + 1024]]
            model, opt_state, _ = step(model, opt_state, jnp.asarray(x_tr[batch]), jnp.asarray(y_tr[batch]))
        val_rho = float(stats.spearmanr(np.asarray(model(jnp.asarray(x_tr[va]))), y_tr[va]).statistic)
        if np.isfinite(val_rho) and val_rho > best_rho:
            best_rho, best_model, stale = val_rho, model, 0
        else:
            stale += 1
        if stale >= 2:
            break
    logger.info("mlp probe: best val_rho=%+.4f (epoch cap 40, stopped at %d)", best_rho, epoch)
    return np.asarray(jax.nn.sigmoid(best_model(jnp.asarray(emb[ev]))))


if __name__ == "__main__":
    main()
