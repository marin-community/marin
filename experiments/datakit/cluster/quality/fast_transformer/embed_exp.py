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

The non-probe arms compose with the embedding-table and trunk levers that
:mod:`train_exp` studied separately: ``--tokenizer`` swaps the tokenizer,
``--init-embed`` warm-starts the embedding table from a donor model with
``--embed-treatment`` selecting the donor treatment (fixed PCA; clean+whiten
per the clean_gemma_embeddings recipe; PCA from components 2..embed_dim+1; or
a frozen donor table behind a learned projection initialized at the PCA
solution), ``--gigatoken`` tokenizes with the gigatoken Rust backend behind an exact
token-id parity gate against the HF tokenizer, and
``--hidden-dim/--num-layers/--num-heads/--pool-window/--mlp-ratio`` size the
trunk (defaults reproduce the deployed config). ``--domain-mlp`` types every
document from its stored embedding (:mod:`domain_mlp`) and adds per-type
tables keyed by that prediction; ``--baseline-model-dir`` scores a deployed
text-only model on the identical holdout rows so the arm's tables read as
deltas against it.

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
import time

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

from experiments.datakit.cluster.quality.fast_transformer import domain_mlp
from experiments.datakit.cluster.quality.fast_transformer.data import (
    NUM_RESERVED,
    encode_texts,
    encode_texts_fast,
    pack,
)
from experiments.datakit.cluster.quality.fast_transformer.gate_model import source_signal
from experiments.datakit.cluster.quality.fast_transformer.inference import predict
from experiments.datakit.cluster.quality.fast_transformer.joined_labels import (
    DEFAULT_JOINED,
    EMBED_DIM,
    embedding_matrix,
    load_joined,
)
from experiments.datakit.cluster.quality.fast_transformer.model import FastTransformer, FastTransformerConfig
from experiments.datakit.cluster.quality.fast_transformer.scorer import load_pooled_scorer, score_bme
from experiments.datakit.cluster.quality.fast_transformer.train import (
    DEPLOY_CONFIG,
    MAX_TOKENS,
    TOKENIZER,
    TrainHParams,
    _save_scorer,
    train_regressor,
)
from experiments.datakit.cluster.quality.fast_transformer.train_exp import (
    DONOR_TOKENIZERS,
    DONORS,
    EMBED_INIT_STD,
    EVAL_FRAC,
    TRAIN_SEED,
    clean_whiten,
    donor_embedding_table,
    full_vocab_remap,
    pca_basis,
    pca_project,
    warm_start,
)

logger = logging.getLogger(__name__)

DEFAULT_LABELS = "s3://marin-us-east-02a/marin/user/rav/quality_v2/glm52_labels_88k.parquet"
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
# Donor-embedding treatments for --init-embed warm starts. "pca" is the
# original fixed PCA down-projection; the others vary only that step.
TREATMENTS = ("pca", "clean-whiten", "drop-top-pc", "learned-proj")
PARITY_DOCS_PER_SOURCE = 16


def holdout_id_set(labels_path: str) -> set[str]:
    """Ids of the rows every arm of the campaign holds out (seed 0, 1/7 of rows)."""
    with StoragePath(labels_path).open("rb") as fh:
        ids = pq.read_table(fh, columns=["id"]).column("id").to_pylist()
    perm = np.random.default_rng(TRAIN_SEED).permutation(len(ids))
    return {ids[i] for i in perm[: max(1, int(len(ids) * EVAL_FRAC))]}


def check_gigatoken_parity(tokenizer_name: str, texts: list[str], sources: np.ndarray) -> None:
    """Fail loudly unless gigatoken reproduces the HF tokenizer's ids exactly.

    The sample spans every source in the corpus, so tokenizer-hostile content
    (CJK, LaTeX, tool logs) is represented rather than just prose.
    """
    rng = np.random.default_rng(TRAIN_SEED)
    sample: list[int] = []
    for name in sorted(set(sources.tolist())):
        idx = np.flatnonzero(sources == name)
        sample.extend(rng.choice(idx, size=min(PARITY_DOCS_PER_SOURCE, len(idx)), replace=False).tolist())
    docs = [texts[i] for i in sample]
    # Warm both tokenizer caches so the timing below measures encoding, not loading.
    encode_texts(tokenizer_name, ["warmup"], MAX_TOKENS)
    encode_texts_fast(tokenizer_name, ["warmup"], MAX_TOKENS)
    t0 = time.time()
    hf_ids = encode_texts(tokenizer_name, docs, MAX_TOKENS)
    t1 = time.time()
    fast_ids = encode_texts_fast(tokenizer_name, docs, MAX_TOKENS)
    t2 = time.time()
    mismatched = [i for i, (a, b) in enumerate(zip(hf_ids, fast_ids, strict=True)) if list(a) != list(b)]
    if mismatched:
        first = mismatched[0]
        raise ValueError(
            f"gigatoken diverges from the HF tokenizer on {len(mismatched)}/{len(docs)} sampled documents; "
            f"first mismatch: source={sources[sample[first]]!r} "
            f"hf={hf_ids[first][:8]}... gigatoken={fast_ids[first][:8]}..."
        )
    logger.info(
        "gigatoken parity: %d/%d documents identical across %d sources (hf %.2fs, gigatoken %.2fs, %.1fx)",
        len(docs),
        len(docs),
        len(set(sources.tolist())),
        t1 - t0,
        t2 - t1,
        (t1 - t0) / max(t2 - t1, 1e-9),
    )


def grouped_spearman(preds: np.ndarray, quality: np.ndarray, groups: np.ndarray, min_n: int) -> dict[str, float]:
    out = {}
    for name in sorted(set(groups.tolist())):
        mask = groups == name
        if int(mask.sum()) < min_n:
            continue
        rho = float(stats.spearmanr(preds[mask], quality[mask]).statistic)
        # A constant input (a scorer flat on a group, or a single-level group)
        # yields nan; drop it rather than poisoning the summary stats.
        if np.isfinite(rho):
            out[name] = rho
    return out


def report_arm(
    name: str,
    preds: np.ndarray,
    quality: np.ndarray,
    types: np.ndarray,
    sources: np.ndarray,
    domains: np.ndarray | None = None,
) -> None:
    """The holdout table one arm contributes; grep for ``EMBED_EXP`` to harvest.

    ``type`` rows are keyed by the oracle's own content type; ``mlp_domain``
    rows (when a domain typer is given) by the embedding-MLP prediction, which
    is what inference-side calibration would have.
    """
    overall = float(stats.spearmanr(preds, quality).statistic)
    by_type = grouped_spearman(preds, quality, types, MIN_TYPE_LABELS)
    by_source = grouped_spearman(preds, quality, sources, MIN_SOURCE_LABELS)
    stds = [float(preds[sources == s].std()) for s in sorted(set(sources.tolist()))]
    flat = sum(1 for s in stds if s < FLAT_STD)
    logger.info("EMBED_EXP %s overall_rho=%+.4f", name, overall)
    for type_name in sorted(by_type):
        logger.info("EMBED_EXP %s type %-14s %+.3f", name, type_name, by_type[type_name])
    if domains is not None:
        by_domain = grouped_spearman(preds, quality, domains, MIN_TYPE_LABELS)
        for domain_name in sorted(by_domain):
            logger.info("EMBED_EXP %s mlp_domain %-14s %+.3f", name, domain_name, by_domain[domain_name])
    if by_source:
        logger.info(
            "EMBED_EXP %s source_rho mean=%+.3f median=%+.3f min=%+.3f (%d sources)",
            name,
            float(np.mean(list(by_source.values()))),
            float(np.median(list(by_source.values()))),
            float(np.min(list(by_source.values()))),
            len(by_source),
        )
    else:
        # A subsampled smoke holdout can leave every source under the floor.
        logger.info("EMBED_EXP %s source_rho: no source met the %d-label floor", name, MIN_SOURCE_LABELS)
    logger.info("EMBED_EXP %s source_signal=%+.3f", name, source_signal(preds, quality, sources))
    logger.info(
        "EMBED_EXP %s per-source pred std mean=%.4f min=%.4f flat(<%.2f)=%d/%d",
        name,
        float(np.mean(stds)),
        float(np.min(stds)),
        FLAT_STD,
        flat,
        len(stds),
    )


def apply_embed_treatment(
    model: FastTransformer,
    config: FastTransformerConfig,
    treatment: str,
    donor: np.ndarray,
    remap: dict[int, int],
) -> tuple[FastTransformer, object | None]:
    """Warm-start the token embedding per the requested donor treatment.

    Returns ``(model, params_filter)``; the filter is non-None only for
    ``learned-proj``, whose frozen donor table must be masked out of the
    optimizer (no gradients and, crucially, no weight decay).
    """
    if treatment == "pca":
        return warm_start(model, pca_project(donor, config.embed_dim), remap), None
    if treatment == "drop-top-pc":
        # PCA to embed_dim from components 2..embed_dim+1: PC1 is projected out
        # and the basis is the next embed_dim components.
        return warm_start(model, pca_project(donor, config.embed_dim, skip_top=1), remap), None
    if treatment == "clean-whiten":
        cleaned, kept = clean_whiten(donor)
        projected = np.zeros((len(donor), config.embed_dim), dtype=np.float32)
        projected[kept] = pca_project(cleaned, config.embed_dim)
        kept_mask = np.zeros(len(donor), dtype=bool)
        kept_mask[kept] = True
        # Rows the cleaning dropped keep their cold-start init, like any token
        # the donor lacks.
        clean_remap = {r: d for r, d in remap.items() if r < len(donor) and kept_mask[r]}
        return warm_start(model, projected, clean_remap), None
    # learned-proj: frozen centered donor rows behind a learnable projection
    # initialized at the PCA solution, so training starts at the pca arm's warm
    # start and differs only in what is trainable (the 640x256 projection
    # instead of the full 262K x 256 table).
    centered = (donor - donor.mean(axis=0)).astype(np.float32)
    table = np.zeros((config.vocab_size, donor.shape[1]), dtype=np.float32)
    raw = np.fromiter(remap.keys(), dtype=np.int64, count=len(remap))
    dense = np.fromiter(remap.values(), dtype=np.int64, count=len(remap))
    keep = raw < len(donor)
    table[dense[keep]] = centered[raw[keep]]
    basis = pca_basis(donor, config.embed_dim)
    scale = EMBED_INIT_STD / (centered.astype(np.float64) @ basis).std()
    model = eqx.tree_at(
        lambda m: (m.donor_embed, m.donor_proj),
        model,
        (jnp.asarray(table), jnp.asarray((basis * scale).astype(np.float32))),
    )
    params_filter = jax.tree_util.tree_map(eqx.is_inexact_array, model)
    params_filter = eqx.tree_at(lambda m: m.donor_embed, params_filter, replace=False)
    logger.info(
        "learned-proj: froze %d donor rows behind a trainable %dx%d projection",
        int(keep.sum()),
        donor.shape[1],
        config.embed_dim,
    )
    return model, params_filter


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
    p.add_argument("--tokenizer", default=TOKENIZER)
    p.add_argument("--init-embed", choices=["none", *DONORS], default="none")
    p.add_argument("--embed-treatment", choices=TREATMENTS, default="pca", help="donor treatment for --init-embed")
    p.add_argument("--gigatoken", action="store_true", help="tokenize with gigatoken (exact-parity gated)")
    p.add_argument("--domain-mlp", default=None, help="domain_mlp npz; adds per-type tables keyed by its predictions")
    p.add_argument("--baseline-model-dir", default=None, help="text-only pooled scorer reported on the same holdout")
    p.add_argument("--hidden-dim", type=int, default=DEPLOY_CONFIG["hidden_dim"])
    p.add_argument("--num-layers", type=int, default=DEPLOY_CONFIG["num_layers"])
    p.add_argument("--num-heads", type=int, default=DEPLOY_CONFIG["num_heads"])
    p.add_argument("--pool-window", type=int, default=DEPLOY_CONFIG["pool_window"])
    p.add_argument("--mlp-ratio", type=int, default=4, help="transformer MLP expansion ratio")
    p.add_argument("--epochs", type=int, default=None, help="override the epoch cap (smoke runs)")
    p.add_argument("--limit", type=int, default=0, help="use only the first N joined rows (0 = all; smoke runs)")
    args = p.parse_args()
    configure_logging(logging.INFO)
    configure_coreweave_s3()

    holdout_ids = holdout_id_set(args.labels)
    joined = load_joined(args.joined_dir)
    if args.limit:
        joined = {c: v[: args.limit] for c, v in joined.items()}
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

    domains = None
    if args.domain_mlp:
        typer, typer_labels = domain_mlp.load(args.domain_mlp)
        domains = domain_mlp.predict(typer, typer_labels, joined["embedding"])
        logger.info("domain_mlp typing: agreement with oracle content types %.3f", float((domains == types).mean()))

    tr, ev = ~is_eval, is_eval
    ev_domains = domains[ev] if domains is not None else None
    if args.arm == "probe":
        w = fit_ridge(emb[tr], target[tr])
        ridge_preds = np.concatenate([emb[ev], np.ones((int(ev.sum()), 1), dtype=np.float32)], axis=1) @ w
        report_arm("probe_ridge", ridge_preds, quality[ev], types[ev], sources[ev], ev_domains)
        mlp_preds = _mlp_probe_preds(emb, target, tr, ev)
        report_arm("probe_mlp", mlp_preds, quality[ev], types[ev], sources[ev], ev_domains)
        return

    if args.arm in ("head", "token") and (not args.out_dir or not args.name):
        raise ValueError("fusion arms need --out-dir and --name")

    texts = [t or "" for t in joined["text"]]
    hp = TrainHParams(seed=TRAIN_SEED) if args.epochs is None else TrainHParams(seed=TRAIN_SEED, epochs=args.epochs)
    tr_idx, ev_idx = np.flatnonzero(tr), np.flatnonzero(ev)

    if args.baseline_model_dir:
        # Scored before training so a broken baseline path fails in minutes, and
        # bme-windowed exactly as the deployed scoring path (gate_model) runs it.
        baseline_preds = score_bme(load_pooled_scorer(args.baseline_model_dir), [texts[i] for i in ev_idx])
        report_arm("baseline", baseline_preds, quality[ev], types[ev], sources[ev], ev_domains)

    encode = encode_texts
    if args.gigatoken:
        check_gigatoken_parity(args.tokenizer, texts, sources)
        encode = encode_texts_fast
    remap = full_vocab_remap(args.tokenizer)
    vocab = len(remap) + NUM_RESERVED
    tr_raw = encode(args.tokenizer, [texts[i] for i in tr_idx], MAX_TOKENS)
    ev_raw = encode(args.tokenizer, [texts[i] for i in ev_idx], MAX_TOKENS)
    tr_pack = pack(tr_raw, remap, target[tr_idx], MAX_TOKENS)
    ev_pack = pack(ev_raw, remap, target[ev_idx], MAX_TOKENS)

    if args.init_embed != "none" and args.tokenizer != DONOR_TOKENIZERS[args.init_embed]:
        # Donor row i is the embedding of *its own* tokenizer's token i, and the
        # warm start routes rows through the vocab remap by raw id. Under any
        # other tokenizer that mapping is silently wrong rather than merely
        # suboptimal, so refuse instead of training on scrambled embeddings.
        raise ValueError(
            f"--init-embed {args.init_embed} must be paired with --tokenizer "
            f"{DONOR_TOKENIZERS[args.init_embed]}, got {args.tokenizer}"
        )
    donor = donor_embedding_table(args.init_embed) if args.init_embed != "none" else None
    if args.embed_treatment != "pca" and donor is None:
        raise ValueError(f"--embed-treatment {args.embed_treatment} requires --init-embed")
    config = FastTransformerConfig(
        vocab_size=vocab,
        max_tokens=MAX_TOKENS,
        dropout=0.1,
        final_pool="mean",
        embed_dim=DEPLOY_CONFIG["embed_dim"],
        pool_kind=DEPLOY_CONFIG["pool_kind"],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        pool_window=args.pool_window,
        mlp_ratio=args.mlp_ratio,
        doc_embed_dim=0 if args.arm == "control" else EMBED_DIM,
        doc_embed_super_token=args.arm == "token",
        frozen_donor_dim=donor.shape[1] if args.embed_treatment == "learned-proj" else 0,
    )
    model = FastTransformer(config, key=jr.PRNGKey(hp.seed))
    params_filter = None
    if donor is not None:
        model, params_filter = apply_embed_treatment(model, config, args.embed_treatment, donor, remap)
    logger.info(
        "arm %s: tokenizer=%s vocab=%d init=%s treatment=%s gigatoken=%s trunk d=%d L=%d h=%d w=%d mlp=%d "
        "(%.0fK flops/token)",
        args.arm,
        args.tokenizer,
        vocab,
        args.init_embed,
        args.embed_treatment,
        args.gigatoken,
        config.hidden_dim,
        config.num_layers,
        config.num_heads,
        config.pool_window,
        config.mlp_ratio,
        config.flops_per_token() / 1e3,
    )

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
        params_filter=params_filter,
    )
    preds = predict(best, ev_pack.ids, doc_embed=emb[ev_idx] if use_emb else None)
    logger.info("arm %s: best_epoch=%d train_seconds=%.0f", args.arm, best_epoch, seconds)
    report_arm(args.arm, preds, quality[ev], types[ev], sources[ev], ev_domains)
    if args.out_dir and args.name:
        _save_scorer(best, remap, args.tokenizer, config, args.out_dir, args.name)


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

    best_rho, best_model, stale, stopped_at = -2.0, model, 0, 0
    for epoch in range(40):
        stopped_at = epoch
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
    logger.info("mlp probe: best val_rho=%+.4f (epoch cap 40, stopped at %d)", best_rho, stopped_at)
    return np.asarray(jax.nn.sigmoid(best_model(jnp.asarray(emb[ev]))))


if __name__ == "__main__":
    main()
