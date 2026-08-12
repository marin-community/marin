# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retrain the winning fusion arm on the scaled window-level label set.

Two arms, sharing the 88k campaign's winning configuration (bigger trunk,
Gemma full vocab via gigatoken behind the exact-parity gate, frozen centered
Gemma donor table behind a learned projection initialized at the PCA solution,
doc-embedding super-token fusion):

* ``--arm learnedproj``: that configuration retrained unchanged on the window
  dataset (:mod:`window_dataset`, ~368k window examples over ~212k docs).
* ``--arm moe``: the same base plus the per-sequence-routed FFN mixture — each
  layer keeps its shared FFN and adds four routed experts, soft top-2, router
  fed by the 1024-d document embedding, all behind per-layer zero-init gates so
  training starts exactly as the learnedproj arm.

Evaluation is the prior campaign's, unchanged: the seed-0 id-set holdout of
the 88k labels, scored on the full joined document text and reported through
:func:`embed_exp.report_arm` (grouped Spearman under both oracle types and
domain-MLP-predicted types, per-source stats, source signal, flat-source
count), so numbers read directly against the v3 baseline (0.634) and the 88k
learned-proj arm (0.717). Both baselines are re-scored on the identical
holdout rows in the same run — v3 through the deployed bme scoring path, the
88k fused artifact through the arm's own predict path — so every table in the
comparison comes from one process on one row set. The internal early-stopping
validation split is grouped by doc id so sibling windows of one document never
straddle it.
"""

import argparse
import logging

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging

from experiments.datakit.cluster.quality.fast_transformer import bme_windows, domain_mlp
from experiments.datakit.cluster.quality.fast_transformer.data import NUM_RESERVED, encode_texts_fast, pack
from experiments.datakit.cluster.quality.fast_transformer.embed_exp import (
    DEFAULT_LABELS,
    apply_embed_treatment,
    check_gigatoken_parity,
    holdout_id_set,
    report_arm,
)
from experiments.datakit.cluster.quality.fast_transformer.inference import predict
from experiments.datakit.cluster.quality.fast_transformer.joined_labels import (
    DEFAULT_JOINED,
    EMBED_DIM,
    embedding_matrix,
    load_joined,
)
from experiments.datakit.cluster.quality.fast_transformer.model import (
    FastTransformer,
    FastTransformerConfig,
    expert_utilization,
)
from experiments.datakit.cluster.quality.fast_transformer.scorer import load_pooled_scorer, score_bme
from experiments.datakit.cluster.quality.fast_transformer.train import (
    MAX_TOKENS,
    TrainHParams,
    _save_scorer,
    train_regressor,
)
from experiments.datakit.cluster.quality.fast_transformer.train_exp import (
    TRAIN_SEED,
    donor_embedding_table,
    full_vocab_remap,
)
from experiments.datakit.cluster.quality.fast_transformer.window_dataset import (
    SCALEUP_JOINED,
    WINDOW_LABELS,
    assemble_training_windows,
    begin_window_texts,
    drop_cut_artifact_grades,
    load_window_labels,
    subsample_mask,
)

logger = logging.getLogger(__name__)

ARMS = ("learnedproj", "moe")
DEFAULT_OUT = "s3://marin-us-east-02a/marin/user/muchanem/quality_exp/scaled_retrain"
DEFAULT_DOMAIN_MLP = "s3://marin-us-east-02a/marin/user/muchanem/quality_exp/domain_mlp/domain_mlp.npz"
# The 88k campaign's winning trunk (bigger_fused learnedproj).
WINNER_TRUNK = {"hidden_dim": 384, "num_layers": 4, "num_heads": 8, "pool_window": 16, "mlp_ratio": 4}
MOE_EXPERTS = 4
MOE_TOP_K = 2
MOE_EXPERT_RATIO = 1
# The two prior scorers every run re-reports on the identical holdout rows.
V3_BASELINE_DIR = "s3://marin-us-east-02a/marin/user/rav/quality_v2/models/pooled_glm52_v3"
FUSED_88K_BASELINE_DIR = "s3://marin-us-east-02a/marin/user/muchanem/quality_exp/bigger_fused/treatments/learnedproj"


def grouped_val_split(ids: list[str], val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(fit_idx, val_idx) with no doc id straddling the split.

    Window-level rows share a document embedding across siblings, so a
    row-level split would let the model meet a val document's embedding during
    fitting and stop late on the inflated score.
    """
    unique = sorted(set(ids))
    perm = np.random.default_rng(seed).permutation(len(unique))
    val_ids = {unique[i] for i in perm[: max(1, int(len(unique) * val_frac))]}
    is_val = np.array([i in val_ids for i in ids])
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)


def report_routing(model: FastTransformer, doc_embed: np.ndarray) -> None:
    """Expert-utilization table over a document batch; grep for ``MOE_ROUTING``."""
    util = np.asarray(expert_utilization(model, jnp.asarray(doc_embed)))
    for layer_index, layer in enumerate(model.layers):
        mix = np.asarray(layer.expert_mixture(jnp.asarray(doc_embed)))
        top1 = np.bincount(mix.argmax(axis=1), minlength=util.shape[1]) / len(mix)
        logger.info(
            "MOE_ROUTING layer %d gate=%+.4f mean_weight=%s top1_share=%s",
            layer_index,
            float(layer.moe_gate),
            np.array2string(util[layer_index], precision=3),
            np.array2string(top1, precision=3),
        )
    collapsed = bool((util.max(axis=1) > 0.99).all())
    logger.info("MOE_ROUTING min_expert_weight=%.4f collapsed=%s", float(util.min()), collapsed)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--out-dir", default=None, help="artifact dir (default <scaled_retrain>/<arm>)")
    p.add_argument("--name", default=None, help="artifact stem (default scaled_<arm>)")
    p.add_argument("--labels", default=DEFAULT_LABELS, help="88k label parquet defining the holdout id set")
    p.add_argument("--legacy-joined", default=DEFAULT_JOINED, help="88k labels-x-embeddings join root")
    p.add_argument("--scaleup-joined", default=SCALEUP_JOINED, help="scale-up labels-x-embeddings join root")
    p.add_argument("--window-labels", default=WINDOW_LABELS, help="scale-up window labels parquet")
    p.add_argument("--domain-mlp", default=DEFAULT_DOMAIN_MLP, help="existing typer npz (kept for comparability)")
    p.add_argument(
        "--baseline-model-dir", default=V3_BASELINE_DIR, help="text-only scorer re-reported (bme path); '' skips"
    )
    p.add_argument(
        "--fused-baseline-model-dir", default=FUSED_88K_BASELINE_DIR, help="fused scorer re-reported; '' skips"
    )
    p.add_argument("--epochs", type=int, default=None, help="override the epoch cap (smoke runs)")
    p.add_argument("--subsample", type=int, default=1, help="keep 1-in-N doc ids everywhere (smoke runs)")
    p.add_argument(
        "--train-windows",
        choices=("all", "begin", "legacy-begin"),
        default="all",
        help="ablations: 'begin' drops middle/end training rows; 'legacy-begin' additionally drops every "
        "scale-up window, leaving only the recut 88k begin grades",
    )
    p.add_argument(
        "--drop-cut-artifacts",
        action="store_true",
        help="drop invalid scale-up grades whose rationale blames the window cut (harness artifact, not content)",
    )
    p.add_argument(
        "--moe-experts",
        type=int,
        default=MOE_EXPERTS,
        help="routed experts per layer for --arm moe (top-2 stays fixed, so active flops do not grow with it)",
    )
    args = p.parse_args()
    configure_logging(logging.INFO)
    configure_coreweave_s3()
    out_dir = args.out_dir or f"{DEFAULT_OUT}/{args.arm}"
    name = args.name or f"scaled_{args.arm}"

    holdout_ids = holdout_id_set(args.labels)
    legacy = load_joined(args.legacy_joined)
    scaleup = load_joined(args.scaleup_joined, columns=["id", "embedding"])
    windows = load_window_labels(args.window_labels)
    if args.drop_cut_artifacts:
        windows = drop_cut_artifact_grades(windows)
    if args.train_windows != "all":
        keep = (
            [] if args.train_windows == "legacy-begin" else [i for i, w in enumerate(windows["window"]) if w == "begin"]
        )
        windows = {c: [windows[c][i] for i in keep] for c in windows}
        logger.info("train-windows ablation %s: %d scale-up rows kept", args.train_windows, len(windows["id"]))
    if args.subsample > 1:
        for table in (legacy, scaleup, windows):
            keep = subsample_mask(table["id"], args.subsample)
            for c in table:
                table[c] = [v for v, k in zip(table[c], keep, strict=True) if k]
        logger.info(
            "subsample 1/%d: %d legacy, %d scale-up docs, %d windows",
            args.subsample,
            len(legacy["id"]),
            len(scaleup["id"]),
            len(windows["id"]),
        )

    legacy_texts = [t or "" for t in legacy["text"]]
    bme_windows.check_gigatoken_parity(legacy_texts, seed=TRAIN_SEED)
    begin_texts = begin_window_texts(legacy_texts)
    train, stats = assemble_training_windows(windows, legacy, begin_texts, scaleup, holdout_ids)

    is_eval = np.array([doc_id in holdout_ids for doc_id in legacy["id"]])
    ev_idx = np.flatnonzero(is_eval)
    quality = np.array(legacy["glm52_quality"], dtype=float)[ev_idx]
    types = np.array(legacy["glm52_content_type"])[ev_idx]
    sources = np.array(legacy["glm52_source"])[ev_idx]
    ev_texts = [legacy_texts[i] for i in ev_idx]
    ev_emb_rows = [legacy["embedding"][i] for i in ev_idx]
    logger.info(
        "split: %d train windows / %d holdout docs (of %d holdout ids); train positions: %s",
        len(train.ids),
        len(ev_idx),
        len(holdout_ids),
        {p_: int((train.positions == p_).sum()) for p_ in ("begin", "middle", "end")},
    )
    assert not set(train.ids) & holdout_ids, "holdout doc ids leaked into the training windows"

    typer, typer_labels = domain_mlp.load(args.domain_mlp)
    domains = domain_mlp.predict(typer, typer_labels, ev_emb_rows)

    tokenizer = bme_windows.GEMMA_TOKENIZER
    check_gigatoken_parity(tokenizer, train.texts, train.sources)
    ev_targets = np.array([legacy["glm52_score_normalized"][i] for i in ev_idx], dtype=np.float32)
    ev_emb = embedding_matrix(ev_emb_rows)

    # Both prior scorers re-reported on the identical holdout rows, before
    # training so a broken baseline path fails in minutes: v3 through the
    # deployed bme scoring path (how its 0.634 was measured), the 88k fused
    # artifact through the fusion arms' own predict path (how its 0.717 was).
    if args.baseline_model_dir:
        baseline = load_pooled_scorer(args.baseline_model_dir)
        report_arm("baseline_v3", score_bme(baseline, ev_texts), quality, types, sources, domains)
    if args.fused_baseline_model_dir:
        fused = load_pooled_scorer(args.fused_baseline_model_dir)
        fused_raw = encode_texts_fast(fused.tokenizer_name, ev_texts, fused.max_tokens)
        fused_pack = pack(fused_raw, fused.remap, ev_targets, fused.max_tokens)
        fused_preds = predict(fused.model, fused_pack.ids, doc_embed=ev_emb)
        report_arm("baseline_88k_fused", fused_preds, quality, types, sources, domains)

    remap = full_vocab_remap(tokenizer)
    vocab = len(remap) + NUM_RESERVED
    hp = TrainHParams(seed=TRAIN_SEED) if args.epochs is None else TrainHParams(seed=TRAIN_SEED, epochs=args.epochs)
    tr_pack = pack(encode_texts_fast(tokenizer, train.texts, MAX_TOKENS), remap, train.targets, MAX_TOKENS)
    ev_pack = pack(encode_texts_fast(tokenizer, ev_texts, MAX_TOKENS), remap, ev_targets, MAX_TOKENS)
    tr_emb = embedding_matrix(train.embeddings)

    donor = donor_embedding_table("gemma")
    config = FastTransformerConfig(
        vocab_size=vocab,
        max_tokens=MAX_TOKENS,
        dropout=0.1,
        final_pool="mean",
        embed_dim=256,
        pool_kind="meanmaxmin",
        doc_embed_dim=EMBED_DIM,
        doc_embed_super_token=True,
        frozen_donor_dim=donor.shape[1],
        moe_experts=args.moe_experts if args.arm == "moe" else 0,
        moe_expert_ratio=MOE_EXPERT_RATIO,
        moe_top_k=MOE_TOP_K,
        **WINNER_TRUNK,
    )
    model = FastTransformer(config, key=jr.PRNGKey(hp.seed))
    model, params_filter = apply_embed_treatment(model, config, "learned-proj", donor, remap)
    logger.info(
        "arm %s: vocab=%d trunk d=%d L=%d h=%d w=%d mlp=%d experts=%d (%.0fK active flops/token)",
        args.arm,
        vocab,
        config.hidden_dim,
        config.num_layers,
        config.num_heads,
        config.pool_window,
        config.mlp_ratio,
        config.moe_experts,
        config.flops_per_token() / 1e3,
    )

    fit_i, val_i = grouped_val_split(train.ids, hp.val_frac, hp.seed)
    best, best_epoch, seconds = train_regressor(
        model,
        tr_pack.ids[fit_i],
        tr_pack.scores[fit_i],
        tr_pack.ids[val_i],
        tr_pack.scores[val_i],
        hp,
        tr_doc_embed=tr_emb[fit_i],
        val_doc_embed=tr_emb[val_i],
        params_filter=params_filter,
    )
    logger.info("arm %s: best_epoch=%d train_seconds=%.0f", args.arm, best_epoch, seconds)
    preds = predict(best, ev_pack.ids, doc_embed=ev_emb)
    report_arm(f"scaled_{args.arm}", preds, quality, types, sources, domains)
    if args.arm == "moe":
        report_routing(best, ev_emb)
    _save_scorer(best, remap, tokenizer, config, out_dir, name)
    logger.info(
        "assembly stats: legacy_begin=%d scaleup_windows=%d holdout_excluded=%d "
        "missing_embedding=%d begin_regrades=%d",
        stats.legacy_begin,
        stats.scaleup_windows,
        stats.holdout_excluded,
        stats.missing_embedding,
        stats.begin_regrades_skipped,
    )


if __name__ == "__main__":
    main()
