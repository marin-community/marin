# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Retrain the fusion scorer on the clean bme2048 window labels.

Every arm shares one supervision source — the ``glm52_rubric_v2_bme2048``
grades, 2048-token begin/middle/end windows drawn with the seed-0 holdout ids
excluded — and one recipe: the Nemotron donor table frozen behind a learned
projection initialized at the PCA solution, the Nemotron tokenizer via
gigatoken behind the exact-parity gate, the 88k campaign's winning trunk, and
the 1024-d harrier document embedding fused as a zero-init-gated super token.
Nemotron rather than Gemma because it reached parity on the 88k set (0.7131
against 0.7170) with half the deployed embedding table.

What varies is only the supervision the trunk is pointed at:

* ``--train-windows begin`` versus ``all`` — whether middle and end grades are
  training rows at all.
* ``--target own`` versus ``begin`` — whether a middle/end window is trained
  against its own grade or against its document's begin grade. The second
  isolates whether the middle/end grades carry signal or noise: the model sees
  the same inputs either way.
* ``--max-tokens`` — 512 keeps the deployed context, so the model reads the
  first 512 tokens of a window graded over 2048; 2048 lets it read the whole
  graded window. ``pool_window`` is held fixed so the token-to-super-token
  mapping is identical and only the sequence gets longer.
* ``--filter`` — which suspect grades are dropped (:mod:`window_dataset`).

Evaluation is the prior campaigns', unchanged: the seed-0 id-set holdout of the
88k labels, reported through :func:`embed_exp.report_arm` (grouped Spearman
under both oracle types and domain-MLP-predicted types, per-source stats,
source signal, flat-source count). Every arm is scored three ways on those same
rows — the document's begin window, the mean over its bme windows under the
label set's own geometry, and the deployed character-window bme — so a
difference between arms can be attributed to the model rather than to the
scoring path. The v3 scorer is re-anchored in-run through its own deployed path
so the whole table reads against its 0.634.
"""

import argparse
import logging

import jax.random as jr
import numpy as np
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging

from experiments.datakit.cluster.quality.fast_transformer import bme_windows, domain_mlp
from experiments.datakit.cluster.quality.fast_transformer.bme_windows import (
    GEOMETRY_2048,
    doc_windows,
    encode_documents,
)
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
from experiments.datakit.cluster.quality.fast_transformer.model import FastTransformer, FastTransformerConfig
from experiments.datakit.cluster.quality.fast_transformer.scaled_exp import (
    DEFAULT_DOMAIN_MLP,
    FUSED_88K_BASELINE_DIR,
    V3_BASELINE_DIR,
    WINNER_TRUNK,
    grouped_val_split,
)
from experiments.datakit.cluster.quality.fast_transformer.scorer import bme_chunks, load_pooled_scorer, score_bme
from experiments.datakit.cluster.quality.fast_transformer.train import TrainHParams, _save_scorer, train_regressor
from experiments.datakit.cluster.quality.fast_transformer.train_exp import (
    DONOR_TOKENIZERS,
    TRAIN_SEED,
    donor_embedding_table,
    full_vocab_remap,
)
from experiments.datakit.cluster.quality.fast_transformer.window_dataset import (
    BME2048_COLUMNS,
    BME2048_WINDOW_LABELS,
    SCALEUP_JOINED,
    drop_cross_window_disagreements,
    drop_cut_window_invalids,
    load_window_labels,
    subsample_mask,
)

logger = logging.getLogger(__name__)

DEFAULT_OUT = "s3://marin-us-east-02a/marin/user/muchanem/quality_exp/bme2048_retrain"
DONOR = "nemotron"
BEGIN = "begin"
FILTERS = ("none", "cross-window", "cut-invalid")
# The bar the cross-window filter draws at: both siblings valid and averaging at
# least this quality. Set from the label set's own cross-window table
# (window_label_report's `cross_window` section).
CROSS_WINDOW_SIBLING_QUALITY = 3.0
# Documents whose graded window is wider than the model's context are trained on
# a label describing text the model never reads; this samples that gap.
COVERAGE_SAMPLE = 8_192
COVERAGE_MAX_TOKENS = 4_096
# 2048-token context quadruples the per-example activation, so the forward is
# gradient-checkpointed there rather than at the deployed context.
REMAT_ABOVE_TOKENS = 512


def window_target_from_begin(windows: dict[str, list]) -> tuple[np.ndarray, int]:
    """Each window's document's begin-window score, and how many windows lack one.

    Rows whose document has no begin grade cannot be retargeted, so they are
    marked with NaN for the caller to drop.
    """
    begin_score = {
        doc_id: score
        for doc_id, window, score in zip(windows["id"], windows["window"], windows["score_normalized"], strict=True)
        if window == BEGIN
    }
    targets = np.array([begin_score.get(doc_id, np.nan) for doc_id in windows["id"]], dtype=np.float32)
    return targets, int(np.isnan(targets).sum())


def log_window_coverage(tokenizer: str, texts: list[str], max_tokens: int, seed: int) -> None:
    """How much of each graded window the model's context actually reaches.

    The grades describe a 2048-token window whatever the model's context is, so
    a 512-token model is trained on a label covering text it never sees. Grep
    for ``WINDOW_COVERAGE``.
    """
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(texts), size=min(COVERAGE_SAMPLE, len(texts)), replace=False)
    lengths = np.array(
        [len(row) for row in encode_texts_fast(tokenizer, [texts[i] for i in sample], COVERAGE_MAX_TOKENS)]
    )
    covered = np.minimum(lengths, max_tokens) / np.maximum(lengths, 1)
    logger.info(
        "WINDOW_COVERAGE context=%d tokens: window length median=%d mean=%.0f p90=%d; "
        "%.1f%% of windows exceed the context; mean covered fraction=%.3f (n=%d)",
        max_tokens,
        int(np.median(lengths)),
        float(lengths.mean()),
        int(np.percentile(lengths, 90)),
        100.0 * float((lengths > max_tokens).mean()),
        float(covered.mean()),
        len(sample),
    )


def token_bme_chunks(texts: list[str]) -> tuple[list[str], list[tuple[int, int]]]:
    """Every document's bme windows under the label set's own 2048-token geometry.

    Returns the flat window list and each document's ``[start, end)`` span in
    it, matching :func:`scorer.bme_chunks`' contract. Documents under the
    geometry's long-document threshold yield their begin window alone, exactly
    as the labeling campaign cut them.
    """
    flat: list[str] = []
    spans: list[tuple[int, int]] = []
    for row in encode_documents(texts):
        windows = doc_windows(row, GEOMETRY_2048)
        spans.append((len(flat), len(flat) + len(windows)))
        flat.extend(w.text for w in windows)
    three = sum(1 for a, b in spans if b - a == 3)
    logger.info(
        "token bme: %d windows over %d documents (%d documents reached three windows)", len(flat), len(spans), three
    )
    return flat, spans


def score_chunks(
    model: FastTransformer,
    tokenizer: str,
    remap: dict[int, int],
    max_tokens: int,
    flat: list[str],
    spans: list[tuple[int, int]],
    doc_embed: np.ndarray,
) -> np.ndarray:
    """Mean of the model's score over each document's windows.

    Windows share their document's embedding, which is how the fusion model is
    fed at training time and how a deployment would feed it.
    """
    packed = pack(
        encode_texts_fast(tokenizer, flat, max_tokens), remap, np.zeros(len(flat), dtype=np.float32), max_tokens
    )
    repeated = np.concatenate([np.repeat(doc_embed[i : i + 1], b - a, axis=0) for i, (a, b) in enumerate(spans)])
    scores = predict(model, packed.ids, doc_embed=repeated)
    return np.array([scores[a:b].mean() for a, b in spans])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", required=True, help="arm name; also the artifact stem")
    p.add_argument("--out-dir", default=None, help="artifact dir (default <bme2048_retrain>/<name>)")
    p.add_argument("--window-labels", default=BME2048_WINDOW_LABELS)
    p.add_argument("--labels", default=DEFAULT_LABELS, help="88k label parquet defining the holdout id set")
    p.add_argument("--legacy-joined", default=DEFAULT_JOINED, help="88k labels-x-embeddings join root")
    p.add_argument("--scaleup-joined", default=SCALEUP_JOINED, help="scale-up labels-x-embeddings join root")
    p.add_argument("--domain-mlp", default=DEFAULT_DOMAIN_MLP, help="existing typer npz (kept for comparability)")
    p.add_argument("--train-windows", choices=("begin", "all"), default="begin")
    p.add_argument("--target", choices=("own", "begin"), default="own")
    p.add_argument(
        "--max-tokens", type=int, default=512, help="model context; the grades describe 2048 tokens either way"
    )
    p.add_argument("--filter", choices=FILTERS, default="none", help="which suspect grades to drop before training")
    p.add_argument(
        "--baseline-model-dir", default=V3_BASELINE_DIR, help="text-only scorer re-anchored (bme path); '' skips"
    )
    p.add_argument("--fused-baseline-model-dir", default=FUSED_88K_BASELINE_DIR, help="fused scorer re-scored; '' skips")
    p.add_argument("--epochs", type=int, default=None, help="override the epoch cap (smoke runs)")
    p.add_argument("--subsample", type=int, default=1, help="keep 1-in-N doc ids everywhere (smoke runs)")
    args = p.parse_args()
    configure_logging(logging.INFO)
    configure_coreweave_s3()
    out_dir = args.out_dir or f"{DEFAULT_OUT}/{args.name}"

    holdout_ids = holdout_id_set(args.labels)
    legacy = load_joined(args.legacy_joined)
    scaleup = load_joined(args.scaleup_joined, columns=["id", "embedding"])
    windows = load_window_labels(args.window_labels, columns=BME2048_COLUMNS)
    graded = len(windows["id"])
    if args.filter == "cross-window":
        windows = drop_cross_window_disagreements(windows, CROSS_WINDOW_SIBLING_QUALITY)
    elif args.filter == "cut-invalid":
        windows = drop_cut_window_invalids(windows)
    if args.train_windows == BEGIN:
        keep = [i for i, w in enumerate(windows["window"]) if w == BEGIN]
        windows = {c: [windows[c][i] for i in keep] for c in windows}
    if args.subsample > 1:
        for table in (legacy, scaleup, windows):
            keep_mask = subsample_mask(table["id"], args.subsample)
            for c in table:
                table[c] = [v for v, k in zip(table[c], keep_mask, strict=True) if k]
    logger.info(
        "labels: %d graded windows -> %d after filter=%s train-windows=%s subsample=%d",
        graded,
        len(windows["id"]),
        args.filter,
        args.train_windows,
        args.subsample,
    )

    targets = np.array(windows["score_normalized"], dtype=np.float32)
    if args.target == BEGIN:
        targets, orphans = window_target_from_begin(windows)
        logger.info("begin-label targets: %d windows have no begin grade for their document and are dropped", orphans)

    embedding_by_id = dict(zip(scaleup["id"], scaleup["embedding"], strict=True))
    embedding_by_id.update(zip(legacy["id"], legacy["embedding"], strict=True))
    train_index: list[int] = []
    holdout_hits = 0
    missing_embedding = 0
    for i, doc_id in enumerate(windows["id"]):
        if doc_id in holdout_ids:
            holdout_hits += 1
            continue
        if doc_id not in embedding_by_id:
            missing_embedding += 1
            continue
        if np.isnan(targets[i]):
            continue
        train_index.append(i)
    train_ids = [windows["id"][i] for i in train_index]
    train_texts = [windows["text"][i] for i in train_index]
    train_positions = np.array([windows["window"][i] for i in train_index])
    train_targets = targets[train_index]
    train_embed = embedding_matrix([embedding_by_id[doc_id] for doc_id in train_ids])
    logger.info(
        "assembled %d training windows over %d documents (%d holdout, %d missing embedding); positions: %s",
        len(train_index),
        len(set(train_ids)),
        holdout_hits,
        missing_embedding,
        {p_: int((train_positions == p_).sum()) for p_ in ("begin", "middle", "end")},
    )
    # The draw excluded the holdout by construction; assert it rather than trust it.
    assert not set(train_ids) & holdout_ids, "holdout doc ids leaked into the bme2048 training windows"
    assert holdout_hits == 0, f"{holdout_hits} bme2048 grades fell on holdout documents"

    is_eval = np.array([doc_id in holdout_ids for doc_id in legacy["id"]])
    ev_idx = np.flatnonzero(is_eval)
    legacy_texts = [t or "" for t in legacy["text"]]
    ev_texts = [legacy_texts[i] for i in ev_idx]
    quality = np.array(legacy["glm52_quality"], dtype=float)[ev_idx]
    types = np.array(legacy["glm52_content_type"])[ev_idx]
    sources = np.array(legacy["glm52_source"])[ev_idx]
    ev_emb_rows = [legacy["embedding"][i] for i in ev_idx]
    ev_emb = embedding_matrix(ev_emb_rows)
    typer, typer_labels = domain_mlp.load(args.domain_mlp)
    domains = domain_mlp.predict(typer, typer_labels, ev_emb_rows)
    logger.info(
        "split: %d training windows / %d holdout documents (of %d holdout ids)",
        len(train_ids),
        len(ev_idx),
        len(holdout_ids),
    )

    tokenizer = DONOR_TOKENIZERS[DONOR]
    bme_windows.check_gigatoken_parity(ev_texts, seed=TRAIN_SEED)
    check_gigatoken_parity(
        tokenizer, train_texts, np.array([windows["source"][i] for i in train_index]), max_tokens=args.max_tokens
    )
    log_window_coverage(tokenizer, train_texts, args.max_tokens, TRAIN_SEED)

    # Re-anchored before training so a broken baseline path fails in minutes:
    # v3 through the deployed bme scoring path (how its 0.634 was measured), the
    # 88k fused artifact through the fusion arms' own begin-window predict path
    # (how its 0.717 was).
    if args.baseline_model_dir:
        report_arm(
            "baseline_v3",
            score_bme(load_pooled_scorer(args.baseline_model_dir), ev_texts),
            quality,
            types,
            sources,
            domains,
        )
    if args.fused_baseline_model_dir:
        fused = load_pooled_scorer(args.fused_baseline_model_dir)
        fused_packed = pack(
            encode_texts_fast(fused.tokenizer_name, ev_texts, fused.max_tokens),
            fused.remap,
            np.zeros(len(ev_texts), dtype=np.float32),
            fused.max_tokens,
        )
        report_arm(
            "baseline_88k_fused",
            predict(fused.model, fused_packed.ids, doc_embed=ev_emb),
            quality,
            types,
            sources,
            domains,
        )
        del fused

    remap = full_vocab_remap(tokenizer)
    vocab = len(remap) + NUM_RESERVED
    donor = donor_embedding_table(DONOR)
    config = FastTransformerConfig(
        vocab_size=vocab,
        max_tokens=args.max_tokens,
        dropout=0.1,
        final_pool="mean",
        embed_dim=256,
        pool_kind="meanmaxmin",
        doc_embed_dim=EMBED_DIM,
        doc_embed_super_token=True,
        frozen_donor_dim=donor.shape[1],
        **WINNER_TRUNK,
    )
    hp = TrainHParams(seed=TRAIN_SEED, remat=args.max_tokens > REMAT_ABOVE_TOKENS)
    if args.epochs is not None:
        hp = TrainHParams(seed=TRAIN_SEED, remat=hp.remat, epochs=args.epochs)
    model = FastTransformer(config, key=jr.PRNGKey(hp.seed))
    model, params_filter = apply_embed_treatment(model, config, "learned-proj", donor, remap)
    logger.info(
        "arm %s: vocab=%d context=%d super_tokens=%d trunk d=%d L=%d h=%d w=%d mlp=%d remat=%s (%.0fK flops/token)",
        args.name,
        vocab,
        config.max_tokens,
        config.num_super_tokens,
        config.hidden_dim,
        config.num_layers,
        config.num_heads,
        config.pool_window,
        config.mlp_ratio,
        hp.remat,
        config.flops_per_token() / 1e3,
    )

    tr_pack = pack(encode_texts_fast(tokenizer, train_texts, args.max_tokens), remap, train_targets, args.max_tokens)
    fit_i, val_i = grouped_val_split(train_ids, hp.val_frac, hp.seed)
    best, best_epoch, seconds = train_regressor(
        model,
        tr_pack.ids[fit_i],
        tr_pack.scores[fit_i],
        tr_pack.ids[val_i],
        tr_pack.scores[val_i],
        hp,
        tr_doc_embed=train_embed[fit_i],
        val_doc_embed=train_embed[val_i],
        params_filter=params_filter,
    )
    logger.info("arm %s: best_epoch=%d train_seconds=%.0f", args.name, best_epoch, seconds)

    # Three scoring paths on the identical rows: the document's begin window (how
    # the 88k fused arms were measured), the mean over its bme windows under the
    # label set's own geometry (what a bme arm would deploy as), and the deployed
    # character-window bme (how v3's 0.634 was measured).
    begin_packed = pack(
        encode_texts_fast(tokenizer, ev_texts, args.max_tokens),
        remap,
        np.zeros(len(ev_texts), dtype=np.float32),
        args.max_tokens,
    )
    report_arm(f"{args.name}_begin", predict(best, begin_packed.ids, doc_embed=ev_emb), quality, types, sources, domains)
    token_flat, token_spans = token_bme_chunks(ev_texts)
    report_arm(
        f"{args.name}_bme2048",
        score_chunks(best, tokenizer, remap, args.max_tokens, token_flat, token_spans, ev_emb),
        quality,
        types,
        sources,
        domains,
    )
    char_flat, char_spans = bme_chunks(ev_texts)
    report_arm(
        f"{args.name}_bmechars",
        score_chunks(best, tokenizer, remap, args.max_tokens, char_flat, char_spans, ev_emb),
        quality,
        types,
        sources,
        domains,
    )
    _save_scorer(best, remap, tokenizer, config, out_dir, args.name)


if __name__ == "__main__":
    main()
