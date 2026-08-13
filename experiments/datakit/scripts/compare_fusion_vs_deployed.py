# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare a doc-embedding fusion quality scorer against the deployed text-only one.

Two stages, run in this order.

``holdout`` measures ranking against the GLM-5.2 oracle on the shared seed-0
holdout of the 88k label set (the 1/7 ``train.py`` reserves, intersected with the
rows that survived the harrier embedding join). Both models are scored in-process
on byte-identical documents, each through its own path: the deployed model sees
begin/middle/end windows of the whole document, the fusion model sees the first
512 tokens plus the document's stored 1024-d embedding, which is what it was
trained and evaluated on. The stage also fits the fusion model's per-type
calibration on those holdout rows, typed by the embedding domain MLP, and writes
it next to the model.

``evalset`` scores the 80,000-document domain evaluation set — the population
every earlier quality comparison used — with both models and reports what the
buckets do: distributions, the bucket-to-bucket transition, per-source and
per-domain mixes, and the documents whose bucket moves furthest. The fusion model
can only score it because :mod:`join_eval_docs_harrier_embeddings` recovered the
embedding column, which the evaluation shards do not carry.

Each stage writes one JSON blob of everything the report needs. Nothing is
recomputed at read time, so the report and the numbers cannot drift apart.
"""

import argparse
import collections
import functools
import json
import logging

import numpy as np
import pyarrow.parquet as pq
from rigging.filesystem import StoragePath, open_url
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging
from scipy import stats

from experiments.datakit.cluster.quality.fast_transformer import content_type, domain_mlp
from experiments.datakit.cluster.quality.fast_transformer.artifact import BUCKET_EDGES
from experiments.datakit.cluster.quality.fast_transformer.calibrate import (
    DEFAULT_MIN_PER_TYPE,
    apply_calibration,
    parity_ratio,
    per_type_knots,
)
from experiments.datakit.cluster.quality.fast_transformer.embed_exp import (
    DEFAULT_LABELS,
    FLAT_STD,
    MIN_SOURCE_LABELS,
    MIN_TYPE_LABELS,
    grouped_spearman,
    holdout_id_set,
)
from experiments.datakit.cluster.quality.fast_transformer.gate_model import source_signal
from experiments.datakit.cluster.quality.fast_transformer.joined_labels import (
    DEFAULT_JOINED,
    embedding_matrix,
    load_joined,
)
from experiments.datakit.cluster.quality.fast_transformer.scorer import load_pooled_scorer, score_bme

logger = logging.getLogger(__name__)

FUSION_DIR = "s3://marin-us-east-02a/marin/user/muchanem/quality_exp/bigger_fused/treatments/learnedproj"
DEPLOYED_DIR = "s3://marin-us-east-02a/marin/user/rav/quality_v2/models/pooled_glm52_v3"
DOMAIN_MLP = "s3://marin-us-east-02a/marin/user/muchanem/quality_exp/domain_mlp/domain_mlp.npz"
EVAL_DOCS = "s3://marin-us-east-02a/marin/user/rav/quality_v2/domain_eval/docs_v2"
EVAL_EMBEDDINGS = "s3://marin-us-east-02a/marin/user/muchanem/quality_v2/domain_eval_docs_v2-x-harrier-embeddings"
DEPLOYED_SCORED = "s3://marin-us-east-02a/marin/user/rav/quality_v2/domain_eval/v2set_scored_v3b/outputs/main"
REPORT_ROOT = "s3://marin-us-east-02a/marin/user/muchanem/quality_exp/fusion_vs_v3"
# The fusion model's calibration is typed by the embedding MLP, not by the text
# classifier the deployed model ships, so it does not overwrite `calib_bme.json`
# — a scoring path that picked it up would route through the wrong typer.
FUSION_CALIB_NAME = "calib_bme_domainmlp.json"

SCORE_BATCH = 256
# Documents kept per direction for the read test, and the text carried with them.
EXAMPLES_PER_DIRECTION = 40
EXAMPLE_CHARS = 1_400
HIST_BINS = 50


def _walk_parquet(root: str, max_depth: int = 5) -> list[str]:
    """Every ``*.parquet`` under ``root`` via single-level globs (a recursive glob
    HeadObjects the prefix, which the CW store answers with a 400)."""
    shards: list[str] = []
    dirs = [root.rstrip("/")]
    for _ in range(max_depth):
        nxt: list[str] = []
        for d in dirs:
            for entry in sorted(str(m) for m in StoragePath(f"{d}/*").glob()):
                (shards if entry.endswith(".parquet") else nxt).append(entry)
        dirs = nxt
        if not dirs:
            break
    return shards


def _histogram(values: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> dict:
    counts, edges = np.histogram(values, bins=HIST_BINS, range=(lo, hi))
    return {"edges": [float(e) for e in edges], "counts": [int(c) for c in counts]}


def _quantiles(values: np.ndarray) -> dict:
    qs = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    return {str(q): float(np.quantile(values, q)) for q in qs}


def _buckets(calibrated: np.ndarray) -> np.ndarray:
    return np.digitize(calibrated, BUCKET_EDGES)


def _bucket_shares(buckets: np.ndarray) -> list[float]:
    return [float((buckets == b).mean()) for b in range(5)]


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    rho = stats.spearmanr(a, b).statistic
    return float(rho) if np.isfinite(rho) else float("nan")


def _group_table(preds: np.ndarray, quality: np.ndarray, groups: np.ndarray, min_n: int) -> dict:
    """Per-group Spearman plus the counts and prediction spread behind each one."""
    table = {}
    for name in sorted(set(groups.tolist())):
        mask = groups == name
        n = int(mask.sum())
        if n < min_n:
            continue
        table[name] = {"n": n, "rho": _spearman(preds[mask], quality[mask]), "std": float(preds[mask].std())}
    return table


def _ranking_report(preds: np.ndarray, quality: np.ndarray, types: np.ndarray, sources: np.ndarray) -> dict:
    by_source = grouped_spearman(preds, quality, sources, MIN_SOURCE_LABELS)
    stds = {name: float(preds[sources == name].std()) for name in sorted(set(sources.tolist()))}
    flat = [name for name, s in stds.items() if s < FLAT_STD]
    return {
        "n": len(preds),
        "overall_rho": _spearman(preds, quality),
        "by_type": _group_table(preds, quality, types, MIN_TYPE_LABELS),
        "by_source": _group_table(preds, quality, sources, MIN_SOURCE_LABELS),
        "source_signal": source_signal(preds, quality, sources),
        "source_rho_mean": float(np.mean(list(by_source.values()))) if by_source else float("nan"),
        "source_rho_median": float(np.median(list(by_source.values()))) if by_source else float("nan"),
        "source_rho_min": float(np.min(list(by_source.values()))) if by_source else float("nan"),
        "source_rho_count": len(by_source),
        "source_pred_std": stds,
        "flat_sources": sorted(flat),
        "flat_source_count": len(flat),
        "score_quantiles": _quantiles(preds),
        "score_histogram": _histogram(preds),
    }


def _top_bucket_shares(buckets: np.ndarray, types: np.ndarray) -> dict:
    return {t: float((buckets[types == t] == 4).mean()) for t in sorted(set(types.tolist()))}


def _json_safe(value):
    """NaN as null. A Spearman over a constant slice is NaN, and ``NaN`` is not
    JSON — a reader that is not Python rejects the whole document over it."""
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _write_json(path: str, payload: dict) -> None:
    with StoragePath(path).open("w") as fh:
        json.dump(_json_safe(payload), fh, allow_nan=False)
    logger.info("wrote %s", path)


@functools.cache
def _scorer(model_dir: str):
    """One load per model per process: the fusion artifact is 700 MB."""
    return load_pooled_scorer(model_dir)


def _score_deployed(model_dir: str, texts: list[str]) -> np.ndarray:
    return score_bme(_scorer(model_dir), texts)


def _score_fusion(model_dir: str, texts: list[str], embeddings: np.ndarray) -> np.ndarray:
    scorer = _scorer(model_dir)
    if scorer.model.config.doc_embed_dim != embeddings.shape[1]:
        raise ValueError(
            f"{model_dir} expects a {scorer.model.config.doc_embed_dim}-d document embedding, "
            f"got {embeddings.shape[1]}"
        )
    return scorer.score(texts, batch_size=SCORE_BATCH, doc_embed=embeddings)


def run_holdout(args) -> dict:
    """Ranking against the oracle on the shared holdout, and the fusion calibration."""
    holdout_ids = holdout_id_set(args.labels)
    joined = load_joined(args.joined_dir)
    is_eval = np.array([doc_id in holdout_ids for doc_id in joined["id"]])
    if args.limit_rows:
        keep = np.flatnonzero(is_eval)[: args.limit_rows]
        is_eval = np.zeros_like(is_eval)
        is_eval[keep] = True

    quality = np.array(joined["glm52_quality"], dtype=float)
    types = np.array(joined["glm52_content_type"])
    sources = np.array(joined["glm52_source"])
    texts = [t or "" for t in joined["text"]]
    ev = np.flatnonzero(is_eval)
    ev_texts = [texts[i] for i in ev]
    ev_embed = embedding_matrix([joined["embedding"][i] for i in ev])
    logger.info(
        "holdout: %d joined rows, %d in the seed-0 holdout (of %d holdout ids in the label parquet)",
        len(quality),
        len(ev),
        len(holdout_ids),
    )

    typer, typer_labels = domain_mlp.load(args.domain_mlp)
    mlp_types = domain_mlp.predict(typer, typer_labels, [joined["embedding"][i] for i in ev])
    text_typer = content_type.load(f"{args.deployed_dir.rstrip('/')}/content_type.npz")
    text_types = np.array(content_type.predict(text_typer, ev_texts))

    fusion = _score_fusion(args.fusion_dir, ev_texts, ev_embed)
    deployed = _score_deployed(args.deployed_dir, ev_texts)

    ev_quality, ev_types, ev_sources = quality[ev], types[ev], sources[ev]
    report = {
        "population": {
            "joined_rows": len(quality),
            "holdout_rows": len(ev),
            "label_parquet_holdout_ids": len(holdout_ids),
            "sources": len(set(ev_sources.tolist())),
            "oracle_type_counts": {k: int(v) for k, v in collections.Counter(ev_types.tolist()).items()},
            "oracle_level_counts": {str(int(k)): int(v) for k, v in collections.Counter(ev_quality.tolist()).items()},
        },
        "typing": {
            "mlp_vs_oracle_agreement": float((mlp_types == ev_types).mean()),
            "text_vs_oracle_agreement": float((text_types == ev_types).mean()),
            "mlp_vs_text_agreement": float((mlp_types == text_types).mean()),
            "mlp_type_counts": {k: int(v) for k, v in collections.Counter(mlp_types.tolist()).items()},
            "text_type_counts": {k: int(v) for k, v in collections.Counter(text_types.tolist()).items()},
        },
        "fusion": _ranking_report(fusion, ev_quality, ev_types, ev_sources),
        "deployed": _ranking_report(deployed, ev_quality, ev_types, ev_sources),
        "agreement": {"fusion_vs_deployed_rho": _spearman(fusion, deployed)},
        "by_mlp_type": {
            "fusion": _group_table(fusion, ev_quality, mlp_types, MIN_TYPE_LABELS),
            "deployed": _group_table(deployed, ev_quality, mlp_types, MIN_TYPE_LABELS),
        },
    }

    # Calibration for the fusion model, fitted where the deployed model's was: on
    # labelled documents, typed by the typer that will route them at scoring time.
    # Held-out rows only, so the cutpoints are not read off the model's own
    # training documents; the full-join fit is reported beside it as a sensitivity
    # check, since the deployed model's shipped calibration was fitted in-sample.
    levels = ev_quality
    knots = per_type_knots(fusion, levels, mlp_types, min_per_type=args.min_per_type)
    calibrated = apply_calibration(fusion, mlp_types, knots)
    buckets = _buckets(calibrated)
    oracle_bucket = np.clip((levels - 1).astype(int), 0, 4)

    with open_url(f"{args.deployed_dir.rstrip('/')}/calib_bme.json", "r") as fh:
        deployed_calib = json.loads(fh.read())
    deployed_cal = apply_calibration(deployed, text_types, deployed_calib)
    deployed_buckets = _buckets(deployed_cal)

    # The shipped calibration was fitted on all 88k labels through the text
    # classifier, so a parity comparison against a fresh 11k MLP-typed fit would
    # be reading the fitting conditions, not the models. Fit the deployed model
    # one more time under the fusion model's exact conditions and report both.
    matched_knots = per_type_knots(deployed, levels, mlp_types, min_per_type=args.min_per_type)
    matched_buckets = _buckets(apply_calibration(deployed, mlp_types, matched_knots))

    report["calibration"] = {
        "fusion_knots": knots,
        "fusion_typing": "domain_mlp",
        "deployed_typing": "content_type.npz",
        "fusion_bucket_shares": _bucket_shares(buckets),
        "deployed_bucket_shares": _bucket_shares(deployed_buckets),
        "deployed_matched_bucket_shares": _bucket_shares(matched_buckets),
        "fusion_bucket_vs_oracle_exact": float((buckets == oracle_bucket).mean()),
        "fusion_bucket_vs_oracle_within1": float((np.abs(buckets - oracle_bucket) <= 1).mean()),
        "deployed_bucket_vs_oracle_exact": float((deployed_buckets == oracle_bucket).mean()),
        "deployed_bucket_vs_oracle_within1": float((np.abs(deployed_buckets - oracle_bucket) <= 1).mean()),
        "deployed_matched_bucket_vs_oracle_exact": float((matched_buckets == oracle_bucket).mean()),
        "deployed_matched_bucket_vs_oracle_within1": float((np.abs(matched_buckets - oracle_bucket) <= 1).mean()),
        "fusion_top_bucket_share_by_type": _top_bucket_shares(buckets, mlp_types),
        "deployed_top_bucket_share_by_type": _top_bucket_shares(deployed_buckets, text_types),
        "deployed_matched_top_bucket_share_by_type": _top_bucket_shares(matched_buckets, mlp_types),
        "parity_ratio": {
            "fusion": parity_ratio(buckets, mlp_types),
            "deployed_shipped": parity_ratio(deployed_buckets, text_types),
            "deployed_matched": parity_ratio(matched_buckets, mlp_types),
        },
    }

    if not args.limit_rows:
        _write_json(f"{args.fusion_dir.rstrip('/')}/{FUSION_CALIB_NAME}", knots)
    return report


def _load_embedding_index(root: str) -> tuple[dict[str, int], np.ndarray]:
    """``(id -> row, int8 matrix)`` for the joined evaluation-set embeddings."""
    shards = _walk_parquet(f"{root.rstrip('/')}/outputs")
    if not shards:
        raise ValueError(f"no embedding shards under {root}/outputs/")
    index: dict[str, int] = {}
    rows: list = []
    for shard in shards:
        with StoragePath(shard).open("rb") as fh:
            table = pq.ParquetFile(fh).read(columns=["id", "embedding"])
        for doc_id, vec in zip(table.column("id").to_pylist(), table.column("embedding").to_pylist(), strict=True):
            if doc_id in index:
                continue
            index[doc_id] = len(rows)
            rows.append(vec)
    logger.info("evalset: %d embeddings from %d shards", len(rows), len(shards))
    return index, np.asarray(rows, dtype=np.int8)


def run_evalset(args) -> dict:
    """Both models over the 80k domain evaluation set, scored shard by shard."""
    index, embed_int8 = _load_embedding_index(args.eval_embeddings)
    typer, typer_labels = domain_mlp.load(args.domain_mlp)
    text_typer = content_type.load(f"{args.deployed_dir.rstrip('/')}/content_type.npz")
    with open_url(f"{args.fusion_dir.rstrip('/')}/{FUSION_CALIB_NAME}", "r") as fh:
        fusion_calib = json.loads(fh.read())
    with open_url(f"{args.deployed_dir.rstrip('/')}/calib_bme.json", "r") as fh:
        deployed_calib = json.loads(fh.read())

    shards = sorted(str(m) for m in StoragePath(f"{args.eval_docs.rstrip('/')}/*.parquet").glob())
    if args.limit_shards:
        shards = shards[: args.limit_shards]
    logger.info("evalset: %d shards", len(shards))

    columns = ("id", "source", "domain_id", "cluster_5000")
    kept: dict[str, list] = {c: [] for c in columns}
    kept["chars"] = []
    kept["snippet"] = []
    fusion_raw: list[np.ndarray] = []
    deployed_raw: list[np.ndarray] = []
    fusion_types: list[np.ndarray] = []
    deployed_types: list[np.ndarray] = []
    missing_embedding = 0
    seen: set[str] = set()

    for n, shard in enumerate(shards, 1):
        with StoragePath(shard).open("rb") as fh:
            table = pq.ParquetFile(fh).read(columns=[*columns, "text"])
        data = {c: table.column(c).to_pylist() for c in (*columns, "text")}
        rows = []
        for i, doc_id in enumerate(data["id"]):
            if doc_id in seen:
                continue
            if doc_id not in index:
                missing_embedding += 1
                continue
            seen.add(doc_id)
            rows.append(i)
        texts = [data["text"][i] or "" for i in rows]
        raw_embed = [embed_int8[index[data["id"][i]]] for i in rows]
        embeddings = embedding_matrix(raw_embed)

        fusion_raw.append(_score_fusion(args.fusion_dir, texts, embeddings))
        deployed_raw.append(_score_deployed(args.deployed_dir, texts))
        fusion_types.append(domain_mlp.predict(typer, typer_labels, raw_embed))
        deployed_types.append(np.array(content_type.predict(text_typer, texts)))
        for c in columns:
            kept[c].extend(data[c][i] for i in rows)
        kept["chars"].extend(len(t) for t in texts)
        kept["snippet"].extend(t[:EXAMPLE_CHARS] for t in texts)
        logger.info("evalset: shard %d/%d, %d rows scored", n, len(shards), len(rows))

    ids = np.array(kept["id"])
    sources = np.array(kept["source"])
    domains = np.array([str(d) for d in kept["domain_id"]])
    chars = np.array(kept["chars"])
    fusion = np.concatenate(fusion_raw)
    deployed = np.concatenate(deployed_raw)
    ftypes = np.concatenate(fusion_types)
    dtypes = np.concatenate(deployed_types)
    fusion_cal = apply_calibration(fusion, ftypes, fusion_calib)
    deployed_cal = apply_calibration(deployed, dtypes, deployed_calib)
    fb, db = _buckets(fusion_cal), _buckets(deployed_cal)

    transition = [[int(((db == a) & (fb == b)).sum()) for b in range(5)] for a in range(5)]
    per_source = {}
    for name in sorted(set(sources.tolist())):
        mask = sources == name
        per_source[name] = {
            "n": int(mask.sum()),
            "deployed_shares": _bucket_shares(db[mask]),
            "fusion_shares": _bucket_shares(fb[mask]),
            "deployed_mean": float(deployed_cal[mask].mean()),
            "fusion_mean": float(fusion_cal[mask].mean()),
            "deployed_raw_std": float(deployed[mask].std()),
            "fusion_raw_std": float(fusion[mask].std()),
            "rank_agreement": _spearman(fusion[mask], deployed[mask]),
        }
    per_domain = {}
    for name in sorted(set(domains.tolist()), key=lambda d: int(d)):
        mask = domains == name
        per_domain[name] = {
            "n": int(mask.sum()),
            "deployed_shares": _bucket_shares(db[mask]),
            "fusion_shares": _bucket_shares(fb[mask]),
            "deployed_top2": float((db[mask] >= 3).mean()),
            "fusion_top2": float((fb[mask] >= 3).mean()),
        }

    # Length bands: the deployed model reads three windows of the document and the
    # fusion model reads one window plus a whole-document embedding, so what each
    # does with a stub and with a very long document is the sharpest place they
    # can differ, and the one an aggregate hides.
    length_bands = []
    for lo, hi in (
        (0, 200),
        (200, 500),
        (500, 1_000),
        (1_000, 2_000),
        (2_000, 4_000),
        (4_000, 8_000),
        (8_000, 16_000),
        (16_000, 10**9),
    ):
        mask = (chars >= lo) & (chars < hi)
        if not mask.any():
            continue
        length_bands.append(
            {
                "lo": lo,
                "hi": hi,
                "n": int(mask.sum()),
                "deployed_shares": _bucket_shares(db[mask]),
                "fusion_shares": _bucket_shares(fb[mask]),
                "deployed_top2": float((db[mask] >= 3).mean()),
                "fusion_top2": float((fb[mask] >= 3).mean()),
            }
        )

    move = fb.astype(int) - db.astype(int)
    order = np.argsort(move)
    examples = {"promoted": [], "demoted": []}
    for label, picks in (
        ("demoted", order[:EXAMPLES_PER_DIRECTION]),
        ("promoted", order[::-1][:EXAMPLES_PER_DIRECTION]),
    ):
        for i in picks:
            examples[label].append(
                {
                    "id": str(ids[i]),
                    "source": str(sources[i]),
                    "domain_id": str(domains[i]),
                    "chars": int(chars[i]),
                    "deployed_bucket": int(db[i]),
                    "fusion_bucket": int(fb[i]),
                    "deployed_score": float(deployed_cal[i]),
                    "fusion_score": float(fusion_cal[i]),
                    "deployed_type": str(dtypes[i]),
                    "fusion_type": str(ftypes[i]),
                    "text": kept["snippet"][i],
                }
            )

    stored = _compare_against_stored(ids, db, deployed_cal, args.deployed_scored)

    return {
        "population": {
            "rows_scored": len(ids),
            "missing_embedding": missing_embedding,
            "sources": len(set(sources.tolist())),
            "domains": len(set(domains.tolist())),
            "char_quantiles": _quantiles(chars.astype(float)),
        },
        "buckets": {
            "deployed_counts": [int((db == b).sum()) for b in range(5)],
            "fusion_counts": [int((fb == b).sum()) for b in range(5)],
            "deployed_top2_share": float((db >= 3).mean()),
            "fusion_top2_share": float((fb >= 3).mean()),
            "transition": transition,
            "unchanged": float((fb == db).mean()),
        },
        "scores": {
            "deployed_raw_histogram": _histogram(deployed),
            "fusion_raw_histogram": _histogram(fusion),
            "deployed_calibrated_histogram": _histogram(deployed_cal),
            "fusion_calibrated_histogram": _histogram(fusion_cal),
            "rank_agreement": _spearman(fusion, deployed),
            "deployed_length_rho": _spearman(deployed, chars.astype(float)),
            "fusion_length_rho": _spearman(fusion, chars.astype(float)),
        },
        "typing": {
            "mlp_counts": {k: int(v) for k, v in collections.Counter(ftypes.tolist()).items()},
            "text_counts": {k: int(v) for k, v in collections.Counter(dtypes.tolist()).items()},
            "agreement": float((ftypes == dtypes).mean()),
        },
        "per_source": per_source,
        "per_domain": per_domain,
        "length_bands": length_bands,
        "examples": examples,
        "stored_deployed_check": stored,
    }


def _compare_against_stored(ids: np.ndarray, buckets: np.ndarray, calibrated: np.ndarray, root: str) -> dict:
    """Reproduce check: the deployed model's stored scores for this population."""
    shards = sorted(str(m) for m in StoragePath(f"{root.rstrip('/')}/*.parquet").glob())
    if not shards:
        return {"status": "absent", "root": root}
    stored: dict[str, tuple[float, int]] = {}
    for shard in shards:
        with StoragePath(shard).open("rb") as fh:
            table = pq.ParquetFile(fh).read(columns=["id", "score", "quality_bucket"])
        for doc_id, score, bucket in zip(
            table.column("id").to_pylist(),
            table.column("score").to_pylist(),
            table.column("quality_bucket").to_pylist(),
            strict=True,
        ):
            stored.setdefault(doc_id, (float(score), int(bucket)))
    matched = [(i, stored[d]) for i, d in enumerate(ids) if d in stored]
    if not matched:
        return {"status": "no_overlap", "stored_rows": len(stored)}
    idx = np.array([i for i, _ in matched])
    stored_scores = np.array([s for _, (s, _) in matched])
    stored_buckets = np.array([b for _, (_, b) in matched])
    return {
        "status": "compared",
        "stored_rows": len(stored),
        "matched_rows": len(idx),
        "bucket_agreement": float((stored_buckets == buckets[idx]).mean()),
        "max_score_delta": float(np.abs(stored_scores - calibrated[idx]).max()),
        "stored_bucket_counts": [int((stored_buckets == b).sum()) for b in range(5)],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", required=True, choices=("holdout", "evalset"))
    p.add_argument("--fusion-dir", default=FUSION_DIR)
    p.add_argument("--deployed-dir", default=DEPLOYED_DIR)
    p.add_argument("--domain-mlp", default=DOMAIN_MLP)
    p.add_argument("--joined-dir", default=DEFAULT_JOINED)
    p.add_argument("--labels", default=DEFAULT_LABELS)
    p.add_argument("--eval-docs", default=EVAL_DOCS)
    p.add_argument("--eval-embeddings", default=EVAL_EMBEDDINGS)
    p.add_argument("--deployed-scored", default=DEPLOYED_SCORED)
    p.add_argument("--out", default=None, help="report JSON path (default: REPORT_ROOT/<stage>.json)")
    p.add_argument("--min-per-type", type=int, default=DEFAULT_MIN_PER_TYPE)
    p.add_argument("--limit-rows", type=int, default=0, help="holdout smoke run: score only N rows")
    p.add_argument("--limit-shards", type=int, default=0, help="evalset smoke run: score only N shards")
    args = p.parse_args()
    configure_logging(logging.INFO)
    configure_coreweave_s3()

    report = run_holdout(args) if args.stage == "holdout" else run_evalset(args)
    report["artifacts"] = {
        "fusion_model": args.fusion_dir,
        "deployed_model": args.deployed_dir,
        "domain_mlp": args.domain_mlp,
        "joined_labels": args.joined_dir,
        "label_parquet": args.labels,
        "eval_docs": args.eval_docs,
        "eval_embeddings": args.eval_embeddings,
    }
    out = args.out or f"{REPORT_ROOT}/{args.stage}.json"
    _write_json(out, report)
    logger.info("COMPARE %s: %s", args.stage, json.dumps(report.get("buckets") or report.get("fusion", {}))[:2000])


if __name__ == "__main__":
    main()
