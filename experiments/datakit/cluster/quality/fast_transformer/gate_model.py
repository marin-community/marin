# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Decide whether a trained quality model is better than the deployed one.

The predecessor to this model won every aggregate on the evaluation set — wider
bucket spread, better cross-domain parity, higher within-domain variance — and was
still wrong: reading its disagreements against the deployed model showed it demoting
worked derivations and documented code while promoting scraped boilerplate. So the
checks here are chosen to be the ones that would have caught it, and none of them is
a summary of "how different" the two models are.

Three properties, each measured against the deployed model on the same documents:

* **Within-type ranking.** The predecessor's failure was legible only per type: it
  ranked prose at Spearman 0.742 and code at 0.586, math at 0.498. A model that
  cannot order documents inside a type will shuffle good and bad ones there whatever
  its aggregate looks like.
* **Cross-type parity.** Types must reach the top bucket at comparable rates, which
  is what per-type calibration is for. Measured on *predicted* types, because that
  is what inference has.
* **Source signal preserved.** Sources genuinely differ — oracle means run 2.13 to
  4.72 — and that difference is the most reliable signal the filter has. A model
  that flattens it has not become fairer, it has gone blind. This is also what
  distinguishes calibration from bucketing by rank, which would flatten it by
  construction.

Ranking is measured on the holdout ``train.py`` set aside, reproduced from the same
seed and fraction, so a model is never judged on documents it was fit to.

Passing this is necessary and not sufficient: the acceptance test is still reading
the disagreements (:mod:`sample_disagreements`).
"""

import argparse
import json
import logging
from dataclasses import dataclass

import numpy as np
import pyarrow.parquet as pq
from rigging.filesystem import StoragePath
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging
from scipy import stats

from experiments.datakit.cluster.quality.fast_transformer import content_type
from experiments.datakit.cluster.quality.fast_transformer.artifact import BUCKET_EDGES
from experiments.datakit.cluster.quality.fast_transformer.calibrate import apply_calibration
from experiments.datakit.cluster.quality.fast_transformer.scorer import load_pooled_scorer, score_bme

logger = logging.getLogger(__name__)

# Must match train.py, or the "holdout" contains documents the model was fit to.
TRAIN_SEED = 0
TRAIN_EVAL_FRAC = 1 / 7

MIN_TYPE_LABELS = 300
MIN_WITHIN_TYPE_RHO = 0.65
MAX_PARITY_RATIO = 2.0
MIN_SOURCE_RHO = 0.70


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str

    def line(self) -> str:
        return f"[{'PASS' if self.passed else 'FAIL'}] {self.name}: {self.detail}"


def holdout_indices(n_rows: int) -> np.ndarray:
    """The rows ``train.py`` held out, reproduced from its seed and fraction."""
    perm = np.random.default_rng(TRAIN_SEED).permutation(n_rows)
    return perm[: max(1, int(n_rows * TRAIN_EVAL_FRAC))]


def within_type_ranking(scores: np.ndarray, quality: np.ndarray, types: np.ndarray) -> dict[str, float]:
    """Spearman between score and oracle quality, inside each well-supported type."""
    out = {}
    for name in sorted(set(types.tolist())):
        mask = types == name
        if int(mask.sum()) >= MIN_TYPE_LABELS:
            out[name] = float(stats.spearmanr(scores[mask], quality[mask]).statistic)
    return out


def top_bucket_share(calibrated: np.ndarray, types: np.ndarray) -> dict[str, float]:
    buckets = np.digitize(calibrated, BUCKET_EDGES)
    return {name: float((buckets[types == name] == 4).mean()) for name in sorted(set(types.tolist()))}


def source_signal(scores: np.ndarray, quality: np.ndarray, sources: np.ndarray) -> float:
    """How well per-source predicted means track per-source oracle means."""
    names = sorted(set(sources.tolist()))
    predicted = [scores[sources == s].mean() for s in names]
    oracle = [quality[sources == s].mean() for s in names]
    if len(names) < 3:
        return float("nan")
    return float(stats.spearmanr(predicted, oracle).statistic)


def evaluate(*, texts, quality, sources, model_dir, calibration, type_model) -> dict:
    """Everything one model contributes to the comparison."""
    scorer = load_pooled_scorer(model_dir)
    raw = score_bme(scorer, texts)
    types = np.array(content_type.predict(type_model, texts)) if type_model else np.array(["all"] * len(texts))
    calibrated = apply_calibration(raw, types, calibration) if calibration else raw
    return {
        "raw": raw,
        "types": types,
        "calibrated": calibrated,
        "within_type": within_type_ranking(raw, quality, types),
        "top_share": top_bucket_share(calibrated, types),
        "source_rho": source_signal(raw, quality, sources),
        "overall_rho": float(stats.spearmanr(raw, quality).statistic),
    }


def gate(candidate: dict, baseline: dict | None) -> list[Check]:
    """The three model gates, stated against the baseline where one is given."""
    ranks = candidate["within_type"]
    worst = min(ranks, key=ranks.get) if ranks else None
    checks = [
        Check(
            "within-type ranking",
            bool(ranks) and all(v >= MIN_WITHIN_TYPE_RHO for v in ranks.values()),
            (
                f"worst type {worst} at {ranks[worst]:+.3f} (floor {MIN_WITHIN_TYPE_RHO}) over {len(ranks)} types"
                if ranks
                else "no type had enough holdout labels to measure"
            ),
        )
    ]

    shares = [v for v in candidate["top_share"].values()]
    if shares:
        median = float(np.median(shares))
        ratio = (max(shares) / max(min(shares), 1e-6)) if median > 0 else float("inf")
        checks.append(
            Check(
                "cross-type parity",
                ratio <= MAX_PARITY_RATIO,
                f"top-bucket share spans {min(shares):.1%}..{max(shares):.1%} = {ratio:.1f}x "
                f"(ceiling {MAX_PARITY_RATIO:.0f}x)",
            )
        )

    checks.append(
        Check(
            "source signal preserved",
            candidate["source_rho"] >= MIN_SOURCE_RHO,
            f"per-source Spearman {candidate['source_rho']:+.3f} (floor {MIN_SOURCE_RHO})",
        )
    )

    if baseline is not None:
        checks.append(
            Check(
                "beats the deployed model overall",
                candidate["overall_rho"] > baseline["overall_rho"],
                f"Spearman vs oracle {candidate['overall_rho']:+.3f} against {baseline['overall_rho']:+.3f}",
            )
        )
    return checks


def _report(candidate: dict, baseline: dict | None) -> None:
    logger.info("within-type ranking (holdout, higher is better)")
    for name in sorted(candidate["within_type"]):
        base = baseline["within_type"].get(name) if baseline else None
        suffix = f"   v0 {base:+.3f}" if base is not None else ""
        logger.info("  %-14s %+.3f%s", name, candidate["within_type"][name], suffix)
    logger.info("top-bucket share by predicted type (parity target: similar across types)")
    for name in sorted(candidate["top_share"]):
        logger.info("  %-14s %.1f%%", name, 100 * candidate["top_share"][name])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, help="the label set the model was trained on")
    parser.add_argument("--model-dir", required=True, help="candidate scorer directory")
    parser.add_argument("--baseline-model-dir", default=None, help="the deployed scorer to beat")
    parser.add_argument("--calibration", default=None, help="calibration json (per-type or global)")
    parser.add_argument("--content-type-model", default=None, help="content_type classifier npz")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    configure_logging(logging.INFO)
    configure_coreweave_s3()

    with StoragePath(args.labels).open("rb") as handle:
        table = pq.ParquetFile(handle).read(columns=["text", "quality", "source"])
    idx = holdout_indices(table.num_rows)
    rows = table.take(idx)
    texts = [t or "" for t in rows.column("text").to_pylist()]
    quality = np.array(rows.column("quality").to_pylist(), dtype=float)
    sources = np.array(rows.column("source").to_pylist())
    logger.info("gate_model: %d holdout documents (of %d labels)", len(texts), table.num_rows)

    calibration = None
    if args.calibration:
        with StoragePath(args.calibration).open("r") as handle:
            calibration = json.load(handle)
    type_model = content_type.load(args.content_type_model) if args.content_type_model else None

    candidate = evaluate(
        texts=texts,
        quality=quality,
        sources=sources,
        model_dir=args.model_dir,
        calibration=calibration,
        type_model=type_model,
    )
    baseline = None
    if args.baseline_model_dir:
        baseline = evaluate(
            texts=texts,
            quality=quality,
            sources=sources,
            model_dir=args.baseline_model_dir,
            calibration=None,
            type_model=type_model,
        )

    _report(candidate, baseline)
    checks = gate(candidate, baseline)
    for check in checks:
        logger.info("%s", check.line())
    failed = [c.name for c in checks if not c.passed]
    if failed:
        raise SystemExit(f"model gate FAILED: {', '.join(failed)}")
    logger.info("model gate passed — still read the disagreements before shipping")


if __name__ == "__main__":
    main()
