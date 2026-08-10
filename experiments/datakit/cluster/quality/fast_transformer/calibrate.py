# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fit the monotonic calibration used by ``score.py``.

The raw pooled-FT score is bell-shaped, so slicing it at fixed 0.2 cutpoints would
pile ~everything into the middle buckets. This fits a piecewise-linear remap that
warps the raw score so the fixed 0.2 boundaries land on the oracle quality levels:
score the labeled docs with the same whole-doc (bme) scoring the production step
uses, take the median raw score per oracle level (1..5), place a cutpoint at each
adjacent-level midpoint, and map those cutpoints onto ``[0, .2, .4, .6, .8, 1]``.

The remap is monotonic, so it does not change document ranking -- it only makes the
fixed-bucket quantization quality-coherent. Writes ``{"xk": [...], "yk": [...]}``
consumed by ``np.interp`` in ``score.py``.

    python -m experiments.datakit.cluster.quality.fast_transformer.calibrate \\
        --labels    s3://marin-us-east-02a/marin/datakit/quality_labels_20260709.parquet \\
        --model-dir s3://marin-us-east-02a/marin/datakit/models/quality/pooled_junkgate2 \\
        --out       s3://marin-us-east-02a/marin/datakit/models/quality/pooled_junkgate2/calib_bme.json
"""

import argparse
import json
import logging

import numpy as np
import pyarrow.parquet as pq
from rigging.filesystem import StoragePath
from rigging.log_setup import configure_logging

from experiments.datakit.cluster.quality.fast_transformer import content_type
from experiments.datakit.cluster.quality.fast_transformer.artifact import BUCKET_EDGES
from experiments.datakit.cluster.quality.fast_transformer.scorer import load_pooled_scorer, score_bme

logger = logging.getLogger(__name__)

DEFAULT_LABELS = "s3://marin-us-east-02a/marin/datakit/quality_labels_20260709.parquet"
YK = [0.0, *BUCKET_EDGES, 1.0]  # the interior IS BUCKET_EDGES, so the two can't drift
# Labels a type needs before it gets its own cutpoints. Five cutpoints from a
# handful of documents move with the sample rather than with the type.
DEFAULT_MIN_PER_TYPE = 400
# The rubric's catch-all. It is not a content type with its own standard of
# excellence, so it is held to the quality scale but not to cross-type parity.
RESIDUAL_TYPE = "other"


def apply_calibration(raw: np.ndarray, types: np.ndarray | None, knots: dict) -> np.ndarray:
    """Calibrated scores for ``raw``, using each document's own type when there is one.

    Accepts both calibration shapes so callers do not branch: a global ``{xk, yk}``
    is applied to everything, and a per-type ``{default, types}`` routes each
    document to its type's remap, falling back to the default for a type that was
    never fitted.
    """
    if "types" not in knots:
        return np.interp(raw, knots["xk"], knots["yk"])
    default = knots["default"]
    if types is None:
        return np.interp(raw, default["xk"], default["yk"])
    out = np.empty_like(raw, dtype=float)
    for name in set(types.tolist()):
        mask = types == name
        k = knots["types"].get(name, default)
        out[mask] = np.interp(raw[mask], k["xk"], k["yk"])
    return out


def fit_cutpoints(raw: np.ndarray, levels: np.ndarray) -> tuple[dict[int, float], list[float]]:
    """Return (per-level medians, cutpoints). The cutpoint between level k and k+1 is
    the midpoint of the two level medians; the cutpoints are enforced non-decreasing.
    All five oracle levels must be present -- a missing level would make the bucket
    boundaries ambiguous, so fail loudly rather than KeyError."""
    present = {int(v) for v in np.unique(levels)}
    missing = {1, 2, 3, 4, 5} - present
    if missing:
        raise ValueError(f"calibration labels missing oracle level(s) {sorted(missing)}; all of 1..5 required")
    med = {level: float(np.median(raw[levels == level])) for level in (1, 2, 3, 4, 5)}
    cuts = [(med[k] + med[k + 1]) / 2 for k in (1, 2, 3, 4)]
    return med, [float(c) for c in np.maximum.accumulate(cuts)]


def calibration_knots(raw: np.ndarray, levels: np.ndarray) -> dict:
    _, cuts = fit_cutpoints(raw, levels)
    xk = [float(raw.min()) - 1e-6, *cuts, float(raw.max()) + 1e-6]
    return {"xk": xk, "yk": YK}


def per_type_knots(raw: np.ndarray, levels: np.ndarray, types: np.ndarray, *, min_per_type: int) -> dict:
    """One remap per content type, plus a default for the types that lack support.

    The oracle grades each document as an example of its own type, and the types do
    not share a ceiling: solved agent trajectories reach the top score 3.4% of the
    time against prose's 26%. A single global remap therefore encodes prose's scale
    and pushes whole types out of the top bucket however good their members are.

    Each type gets its own cutpoints, so a score means the same quality whichever
    type produced it. The remap stays monotonic within a type, so it reorders
    nothing — unlike bucketing by quantile within a group, which would force every
    group to the same shape and rank a weak group's median above a strong group's.

    ``types`` must be *predicted* types, not the oracle's. The calibration is applied
    downstream to whatever the classifier says, so fitting it on the same predicted
    population is what makes the two agree — a type that quietly absorbs some prose
    gets cutpoints fitted on that mixture rather than on a cleaner one it will never
    see.

    A type with fewer than ``min_per_type`` labels cannot place five cutpoints
    stably, so it falls back to the global remap rather than to a noisy one.
    """
    knots = {"default": calibration_knots(raw, levels), "types": {}}
    for name in sorted(set(types.tolist())):
        mask = types == name
        if int(mask.sum()) < min_per_type:
            logger.info("calibrate: %-14s n=%-5d below floor, using the default remap", name, int(mask.sum()))
            continue
        try:
            knots["types"][name] = calibration_knots(raw[mask], levels[mask])
        except ValueError as e:
            # A type missing an oracle level entirely has ambiguous boundaries.
            logger.info("calibrate: %-14s n=%-5d %s", name, int(mask.sum()), e)
    return knots


def _parity_ratio(buckets: np.ndarray, types: np.ndarray) -> float:
    """Widest-to-narrowest top-bucket share across types; 1.0 is perfect parity.

    ``other`` is excluded. The rubric defines it as the residual bin — what is left
    when a document is not prose, code, math, multilingual, structured or a
    trajectory — and its members are junk (mean oracle quality 1.97). Requiring it to
    reach the top bucket as often as math would be requiring the filter to promote
    junk, so including it measures the opposite of what parity is for.
    """
    shares = [float((buckets[types == t] == 4).mean()) for t in sorted(set(types.tolist())) if t != RESIDUAL_TYPE]
    if not shares:
        return float("nan")
    return max(shares) / max(min(shares), 1e-6)


def _report_classifier_cost(raw: np.ndarray, levels: np.ndarray, predicted: np.ndarray, args) -> None:
    """How much parity the *predicted* type costs against the oracle's own type.

    Raw classifier accuracy is the wrong thing to gate on. What matters is whether
    calibrating through a predicted type still delivers parity, and confusions
    between types whose scales already agree cost nothing. Fitting both ways puts a
    number on the gap, so a classifier can be judged by what it costs rather than by
    a threshold chosen in advance.
    """
    with StoragePath(args.labels).open("rb") as fh:
        names = pq.read_table(fh).schema.names
    if "content_type" not in names:
        return
    with StoragePath(args.labels).open("rb") as fh:
        true_types = np.array(pq.read_table(fh, columns=["content_type"]).column("content_type").to_pylist())

    predicted_buckets = np.digitize(
        apply_calibration(raw, predicted, per_type_knots(raw, levels, predicted, min_per_type=args.min_per_type)),
        BUCKET_EDGES,
    )
    true_buckets = np.digitize(
        apply_calibration(raw, true_types, per_type_knots(raw, levels, true_types, min_per_type=args.min_per_type)),
        BUCKET_EDGES,
    )
    logger.info(
        "parity ratio (lower is better): %.2fx with predicted types, %.2fx with oracle types; "
        "classifier agrees with the oracle on %.1f%% of documents",
        _parity_ratio(predicted_buckets, predicted),
        _parity_ratio(true_buckets, true_types),
        100 * float((predicted == true_types).mean()),
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", default=DEFAULT_LABELS, help="labels parquet (source/text/quality/score_normalized)")
    p.add_argument("--model-dir", required=True, help="dir with the scorer artifacts to calibrate")
    p.add_argument("--out", required=True, help="output calibration json path")
    p.add_argument(
        "--content-type-model",
        default=None,
        help="content_type classifier npz; fits one remap per predicted type instead of one global remap",
    )
    p.add_argument("--min-per-type", type=int, default=DEFAULT_MIN_PER_TYPE)
    args = p.parse_args()
    configure_logging(logging.INFO)

    with StoragePath(args.labels).open("rb") as fh:
        table = pq.read_table(fh, columns=["text", "quality"])
    texts = [t or "" for t in table.column("text").to_pylist()]
    levels = np.array(table.column("quality").to_pylist(), dtype=float)

    scorer = load_pooled_scorer(args.model_dir)
    raw = score_bme(scorer, texts)

    if args.content_type_model:
        predicted = np.array(content_type.predict(content_type.load(args.content_type_model), texts))
        knots = per_type_knots(raw, levels, predicted, min_per_type=args.min_per_type)
        cal = apply_calibration(raw, predicted, knots)
        logger.info("fit per-type on %d labels; %d types have their own remap", len(texts), len(knots["types"]))
    else:
        knots = calibration_knots(raw, levels)
        cal = np.interp(raw, knots["xk"], knots["yk"])
        logger.info("fit on %d labels; cutpoints %s", len(texts), [round(x, 3) for x in knots["xk"][1:-1]])

    cb = np.digitize(cal, BUCKET_EDGES)
    ob = np.clip((levels - 1).astype(int), 0, 4)
    logger.info(
        "calibrated-bucket vs oracle-level: exact %.3f  within-1 %.3f", np.mean(cb == ob), np.mean(np.abs(cb - ob) <= 1)
    )
    if args.content_type_model:
        # Parity is the point of fitting per type, so report it rather than leaving
        # it to be discovered downstream. Measured on predicted types throughout.
        logger.info("top-bucket share by predicted type (parity target: similar across types)")
        for content_type_name in sorted(set(predicted.tolist())):
            mask = predicted == content_type_name
            logger.info(
                "  %-14s n=%-5d top-share=%.1f%%", content_type_name, int(mask.sum()), 100 * (cb[mask] == 4).mean()
            )
        _report_classifier_cost(raw, levels, predicted, args)

    with StoragePath(args.out).open("w") as fh:
        json.dump(knots, fh)
    logger.info("wrote calibration -> %s", args.out)


if __name__ == "__main__":
    main()
