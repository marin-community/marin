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
        --model-dir s3://marin-us-east-02a/marin/user/rav/quality/pooled_junkgate2 \\
        --out       s3://marin-us-east-02a/marin/user/rav/quality/pooled_junkgate2/calib_bme.json
"""

import argparse
import json
import logging

import numpy as np
import pyarrow.parquet as pq
from rigging.filesystem import StoragePath
from rigging.log_setup import configure_logging

from experiments.datakit.cluster.quality.fast_transformer.scorer import BUCKET_EDGES, load_pooled_scorer, score_bme

logger = logging.getLogger(__name__)

DEFAULT_LABELS = "s3://marin-us-east-02a/marin/datakit/quality_labels_20260709.parquet"
YK = [0.0, *BUCKET_EDGES, 1.0]  # the interior IS BUCKET_EDGES, so the two can't drift


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


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", default=DEFAULT_LABELS, help="labels parquet (source/text/quality/score_normalized)")
    p.add_argument("--model-dir", required=True, help="dir with the scorer artifacts to calibrate")
    p.add_argument("--out", required=True, help="output calibration json path")
    args = p.parse_args()
    configure_logging(logging.INFO)

    with StoragePath(args.labels).open("rb") as fh:
        table = pq.read_table(fh, columns=["text", "quality"])
    texts = [t or "" for t in table.column("text").to_pylist()]
    levels = np.array(table.column("quality").to_pylist(), dtype=float)

    scorer = load_pooled_scorer(args.model_dir)
    raw = score_bme(scorer, texts)
    knots = calibration_knots(raw, levels)

    cal = np.interp(raw, knots["xk"], knots["yk"])
    cb = np.digitize(cal, BUCKET_EDGES)
    ob = np.clip((levels - 1).astype(int), 0, 4)
    logger.info("fit on %d labels; cutpoints %s", len(texts), [round(x, 3) for x in knots["xk"][1:-1]])
    logger.info(
        "calibrated-bucket vs oracle-level: exact %.3f  within-1 %.3f", np.mean(cb == ob), np.mean(np.abs(cb - ob) <= 1)
    )

    with StoragePath(args.out).open("w") as fh:
        json.dump(knots, fh)
    logger.info("wrote calibration -> %s", args.out)


if __name__ == "__main__":
    main()
