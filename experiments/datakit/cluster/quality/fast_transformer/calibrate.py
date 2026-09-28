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
fixed-bucket quantization quality-coherent. Writes the global ``{"xk", "yk"}``
layout of :class:`Calibration`.

    python -m experiments.datakit.cluster.quality.fast_transformer.calibrate \\
        --labels    s3://marin-us-east-02a/marin/datakit/quality_labels_20260709.parquet \\
        --model-dir s3://marin-us-east-02a/marin/datakit/models/quality/pooled_junkgate2 \\
        --out       s3://marin-us-east-02a/marin/datakit/models/quality/pooled_junkgate2/calib_bme.json
"""

import argparse
import dataclasses
import json
import logging
from dataclasses import dataclass, field

import numpy as np
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath
from rigging.log_setup import configure_logging

from experiments.datakit.cluster.quality.fast_transformer.artifact import BUCKET_EDGES
from experiments.datakit.cluster.quality.fast_transformer.quality_model import CALIBRATION_FILE
from experiments.datakit.cluster.quality.fast_transformer.scorer import load_pooled_scorer, score_bme

logger = logging.getLogger(__name__)

DEFAULT_LABELS = "s3://marin-us-east-02a/marin/datakit/quality_labels_20260709.parquet"
YK = [0.0, *BUCKET_EDGES, 1.0]  # the interior IS BUCKET_EDGES, so the two can't drift


@dataclass(frozen=True)
class Curve:
    """A monotonic piecewise-linear remap of the raw score, applied with ``np.interp``."""

    xk: list[float]
    yk: list[float]

    def __call__(self, raw: np.ndarray) -> np.ndarray:
        return np.interp(raw, self.xk, self.yk)


@dataclass(frozen=True)
class Calibration:
    """The remap from raw scorer output to the calibrated score.

    A document routes through its content type's curve in ``types`` and through
    ``default`` when its type has none. A global calibration has no ``types`` and
    needs no content types to apply.
    """

    default: Curve
    types: dict[str, Curve] = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: dict) -> "Calibration":
        """Parse a calibration file: the global ``{xk, yk}`` or the per-type ``{default, types}`` layout."""
        if "types" not in data:
            return cls(default=Curve(**data))
        return cls(default=Curve(**data["default"]), types={name: Curve(**c) for name, c in data["types"].items()})

    def to_json(self) -> dict:
        """The layout :meth:`from_json` reads; a global calibration writes the bare curve."""
        if not self.types:
            return dataclasses.asdict(self.default)
        return {
            "default": dataclasses.asdict(self.default),
            "types": {n: dataclasses.asdict(c) for n, c in self.types.items()},
        }

    def apply(self, raw: np.ndarray, types: np.ndarray | None) -> np.ndarray:
        """Remap ``raw``; ``types`` holds one content type per row and is required when the calibration is per-type."""
        if not self.types:
            return self.default(raw)
        if types is None:
            raise ValueError("a per-type calibration needs one content type per document")
        out = np.empty(len(raw), dtype=np.float64)
        for name in set(types.tolist()):
            mask = types == name
            out[mask] = self.types.get(name, self.default)(raw[mask])
        return out


def load_calibration(model_dir: str, calib_file: str = CALIBRATION_FILE) -> Calibration:
    """Read the calibration file under ``model_dir``."""
    with (StoragePath(model_dir) / calib_file).open("r") as fh:
        return Calibration.from_json(json.loads(fh.read()))


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


def fit_calibration(raw: np.ndarray, levels: np.ndarray) -> Calibration:
    """A global calibration whose curve maps the per-level cutpoints onto :data:`BUCKET_EDGES`."""
    _, cuts = fit_cutpoints(raw, levels)
    xk = [float(raw.min()) - 1e-6, *cuts, float(raw.max()) + 1e-6]
    return Calibration(default=Curve(xk=xk, yk=YK))


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
    calibration = fit_calibration(raw, levels)

    cal = calibration.apply(raw, None)
    cb = np.digitize(cal, BUCKET_EDGES)
    ob = np.clip((levels - 1).astype(int), 0, 4)
    logger.info("fit on %d labels; cutpoints %s", len(texts), [round(x, 3) for x in calibration.default.xk[1:-1]])
    logger.info(
        "calibrated-bucket vs oracle-level: exact %.3f  within-1 %.3f", np.mean(cb == ob), np.mean(np.abs(cb - ob) <= 1)
    )

    with StoragePath(args.out).open("w") as fh:
        json.dump(calibration.to_json(), fh)
    logger.info("wrote calibration -> %s", args.out)


if __name__ == "__main__":
    main()
