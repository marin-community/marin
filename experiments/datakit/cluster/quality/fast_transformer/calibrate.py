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
        --model-dir s3://marin-us-east-02a/marin/datakit/models/quality/pooled_junkgate2_marin \\
        --out       s3://marin-us-east-02a/marin/datakit/models/quality/pooled_junkgate2_marin/calib_bme.json

``train.py`` runs :func:`calibrate_model` right after training, so this CLI is for
recalibrating an existing model dir.

Knot validation (added 2026-10-30): ``np.interp`` assumes increasing ``xk`` and returns
silently wrong values otherwise, so every :class:`Curve` is checked at construction and
every curve of a loaded file is checked before any scoring. The fitter anchors its
floor and ceiling knots so they cannot land inside the cutpoints.  One released file,
``nemotron_88k/calib_bme.json`` (sha256 b89b7b78…), carries an inverted first knot on
its ``agentic`` curve; :data:`KNOWN_INVERTED_KNOT_DIGESTS` lets exactly that file load
with a warning, because its ``quality_bucket`` output is identical to the intended
curve's for every raw score (only the calibrated ``score`` column differs), and a
refit would change the recipe.
"""

import argparse
import contextvars
import dataclasses
import hashlib
import json
import logging
from dataclasses import dataclass, field

import numpy as np
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath
from rigging.log_setup import configure_logging

from experiments.datakit.cluster.quality.fast_transformer.artifact import BUCKET_EDGES, MODEL_CALIB
from experiments.datakit.cluster.quality.fast_transformer.data import encode_texts
from experiments.datakit.cluster.quality.fast_transformer.scorer import load_pooled_scorer, score_bme

logger = logging.getLogger(__name__)

DEFAULT_LABELS = "s3://marin-us-east-02a/marin/datakit/quality_labels_20260709.parquet"
YK = [0.0, *BUCKET_EDGES, 1.0]  # the interior IS BUCKET_EDGES, so the two can't drift
ANCHOR_MARGIN = 1e-6

# Released calibration files whose knots fail :func:`validate_knots` but whose bucket
# output is proven identical to the intended monotone curve's.  Keyed by the file's
# sha256; the value is the finding.  Loading one logs a warning instead of failing.
KNOWN_INVERTED_KNOT_DIGESTS = {
    "b89b7b782e606394fd341e7705d438521e96fcb553f769c6c1dd520331da5758": (
        "nemotron_88k/calib_bme.json: curve 'agentic' has xk[0]=0.31556 above xk[1]=0.28216; "
        "quality_bucket is identical to the sorted curve's for every raw score (both cross 0.2 at raw "
        "0.31556 and 0.4 at 0.47738); only the calibrated score is distorted for raw in [0.28216, 0.47738]"
    ),
}

# Set while :meth:`Calibration.from_json` loads a file listed in KNOWN_INVERTED_KNOT_DIGESTS.
_KNOWN_DEFECT_ALLOWED: contextvars.ContextVar[str | None] = contextvars.ContextVar("known_defect_allowed", default=None)


def validate_knots(xk, yk, name: str = "curve") -> None:
    """A curve ``np.interp`` can be trusted with: at least two knots, as many ``yk`` as ``xk``,
    ``xk`` strictly increasing and ``yk`` non-decreasing.  Names the curve and the knots otherwise."""
    if len(xk) != len(yk) or len(xk) < 2:
        raise ValueError(f"calibration curve {name!r}: need len(xk) == len(yk) >= 2, got {len(xk)} xk and {len(yk)} yk")
    x = np.asarray(xk, dtype=np.float64)
    y = np.asarray(yk, dtype=np.float64)
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError(f"calibration curve {name!r}: knots must be finite, got xk={list(xk)} yk={list(yk)}")
    bad = np.flatnonzero(np.diff(x) <= 0)
    if bad.size:
        i = int(bad[0])
        raise ValueError(
            f"calibration curve {name!r}: xk must be strictly increasing (np.interp assumes it); "
            f"xk[{i}]={x[i]:.6g} >= xk[{i + 1}]={x[i + 1]:.6g} in xk={[round(v, 6) for v in x.tolist()]}"
        )
    bad = np.flatnonzero(np.diff(y) < 0)
    if bad.size:
        i = int(bad[0])
        raise ValueError(
            f"calibration curve {name!r}: yk must be non-decreasing; "
            f"yk[{i}]={y[i]:.6g} > yk[{i + 1}]={y[i + 1]:.6g} in yk={y.tolist()}"
        )


@dataclass(frozen=True)
class Curve:
    """A monotonic piecewise-linear remap of the raw score, applied with ``np.interp``.

    Validated at construction (:func:`validate_knots`); a curve of a file listed in
    :data:`KNOWN_INVERTED_KNOT_DIGESTS` is admitted with a warning while that file loads.
    """

    xk: list[float]
    yk: list[float]

    def __post_init__(self) -> None:
        try:
            validate_knots(self.xk, self.yk)
        except ValueError as error:
            allowed = _KNOWN_DEFECT_ALLOWED.get()
            if allowed is None:
                raise
            logger.warning("admitting a known-defective calibration curve (%s): %s", allowed, error)

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
    def from_json(cls, data: dict, *, digest: str | None = None) -> "Calibration":
        """Parse a calibration file: the global ``{xk, yk}`` or the per-type ``{default, types}`` layout.

        Every curve is validated here, before any scoring, and a defect names its curve.  ``digest`` is the
        file's sha256 when known; a digest in :data:`KNOWN_INVERTED_KNOT_DIGESTS` admits that file's known
        defect with a warning (the finding is logged), anything else fails.
        """
        token = _KNOWN_DEFECT_ALLOWED.set(KNOWN_INVERTED_KNOT_DIGESTS.get(digest) if digest else None)
        try:
            if "types" not in data:
                return cls(default=_curve(data, "default"))
            return cls(
                default=_curve(data["default"], "default"),
                types={name: _curve(c, name) for name, c in data["types"].items()},
            )
        finally:
            _KNOWN_DEFECT_ALLOWED.reset(token)

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


def _curve(data: dict, name: str) -> Curve:
    """Build one curve of a file, naming it in the error when its knots are invalid."""
    try:
        return Curve(**data)
    except ValueError as error:
        raise ValueError(str(error).replace("calibration curve 'curve'", f"calibration curve {name!r}", 1)) from None


def load_calibration(model_dir: str, calib_file: str = MODEL_CALIB) -> Calibration:
    """Read the calibration file under ``model_dir``; its curves are validated before it is returned."""
    with (StoragePath(model_dir) / calib_file).open("rb") as fh:
        raw = fh.read()
    return Calibration.from_json(json.loads(raw), digest=hashlib.sha256(raw).hexdigest())


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


def anchored_knots(raw: np.ndarray, cuts: list[float]) -> list[float]:
    """The curve's ``xk``: a floor anchor below both the smallest raw score and the first cutpoint, the
    cutpoints, and a ceiling anchor above both the largest raw score and the last cutpoint.

    Anchoring to the raw extremes alone (the original rule) puts the floor *inside* the cutpoints whenever
    the scores being anchored do not reach below the first cutpoint -- the case when shared cutpoints are
    anchored per content type on that type's own raw range, which is how ``nemotron_88k``'s ``agentic``
    curve got xk[0] = 0.31556 > xk[1] = 0.28216.  Equal cutpoints (``np.maximum.accumulate`` ties) are
    spread by ``ANCHOR_MARGIN`` so the knots stay strictly increasing.
    """
    floor = min(float(raw.min()), cuts[0]) - ANCHOR_MARGIN
    ceiling = max(float(raw.max()), cuts[-1]) + ANCHOR_MARGIN
    knots = [floor]
    for cut in cuts:
        knots.append(max(cut, knots[-1] + ANCHOR_MARGIN))
    knots.append(max(ceiling, knots[-1] + ANCHOR_MARGIN))
    validate_knots(knots, YK, "fitted")
    return knots


def fit_calibration(raw: np.ndarray, levels: np.ndarray) -> Calibration:
    """A global calibration whose curve maps the per-level cutpoints onto :data:`BUCKET_EDGES`."""
    med, cuts = fit_cutpoints(raw, levels)
    xk = anchored_knots(raw, cuts)
    logger.info(
        "calibration fit: medians %s cutpoints %s anchors [%.6g, %.6g] (raw range [%.6g, %.6g])",
        {k: round(v, 5) for k, v in med.items()},
        [round(c, 5) for c in cuts],
        xk[0],
        xk[-1],
        float(raw.min()),
        float(raw.max()),
    )
    return Calibration(default=Curve(xk=xk, yk=YK))


def calibrate_model(labels_path: str, model_dir: str, out_path: str) -> Calibration:
    """Score the labels with the model in ``model_dir`` (encoding the label text the
    way the tokenize stage does), fit the global calibration, write it to
    ``out_path`` and return it."""
    with StoragePath(labels_path).open("rb") as fh:
        table = pq.read_table(fh, columns=["text", "quality"])
    texts = [t or "" for t in table.column("text").to_pylist()]
    levels = np.array(table.column("quality").to_pylist(), dtype=float)

    scorer = load_pooled_scorer(model_dir)
    docs = [np.asarray(ids, dtype=np.int32) for ids in encode_texts(scorer.tokenizer_name, texts)]
    raw = score_bme(scorer, docs)
    calibration = fit_calibration(raw, levels)

    cal = calibration.apply(raw, None)
    cb = np.digitize(cal, BUCKET_EDGES)
    ob = np.clip((levels - 1).astype(int), 0, 4)
    logger.info("fit on %d labels; cutpoints %s", len(texts), [round(x, 3) for x in calibration.default.xk[1:-1]])
    logger.info(
        "calibrated-bucket vs oracle-level: exact %.3f  within-1 %.3f", np.mean(cb == ob), np.mean(np.abs(cb - ob) <= 1)
    )

    with StoragePath(out_path).open("w") as fh:
        json.dump(calibration.to_json(), fh)
    logger.info("wrote calibration -> %s", out_path)
    return calibration


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", default=DEFAULT_LABELS, help="labels parquet (source/text/quality/score_normalized)")
    p.add_argument("--model-dir", required=True, help="dir with the scorer artifacts to calibrate")
    p.add_argument("--out", required=True, help="output calibration json path")
    args = p.parse_args()
    configure_logging(logging.INFO)
    calibrate_model(args.labels, args.model_dir, args.out)


if __name__ == "__main__":
    main()
