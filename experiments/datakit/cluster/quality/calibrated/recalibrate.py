# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Content-type calibration of an existing document-quality score.

The deployed fasttext classifier ranks documents reasonably *within* a content
type, but places whole types on different absolute scales — e.g. it scores
non-English text near 0 regardless of quality, so excellent multilingual and code
documents can never reach the top quality bucket while mediocre English prose can.
The buckets therefore sort by domain, not quality.

This module removes that per-type offset without retraining the quality model:

    1. a cheap fasttext content-type classifier assigns each document a type;
    2. a per-type affine map rescales the raw score onto a common quality scale
       (fit against type-aware, source-blind oracle labels — see ``rubric.py``);
    3. global cutpoints + an absolute q4 excellence floor turn the calibrated
       score into 5 buckets that are comparable across types.

All three pieces are fasttext-cheap, so the scorer stays well under the deployed
FLOPs/token budget. See the module README for the validation evidence.
"""

import argparse
import json
import re
import tempfile
from dataclasses import asdict, dataclass

import fasttext
import numpy as np
import pyarrow.parquet as pq
from rigging.filesystem import open_url

from experiments.datakit.cluster.quality.calibrated.rubric import CONTENT_TYPES

_WS = re.compile(r"\s+")


# fasttext-wheel 0.9.2 calls ``np.array(..., copy=False)`` which NumPy 2 rejects.
def _patch_numpy_copy_compat() -> None:
    if getattr(np, "_calibrated_copy_compat", False):
        return
    _orig = np.array

    def _shim(*args, **kwargs):
        if kwargs.get("copy") is False:
            kwargs["copy"] = None
        return _orig(*args, **kwargs)

    np.array = _shim
    np._calibrated_copy_compat = True


def _clean(text: str, max_chars: int = 2000) -> str:
    return _WS.sub(" ", text).strip()[:max_chars]


@dataclass(frozen=True)
class TypeStats:
    """Per-type moments used to affine-map a raw score onto the quality scale."""

    raw_mean: float
    raw_std: float
    quality_mean: float
    quality_std: float
    n: int


@dataclass(frozen=True)
class Calibration:
    """Fitted calibration: per-type rescaling + global bucketing."""

    stats: dict[str, TypeStats]
    cutpoints: list[float]  # 4 ascending global cutpoints on the calibrated score
    q4_floor: float  # a doc may only enter q4 if its calibrated score exceeds this

    def to_json(self) -> str:
        return json.dumps(
            {
                "stats": {k: asdict(v) for k, v in self.stats.items()},
                "cutpoints": self.cutpoints,
                "q4_floor": self.q4_floor,
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, blob: str) -> "Calibration":
        d = json.loads(blob)
        return cls(
            stats={k: TypeStats(**v) for k, v in d["stats"].items()},
            cutpoints=list(d["cutpoints"]),
            q4_floor=float(d["q4_floor"]),
        )


class CalibratedQuality:
    """Turn a raw quality score into a content-type-calibrated score and bucket."""

    def __init__(self, type_model, calibration: Calibration):
        self._type_model = type_model
        self._cal = calibration

    @classmethod
    def load(cls, type_model_path: str, calibration_path: str) -> "CalibratedQuality":
        _patch_numpy_copy_compat()
        with open_url(calibration_path, "r") as fh:
            cal = Calibration.from_json(fh.read())
        # fasttext needs a local path; callers pass a local file or pre-fetch.
        return cls(fasttext.load_model(type_model_path), cal)

    def content_type(self, text: str) -> str:
        label = self._type_model.predict(_clean(text), k=1)[0][0]
        return label.replace("__label__", "")

    def recalibrate(self, raw_score: float, content_type: str) -> float:
        s = self._cal.stats.get(content_type)
        if s is None:
            return float(np.clip(raw_score, 0.0, 1.0))
        z = (raw_score - s.raw_mean) / s.raw_std
        return float(np.clip(z * s.quality_std + s.quality_mean, 0.0, 1.0))

    def bucket(self, calibrated: float) -> int:
        b = int(np.digitize(calibrated, self._cal.cutpoints))
        if b == 4 and calibrated < self._cal.q4_floor:
            return 3
        return b

    def score(self, text: str, raw_score: float) -> tuple[float, int]:
        """Return (calibrated_score, bucket) for one document."""
        ct = self.content_type(text)
        cal = self.recalibrate(raw_score, ct)
        return cal, self.bucket(cal)


def fit_calibration(
    texts: list[str],
    content_types: list[str],
    raw_scores: np.ndarray,
    quality_norm: np.ndarray,
    *,
    q4_floor: float = 0.70,
    type_epochs: int = 30,
):
    """Fit the type classifier + per-type calibration from oracle-labeled docs.

    ``quality_norm`` is the type-aware oracle quality in ``[0, 1]`` ((rating-1)/4).
    Returns ``(fasttext_type_model, Calibration)``.
    """
    _patch_numpy_copy_compat()
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        for text, ct in zip(texts, content_types, strict=True):
            fh.write(f"__label__{ct} {_clean(text)}\n")
        type_train_path = fh.name
    type_model = fasttext.train_supervised(type_train_path, epoch=type_epochs, lr=0.5, wordNgrams=2, dim=100, minCount=2)

    stats: dict[str, TypeStats] = {}
    for ct in CONTENT_TYPES:
        mask = np.array([c == ct for c in content_types])
        if mask.sum() < 10:
            continue
        raw, q = raw_scores[mask], quality_norm[mask]
        stats[ct] = TypeStats(
            float(raw.mean()), float(raw.std() + 1e-6), float(q.mean()), float(q.std() + 1e-6), int(mask.sum())
        )

    calibrated = np.array(
        [
            (
                (rs - stats[ct].raw_mean) / stats[ct].raw_std * stats[ct].quality_std + stats[ct].quality_mean
                if ct in stats
                else rs
            )
            for rs, ct in zip(raw_scores, content_types, strict=True)
        ]
    )
    calibrated = np.clip(calibrated, 0.0, 1.0)
    cutpoints = [float(np.quantile(calibrated, q)) for q in (0.2, 0.4, 0.6, 0.8)]
    return type_model, Calibration(stats=stats, cutpoints=cutpoints, q4_floor=q4_floor)


def _rescore_parquet(
    input_path: str, output_path: str, type_model_path: str, calibration_path: str, text_col: str, score_col: str
) -> None:
    scorer = CalibratedQuality.load(type_model_path, calibration_path)
    table = pq.read_table(input_path)
    texts = table.column(text_col).to_pylist()
    raw = table.column(score_col).to_pylist()
    cal, buckets = zip(*(scorer.score(t, s) for t, s in zip(texts, raw, strict=True)), strict=True)
    out = table.append_column("calibrated_score", [list(cal)]).append_column("quality_bucket", [list(buckets)])
    pq.write_table(out, output_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-score a parquet with content-type-calibrated quality buckets.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--type-model", required=True, help="local path to calib_type.bin")
    ap.add_argument("--calibration", required=True, help="path to calib.json")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--score-col", default="score")
    args = ap.parse_args()
    _rescore_parquet(args.input, args.output, args.type_model, args.calibration, args.text_col, args.score_col)


if __name__ == "__main__":
    main()
