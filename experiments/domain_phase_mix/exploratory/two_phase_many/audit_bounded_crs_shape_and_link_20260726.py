# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Decompose the bounded-CRS Observatory fit into its shape choice and its link choice.

``crs_bounded`` changes two things at once relative to compact retained state: it
fits on a log-deficit link, and it pins the nonlinear shape to the configuration
chosen by 60M-to-300M transfer rather than by fit-panel cross-validation. It fits
the Observatory panels worse than the baseline, so the useful question is which of
the two changes pays for that.

Crossing the two factors on the same design block answers it directly, and the
answer decides whether the model is salvageable: a bad link would be a modeling
error, whereas a shape that trades panel fit for transfer is the documented and
intended behaviour.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from swarm39_harness_20260725 import Design, Model, fit_model, load_scale  # noqa: E402

OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "bounded_crs_shape_and_link_20260726"

# Shared by both arms; only power and late_multiplier differ between selectors.
BASE_SHAPE = {"rate": 0.25, "forgetting_rate": 1.0, "deficit_floor_fraction": 0.95}
CV_SHAPE = {**BASE_SHAPE, "power": 1.0, "late_multiplier": 4.0}
TRANSFER_SHAPE = {**BASE_SHAPE, "power": 0.7, "late_multiplier": 8.0}

SCALES = ("delphi_3e18", "300m")
TARGETS = ("uncheatable_bpb", "table9_macro_bpb")
LOWER_TAIL_FRACTION = 0.15
LOWER_TAIL_MIN_COUNT = 5


def weibull(exposure: np.ndarray, rate: float, power: float) -> np.ndarray:
    return -np.expm1(-((np.maximum(rate * exposure, 0.0)) ** power))


def build_crs(panel, shape: dict) -> Design:
    """Compact retained state's design block, unchanged."""
    early = panel.phase0 * panel.c0
    late = panel.phase1 * panel.c1
    revisit = np.clip(panel.phase1, 0.0, 1.0)
    state = np.maximum(
        np.exp(-shape["forgetting_rate"] * (1.0 - revisit)) * early + shape["late_multiplier"] * late, 0.0
    )
    total = early + late
    benefit = weibull(state, shape["rate"], shape["power"])
    replay = np.sum(np.maximum(total - 1.0, 0.0) ** 2, axis=1, keepdims=True)
    return Design(
        matrix=np.hstack([-benefit, replay]),
        names=tuple([*(f"retained_benefit:{bucket}" for bucket in panel.buckets), "shared_literal_replay"]),
    )


def heldout_metrics(prediction: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    finite = np.isfinite(prediction) & np.isfinite(observed)
    prediction, observed = prediction[finite], observed[finite]
    residual = prediction - observed
    order = np.argsort(prediction)
    tail = order[: max(LOWER_TAIL_MIN_COUNT, int(LOWER_TAIL_FRACTION * len(order)))]
    ranks = lambda values: np.argsort(np.argsort(values))  # noqa: E731
    return {
        "heldout_rmse": float(np.sqrt(np.mean(residual**2))),
        "heldout_spearman": float(np.corrcoef(ranks(prediction), ranks(observed))[0, 1]),
        "low_tail_rmse": float(np.sqrt(np.mean(residual[tail] ** 2))),
        "low_tail_optimism": float(np.mean(observed[tail] - prediction[tail])),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for scale in SCALES:
        fit_panel, heldout = load_scale(scale)
        for target in TARGETS:
            for shape_name, shape in (("cv", CV_SHAPE), ("transfer", TRANSFER_SHAPE)):
                for link in ("identity", "log_deficit"):
                    model = Model(
                        name=f"crs_{shape_name}_{link}",
                        build=build_crs,
                        shapes=lambda bound=shape: (bound,),
                        link=link,
                    )
                    fit = fit_model(fit_panel, model, target)
                    rows.append(
                        {
                            "scale": scale,
                            "target": target,
                            "shape": shape_name,
                            "link": link,
                            "oof_rmse": fit.oof_rmse,
                            **heldout_metrics(fit.predict(heldout, model), heldout.targets[target]),
                        }
                    )

    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT_DIR / "shape_link_crossing.csv", index=False)

    print("=== crossed shape and link on the compact retained-state design ===")
    print(frame.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    print("\n=== main effects (mean over scale x target) ===")
    for factor in ("link", "shape"):
        print(f"\n-- {factor} --")
        print(
            frame.groupby(factor)[["oof_rmse", "heldout_rmse", "heldout_spearman", "low_tail_rmse"]]
            .mean()
            .to_string(float_format=lambda v: f"{v:.5f}")
        )

    print("\n=== paired effect of each change, cell by cell ===")
    wide = frame.set_index(["scale", "target", "shape", "link"])
    for scale in SCALES:
        for target in TARGETS:
            for shape_name in ("cv", "transfer"):
                bounded = wide.loc[(scale, target, shape_name, "log_deficit")]
                identity = wide.loc[(scale, target, shape_name, "identity")]
                delta = bounded["heldout_rmse"] - identity["heldout_rmse"]
                print(f"  link  {scale:12s} {target:18s} {shape_name:9s} heldout rmse {delta:+.5f}")
    for scale in SCALES:
        for target in TARGETS:
            for link in ("identity", "log_deficit"):
                transfer = wide.loc[(scale, target, "transfer", link)]
                cv = wide.loc[(scale, target, "cv", link)]
                delta = transfer["heldout_rmse"] - cv["heldout_rmse"]
                print(f"  shape {scale:12s} {target:18s} {link:11s} heldout rmse {delta:+.5f}")


if __name__ == "__main__":
    main()
