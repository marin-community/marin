# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "scikit-learn",
#   "scipy",
# ]
# ///
"""Re-score every scale and target with the real Observatory DSP.

The ``effective_exposure_dsp`` baseline used everywhere else in this experiment
was a 43-column strawman with a shared power exponent. The Observatory model is
``standalone_code/dsp_exact.py``, 158 parameters at 39 buckets with a per-bucket
saturation rate and overexposure threshold. This script produces the replacement
rows so the affected comparisons can be corrected.

Out-of-fold RMSE refits the 78 nonlinear parameters inside every fold, so a fit
costs five nonlinear optimizations plus one. At roughly 38 seconds each on the
280-row panel this is minutes per cell, not seconds.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dsp_exact_baseline_20260726 import MODEL_NAME, dsp_exact_model, fit_dsp_exact
from swarm39_harness_20260725 import (
    REFERENCE_OUTPUTS,
    TABLE9,
    UNCHEATABLE,
    evaluate,
    load_scale,
    provenance,
    support_distance,
)

DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "dsp_exact_replacement_20260726"
SCALES = ("60m", "300m", "delphi_3e18")
TARGETS = (UNCHEATABLE, TABLE9)
CV_SPLITS = 5
CV_SEED = 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scales", nargs="*", default=list(SCALES))
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    model = dsp_exact_model()
    frames, fits = [], []
    for scale in args.scales:
        fit_panel, heldout = load_scale(scale)
        for target in TARGETS:
            if not np.isfinite(fit_panel.targets[target]).any():
                continue
            if not np.isfinite(heldout.targets[target]).any():
                continue
            started = time.time()
            fit = fit_dsp_exact(fit_panel, target, n_splits=CV_SPLITS, seed=CV_SEED)
            elapsed = time.time() - started
            usable = np.isfinite(heldout.targets[target])
            evaluation = heldout.subset(usable)
            distance = support_distance(fit_panel, evaluation)
            frames.append(evaluate(fit, model, fit_panel, heldout, target, support=distance))
            fits.append(
                {
                    "scale": scale,
                    "target": target,
                    "model": MODEL_NAME,
                    "oof_rmse": fit.oof_rmse,
                    "n_parameters": len(fit.coefficients) + 2 * len(fit_panel.buckets) + 2,
                    "n_active_features": int(np.count_nonzero(fit.coefficients)),
                    "intercept": fit.intercept,
                    "gamma": float(fit.shape["gamma"]),
                    "rho_at_upper_bound": int(np.sum(np.asarray(fit.shape["rho"]) > 1.999)),
                    "fit_seconds": elapsed,
                }
            )
            print(f"{scale:12s} {target:18s} oof={fit.oof_rmse:.6f}  ({elapsed:.0f}s)", flush=True)

    metrics = pd.concat(frames, ignore_index=True)
    metrics.to_csv(output / "heldout_metrics.csv", index=False)
    pd.DataFrame(fits).to_csv(output / "selected_fits.csv", index=False)
    (output / "protocol.json").write_text(
        json.dumps(
            {
                "model": "standalone_code/dsp_exact.py effective_exposure variant, unmodified",
                "adapter": "dsp_exact_baseline_20260726, verified against dsp_exact.predict to 1e-9",
                "parameters_at_39_buckets": 158,
                "replaces": "swarm39_models_20260725 build_effective_exposure_dsp (43-column strawman)",
                "cv_splits": CV_SPLITS,
                "cv_seed": CV_SEED,
                "nonlinear_parameters_refit_per_fold": True,
                "provenance_sha256": provenance(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    pooled = metrics[metrics["stratum_type"] == "pooled"]
    pd.set_option("display.width", 220)
    print("\n=== real DSP pooled heldout metrics ===")
    print(
        pooled[["scale", "target", "n", "oof_rmse", "rmse", "low_tail_rmse", "spearman", "regret_at_1"]].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
