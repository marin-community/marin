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
"""Audit raw two-phase optima using the cross-scale-selected shapes.

The earlier optimum audit re-selected each model's shape per panel by single-panel
out-of-fold RMSE, which is the criterion shown to cause fantasy optima. This audit
instead reads the shape chosen by the heldout-free cross-scale criterion, so the
optimum reported is the one a clean protocol would actually deploy.

It answers the one open question from the previous round: whether the family-benefit
term keeps the optimum plausible at the 4-to-8-epoch saturation scale the data
prefers, rather than only at the 2 epochs that heldout metrics preferred.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from audit_swarm39_raw_optima_20260725 import describe, optimize
from benchmark_swarm39_crossscale_20260725 import fit_with_fixed_shape, select_shape_cross_scale
from swarm39_harness_20260725 import REFERENCE_OUTPUTS, TABLE9, UNCHEATABLE, load_scale, provenance
from swarm39_models_20260725 import (
    breadth_candidates,
    candidates,
    hierarchical_candidates,
    log_ratio_candidates,
    multiplicative_candidates,
    nested_candidates,
    observatory_baselines,
    shrinkage_candidates,
    structured_candidates,
)

DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "swarm39_crossscale_optima_20260725"
SCALES = ("60m", "300m", "delphi_3e18")
TARGETS = (UNCHEATABLE, TABLE9)
OPTIMIZER_SEED = 20260725
# Thresholds for calling an optimum deployable, fixed before reading results.
MAX_PLAUSIBLE_EPOCHS = 8.0
MIN_PLAUSIBLE_EFFECTIVE_BUCKETS = 10.0
MAX_PLAUSIBLE_SUPPORT_RATIO = 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--optimum-scale", default="delphi_3e18")
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(OPTIMIZER_SEED)

    loaded = {scale: load_scale(scale) for scale in SCALES}
    panels = {scale: pair[0] for scale, pair in loaded.items()}
    heldouts = {scale: pair[1] for scale, pair in loaded.items()}
    models = (
        observatory_baselines(panels["delphi_3e18"])
        + candidates()
        + hierarchical_candidates()
        + nested_candidates()
        + structured_candidates()
        + multiplicative_candidates()
        + shrinkage_candidates()
        + log_ratio_candidates()
        + breadth_candidates()
    )
    models = [m for m in models if m.name != "separate_heads"]

    scale = args.optimum_scale
    rows = []
    for model in models:
        for target in TARGETS:
            shape, _, _ = select_shape_cross_scale(panels, model, target)
            fit = fit_with_fixed_shape(panels[scale], model, target, shape)
            optimum = optimize(fit, model, panels[scale], rng)
            described = describe(optimum, panels[scale], heldouts[scale], target)
            rows.append(
                {
                    "scale": scale,
                    "target": target,
                    "model": model.name,
                    "shape": json.dumps({k: v for k, v in shape.items() if not isinstance(v, list)}, sort_keys=True),
                    "oof_rmse": fit.oof_rmse,
                    **described,
                    "deployable": bool(
                        described["max_simulated_epochs"] <= MAX_PLAUSIBLE_EPOCHS
                        and described["effective_bucket_count"] >= MIN_PLAUSIBLE_EFFECTIVE_BUCKETS
                        and described["support_distance_over_p95"] <= MAX_PLAUSIBLE_SUPPORT_RATIO
                    ),
                }
            )
    optima = pd.DataFrame(rows)
    optima.to_csv(output / "cross_scale_optima.csv", index=False)

    protocol = {
        "shape_source": "cross-scale selection over the 60M, 300M and 3e18 fit panels; no heldout used",
        "optimum_scale": scale,
        "plausibility_thresholds": {
            "max_simulated_epochs": MAX_PLAUSIBLE_EPOCHS,
            "min_effective_buckets": MIN_PLAUSIBLE_EFFECTIVE_BUCKETS,
            "max_support_ratio": MAX_PLAUSIBLE_SUPPORT_RATIO,
        },
        "thresholds_fixed_before_reading_results": True,
        "sealed_targeted_pairwise_panel_accessed": False,
        "provenance_sha256": provenance(),
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")

    pd.set_option("display.width", 250)
    columns = [
        "model",
        "predicted_minus_best_observed",
        "phase_tv",
        "max_bucket_weight",
        "max_simulated_epochs",
        "effective_bucket_count",
        "support_distance_over_p95",
        "deployable",
    ]
    for target in TARGETS:
        block = optima[optima["target"] == target]
        print(f"\n=== {target} at {scale}, cross-scale shapes ===")
        print(block.sort_values("max_simulated_epochs")[columns].to_string(index=False))
    print("\n=== top buckets ===")
    for row in optima.itertuples():
        print(f"{row.target:18s} {row.model:28s} {row.top_buckets}")


if __name__ == "__main__":
    main()
