# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""Stage-0 residual test for quality-pair churn on 300M Uncheatable."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import benchmark_aggregate_conditioned_replay_control_20260730 as benchmark  # noqa: E402
import benchmark_centered_hierarchical_rpl_20260730 as centered_benchmark  # noqa: E402

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    quality_pair_churn_hazard_rpl_20260730 as candidate,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    retained_power_law_model_20260728 as rpl,
)

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "quality_pair_churn_hazard_rpl_20260730"
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260730


def geometry_features(
    weights: np.ndarray,
    geometry: rpl.Geometry,
) -> np.ndarray:
    """Return generic phase-asymmetry features used for residualization."""
    phase_0, phase_1 = weights[:, 0, :], weights[:, 1, :]
    total_variation = 0.5 * np.abs(phase_1 - phase_0).sum(axis=1)
    global_hellinger = 1.0 - np.sqrt(phase_0 * phase_1).sum(axis=1)
    concentration = rpl.concentration_gap(weights, geometry)
    return np.column_stack(
        [
            np.ones(len(weights)),
            concentration,
            total_variation,
            global_hellinger,
        ]
    )


def residualized_churn(
    churn: np.ndarray,
    features: np.ndarray,
) -> tuple[np.ndarray, float]:
    coefficients = np.linalg.lstsq(features, churn, rcond=None)[0]
    residual = churn - features @ coefficients
    denominator = np.sum((churn - churn.mean()) ** 2)
    r_squared = 1.0 - np.sum(residual**2) / denominator
    return residual, float(r_squared)


def bootstrap_correlation(
    pair_residual: np.ndarray,
    churn: np.ndarray,
    features: np.ndarray,
) -> np.ndarray:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    correlations = np.empty(BOOTSTRAP_SAMPLES)
    for sample_id in range(BOOTSTRAP_SAMPLES):
        rows = rng.integers(0, len(pair_residual), size=len(pair_residual))
        residualized, _ = residualized_churn(churn[rows], features[rows])
        correlations[sample_id] = float(spearmanr(pair_residual[rows], residualized).statistic)
    return correlations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = benchmark.load_300m("uncheatable")
    geometry = benchmark.geometry_300m(dataset)
    baseline = centered_benchmark.frozen_rpl_prediction(
        "uncheatable",
        dataset,
    )
    tied_rows, asymmetric_rows = centered_benchmark.exact_pair_indices(
        dataset.frame,
        dataset.weights,
    )
    if len(tied_rows) != 238:
        raise ValueError(f"expected 238 exact pairs, found {len(tied_rows)}")

    observed_delta = dataset.y[asymmetric_rows] - dataset.y[tied_rows]
    predicted_delta = baseline[asymmetric_rows] - baseline[tied_rows]
    pair_residual = observed_delta - predicted_delta
    churn_families = candidate.quality_pair_families(dataset.domain_names)
    churn = candidate.conditional_family_churn(
        dataset.weights[asymmetric_rows],
        churn_families,
    ).sum(axis=1)
    features = geometry_features(
        dataset.weights[asymmetric_rows],
        geometry,
    )
    residualized, r_squared = residualized_churn(churn, features)
    raw_correlation = float(spearmanr(pair_residual, churn).statistic)
    correlation = float(spearmanr(pair_residual, residualized).statistic)
    standardized_slope = float(
        np.linalg.lstsq(
            np.column_stack(
                [
                    np.ones(len(residualized)),
                    residualized / np.std(residualized, ddof=1),
                ]
            ),
            pair_residual,
            rcond=None,
        )[0][1]
    )
    bootstrap = bootstrap_correlation(pair_residual, churn, features)
    ci_low, ci_high = np.quantile(bootstrap, (0.025, 0.975))
    passed = bool(correlation > 0.0 and ci_low > 0.0)

    pairs = pd.DataFrame(
        {
            "phase_correspondence_key": (
                dataset.frame.iloc[asymmetric_rows]["phase_correspondence_key"].astype(str).to_numpy()
            ),
            "tied_run": dataset.frame.iloc[tied_rows]["run_name"].astype(str).to_numpy(),
            "asymmetric_run": dataset.frame.iloc[asymmetric_rows]["run_name"].astype(str).to_numpy(),
            "observed_delta": observed_delta,
            "predicted_delta_rpl": predicted_delta,
            "pair_residual_observed_minus_predicted": pair_residual,
            "quality_pair_churn": churn,
            "residualized_quality_pair_churn": residualized,
        }
    )
    pairs.to_csv(args.output_dir / "stage0_pairs.csv", index=False)
    summary = {
        "target": "uncheatable",
        "pairs": len(pairs),
        "pair_residual_bias": float(np.mean(pair_residual)),
        "geometry_r_squared": r_squared,
        "raw_spearman": raw_correlation,
        "residualized_spearman": correlation,
        "standardized_linear_slope_bpb": standardized_slope,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_ci_low": float(ci_low),
        "bootstrap_ci_high": float(ci_high),
        "bootstrap_probability_positive": float(np.mean(bootstrap > 0.0)),
        "passed": passed,
        "next_action": (
            "freeze and run the nonzero-hazard comparison"
            if passed
            else "close the quality-pair churn route without fitting gamma"
        ),
    }
    (args.output_dir / "stage0_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = f"""# Quality-pair churn hazard RPL: Stage-0 result

- exact pairs: {len(pairs)}
- RPL pair-residual bias, observed minus predicted:
  {summary["pair_residual_bias"]:+.6f} BPB
- churn R-squared on global geometry:
  {r_squared:.3f}
- raw churn Spearman:
  {raw_correlation:+.3f}
- residualized churn Spearman:
  {correlation:+.3f}
- bootstrap 95% interval:
  [{ci_low:+.3f}, {ci_high:+.3f}]
- probability correlation is positive:
  {summary["bootstrap_probability_positive"]:.3f}
- standardized linear slope:
  {standardized_slope:+.6f} BPB per residualized-churn SD

## Decision

**{"PROCEED TO NONZERO-HAZARD FITTING" if passed else "CLOSE WITHOUT FITTING GAMMA"}**.

Table-9 was not read by this stage.
"""
    (args.output_dir / "stage0_report.md").write_text(report)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
