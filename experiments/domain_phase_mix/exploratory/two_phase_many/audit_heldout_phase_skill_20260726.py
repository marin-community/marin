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
"""Transfer phase skill on the heldout archive's own aggregate-matched pairs.

The fit panel supplies 238 matched pairs, and at that size the transfer ranking is
unresolvable: bootstrapping the skill score leaves 8 of 9 Uncheatable challengers
and 9 of 9 Table-9 challengers overlapping the leader.

The heldout archive contains its own pairs. Of 1963 rows, 430 are phase-tied, and
949 two-phase rows have a tied row matching their aggregate to L2 below 1e-9.
Those 949 pairs are exact, disjoint from the fit panel, and four times the
previous sample, which narrows every interval by about a factor of two.

The tied twin of a matched row is by construction the policy with
``phase_0 = phase_1 = aggregate``, so the model is asked for exactly the contrast
that was observed and ``Delta`` carries no aggregate-model error.

Models are fitted on the 300M panel and judged here, so no 3e18 observation of any
kind enters the fit. That closes the leakage path found in the within-scale arm,
where the paired panel's group ids were unique per row and grouped_splits silently
degraded to plain KFold over a single design series.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from audit_crossscale_phase_skill_20260726 import fit_any
from audit_proposal_metric_suite_20260726 import tied_twin
from proposal_metrics_20260726 import phase_decision_skill
from scipy.spatial import cKDTree
from swarm39_harness_20260725 import (
    REFERENCE_OUTPUTS,
    TABLE9,
    UNCHEATABLE,
    Panel,
    load_scale,
    provenance,
)
from swarm39_models_20260725 import crs_plus_extensions, nested_candidates, observatory_baselines

DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "heldout_phase_skill_20260726"
FIT_SCALE = "300m"
JUDGE_SCALE = "delphi_3e18"
TARGETS = (UNCHEATABLE, TABLE9)
TIED_TOLERANCE = 1e-9
MATCH_TOLERANCE = 1e-9
BOOTSTRAP_DRAWS = 4000
SEED = 20260726


def heldout_pairs(panel: Panel, target: str) -> tuple[Panel, np.ndarray]:
    """Two-phase heldout rows paired with an exactly aggregate-matched tied row."""
    observed = panel.targets[target]
    tied = np.abs(panel.phase0 - panel.phase1).max(axis=1) < TIED_TOLERANCE
    tied_index = np.flatnonzero(tied & np.isfinite(observed))
    two_phase_index = np.flatnonzero(~tied & np.isfinite(observed))
    if len(tied_index) == 0 or len(two_phase_index) == 0:
        return panel.subset(np.zeros(len(panel), dtype=bool)), np.zeros(0)

    distance, nearest = cKDTree(panel.aggregate[tied_index]).query(panel.aggregate[two_phase_index], k=1)
    matched = distance < MATCH_TOLERANCE
    rows = two_phase_index[matched]
    twins = tied_index[nearest[matched]]
    # The tied twin must be the aggregate itself, which is what tied_twin builds.
    residual = float(np.abs(panel.aggregate[rows] - panel.phase0[twins]).max())
    assert residual < MATCH_TOLERANCE, f"tied twin is not the aggregate (residual {residual:.3e})"

    mask = np.zeros(len(panel), dtype=bool)
    mask[rows] = True
    subset = panel.subset(mask)
    # subset preserves order, so reorder delta onto it.
    order = np.argsort(rows)
    delta = observed[rows][order] - observed[twins][order]
    return subset, delta


def bootstrap_skill(predicted: np.ndarray, observed: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    def skill(index: np.ndarray) -> float:
        oracle = float(np.mean(np.minimum(observed[index], 0.0)))
        if oracle >= 0.0:
            return float("nan")
        return float(np.mean(np.where(predicted[index] < 0.0, observed[index], 0.0)) / oracle)

    draws = np.array([skill(rng.integers(0, len(predicted), len(predicted))) for _ in range(BOOTSTRAP_DRAWS)])
    return {
        "skill_ci_low": float(np.nanquantile(draws, 0.025)),
        "skill_ci_high": float(np.nanquantile(draws, 0.975)),
        "skill_sd": float(np.nanstd(draws)),
        "probability_positive_skill": float(np.mean(draws > 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    fit_panel_full = load_scale(FIT_SCALE)[0]
    judge_archive = load_scale(JUDGE_SCALE)[1]
    catalogue = {m.name: m for m in observatory_baselines(fit_panel_full) + nested_candidates() + crs_plus_extensions()}
    names = list(catalogue)

    rows, deltas = [], []
    for target in TARGETS:
        fit_panel = fit_panel_full.subset(np.isfinite(fit_panel_full.targets[target]))
        pairs, observed_delta = heldout_pairs(judge_archive, target)
        print(f"{target}: {len(pairs)} heldout matched pairs", flush=True)
        for name in names:
            fit, model = fit_any(fit_panel, name, catalogue, target)
            predicted = fit.predict(pairs, model) - fit.predict(tied_twin(pairs), model)
            keep = np.isfinite(predicted) & np.isfinite(observed_delta)
            record = {
                "model": name,
                "target": target,
                **phase_decision_skill(predicted[keep], observed_delta[keep]),
                **bootstrap_skill(predicted[keep], observed_delta[keep], rng),
            }
            rows.append(record)
            deltas.append(
                pd.DataFrame(
                    {
                        "model": name,
                        "target": target,
                        "predicted_delta": predicted[keep],
                        "observed_delta": observed_delta[keep],
                    }
                )
            )
            print(
                f"  {name:28s} skill={record['phase_skill_score']:+.3f} "
                f"[{record['skill_ci_low']:+.3f}, {record['skill_ci_high']:+.3f}]",
                flush=True,
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "heldout_phase_skill.csv", index=False)
    pd.concat(deltas, ignore_index=True).to_csv(output / "heldout_deltas.csv", index=False)
    (output / "protocol.json").write_text(
        json.dumps(
            {
                "fit_scale": FIT_SCALE,
                "judge_scale": f"{JUDGE_SCALE} heldout archive, self-paired",
                "pair_construction": (
                    "two-phase heldout row matched to a tied heldout row with identical aggregate (L2 < 1e-9)"
                ),
                "no_3e18_data_in_fit": True,
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "seed": SEED,
                "provenance_sha256": provenance(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    pd.set_option("display.width", 240)
    for target in TARGETS:
        block = frame[frame["target"] == target].sort_values("phase_skill_score", ascending=False)
        print(f"\n=== {target}: heldout transfer phase skill (n={block['n_pairs'].iloc[0]}) ===")
        print(
            block[
                [
                    "model",
                    "phase_skill_score",
                    "skill_ci_low",
                    "skill_ci_high",
                    "probability_positive_skill",
                    "decision_accuracy",
                    "always_tied_accuracy",
                ]
            ]
            .round(4)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
