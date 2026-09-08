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
"""Does the explicit odd/even phase law beat the incumbents at 3e18?

Fits every model on the 300M panel and judges at 3e18, which is the operating
protocol. Three questions, in the order they should be asked.

1. Phase skill on the 238 aggregate-matched fit-panel pairs. Those pairs have 238
   distinct aggregates, one per row, so a row bootstrap is valid on them. (The 948
   heldout pairs are NOT usable for this: they come from 20 aggregates with 78
   percent of rows at two anchors, and their median phase TV is 0.046 against the
   fit panel's 0.510.)
2. Proposal quality on the 3e18 heldout archive, as top-5 percentile inside a KL
   ball. Reported over a kappa sweep because the response is non-monotone in
   kappa: models pick in the top 5 to 19 percent at kappa=0.25 and near random at
   kappa=0.5.
3. The optimum sanity gate, which is pass/fail and independent of the above.

A candidate has to win on 1 without losing 2 or 3. Winning 2 alone is what
crs_plus did, and its Table-9 phase skill is the worst of the incumbents.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from audit_crossscale_phase_skill_20260726 import fit_any
from audit_proposal_metric_suite_20260726 import constrained_optimum, paired_subset, tied_twin
from audit_proposal_regret_convergence_20260726 import kl_to_proportional
from phase_law_model_20260726 import phase_law_hybrid_model, phase_law_model
from phase_order_spine_20260725 import load_paired_panel
from proposal_metrics_20260726 import optimum_sanity, phase_decision_skill, regret_in_sigma, top_k_percentile
from swarm39_harness_20260725 import (
    REFERENCE_OUTPUTS,
    TABLE9,
    UNCHEATABLE,
    load_scale,
    provenance,
    support_distance,
)
from swarm39_models_20260725 import crs_plus_extensions, nested_candidates, observatory_baselines

DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "phase_law_evaluation_20260726"
FIT_SCALE = "300m"
JUDGE_SCALE = "delphi_3e18"
TARGETS = (UNCHEATABLE, TABLE9)
TIED_PREFIX = r"^singleavg_fit_\d+_"
KAPPA_GRID = (0.25, 0.5, 1.0)
RUN_SIGMA = {UNCHEATABLE: 0.000963, TABLE9: 0.003121}
INCUMBENTS = ("compact_retained_state", "crs_plus", "crs_plus_phase", "hierarchical_phase_replay")
CANDIDATE = "phase_law_crs"
TOP_K = 5
BOOTSTRAP_DRAWS = 4000
OPTIMUM_LAMBDAS = (0.0, 0.1)
SEED = 20260726


def skill_interval(predicted: np.ndarray, observed: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    def skill(index: np.ndarray) -> float:
        oracle = float(np.mean(np.minimum(observed[index], 0.0)))
        if oracle >= 0.0:
            return float("nan")
        return float(np.mean(np.where(predicted[index] < 0.0, observed[index], 0.0)) / oracle)

    draws = np.array([skill(rng.integers(0, len(predicted), len(predicted))) for _ in range(BOOTSTRAP_DRAWS)])
    return {
        "skill_ci_low": float(np.nanquantile(draws, 0.025)),
        "skill_ci_high": float(np.nanquantile(draws, 0.975)),
        "probability_positive": float(np.nanmean(draws > 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    fit_full = load_scale(FIT_SCALE)[0]
    judge_panel_full, judge_archive = load_scale(JUDGE_SCALE)
    catalogue = {m.name: m for m in observatory_baselines(fit_full) + nested_candidates() + crs_plus_extensions()}
    catalogue[CANDIDATE] = phase_law_model()
    catalogue["phase_law_hybrid"] = phase_law_hybrid_model()
    names = [CANDIDATE, "phase_law_hybrid", *INCUMBENTS]
    paired = load_paired_panel(JUDGE_SCALE, TIED_PREFIX)

    phase_rows, rank_rows, gate_rows, delta_rows = [], [], [], []
    for target in TARGETS:
        fit_panel = fit_full.subset(np.isfinite(fit_full.targets[target]))
        judge_pairs, order = paired_subset(
            judge_panel_full.subset(np.isfinite(judge_panel_full.targets[target])), paired
        )
        observed_delta = paired.delta[target][np.argsort(order)]
        archive = judge_archive.subset(np.isfinite(judge_archive.targets[target]))
        observed = archive.targets[target]
        kl = kl_to_proportional(archive.aggregate, fit_panel.proportional)

        for name in names:
            fit, model = fit_any(fit_panel, name, catalogue, target)

            predicted_delta = fit.predict(judge_pairs, model) - fit.predict(tied_twin(judge_pairs), model)
            keep = np.isfinite(predicted_delta) & np.isfinite(observed_delta)
            phase_rows.append(
                {
                    "model": name,
                    "target": target,
                    **phase_decision_skill(predicted_delta[keep], observed_delta[keep]),
                    **skill_interval(predicted_delta[keep], observed_delta[keep], rng),
                }
            )
            delta_rows.append(
                pd.DataFrame(
                    {
                        "model": name,
                        "target": target,
                        "predicted_delta": predicted_delta[keep],
                        "observed_delta": observed_delta[keep],
                    }
                )
            )

            predicted = fit.predict(archive, model)
            for kappa in KAPPA_GRID:
                feasible = kl <= kappa
                if feasible.sum() < 20:
                    continue
                rank_rows.append(
                    {
                        "model": name,
                        "target": target,
                        "kappa": kappa,
                        **top_k_percentile(predicted, observed, feasible, TOP_K),
                        **regret_in_sigma(predicted, observed, feasible, RUN_SIGMA[target], TOP_K),
                    }
                )

            for lam in OPTIMUM_LAMBDAS:
                aggregate, epochs, row = constrained_optimum(fit, model, fit_panel, lam, rng)
                gate_rows.append(
                    {
                        "model": name,
                        "target": target,
                        "lambda": lam,
                        **optimum_sanity(aggregate, epochs, float(support_distance(fit_panel, row)[0])),
                    }
                )
            print(f"{target} {name}: done", flush=True)

    frames = {
        "phase_skill": pd.DataFrame(phase_rows),
        "ranking": pd.DataFrame(rank_rows),
        "optimum_gate": pd.DataFrame(gate_rows),
    }
    for key, frame in frames.items():
        frame.to_csv(output / f"{key}.csv", index=False)
    pd.concat(delta_rows, ignore_index=True).to_csv(output / "phase_deltas.csv", index=False)
    (output / "protocol.json").write_text(
        json.dumps(
            {
                "candidate": CANDIDATE,
                "incumbents": list(INCUMBENTS),
                "fit_scale": FIT_SCALE,
                "judge_scale": JUDGE_SCALE,
                "phase_pairs": "238 fit-panel aggregate-matched pairs, 238 distinct aggregates, row bootstrap valid",
                "heldout_pairs_excluded_because": (
                    "20 aggregates, 78 percent of rows at two anchors, median phase TV 0.046"
                ),
                "top_k": TOP_K,
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "provenance_sha256": provenance(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    pd.set_option("display.width", 240)
    print("\n=== PHASE SKILL, fit 300M judged 3e18 (238 pairs) ===")
    print(
        frames["phase_skill"][
            ["target", "model", "phase_skill_score", "skill_ci_low", "skill_ci_high", "probability_positive"]
        ]
        .round(4)
        .to_string(index=False)
    )
    print("\n=== PROPOSAL: top-5 percentile in the KL ball (0 = best available) ===")
    print(
        frames["ranking"]
        .pivot_table(index=["target", "model"], columns="kappa", values="picked_percentile")
        .round(4)
        .to_string()
    )
    print("\n=== OPTIMUM SANITY GATE ===")
    print(
        frames["optimum_gate"][["target", "model", "lambda", "effective_buckets", "max_simulated_epochs", "passes_gate"]]
        .round(3)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
