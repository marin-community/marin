# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Local Table-9 candidates between OLMix and the WSPU cap sweep, with predicted component effects.

Candidates are (a) convex interpolations between the OLMix KL=0.005 cap-4 mixture and the frozen WSPU cap-6/7
mixtures, (b) successor optima under floors on synthetic QA and OLMOCR mass, (c) successor optima with the
per-bucket credit rules tested in the remedies script (no credit below a share floor, exposures clamped at panel
support), and (d) a box trust region around OLMix. Every candidate reports predicted per-component deltas from
OLMix under the plain successor and under the bank-calibrated residual model, TV distances, epoch-cap activity,
and the nearest measured bank coordinate. Nothing is launched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_single_phase_observatory_20260902 as harness,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_round3_heldout_selection_20260903 as selection,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_round3_proposals_20260903 as proposals,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_round5_olmix_gap_20260904 as gap,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    single_phase_round5_remedies_20260904 as remedies,
)

BLOCKS = proposals.BLOCKS
CAP = 7.0
INTERPOLATIONS = (0.25, 0.5, 0.75)
QA_FLOORS = (0.20, 0.25, 0.30)
OLMOCR_FLOORS = (0.08, 0.13)
BOX_HALF_WIDTH = 0.05
CALIBRATION_RIDGE = 1.0
KERNEL_BANDWIDTH = 0.2
BANK_TOP = 5


def bucket_costs(
    curves: remedies.Curves,
    exposures_by_count: np.ndarray,
    bucket: int,
    rule: str,
    weights: np.ndarray,
    panel_max: float,
) -> np.ndarray:
    """Cost of allocating each candidate count to ``bucket`` under a credit rule (macro objective)."""
    exposures = exposures_by_count
    if rule == "clamp_panel_max":
        exposures = np.minimum(exposures, panel_max)
    if rule.startswith("share_floor"):
        floor = float(rule.split(":")[1])
        exposures = np.where(weights < floor, 0.0, exposures)  # below the floor the bucket counts as absent
    return proposals.bucket_cost(curves.curves, bucket, exposures)


def solve(
    curves: remedies.Curves,
    inventory: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    rule: str,
    panel_max: np.ndarray,
) -> np.ndarray:
    if lower.sum() > BLOCKS or upper.sum() < BLOCKS:
        raise ValueError("infeasible bounds")
    best = np.full(BLOCKS + 1, np.inf)
    best[0] = 0.0
    choices = np.full((len(inventory), BLOCKS + 1), -1, dtype=np.int32)
    for bucket in range(len(inventory)):
        counts = np.arange(int(lower[bucket]), int(upper[bucket]) + 1)
        weights = counts / BLOCKS
        costs = bucket_costs(curves, inventory[bucket] * weights, bucket, rule, weights, float(panel_max[bucket]))
        updated = np.full(BLOCKS + 1, np.inf)
        selected = np.full(BLOCKS + 1, -1, dtype=np.int32)
        for count, cost in zip(counts, costs, strict=True):
            candidate = best[: BLOCKS + 1 - count] + cost
            target = updated[count:]
            better = candidate < target
            target[better] = candidate[better]
            selected[count:][better] = count
        best, choices[bucket] = updated, selected
    result = np.zeros(len(inventory), dtype=int)
    remaining = BLOCKS
    for bucket in range(len(inventory) - 1, -1, -1):
        count = int(choices[bucket, remaining])
        result[bucket] = count
        remaining -= count
    if remaining != 0:
        raise RuntimeError("broken backpointers")
    return result / BLOCKS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-dir", type=Path, required=True)
    parser.add_argument("--olmix-weights", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path, default=harness.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=gap.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    harness.HELDOUT_DIR = args.registry_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    panel = harness.load_panel(gap.PANEL)
    buckets = tuple(panel.buckets)
    inventory = panel.features.inventory
    panel_max = panel.features.exposures.max(axis=0)
    group = panel.group("table9")
    components = tuple(group.components)
    olmix = gap.olmix_weights(args.olmix_weights, buckets)
    sweep, sweep_active = gap.sweep_weights(buckets)
    full_bank = selection.load_bank(panel, "table9")
    _frame, full_features = harness.heldout_features(panel, "table9")
    bank, bank_features, _keep = remedies.drop_coordinates(
        full_bank, full_features, remedies.matched_seed_coordinates(remedies.MATCHED_SEED_EVAL_RUNS)
    )

    with harness.parallel_config(backend="loky", inner_max_num_threads=1):
        fitted = Parallel(n_jobs=args.workers, verbose=5)(
            delayed(proposals.fit_curve)(remedies.MODEL, "table9", index, None) for index in range(len(components))
        )
    curves = remedies.Curves(tuple(fitted))
    base = remedies.shard_matrix(args.shard_dir, panel, bank.coordinate_id)
    macro_residual = bank.measured - base.mean(axis=1)
    x_bank = remedies.descriptors(bank_features.weights, panel, bank.distance).to_numpy(float)
    calibration = remedies.ridge_fit(x_bank, macro_residual, CALIBRATION_RIDGE)

    order = np.argsort(bank.measured, kind="stable")
    top_rows = order[:BANK_TOP]
    replicated = np.where(bank.run_count >= 5)[0]
    bank_reference = {f"bank_top{rank + 1}": bank_features.weights[index] for rank, index in enumerate(top_rows)}
    bank_reference["bank_top5_mean"] = bank_features.weights[top_rows].mean(axis=0)
    for index in replicated:
        bank_reference[f"bank_replicated_n{int(bank.run_count[index])}"] = bank_features.weights[index]

    cap_blocks = np.floor(CAP / inventory * BLOCKS).astype(int)
    upper_cap = np.minimum(cap_blocks, BLOCKS)
    zeros = np.zeros(len(buckets), dtype=int)
    candidates: dict[str, np.ndarray] = {
        "olmix": olmix,
        "wspu_cap6": sweep[6],
        "wspu_cap7": sweep[7],
        "wspu_cap8": sweep[8],
    }
    for cap in (6, 7):
        for share in INTERPOLATIONS:
            candidates[f"interp_cap{cap}_{share:g}"] = (1 - share) * olmix + share * sweep[cap]
    for qa_floor in QA_FLOORS:
        for olmocr_floor in OLMOCR_FLOORS:
            lower = zeros.copy()
            lower[buckets.index("dolmino_synth_qa")] = int(np.ceil(qa_floor * BLOCKS))
            lower[buckets.index("dolmino_olmocr_pdfs_hq")] = int(np.ceil(olmocr_floor * BLOCKS))
            candidates[f"floor_qa{qa_floor:g}_olmocr{olmocr_floor:g}_cap7"] = solve(
                curves, inventory, lower, upper_cap, "plain", panel_max
            )
    candidates["rule_share_floor0.02_cap7"] = solve(curves, inventory, zeros, upper_cap, "share_floor:0.02", panel_max)
    candidates["rule_clamp_panel_max_cap7"] = solve(curves, inventory, zeros, upper_cap, "clamp_panel_max", panel_max)
    lower_box = np.maximum(np.ceil((olmix - BOX_HALF_WIDTH) * BLOCKS), 0).astype(int)  # inward rounding
    upper_box = np.minimum(np.floor((olmix + BOX_HALF_WIDTH) * BLOCKS).astype(int), upper_cap)
    candidates[f"box{BOX_HALF_WIDTH:g}_around_olmix_cap7"] = solve(
        curves, inventory, lower_box, upper_box, "plain", panel_max
    )
    top_mean = bank_reference["bank_top5_mean"]
    lower_top = np.maximum(np.ceil((top_mean - BOX_HALF_WIDTH) * BLOCKS), 0).astype(int)
    upper_top = np.minimum(np.floor((top_mean + BOX_HALF_WIDTH) * BLOCKS).astype(int), upper_cap)
    candidates[f"box{BOX_HALF_WIDTH:g}_around_bank_top5_cap7"] = solve(
        curves, inventory, lower_top, upper_top, "plain", panel_max
    )
    candidates.update(bank_reference)

    names = list(candidates)
    weights = np.vstack([candidates[name] for name in names])
    exposures = weights * inventory[None, :]
    component_pred = curves.component_matrix(exposures)
    macro_pred = component_pred.mean(axis=1)
    distance_to_panel = np.abs(weights[:, None, :] - panel.features.weights[None, :, :]).sum(-1).min(1)
    corrected = macro_pred + remedies.ridge_predict(
        calibration, remedies.descriptors(weights, panel, distance_to_panel).to_numpy(float)
    )
    kernel_corrected = macro_pred + remedies.kernel_smooth(
        bank_features.weights, macro_residual, weights, KERNEL_BANDWIDTH
    )
    kernel_only = remedies.kernel_smooth(bank_features.weights, bank.measured, weights, KERNEL_BANDWIDTH)
    top_tv = 0.5 * np.abs(weights[:, None, :] - bank_features.weights[None, top_rows, :]).sum(-1)
    olmix_row = names.index("olmix")
    bank_tv = 0.5 * np.abs(weights[:, None, :] - bank_features.weights[None, :, :]).sum(-1)
    summary = []
    effects = []
    for row, name in enumerate(names):
        nearest = int(np.argmin(bank_tv[row]))
        if name.startswith("wspu_cap"):
            active = sorted(sweep_active[int(name[len("wspu_cap") :])])  # the sweep's own cap, as stored
        else:
            counts = np.rint(weights[row] * BLOCKS).astype(int)
            active = [
                buckets[i] for i in range(len(buckets)) if counts[i] >= upper_cap[i]
            ]  # binding at the 7-epoch bound
        positive = np.where(weights[row] > 0, weights[row], 1.0)
        summary.append(
            {
                "candidate": name,
                "predicted_macro": float(macro_pred[row]),
                "predicted_delta_vs_olmix": float(macro_pred[row] - macro_pred[olmix_row]),
                "calibrated_macro": float(corrected[row]),
                "calibrated_delta_vs_olmix": float(corrected[row] - corrected[olmix_row]),
                "kernel_corrected_macro": float(kernel_corrected[row]),
                "kernel_corrected_delta_vs_olmix": float(kernel_corrected[row] - kernel_corrected[olmix_row]),
                "kernel_regression_macro": float(kernel_only[row]),
                "tv_to_bank_top5_min": float(top_tv[row].min()),
                "tv_to_olmix": float(0.5 * np.abs(weights[row] - olmix).sum()),
                "tv_to_nearest_panel_row": float(0.5 * distance_to_panel[row]),
                "tv_to_nearest_bank": float(bank_tv[row, nearest]),
                "nearest_bank_measured": float(bank.measured[nearest]),
                "nearest_bank_source": str(bank.sources[nearest]),
                "effective_buckets": float(np.exp(-(weights[row] * np.log(positive)).sum())),
                "max_epochs": float(exposures[row].max()),
                "cap_active_buckets": ";".join(active),
                "buckets_beyond_panel": int((exposures[row] > panel_max).sum()),
                "share_synth_qa": float(weights[row, buckets.index("dolmino_synth_qa")]),
                "share_olmocr": float(weights[row, buckets.index("dolmino_olmocr_pdfs_hq")]),
                "share_stack": float(
                    weights[row, [buckets.index("dolma3_stack_edu"), buckets.index("dolmino_stack_edu_fim")]].sum()
                ),
            }
        )
        for column, component in enumerate(components):
            effects.append(
                {
                    "candidate": name,
                    "component": gap.short_name(component),
                    "family": gap.family(component),
                    "predicted": float(component_pred[row, column]),
                    "predicted_delta_vs_olmix": float(component_pred[row, column] - component_pred[olmix_row, column]),
                }
            )
    summary_table = pd.DataFrame(summary)
    summary_table.to_csv(args.output_dir / "candidates_summary.csv", index=False)
    pd.DataFrame(effects).to_csv(args.output_dir / "candidates_component_effects.csv", index=False)
    pd.DataFrame(weights, index=names, columns=buckets).to_csv(args.output_dir / "candidates_weights.csv")
    (args.output_dir / "candidates_calibration.json").write_text(
        json.dumps(
            {
                "descriptors": remedies.DESCRIPTOR_NAMES,
                "coefficients_standardized": calibration[2].tolist(),
                "intercept": calibration[3],
                "ridge": CALIBRATION_RIDGE,
            },
            indent=2,
        )
    )
    pd.set_option("display.width", 250)
    shown = [
        "candidate",
        "predicted_macro",
        "predicted_delta_vs_olmix",
        "calibrated_delta_vs_olmix",
        "kernel_corrected_macro",
        "kernel_regression_macro",
        "tv_to_olmix",
        "tv_to_nearest_panel_row",
        "tv_to_nearest_bank",
        "nearest_bank_measured",
        "tv_to_bank_top5_min",
        "effective_buckets",
        "max_epochs",
        "share_synth_qa",
        "share_olmocr",
        "share_stack",
    ]
    print(summary_table[shown].round(4).to_string(index=False))
    print("\nbank reference coordinates (measured):")
    for name in bank_reference:
        row = names.index(name)
        nearest = int(np.argmin(bank_tv[row]))
        print(
            f"  {name}: measured {bank.measured[nearest]:.4f} runs {bank.run_count[nearest]}"
            f" source {bank.sources[nearest]}"
        )
    family_view = (
        pd.DataFrame(effects).groupby(["candidate", "family"])["predicted_delta_vs_olmix"].mean().unstack("family")
    )
    print("\npredicted family-mean deltas vs OLMix:")
    print(family_view.loc[names].round(4).to_string())


if __name__ == "__main__":
    main()
