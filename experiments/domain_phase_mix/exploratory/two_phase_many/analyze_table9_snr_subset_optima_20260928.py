# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Does dropping OlmoBaseEval Easy's low-SNR components move MARINER's optimum?

For each SNR threshold, the objective keeps the components whose swarm standard deviation exceeds the threshold times
the proportional repeat SD (equal weights, as the full objective), refits MARINER's reference procedure on the frozen
3e18 swarm and optimizes it without a cap. Reports each subset optimum's distance to the full optimum, its predicted
loss on the full and subset objectives, each objective's repeat SD over the swarm's proportional repeats, and the
per-component predicted changes.

Writes `policies.csv`, `component_predictions.csv` and `summary.json` to
`reference_outputs/table9_snr_subset_optima_20260928/`. Nothing is launched.

usage: uv run --offline --no-sync python analyze_table9_snr_subset_optima_20260928.py
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OMP_THREAD_LIMIT"] = "1"

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parents[3]))

import materialize_delphi_comparator_proposals_20260909 as mat  # noqa: E402

ms = mat.ms
OUTPUT = SCRIPT_DIR / "reference_outputs" / "table9_snr_subset_optima_20260928"
TARGET = "table9"
THRESHOLDS = (2.0, 5.0, 8.0)


def tv(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).sum() / 2)


def optimum(fit: ms.ObjectiveFit, swarm: ms.Swarm) -> np.ndarray:
    weights, _ = ms.continuous_optimum(fit, swarm.weights)
    counts, _ = ms.runtime_policy(fit, weights)
    return counts / ms.MIXTURE_BLOCK_SIZE


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    panel = mat.bench.load_panel(mat.PANEL)
    swarm, _ = mat.reference_swarm_in_panel_order(panel)
    full = ms.read_objectives(ms.DATA / "objectives.csv")[TARGET]
    anchors = ms.read_anchors(ms.DATA / "anchors.csv", TARGET)
    folds = ms.final_inner_folds(swarm, pd.read_csv(ms.DATA / "splits.csv"))
    snr = {c: float(swarm.outcomes[c].std(ddof=1) / anchors[c].repeat_sd) for c in full.components}
    natural = 1.0 / swarm.inventory
    natural = natural / natural.sum()

    started = time.monotonic()
    full_fit = ms.fit_objective(swarm, full, anchors, folds)
    full_policy = optimum(full_fit, swarm)
    reference = mat.reference_policy(mat.MARINER_POLICY[TARGET], swarm.buckets)
    parity = tv(full_policy, reference)
    print(
        f"full objective: {len(full.components)} components, optimum reproduces the paper's policy to TV {parity:.4f} "
        f"({time.monotonic() - started:.0f}s)",
        flush=True,
    )
    if parity > mat.PARITY_TV:
        raise ValueError(f"reference MARINER fit does not reproduce {mat.MARINER_POLICY[TARGET]}: TV {parity}")
    uncheatable_policy = mat.reference_policy(mat.MARINER_POLICY["uncheatable"], swarm.buckets)

    policies = {"full": full_policy}
    fits = {"full": full_fit}
    components = {"full": full.components}
    for threshold in THRESHOLDS:
        kept = tuple(c for c in full.components if snr[c] >= threshold)
        objective = ms.Objective(f"{TARGET}_snr{threshold:g}", kept, np.full(len(kept), 1.0 / len(kept)))
        started = time.monotonic()
        fit = ms.fit_objective(swarm, objective, anchors, folds)
        policies[objective.name] = optimum(fit, swarm)
        fits[objective.name] = fit
        components[objective.name] = kept
        print(f"{objective.name}: {len(kept)} components kept ({time.monotonic() - started:.0f}s)", flush=True)

    swarm_tv = lambda policy: float(np.abs(swarm.weights - policy[None, :]).sum(axis=1).min() / 2)  # noqa: E731
    # Repeat SD of each equal-weight objective over the swarm's proportional repeats (its calibration rows).
    calibration = swarm.outcomes[swarm.calibration]
    repeat_sd = {name: float(calibration[list(kept)].mean(axis=1).std(ddof=1)) for name, kept in components.items()}
    summary: dict[str, object] = {
        "snr": snr,
        "reference_policy_parity_tv": parity,
        "objective_repeat_sd": repeat_sd,
        "proportional_repeats": int(swarm.calibration.sum()),
        "subsets": {},
    }
    rows = []
    for name, policy in policies.items():
        record = {
            "components": len(fits[name].weights),
            "tv_to_full_optimum": tv(policy, full_policy),
            "tv_to_uncheatable_optimum": tv(policy, uncheatable_policy),
            "tv_to_proportional": tv(policy, natural),
            "tv_to_nearest_swarm_run": swarm_tv(policy),
            "max_epochs": float((policy * swarm.inventory).max()),
            "active_buckets": int((policy > 0).sum()),
            "predicted_full_objective": float(full_fit.predict(policy[None])[0]),
            "predicted_full_objective_at_full_optimum": float(full_fit.predict(full_policy[None])[0]),
        }
        for other, fit in fits.items():
            record[f"predicted_{other}_objective"] = float(fit.predict(policy[None])[0])
        own = fits[name]
        record["own_objective_gain_over_full_optimum"] = float(
            own.predict(full_policy[None])[0] - own.predict(policy[None])[0]
        )
        record["own_objective_gain_in_repeat_sd"] = record["own_objective_gain_over_full_optimum"] / repeat_sd[name]
        summary["subsets"][name] = record
        tasks = full_fit.predict_tasks(policy[None])[0]
        for component, value in zip(full.components, tasks, strict=True):
            rows.append({"policy": name, "component": component, "snr": snr[component], "predicted_bpb": float(value)})
    pd.DataFrame({name: policy for name, policy in policies.items()}, index=list(swarm.buckets)).rename_axis(
        "bucket"
    ).to_csv(OUTPUT / "policies.csv")
    table = pd.DataFrame(rows).pivot(index=["component", "snr"], columns="policy", values="predicted_bpb").reset_index()
    for name in policies:
        if name != "full":
            table[f"{name}_minus_full"] = table[name] - table["full"]
    table.sort_values("snr").to_csv(OUTPUT / "component_predictions.csv", index=False)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary["subsets"], indent=1))
    dropped = table[table.snr < min(THRESHOLDS)]
    print("lowest-SNR components (predicted BPB under each policy):")
    print(dropped.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
