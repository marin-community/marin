# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Did optimizing OlmoBaseEval Easy over its higher-SNR components pay off when trained?

Scores the six trained SNR-subset optima (`delphi_table9_snr_subset_3e18_20260928`: SNR >= 5 and SNR >= 8 subsets,
data seed 662009, trainer seeds 0-2) and MARINER's full-objective runs at the same seeds (frozen-procedure validation
for seed 0, fairness repeats for seeds 1 and 2) on the full 51-component mean and on every subset mean, from the
per-component BPB the collector writes. Each proposal is paired with MARINER by trainer seed; the predicted levels
and gains come from `table9_snr_subset_optima_20260928/summary.json`.

Writes `subset_validation.csv` and `subset_validation_summary.json` to the launch's reference directory.

usage: uv run --offline --no-sync python analyze_table9_snr_subset_validation_20260928.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE = SCRIPT_DIR / "reference_outputs"
LAUNCH = REFERENCE / "delphi_table9_snr_subset_3e18_20260928"
OPTIMA = REFERENCE / "table9_snr_subset_optima_20260928"
MARINER_COMPONENTS = {
    0: (
        REFERENCE / "delphi_frozen_procedure_validation_3e18_20260908/measured_table9_components.csv",
        "lwspu_t9_snc_cap08",
    ),
    1: (REFERENCE / "delphi_fairness_repeats_3e18_20260908/measured_table9_components.csv", "lwspu_t9_snc_cap08@t1"),
    2: (REFERENCE / "delphi_fairness_repeats_3e18_20260908/measured_table9_components.csv", "lwspu_t9_snc_cap08@t2"),
}
POLICIES = {"table9_snr5": "snr5_t9_cap09", "table9_snr8": "snr8_t9_cap07"}
SUBSETS = {"full": 0.0, "table9_snr2": 2.0, "table9_snr5": 5.0, "table9_snr8": 8.0}
SEEDS = (0, 1, 2)
PREFIX, SUFFIX = "olmo_base_eval/easy_bpb/", "/bpb"


def component_name(key: str) -> str:
    """The collector's component name; the prediction table mixes full metric keys with bare names."""
    if key.startswith(PREFIX) and key.endswith(SUFFIX):
        return key[len(PREFIX) : -len(SUFFIX)]
    return key


def subset_members() -> dict[str, list[str]]:
    predictions = pd.read_csv(OPTIMA / "component_predictions.csv")
    names = predictions.component.map(component_name)
    members = {subset: names[predictions.snr >= floor].tolist() for subset, floor in SUBSETS.items()}
    expected = json.loads((OPTIMA / "summary.json").read_text())["subsets"]
    measured = set(pd.read_csv(LAUNCH / "measured_table9_components.csv").component)
    if set(members["full"]) != measured:
        raise ValueError(f"component names differ from the collector's: {set(members['full']) ^ measured}")
    for subset in SUBSETS:
        if len(members[subset]) != expected[subset]["components"]:
            raise ValueError(f"{subset}: {len(members[subset])} components, expected {expected[subset]['components']}")
    return members


def components(path: Path, candidate_id: str) -> pd.Series:
    table = pd.read_csv(path)
    rows = table[table.candidate_id.eq(candidate_id)].set_index("component")["bpb"]
    if len(rows) != 51 or rows.index.duplicated().any():
        raise ValueError(f"{candidate_id} in {path.parent.name}: {len(rows)} components")
    return rows.astype(float)


def run_components() -> dict[tuple[str, int], pd.Series]:
    runs = {("full", seed): components(path, candidate_id) for seed, (path, candidate_id) in MARINER_COMPONENTS.items()}
    measured = pd.read_csv(LAUNCH / "measured_results.csv")
    for policy, candidate_id in POLICIES.items():
        for seed in SEEDS:
            series = components(LAUNCH / "measured_table9_components.csv", f"{candidate_id}@t{seed}")
            logged = measured[measured.candidate_id.eq(candidate_id) & measured.trainer_seed.eq(seed)].table9_macro_bpb
            if len(logged) != 1 or abs(float(logged.iloc[0]) - series.mean()) > 1e-6:
                raise ValueError(f"{candidate_id}@t{seed}: component mean does not reproduce the logged aggregate")
            runs[(policy, seed)] = series
    return runs


def main() -> None:
    members = subset_members()
    runs = run_components()
    predicted = json.loads((OPTIMA / "summary.json").read_text())["subsets"]
    rows = []
    for policy in ("full", *POLICIES):
        for subset, names in members.items():
            levels = pd.Series({seed: float(runs[(policy, seed)].loc[names].mean()) for seed in SEEDS})
            reference = pd.Series({seed: float(runs[("full", seed)].loc[names].mean()) for seed in SEEDS})
            differences = levels - reference
            rows.append(
                {
                    "policy": policy,
                    "subset": subset,
                    "components": len(names),
                    "mean": levels.mean(),
                    "std": levels.std(ddof=1),
                    "seed0": levels[0],
                    "seed1": levels[1],
                    "seed2": levels[2],
                    "difference_vs_full_optimum_mean": differences.mean(),
                    "difference_vs_full_optimum_se": differences.std(ddof=1) / np.sqrt(len(SEEDS)),
                    "predicted": predicted[policy][f"predicted_{subset}_objective"],
                    "predicted_difference_vs_full_optimum": predicted[policy][f"predicted_{subset}_objective"]
                    - predicted["full"][f"predicted_{subset}_objective"],
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(LAUNCH / "subset_validation.csv", index=False)
    summary = {
        policy: {
            "own_subset_measured_gain": -float(
                table[table.policy.eq(policy) & table.subset.eq(policy)].difference_vs_full_optimum_mean.iloc[0]
            ),
            "own_subset_gain_se": float(
                table[table.policy.eq(policy) & table.subset.eq(policy)].difference_vs_full_optimum_se.iloc[0]
            ),
            "own_subset_predicted_gain": -float(
                table[table.policy.eq(policy) & table.subset.eq(policy)].predicted_difference_vs_full_optimum.iloc[0]
            ),
            "full_measured_cost": float(
                table[table.policy.eq(policy) & table.subset.eq("full")].difference_vs_full_optimum_mean.iloc[0]
            ),
            "full_cost_se": float(
                table[table.policy.eq(policy) & table.subset.eq("full")].difference_vs_full_optimum_se.iloc[0]
            ),
            "full_predicted_cost": float(
                table[table.policy.eq(policy) & table.subset.eq("full")].predicted_difference_vs_full_optimum.iloc[0]
            ),
        }
        for policy in POLICIES
    }
    (LAUNCH / "subset_validation_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
