# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Did optimizing OlmoBaseEval Easy's question-answering components pay off when trained?

Scores the six trained QA optima (`delphi_table9_qa_subset_3e18_20260929`: the 15 QA components and QA plus MMLU,
data seed 662009, trainer seeds 0-2) on the full mean and on every component group, from the collector's per-component
BPB, against MARINER's full-objective runs and Olmix's matched OlmoBaseEval Easy runs at the same trainer seeds and
against the ten proportional repeats. Predicted changes come from `table9_qa_subset_optimum_20260929`.

Writes `group_validation.csv`, `component_validation.csv` and `validation_summary.json` to the launch's directory.

usage: uv run --offline --no-sync python analyze_table9_qa_subset_validation_20260929.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plot_component_tradeoffs_obe_20260929 as figure

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE = SCRIPT_DIR / "reference_outputs"
LAUNCH = REFERENCE / "delphi_table9_qa_subset_3e18_20260929"
OPTIMA = REFERENCE / "table9_qa_subset_optimum_20260929"
SEEDS = (0, 1, 2)
POLICIES = {"qa15_t9_cap04": "table9_qa", "qa19_t9_cap05": "table9_qa_mmlu"}
GROUPS = figure.GROUPS


def runs(path: Path, ids: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame([figure.components(path, cid) for cid in ids], index=list(SEEDS))


def main() -> None:
    mariner = pd.DataFrame([figure.components(path, cid) for path, cid in figure.MARINER_T9], index=list(SEEDS))
    olmix = pd.DataFrame([figure.components(path, cid) for path, cid in figure.OLMIX_T9], index=list(SEEDS))
    proportional = pd.read_csv(figure.PROPORTIONAL_MATRIX, index_col=0)
    proportional.columns = [c.split("/")[2] for c in proportional.columns]
    predictions = pd.read_csv(OPTIMA / "component_predictions.csv").set_index("component")
    measured = pd.read_csv(LAUNCH / "measured_results.csv")
    group_rows, component_rows = [], []
    for candidate_id, column in POLICIES.items():
        policy = runs(LAUNCH / "measured_table9_components.csv", tuple(f"{candidate_id}@t{s}" for s in SEEDS))
        logged = measured[measured.candidate_id.eq(candidate_id)].set_index("trainer_seed").table9_macro_bpb
        if (np.abs(policy.mean(axis=1) - logged.loc[list(SEEDS)]) > 1e-6).any():
            raise ValueError(f"{candidate_id}: component means do not reproduce the logged aggregates")
        groups = pd.DataFrame([figure.group_means(policy.loc[s]) for s in SEEDS], index=list(SEEDS))
        references = {
            "mariner": pd.DataFrame([figure.group_means(mariner.loc[s]) for s in SEEDS], index=list(SEEDS)),
            "olmix": pd.DataFrame([figure.group_means(olmix.loc[s]) for s in SEEDS], index=list(SEEDS)),
        }
        baseline = figure.group_means(proportional.mean(axis=0))
        predicted = figure.group_means(predictions[column]) - figure.group_means(predictions["full"])
        for group in GROUPS:
            row = {
                "policy": candidate_id,
                "group": group,
                "mean": groups[group].mean(),
                "std": groups[group].std(ddof=1),
                "vs_proportional": groups[group].mean() - baseline[group],
                "predicted_vs_mariner": predicted[group],
            }
            for name, reference in references.items():
                d = groups[group] - reference[group]
                row[f"vs_{name}"] = d.mean()
                row[f"vs_{name}_se"] = d.std(ddof=1) / np.sqrt(len(d))
            group_rows.append(row)
        for component in policy.columns:
            component_rows.append(
                {
                    "policy": candidate_id,
                    "component": component,
                    "group": figure.group_of(component),
                    "mean": policy[component].mean(),
                    "vs_mariner": (policy[component] - mariner[component]).mean(),
                    "vs_olmix": (policy[component] - olmix[component]).mean(),
                    "predicted_vs_mariner": float(
                        predictions.loc[component, column] - predictions.loc[component, "full"]
                    ),
                }
            )
    groups_table = pd.DataFrame(group_rows)
    groups_table.to_csv(LAUNCH / "group_validation.csv", index=False)
    components_table = pd.DataFrame(component_rows)
    components_table.to_csv(LAUNCH / "component_validation.csv", index=False)
    summary = {
        candidate_id: {
            group: {
                k: float(v)
                for k, v in groups_table[groups_table.policy.eq(candidate_id) & groups_table.group.eq(group)]
                .iloc[0]
                .drop(["policy", "group"])
                .items()
            }
            for group in GROUPS
        }
        for candidate_id in POLICIES
    }
    for candidate_id in POLICIES:
        qa = components_table[components_table.policy.eq(candidate_id) & components_table.group.eq("QA")]
        summary[candidate_id]["qa_components_better_than_mariner"] = int((qa.vs_mariner < 0).sum())
        summary[candidate_id]["qa_components_better_than_olmix"] = int((qa.vs_olmix < 0).sum())
    (LAUNCH / "validation_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(groups_table.round(4).to_string(index=False))
    print(components_table[components_table.group.isin(["QA", "MMLU"])].round(4).to_string(index=False))
    print(
        json.dumps({k: {kk: vv for kk, vv in v.items() if kk.startswith("qa_components")} for k, v in summary.items()})
    )


if __name__ == "__main__":
    main()
