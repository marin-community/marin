# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.3.5", "pandas==2.2.2", "scipy==1.17.0"]
# ///
"""Audit frozen component-floor feasibility before any new phase-link fit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import fit_two_phase_link_transfer_20260907 as previous
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "reference_outputs/two_phase_link_transfer_20260907"
OUTPUT = HERE / "reference_outputs/two_phase_refinement_20260907/floor_feasibility"
CONTEXTS = tuple(
    prefix for context in previous.CONTEXTS for prefix in (context, *(f"{context}_inner{i}" for i in range(3)))
)


def summarize_population(
    panel: dict[str, np.ndarray],
    objective: str,
    context: str,
    population: str,
    asymmetric: np.ndarray,
    tied: np.ndarray,
    floors: np.ndarray,
    deficits: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    response = panel[f"{objective}_outcomes"]
    records, violations = [], []
    for component, floor in enumerate(floors):
        ya, yt = response[asymmetric, component], response[tied, component]
        qa = deficits[asymmetric, component]
        delta = ya - yt
        numerator, denominator = ya - floor, yt - floor
        shifted_delta = qa + delta
        literal_bad = (numerator <= 0) | (denominator <= 0)
        attainable_bad = shifted_delta <= 0
        positive_observed = np.concatenate([numerator[numerator > 0], denominator[denominator > 0]])
        name = str(panel[f"{objective}_components"][component])
        offset = 0 if objective == "uncheatable" else len(panel["uncheatable_components"])
        sd = float(panel["anchor_repeat_sd"][offset + component])
        row = {
            "objective": objective,
            "context": context,
            "population": population,
            "component": component,
            "component_name": name,
            "floor": float(floor),
            "pairs": len(ya),
            "asymmetric_at_or_below_floor": int(np.sum(numerator <= 0)),
            "tied_at_or_below_floor": int(np.sum(denominator <= 0)),
            "literal_log_ratio_invalid_pairs": int(literal_bad.sum()),
            "aggregate_deficit_shift_invalid_pairs": int(attainable_bad.sum()),
            "asymmetric_max_floor_excess": float(np.max(np.maximum(-numerator, 0), initial=0)),
            "tied_max_floor_excess": float(np.max(np.maximum(-denominator, 0), initial=0)),
            "maximum_unattainable_delta_excess": float(np.max(np.maximum(-shifted_delta, 0), initial=0)),
            "repeat_sd": sd,
            "repeat_sd_is_aggregate_approximation": bool(
                panel["anchor_sd_is_aggregate_approximation"][offset + component]
            ),
            "minimum_positive_observed_deficit": float(positive_observed.min()) if len(positive_observed) else None,
            "minimum_positive_deficit_over_repeat_sd": (
                float(positive_observed.min() / sd) if len(positive_observed) and sd > 0 else None
            ),
            "component_irreducible_pair_rmse_from_range": (
                float(np.sqrt(np.mean(np.maximum(-shifted_delta, 0) ** 2))) if len(ya) else None
            ),
        }
        records.append(row)
        for index in np.flatnonzero(literal_bad | attainable_bad):
            violations.append(
                {
                    "objective": objective,
                    "context": context,
                    "population": population,
                    "component": component,
                    "component_name": name,
                    "asymmetric_row": int(asymmetric[index]),
                    "tied_row": int(tied[index]),
                    "group": str(panel["groups"][asymmetric[index]]),
                    "asymmetric_run": str(panel["runs"][asymmetric[index]]),
                    "tied_run": str(panel["runs"][tied[index]]),
                    "asymmetric_observed": float(ya[index]),
                    "tied_observed": float(yt[index]),
                    "floor": float(floor),
                    "predicted_aggregate_deficit": float(qa[index]),
                    "asymmetric_observed_deficit": float(numerator[index]),
                    "tied_observed_deficit": float(denominator[index]),
                    "measured_delta": float(delta[index]),
                    "shifted_delta": float(shifted_delta[index]),
                    "literal_log_ratio_invalid": bool(literal_bad[index]),
                    "direct_pair_range_invalid": bool(attainable_bad[index]),
                }
            )
    return records, violations


def main() -> None:
    module, panel, splits = previous.inputs(str(SOURCE))
    source_paths = [
        Path(__file__),
        SOURCE / "inputs/panel.npz",
        SOURCE / "inputs/splits.npz",
        SOURCE / "inputs/single_phase.py",
        Path(previous.__file__),
    ]
    records, violation_records, endpoint_records, objective_records = [], [], [], []
    for objective in previous.OBJECTIVES:
        n_components = len(panel[f"{objective}_components"])
        for context in CONTEXTS:
            rows = np.arange(len(panel["runs"])) if context == "final" else splits[f"{context}_train"]
            train_a, train_t = previous.selected_pairs(panel, rows)
            assert panel["calibration_mask"][rows].sum() == 2
            test_rows = np.array([], dtype=int) if context == "final" else splits[f"{context}_test"]
            test_a, test_t = previous.selected_pairs(panel, test_rows)
            floors, deficits = [], []
            for component in range(n_components):
                path = SOURCE / "spines" / context / f"{objective}_c{component}.json"
                source_paths.append(path)
                spine = previous.load_spine(SOURCE, context, objective, component)
                floors.append(spine.head.floor)
                design = module.design_matrix(panel["epochs"], spine.shape)
                deficits.append(np.exp(np.clip(spine.head.intercept + design @ spine.head.coefficients, -30, 30)))
                response = panel[f"{objective}_outcomes"][:, component]
                for population, subset in (("train_all", rows), ("test_all", test_rows)):
                    for kind, mask in (
                        ("tied", panel["physical_tied"][subset]),
                        ("asymmetric", ~panel["physical_tied"][subset]),
                    ):
                        eligible = subset[mask]
                        gap = spine.head.floor - response[eligible]
                        endpoint_records.append(
                            {
                                "objective": objective,
                                "context": context,
                                "population": population,
                                "kind": kind,
                                "component": component,
                                "component_name": str(panel[f"{objective}_components"][component]),
                                "n": len(eligible),
                                "floor": float(spine.head.floor),
                                "at_or_below_floor": int(np.sum(gap >= 0)),
                                "max_floor_excess": float(np.max(gap, initial=0)),
                            }
                        )
            floor_array = np.asarray(floors)
            deficit_matrix = np.column_stack(deficits)
            for population, aa, tt in (("train", train_a, train_t), ("test", test_a, test_t)):
                component_rows, violations = summarize_population(
                    panel, objective, context, population, aa, tt, floor_array, deficit_matrix
                )
                records.extend(component_rows)
                violation_records.extend(violations)
                agg_weights = panel[f"{objective}_aggregation_weights"]
                agg_delta = panel[f"{objective}_aggregate"][aa] - panel[f"{objective}_aggregate"][tt]
                aggregate_q = deficit_matrix[aa] @ agg_weights
                objective_records.append(
                    {
                        "objective": objective,
                        "context": context,
                        "population": population,
                        "pairs": len(aa),
                        "aggregate_deficit_shift_invalid_pairs": int(np.sum(aggregate_q + agg_delta <= 0)),
                        "aggregate_maximum_unattainable_delta_excess": float(
                            np.max(np.maximum(-aggregate_q - agg_delta, 0), initial=0)
                        ),
                    }
                )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    components = pd.DataFrame(records)
    violations = pd.DataFrame(violation_records)
    endpoints = pd.DataFrame(endpoint_records)
    components.to_csv(OUTPUT / "component_contexts.csv", index=False)
    violations.to_csv(OUTPUT / "violations.csv", index=False)
    endpoints.to_csv(OUTPUT / "endpoint_contexts.csv", index=False)
    pd.DataFrame(objective_records).to_csv(OUTPUT / "objective_range.csv", index=False)
    totals = (
        components.groupby(["objective", "population"])[
            [
                "pairs",
                "asymmetric_at_or_below_floor",
                "tied_at_or_below_floor",
                "literal_log_ratio_invalid_pairs",
                "aggregate_deficit_shift_invalid_pairs",
            ]
        ]
        .sum()
        .reset_index()
    )
    totals.to_csv(OUTPUT / "summary_counts.csv", index=False)
    summary = {
        "contexts": list(CONTEXTS),
        "component_floors_inspected": 16 * 58,
        "scope": (
            "Only frozen structured520 source outcomes and component spines inspected; "
            "no model fitting, floor adjustment, clipping, dropping, LM jobs or sealed/3e18 outcome reads."
        ),
        "counts_unit": (
            "Component-pair-context incidences; repeated appearances across nested contexts "
            "are not independent measurements."
        ),
        "count_summary": totals.to_dict(orient="records"),
        "sources": {str(path): previous.file_hash(path) for path in source_paths},
    }
    previous.write_json(OUTPUT / "summary.json", summary)
    print(totals.to_string(index=False))
    print(
        components[(components.context == "final") & (components.population == "train")]
        .groupby("objective")[["pairs", "literal_log_ratio_invalid_pairs", "aggregate_deficit_shift_invalid_pairs"]]
        .sum()
        .to_string()
    )


if __name__ == "__main__":
    main()
