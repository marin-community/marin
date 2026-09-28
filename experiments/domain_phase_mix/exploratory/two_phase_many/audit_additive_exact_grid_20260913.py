# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Check additive proposals with the archived exact integer-grid optimizer.

Run after audit_additive_proposals_20260913 with the repository environment:
uv run --no-sync python -m experiments.domain_phase_mix.exploratory.two_phase_many.audit_additive_exact_grid_20260913

The primary proposal remains the matched comparator SLSQP procedure. This audit
checks its objective against exact dynamic programming and separates historical
fit changes from historical caps. It does not submit or modify training.
"""

import argparse
import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import audit_additive_proposals_20260913 as audit
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    materialize_delphi_one_phase_weibull_softplus_epoch_cap_sweep_20260902 as historical,
)

comparator = audit.comparator


def predictor_from_fits(panel, target, fits, source):
    group = panel.group(target)
    components = []
    for index, fitted in enumerate(fits):
        metadata = historical.FitMetadata(
            target=target,
            component_position=index,
            component=str(group.components[index]),
            aggregation_weight=float(group.aggregation_weights[index]),
            source=source,
            train=np.arange(panel.rows),
            test=np.zeros(0, dtype=int),
            shape=fitted.shape,
            ridge=fitted.ridge,
            protocol_hash=source,
            expected_train=np.zeros(0),
            expected_test=np.zeros(0),
        )
        components.append(historical.ComponentFit(metadata, fitted.head, 0.0))
    return historical.AggregatePredictor(target, source, tuple(components), len(panel.buckets))


def old_fits(panel, target, table):
    group = panel.group(target)
    rows = table[(table.target == target) & (table.source == "full")].set_index("component")
    assert set(rows.index) == set(group.components)
    fits = []
    for index, component in enumerate(group.components):
        row = rows.loc[component]
        assert int(row.component_position) == index
        assert math.isclose(float(row.aggregation_weight), group.aggregation_weights[index], abs_tol=1e-14)
        coefficients = np.asarray(json.loads(row.benefit_amplitudes_json) + json.loads(row.harm_amplitudes_json))
        head = comparator.models.FittedHead(float(row.intercept), coefficients, 0.0, int(row.active_coefficients))
        fits.append(
            comparator.models.Fitted(
                shape={key: float(row[key]) for key in ("rate", "power", "threshold")},
                ridge=float(row.ridge),
                head=head,
                diagnostics={},
            )
        )
    return fits


def count_comparison(left, right):
    return {
        "exact_runtime_match": bool(np.array_equal(left, right)),
        "tv_distance": float(np.abs(left - right).sum() / (2 * comparator.ms.MIXTURE_BLOCK_SIZE)),
        "changed_buckets": int(np.count_nonzero(left != right)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=audit.DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir
    panel = comparator.bench.load_panel(comparator.PANEL)
    main_manifest = json.loads((output / "manifest.json").read_text())
    assert list(panel.buckets) == main_manifest["buckets"]
    assert panel.input_hashes == main_manifest["panel_input_hashes"]
    assert len(comparator.bench.calibration_rows(panel)) == 1
    archived_fits = pd.read_csv(audit.HISTORICAL / "component_fits.csv")
    archived_weights = pd.read_csv(audit.HISTORICAL / "candidate_weights.csv")
    archived_summary = pd.read_csv(audit.HISTORICAL / "candidate_summary.csv").set_index("candidate_id")
    for _, rows in archived_weights.groupby("candidate_id", sort=False):
        assert list(rows.domain) == list(panel.buckets)
    source_paths = [Path(__file__), Path(historical.__file__), output / "manifest.json"]
    source_paths.extend(
        audit.HISTORICAL / name for name in ("component_fits.csv", "candidate_weights.csv", "candidate_summary.csv")
    )
    manifest = {
        "primary_proposer": "matched SLSQP plus runtime exchange",
        "diagnostic_proposer": "exact separable integer DP",
        "source_hashes": {str(path): audit.digest(path) for path in source_paths},
    }
    manifest_path = output / "exact_grid_manifest.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest
    else:
        audit.write_json(manifest_path, manifest)
    for target in comparator.TARGET_LABELS:
        result_path = output / f"{target}_exact_grid.json"
        if result_path.exists():
            print(f"{target}: exact-grid audit already complete", flush=True)
            continue
        if not (output / f"{target}_summary.json").exists():
            print(f"{target}: primary refit still incomplete; rerun after completion", flush=True)
            continue
        group = panel.group(target)
        fits = []
        for index in range(len(group.components)):
            with (output / "fits" / target / f"{index:02d}.pickle").open("rb") as handle:
                fits.append(pickle.load(handle))
        previous_fits = old_fits(panel, target, archived_fits)
        current = predictor_from_fits(panel, target, fits, "current_pinned")
        previous = predictor_from_fits(panel, target, previous_fits, "historical")
        surrogate = comparator.ObservatorySurrogate(panel, audit.MODEL_ID, target, fits)
        direct = surrogate.predict(panel.features.weights)
        reproduced = current.predict(panel.features.exposures)
        assert np.allclose(direct, reproduced, atol=1e-12, rtol=0.0)
        original_predictions = previous.predict(panel.features.exposures)
        fit_rows = []
        for component, fitted, old in zip(group.components, fits, previous_fits, strict=True):
            fit_rows.append(
                {
                    "component": component,
                    "same_shape": fitted.shape == old.shape,
                    "same_ridge": fitted.ridge == old.ridge,
                    "new_shape": json.dumps(fitted.shape, sort_keys=True),
                    "old_shape": json.dumps(old.shape, sort_keys=True),
                    "new_ridge": fitted.ridge,
                    "old_ridge": old.ridge,
                    "max_abs_amplitude_change": float(np.max(np.abs(fitted.head.coefficients - old.head.coefficients))),
                }
            )
        pd.DataFrame(fit_rows).to_csv(output / f"{target}_historical_fit_comparison.csv", index=False)
        cap = math.ceil(float(panel.features.inventory.max()))
        exact = historical.exact_runtime_optimum(current, panel.features.inventory, cap)
        old_exact = historical.exact_runtime_optimum(previous, panel.features.inventory, cap)
        primary = pd.read_csv(output / f"{target}_weights.csv").set_index("bucket").loc[list(panel.buckets)]
        primary_counts = primary.runtime_count.to_numpy(int)
        primary_prediction = float(
            current.predict((panel.features.inventory * primary_counts / comparator.ms.MIXTURE_BLOCK_SIZE)[None])[0]
        )
        assert exact.prediction <= primary_prediction + 1e-10
        rows = []
        for candidate_id, saved in archived_weights[archived_weights.target == target].groupby("candidate_id"):
            counts = saved.set_index("domain").loc[list(panel.buckets), "runtime_count"].to_numpy(int)
            old_prediction = float(
                previous.predict((counts * panel.features.inventory / comparator.ms.MIXTURE_BLOCK_SIZE)[None])[0]
            )
            recorded = float(archived_summary.loc[str(candidate_id), "runtime_predicted_bpb"])
            assert math.isclose(old_prediction, recorded, rel_tol=0.0, abs_tol=2e-10)
            rows.append(
                {
                    "candidate_id": candidate_id,
                    **count_comparison(exact.counts, counts),
                    "old_prediction_reproduction_error": abs(old_prediction - recorded),
                }
            )
        original_cap = 6 if target == "uncheatable" else 8
        old_capped = historical.exact_runtime_optimum(previous, panel.features.inventory, original_cap)
        selected_id = f"wspu_{target}_cap{original_cap:02d}"
        selected = (
            archived_weights[archived_weights.candidate_id == selected_id].set_index("domain").loc[list(panel.buckets)]
        )
        assert np.array_equal(old_capped.counts, selected.runtime_count.to_numpy(int))
        pd.DataFrame(
            {
                "bucket": panel.buckets,
                "primary_count": primary_counts,
                "exact_count": exact.counts,
                "historical_uncapped_exact_count": old_exact.counts,
            }
        ).to_csv(output / f"{target}_exact_grid_weights.csv", index=False)
        result = {
            "target": target,
            "uncapped_equivalent_cap": cap,
            "current_exact_vs_primary": count_comparison(exact.counts, primary_counts),
            "current_exact_prediction": exact.prediction,
            "primary_prediction": primary_prediction,
            "primary_minus_exact_predicted_loss": primary_prediction - exact.prediction,
            "current_exact_vs_historical_uncapped": count_comparison(exact.counts, old_exact.counts),
            "historical_uncapped_vs_original_capped": count_comparison(old_exact.counts, old_capped.counts),
            "historical_selected_policy": selected_id,
            "historical_selected_policy_exact_reproduction": True,
            "changed_shape_tasks": sum(not row["same_shape"] for row in fit_rows),
            "changed_ridge_tasks": sum(not row["same_ridge"] for row in fit_rows),
            "swarm_prediction_rmse_change": float(np.sqrt(np.mean((reproduced - original_predictions) ** 2))),
            "swarm_prediction_max_abs_change": float(np.max(np.abs(reproduced - original_predictions))),
            "historical_comparisons": rows,
            "fit_cache_hashes": {
                str(output / "fits" / target / f"{index:02d}.pickle"): audit.digest(
                    output / "fits" / target / f"{index:02d}.pickle"
                )
                for index in range(len(fits))
            },
        }
        audit.write_json(result_path, result)
        print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
