# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["numpy==2.3.5", "pandas==2.2.2"]
# ///
"""Independently reconstruct the reported local transfer metrics and exclusions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
OUTPUT = BASE / "reference_outputs/two_phase_mariner_transfer_20260912"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def crossed_check() -> dict[str, float | int]:
    predictions = pd.read_csv(OUTPUT / "crossed/predictions.csv")
    reported = pd.read_csv(OUTPUT / "crossed/metrics.csv")
    prepared = json.loads((OUTPUT / "crossed/prepared.json").read_text())
    source = pd.read_csv(BASE / "reference_outputs/fixed_checkpoint_branch_wspu_20260907/data/rows.csv")
    errors = []
    comparisons = 0
    for _, metric in reported.iterrows():
        menu = predictions[
            predictions.model.eq(metric.model)
            & predictions.fold.eq(metric.fold)
            & predictions.state_id.eq(metric.state_id)
        ]
        if not metric.with_tied:
            menu = menu[~menu.is_tied]
        assert len(menu) == metric.rows
        assert np.array_equal(source.iloc[menu.row.to_numpy(int)].row_id.to_numpy(), menu.row_id.to_numpy())
        assert np.allclose(source.iloc[menu.row.to_numpy(int)].target, menu.target, rtol=0, atol=1e-14)
        observed, predicted = menu.target.to_numpy(), menu.prediction.to_numpy()
        residual = observed - predicted
        ranking = np.argsort(predicted, kind="stable")
        # Identity for the mean squared difference over distinct unordered pairs.
        pair_rmse = np.sqrt(2 * len(menu) / (len(menu) - 1) * np.var(residual))
        reconstructed = {
            "rmse": np.sqrt(np.mean(residual**2)),
            "mean_optimism": residual.mean(),
            "action_pair_rmse": pair_rmse,
            "top1_regret": observed[ranking[0]] - observed.min(),
            "top3_regret": observed[ranking[:3]].min() - observed.min(),
            "selected_optimism": residual[ranking[0]],
        }
        errors.extend(abs(value - float(metric[key])) for key, value in reconstructed.items())
        assert str(menu.action_id.iloc[ranking[0]]) == str(metric.selected_action)
        comparisons += 1
    assert max(errors) < 1e-12
    inner_count = 0
    for held, context in prepared["contexts"].items():
        train, test = source.iloc[context["train"]], source.iloc[context["test"]]
        if held != "seen_prefix":
            assert set(train.state_id).isdisjoint(test.state_id)
            assert set(train.coordinate_hash).isdisjoint(test.coordinate_hash)
        assert set(train.coordinate_hash).isdisjoint(test.loc[~test.is_tied_control, "coordinate_hash"])
        for inner in context["inner_folds"]:
            inner_train, inner_test = train.iloc[inner["train"]], train.iloc[inner["validation"]]
            assert set(inner_train.row_id).isdisjoint(inner_test.row_id)
            assert set(inner_train.coordinate_hash).isdisjoint(inner_test.coordinate_hash)
            if held != "seen_prefix":
                assert set(inner_train.state_id).isdisjoint(inner_test.state_id)
            inner_count += 1
    return {"metric_menus": comparisons, "max_metric_error": max(errors), "partition_pairs": inner_count}


def boundary_check() -> dict[str, float | int]:
    predictions = pd.read_csv(OUTPUT / "boundary/predictions.csv")
    reported = pd.read_csv(OUTPUT / "boundary/metrics.csv")
    audit = json.loads((OUTPUT / "boundary/input_audit.json").read_text())
    splits = json.loads((OUTPUT / "boundary/split_membership.json").read_text())
    joined = pd.read_csv(OUTPUT / "boundary/joined_rows.csv")
    errors = []
    for model, group in predictions.groupby("model"):
        macro = group[group.target.eq("uncheatable")].set_index("row")
        assert len(macro) == 279 and macro.index.is_unique
        for column in ("observed", "prediction", "observed_tied", "predicted_tied", "observed_gain", "predicted_gain"):
            components = group[~group.target.eq("uncheatable")].pivot(index="row", columns="target", values=column)
            reconstruction = components[audit["components"]].to_numpy() @ np.asarray(audit["aggregation_weights"])
            errors.extend(np.abs(reconstruction - macro.loc[components.index, column]).tolist())
        assert np.max(np.abs(macro.predicted_gain[macro.physical_tied])) < 1e-12
        for fold in range(5):
            actual = set(macro.index[macro.fold.eq(fold)])
            assert actual == set(splits[f"outer{fold}_test"])
        asymmetric = macro[~macro.physical_tied]
        assert len(asymmetric) == 238
        gain = asymmetric.observed_gain.to_numpy()
        predicted_gain = asymmetric.predicted_gain.to_numpy()
        pair_regret = np.where(predicted_gain < 0, np.maximum(gain, 0), np.maximum(-gain, 0)).mean()
        rmse = np.sqrt(np.mean((gain - predicted_gain) ** 2))
        rows = reported[
            reported.model.eq(model)
            & reported.target.eq("uncheatable")
            & reported.fold.eq("oof")
            & reported.population.eq("asymmetric")
        ]
        errors.append(abs(rmse - float(rows.loc[rows.kind.eq("gain"), "rmse"].iloc[0])))
        errors.append(abs(pair_regret - float(rows.loc[rows.kind.eq("selection"), "matched_decision_regret"].iloc[0])))
    partition_count = 0
    for name, train in splits.items():
        if not name.endswith("_train"):
            continue
        test = splits[name.removesuffix("_train") + "_test"]
        assert set(train).isdisjoint(test)
        assert set(joined.iloc[train].group_id).isdisjoint(joined.iloc[test].group_id)
        assert not joined.iloc[test].calibration.any()
        if "_inner" in name:
            outer = name.split("_inner")[0]
            assert set(train + test).issubset(splits[f"{outer}_train"])
        if "_boundary" in name:
            parent = name.split("_boundary")[0]
            assert set(train + test).issubset(splits[f"{parent}_train"])
        partition_count += 1
    assert max(errors) < 1e-12, max(errors)
    return {
        "scored_groups": 279,
        "asymmetric_groups": 238,
        "partition_pairs": partition_count,
        "max_metric_error": max(errors),
    }


def anchored_check() -> dict[str, float | int]:
    predictions = pd.read_csv(OUTPUT / "anchored/finite_predictions.csv")
    reported = pd.read_csv(OUTPUT / "anchored/finite_metrics.csv")
    source = pd.read_csv(BASE / "reference_outputs/fixed_checkpoint_branch_wspu_20260907/data/rows.csv")
    errors = []
    for _, metric in reported.iterrows():
        menu = predictions[predictions.model.eq(metric.model) & predictions.state_id.eq(metric.state_id)]
        if not metric.tied_selectable:
            menu = menu[~menu.is_tied_control]
        assert len(menu) == metric.n
        observed_source = source.set_index("row_id").loc[menu.row_id, "target"].to_numpy()
        assert np.max(np.abs(observed_source - menu.target.to_numpy())) < 1e-12
        observed, predicted = menu.target.to_numpy(), menu.prediction.to_numpy()
        residual = observed - predicted
        ranking = np.argsort(predicted, kind="stable")
        reconstructed = {
            "rmse": np.sqrt(np.mean(residual**2)),
            "pair_rmse": np.sqrt(2 * len(menu) / (len(menu) - 1) * np.var(residual)),
            "regret": observed[ranking[0]] - observed.min(),
            "shortlist3_regret": observed[ranking[:3]].min() - observed.min(),
            "selected_optimism": residual[ranking[0]],
        }
        errors.extend(abs(value - float(metric[key])) for key, value in reconstructed.items())
        assert str(menu.action_id.iloc[ranking[0]]) == str(metric.selected_action)
    for filename in ("finite_composite_terms.csv", "raw_metrics.csv"):
        terms = pd.read_csv(OUTPUT / "anchored" / filename)
        correction = terms.w_policy_prediction - terms.w_tied_aggregate_prediction
        errors.extend(np.abs(correction - terms.phase_correction_bpb).tolist())
        errors.extend(np.abs(terms.anchor_prediction + correction - terms.prediction).tolist())
    weights = pd.read_csv(OUTPUT / "anchored/raw_weights.csv").iloc[:, 2:].to_numpy()
    assert weights.shape == (36, 39)
    assert weights.min() >= -1e-8 and np.max(np.abs(weights.sum(axis=1) - 1)) < 1e-8
    assert max(errors) < 1e-12
    return {"metric_menus": len(reported), "raw_mixtures": len(weights), "max_metric_error": max(errors)}


def main() -> None:
    result: dict[str, dict[str, float | int] | dict[str, str]] = {
        "crossed": crossed_check(),
        "boundary": boundary_check(),
        "anchored": anchored_check(),
    }
    inputs = [
        Path(__file__),
        OUTPUT / "crossed/predictions.csv",
        OUTPUT / "crossed/metrics.csv",
        OUTPUT / "crossed/prepared.json",
        OUTPUT / "boundary/predictions.csv",
        OUTPUT / "boundary/metrics.csv",
        OUTPUT / "boundary/split_membership.json",
        OUTPUT / "boundary/input_audit.json",
        OUTPUT / "boundary/joined_rows.csv",
        OUTPUT / "anchored/finite_predictions.csv",
        OUTPUT / "anchored/finite_metrics.csv",
        OUTPUT / "anchored/finite_composite_terms.csv",
        OUTPUT / "anchored/raw_metrics.csv",
        OUTPUT / "anchored/raw_weights.csv",
    ]
    result["input_hashes"] = {str(path): sha(path) for path in inputs}
    (OUTPUT / "independent_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "input_hashes"}, indent=2))


if __name__ == "__main__":
    main()
