# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "numpy==2.3.5", "pandas==2.2.2", "scipy==1.17.0", "scikit-learn==1.7.2",
#   "cvxpy==1.7.5", "fsspec==2026.1.0", "gcsfs==2026.1.0", "plotly==6.5.1",
#   "tabulate==0.9.0", "threadpoolctl==3.6.0", "matplotlib==3.10.8",
# ]
# ///
"""Compare HPR refitting across Delphi horizons on identical held-out policies."""

from __future__ import annotations

import json
import pickle
from dataclasses import replace
from pathlib import Path

import analyze_tpp40_frontier_gap_20260909 as previous
import numpy as np
import pandas as pd
from fit_two_phase_link_spines_20260907 import write_json_atomic
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

BASE = Path(__file__).resolve().parent
REFERENCE = BASE / "reference_outputs"
OUTPUT = REFERENCE / "delphi_horizon_transfer_20260910"
TPP = REFERENCE / "tpp40_frontier_gap_20260909"
DELPHI = REFERENCE / "two_phase_surrogate_collaborator_packet_20260721/data/canonical/delphi_3e18_two_phase_fit.csv"
CONTEXTS = ("full", "outer0", "outer1", "outer2")


def metrics(frame: pd.DataFrame) -> dict:
    y, p = frame.measured.to_numpy(), frame.prediction.to_numpy()
    residual = y - p
    rmse = np.sqrt(np.mean(residual**2))
    sd = np.std(y)
    return dict(
        n=len(frame),
        rmse=rmse,
        optimism=residual.mean(),
        predictive_r2=1 - np.sum(residual**2) / np.sum((y - y.mean()) ** 2),
        response_sd=sd,
        normalized_rmse=rmse / sd,
        spearman=spearmanr(y, p).statistic,
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    snapshot = json.loads((TPP / "snapshot.json").read_text())
    for name, digest in snapshot["sha256"].items():
        assert previous.sha(TPP / name) == digest, f"TPP40 snapshot changed: {name}"
    dataset, frame, _, calibration, inventory = previous.data()
    shorter = pd.read_csv(DELPHI).iloc[frame.order].reset_index(drop=True)
    assert list(shorter.row_id) == list(frame.source_run)
    weights = np.stack(
        [shorter[[f"phase_{p}_weight::{b}" for b in dataset.domains]].to_numpy() for p in range(2)], axis=1
    )
    parity = float(np.max(abs(weights - dataset.weights)))
    assert parity < 1e-12
    source_paths = [
        Path(__file__),
        OUTPUT / "PROTOCOL.md",
        DELPHI,
        TPP / "snapshot.json",
        Path(previous.__file__),
        Path(previous.HPR.__file__),
        Path(previous.HPR.family_grp.__file__),
        Path(previous.controls.baseline.observatory.__file__),
        previous.OLD / "controls/hierarchical_phase_replay/uncheatable/full/model.pkl",
    ]
    source_paths.extend(TPP / "inputs" / n for n in ("outcomes.csv", "run_specs.json", "buckets.csv", "objectives.csv"))
    for context in CONTEXTS:
        record = json.loads((TPP / "fits" / context / "complete.json").read_text())
        for name, digest in record["identity"].items():
            assert previous.sha(Path(name)) == digest, f"TPP40 fit dependency changed: {name}"
        for name, digest in record["sha256"].items():
            assert previous.sha(TPP / "fits" / context / name) == digest
        source_paths.extend(
            [TPP / "fits" / context / n for n in ("complete.json", "selection.json", "model.pkl", "predictions.csv")]
        )
    identity = {str(p): previous.sha(p) for p in source_paths}
    frozen = OUTPUT / "input_identity.json"
    if frozen.exists():
        assert json.loads(frozen.read_text()) == identity
    else:
        write_json_atomic(frozen, identity)
        shorter.to_csv(OUTPUT / "matched_delphi_rows.csv", index=False)
        frame.to_csv(OUTPUT / "matched_tpp40_rows.csv", index=False)
    short_dataset = replace(
        dataset, target=shorter.uncheatable_bpb.to_numpy(), c0=(2400 / 3007) * inventory, c1=(607 / 3007) * inventory
    )
    predictions, selections = [], []
    max_prediction_parity = 0.0
    for context in CONTEXTS:
        record = json.loads((TPP / "fits" / context / "complete.json").read_text())
        train = np.array(record["train_rows"], dtype=int)
        held = np.flatnonzero(~np.isin(np.arange(dataset.n), train))
        assert np.all(np.isin(np.flatnonzero(calibration), train))
        labels = np.array(json.loads((TPP / "fits" / context / "selection.json").read_text())["inner_labels"])
        local = replace(
            short_dataset,
            frame=frame.iloc[train].reset_index(drop=True),
            target=short_dataset.target[train],
            weights=dataset.weights[train],
        )
        destination = OUTPUT / "fits" / context
        destination.mkdir(parents=True, exist_ok=True)
        marker = destination / "complete.json"
        if marker.exists():
            old = json.loads(marker.read_text())
            assert old["identity"] == identity
            for name, digest in old["hashes"].items():
                assert previous.sha(destination / name) == digest
            with (destination / "model.pkl").open("rb") as handle:
                fitted = pickle.load(handle)
        else:
            fitted, selection = previous.fitted_hpr(local, labels)
            (destination / "model.pkl").write_bytes(pickle.dumps(fitted))
            write_json_atomic(destination / "selection.json", selection)
            write_json_atomic(
                marker,
                dict(
                    identity=identity,
                    train_rows=train.tolist(),
                    held_rows=held.tolist(),
                    hashes={p.name: previous.sha(p) for p in destination.iterdir() if p.is_file()},
                ),
            )
        with (TPP / "fits" / context / "model.pkl").open("rb") as handle:
            long_fit = pickle.load(handle)
        saved = pd.read_csv(TPP / "fits" / context / "predictions.csv")
        error = np.max(abs(long_fit.predict(dataset.weights) - saved.prediction.to_numpy()))
        max_prediction_parity = max(max_prediction_parity, float(error))
        assert error < 1e-12
        for horizon, model, response in [
            ("Delphi 3e18", fitted, short_dataset.target),
            ("Delphi TPP40", long_fit, dataset.target),
        ]:
            table = frame[["order", "run_name", "region", "tied"]].copy()
            table["measured"] = response
            table["prediction"] = model.predict(dataset.weights)
            table["context"], table["horizon"] = context, horizon
            table["in_train"] = np.isin(np.arange(dataset.n), train)
            table["frontier"] = False
            test = table[~table.in_train & ~table.tied]
            if not test.empty:
                frontier = test.nsmallest(int(np.ceil(0.2 * len(test))), "prediction")
                table.loc[frontier.index, "frontier"] = True
                selected = test.sort_values(["prediction", "order"]).iloc[0]
                selections.append(
                    dict(
                        horizon=horizon,
                        context=context,
                        selected_order=int(selected.order),
                        observed=selected.measured,
                        predicted=selected.prediction,
                        optimism=selected.measured - selected.prediction,
                        regret=selected.measured - test.measured.min(),
                    )
                )
            predictions.append(table)
    full = pd.concat(predictions, ignore_index=True)
    full.to_csv(OUTPUT / "predictions.csv", index=False)
    pd.DataFrame(selections).to_csv(OUTPUT / "fold_selections.csv", index=False)
    oof = full[full.context.ne("full") & ~full.in_train & ~full.tied]
    records = []
    for horizon, group in oof.groupby("horizon"):
        assert len(group) == 156 and group.order.nunique() == 156
        records.append(dict(horizon=horizon, population="all_asymmetric_oof", **metrics(group)))
        for selector, points in oof[oof.frontier].groupby("horizon"):
            selected = group[group.order.isin(points.order)]
            records.append(dict(horizon=horizon, population=f"frontier_selected_by_{selector}", **metrics(selected)))
    summary = pd.DataFrame(records)
    summary.to_csv(OUTPUT / "metrics.csv", index=False)
    write_json_atomic(
        OUTPUT / "CHECKS.json",
        dict(
            phase_weight_parity=parity,
            tpp_prediction_parity=max_prediction_parity,
            matching_policies=dataset.n,
            asymmetric_oof=156,
            calibration_rows=int(calibration.sum()),
            matched_outer_and_inner_folds=True,
            verified_tpp_snapshot_files=len(snapshot["sha256"]),
            verified_tpp_fit_contexts=len(CONTEXTS),
        ),
    )
    print(summary.round(6).to_string(index=False))
    print(pd.DataFrame(selections).round(6).to_string(index=False))


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
