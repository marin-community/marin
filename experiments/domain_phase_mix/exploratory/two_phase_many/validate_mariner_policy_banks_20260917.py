# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "joblib"]
# ///
"""Frozen full-fit replay on archived policies and conditional dose curves."""

import argparse
import dataclasses
import functools
import hashlib
import importlib
import json
import math
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_single_phase_observatory_20260902 as bench,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    mariner_harm_screen_models_20260917 as candidates,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    prepare_single_phase_heldout_benchmark_20260902 as builder,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    screen_mariner_harm_forms_20260917 as screen,
)

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "reference_outputs/mariner_local_validation_20260917/banks"
BANK = HERE / "reference_outputs/single_phase_heldout_benchmark_expanded_20260913"
MODEL_IDS = ("hs17_mariner", "hs17_log1", "hs17_bounded_exp_c1")
BOOTSTRAPS = 1000
RANDOM_SEED = 20260917
DOSE_PANEL = "delphi_3e18_39bucket"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


@functools.cache
def bank_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return tuple(
        pd.read_csv(BANK / name)
        for name in ("heldout_coordinates.csv", "heldout_coordinate_components.csv", "heldout_runs.csv")
    )


def bank_query(panel: bench.BenchPanel, target: str):
    coordinates, _, _ = bank_tables()
    count_column, _ = bench.HELDOUT_TARGET_COLUMNS[target]
    bank = coordinates[coordinates.panel.eq(panel.name) & coordinates[count_column].fillna(0).gt(0)].copy()
    bank = bank.reset_index(drop=True)
    weights = bank[[f"weight::{bucket}" for bucket in panel.buckets]].to_numpy(float)
    pools = bank[[f"pool_fraction::{bucket}" for bucket in panel.buckets]].fillna(1).to_numpy(float)
    assert np.isfinite(weights).all() and np.allclose(weights.sum(axis=1), 1, atol=1e-8)
    assert np.all(pools > 0) and np.all(pools <= 1)
    query = dataclasses.replace(
        panel.features,
        weights=weights,
        exposures=weights * panel.features.inventory[None, :] / pools,
        label=f"{panel.name}|validation_bank|{target}",
    )
    return bank, query


@functools.cache
def dose_query():
    panel = bench.load_panel(DOSE_PANEL)
    manifest = pd.read_csv(builder.EPOCH_DOSE_RUN_MANIFEST)
    long_weights = pd.read_csv(builder.EPOCH_DOSE_PHASE_WEIGHTS)
    table = long_weights[long_weights.phase.eq("phase_0")].pivot(index="run_name", columns="domain", values="weight")
    weights = table.reindex(index=manifest.run_name, columns=panel.buckets).to_numpy(float)
    assert len(manifest) == 277 and np.allclose(weights.sum(axis=1), 1, atol=1e-8)
    query = dataclasses.replace(
        panel.features,
        weights=weights,
        exposures=weights * panel.features.inventory[None, :],
        label=f"{DOSE_PANEL}|complete_dose_curves",
    )
    return manifest, query


def fit_path(output: Path, model: str, panel: str, target: str, index: int) -> Path:
    return output / "fits" / model / panel / target / f"component_{index:03d}.joblib"


def make_model(model_id: str, features):
    spec = next(spec for spec in candidates.candidate_specs() if spec.model_id == model_id)
    return candidates.build_model(spec, features)


def prepare(output: Path) -> dict:
    sources = tuple(dict.fromkeys((*screen.source_paths(), Path(__file__), Path(builder.__file__))))
    data_paths = [
        *(
            BANK / name
            for name in (
                "heldout_coordinates.csv",
                "heldout_coordinate_components.csv",
                "heldout_runs.csv",
                "manifest.json",
            )
        ),
        builder.EPOCH_DOSE_RUN_MANIFEST,
        builder.EPOCH_DOSE_PHASE_WEIGHTS,
        builder.EPOCH_DOSE_RESULTS,
        builder.EPOCH_DOSE_UNCHEATABLE_COMPONENTS,
        builder.EPOCH_DOSE_TABLE9_COMPONENTS,
    ]
    panels = {name: bench.load_panel(name) for name in bench.THIRTY_NINE_BUCKET_PANELS}
    manifest = {
        "version": 1,
        "models": MODEL_IDS,
        "sources": {str(path): sha256(path) for path in sources},
        "data": {str(path): sha256(path) for path in data_paths},
        "panels": {},
        "bootstrap_seed": RANDOM_SEED,
        "fit_protocol": (
            "canonical full panel, heldout_inner_folds, canonical seed, same model/floor procedure as screen"
        ),
    }
    for name, panel in panels.items():
        manifest["panels"][name] = {
            "inputs": panel.input_hashes,
            "fit_runs": panel.runs,
            "features": panel.features.cache_key,
            "inner_folds": [(a.tolist(), b.tolist()) for a, b in bench.heldout_inner_folds(panel)],
            "targets": {
                g.name: {"components": g.components, "weights": g.aggregation_weights.tolist()} for g in panel.groups
            },
        }
    manifest = json.loads(json.dumps(manifest))
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Validation source/input/fold identity changed; use a new output identity")
        return manifest
    json_write(path, manifest)
    for source in sources:
        destination = output / "source_snapshot" / source.name
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, destination)
    (output / "PROTOCOL.md").write_text("""# MARINER retrospective policy and curve validation

Frozen before fitting: hs17_mariner, hs17_log1, hs17_bounded_exp_c1; all 58 components on each of three panels
(522 full fits). Fit uses only the canonical panel rows and canonical heldout inner folds/seed, shape grid,
ridge grid and fitted-floor procedure. Bank labels never select fits or hyperparameters. Source, data, rows,
and folds are hashed in manifest.json. Full Fitted objects, heads, CV tables and predictions are cached atomically.

Primary bank objectives are reconstructed from complete atomic components using the panel's current weights.
The registry aggregate is reported separately as sensitivity because 14 Qwen Uncheatable validation coordinates
disagree with current-weight aggregation. Six Uncheatable coordinates and117 Llama160 suite coordinates
lack complete atomic components; their full-bank historical-aggregate readout is separately labeled.
Bank and each source membership are scored separately, with validation:: sources also pooled into a distinct
cohort; multi-source coordinates may occur in several source strata but only once in all-bank summaries.
Primary measures: regret at1, best measured loss among predicted top5/top10, selected optimism, Spearman,
RMSE, and rank among the observed best20% (minimum5) and within .005 BPB of the measured best. These competitive
sets use evaluation labels solely as retrospective diagnostics, never to restrict model selection.

Uncertainty is diagnostic: 1000 paired coordinate bootstrap menus for performance differences; repeated-run
label draws only on coordinates with multiple archived runs (single-run coordinates fixed), not a population
or prospective confidence interval. Original-aggregate run noise is kept distinct from atomic primary labels.

Dose analysis uses all277 designed runs, explicitly marks40 fit-overlapping controls, and reports primary
metrics only on237 nonoverlapping coordinates. Curves are grouped by39 focal buckets. Report direct and
anchor-relative errors, adjacent changes, observed and predicted discrete optimum, one-grid-neighbor optimum
bracket, and upper-half-epoch tail errors. Complete curves show fit controls separately. These are conditional
substitution paths: raising a focal bucket reduces the other shares; they do not isolate intrinsic harm.
No high-level form is promoted on retrospective evidence alone. No training jobs or manuscript changes.
""")
    return manifest


@functools.cache
def verify_sources(output: Path) -> str:
    manifest = json.loads((output / "manifest.json").read_text())
    for path, digest in {**manifest["sources"], **manifest["data"]}.items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"Frozen input changed: {path}")
    return sha256(output / "manifest.json")


def fit_component(model_id: str, panel_name: str, target: str, index: int, output: Path) -> str:
    identity = verify_sources(output)
    path = fit_path(output, model_id, panel_name, target, index)
    evidence = path.with_suffix(".npz")
    panel = bench.load_panel(panel_name)
    group = panel.group(target)
    component = group.components[index]
    features = dataclasses.replace(panel.features, component=component)
    model = make_model(model_id, features)
    bank, query = bank_query(panel, target)
    query = dataclasses.replace(query, component=component)
    if path.exists() and evidence.exists():
        saved = bench.load_shard(evidence)
        if str(saved["manifest_hash"].item()) != identity or str(saved["fit_hash"].item()) != sha256(path):
            raise ValueError(f"Cached fit identity mismatch {path}")
        assert np.isfinite(saved["prediction"]).all() and len(saved["prediction"]) == len(bank)
        return "cached"
    started = time.monotonic()
    fitted = model.fit(
        features,
        group.outcomes[:, index],
        np.arange(panel.rows),
        bench.heldout_inner_folds(panel),
        bench._seed(bench.FitTask(model_id, panel_name, target, index, component, 0, 0)),
    )
    if not fitted.head.converged:
        raise ValueError(f"Final head did not converge {panel_name}/{target}/{index}/{model_id}")
    prediction = model.predict(fitted, query, np.arange(query.rows))
    assert np.isfinite(prediction).all()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    joblib.dump(fitted, temporary)
    temporary.replace(path)
    loaded = joblib.load(path)
    np.testing.assert_allclose(model.predict(loaded, query, np.arange(query.rows)), prediction, rtol=1e-13, atol=1e-13)
    payload = {
        "manifest_hash": identity,
        "fit_hash": sha256(path),
        "model": model_id,
        "panel": panel_name,
        "target": target,
        "component": component,
        "component_index": index,
        "prediction": prediction,
        "coordinate_id": bank.coordinate_id.to_numpy(str),
        "shape_json": json.dumps(fitted.shape, sort_keys=True),
        "ridge": fitted.ridge,
        "diagnostics_json": json.dumps(fitted.diagnostics, sort_keys=True),
        "coefficients": fitted.head.coefficients,
        "intercept": fitted.head.intercept,
        "floor": fitted.head.floor,
        "cv_table": fitted.cv_table,
        "fit_prediction": model.predict(fitted, features, np.arange(panel.rows)),
        "elapsed": time.monotonic() - started,
    }
    if panel_name == DOSE_PANEL:
        _, dose = dose_query()
        dose = dataclasses.replace(dose, component=component)
        payload["dose_prediction"] = model.predict(fitted, dose, np.arange(dose.rows))
    bench.atomic_save(evidence, payload)
    return "fitted"


def safe_rank(observed: np.ndarray, predicted: np.ndarray) -> float:
    if len(observed) < 3 or np.ptp(observed) == 0 or np.ptp(predicted) == 0:
        return float("nan")
    return float(stats.spearmanr(observed, predicted).statistic)


def selection_metrics(frame: pd.DataFrame) -> dict:
    frame = frame.sort_values("coordinate_id")
    loss, guess = frame.measured.to_numpy(float), frame.prediction.to_numpy(float)
    order = np.argsort(guess, kind="stable")
    selected = order[0]
    competitive = np.argsort(loss, kind="stable")[: max(5, math.ceil(0.2 * len(loss)))]
    close = loss <= loss.min() + 0.005
    return {
        "n": len(frame),
        "selected_id": frame.coordinate_id.iloc[selected],
        "best_measured": loss.min(),
        "selected_measured": loss[selected],
        "selected_prediction": guess[selected],
        "regret_at_1": loss[selected] - loss.min(),
        "regret_at_5": loss[order[:5]].min() - loss.min(),
        "regret_at_10": loss[order[:10]].min() - loss.min(),
        "selected_optimism": loss[selected] - guess[selected],
        "selected_rank": stats.rankdata(loss)[selected],
        "rmse": np.sqrt(np.mean((guess - loss) ** 2)),
        "spearman": safe_rank(loss, guess),
        "competitive_n": len(competitive),
        "competitive_spearman": safe_rank(loss[competitive], guess[competitive]),
        "near_best_n": int(close.sum()),
        "near_best_spearman": safe_rank(loss[close], guess[close]),
    }


def source_masks(frame: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
    memberships = [set(str(value).split(";")) for value in frame.sources]
    all_sources = sorted(set.union(*memberships))
    masks = [
        ("all", np.ones(len(frame), bool)),
        (
            "validation_cohort",
            np.array([any(source.startswith("validation::") for source in values) for values in memberships]),
        ),
        (
            "nonvalidation_nondose",
            np.array(
                [
                    not any(
                        source.startswith("validation::") or source == "conditional_epoch_dose_response"
                        for source in values
                    )
                    for values in memberships
                ]
            ),
        ),
    ]
    masks.extend((source, np.array([source in values for values in memberships])) for source in all_sources)
    return [(name, mask) for name, mask in masks if mask.sum() >= 3]


def paired_bootstrap(frame: pd.DataFrame) -> list[dict]:
    rows = []
    rng = np.random.default_rng(RANDOM_SEED)
    reference = frame[frame.model.eq(MODEL_IDS[0])].set_index("coordinate_id")
    for model_id in MODEL_IDS[1:]:
        challenger = frame[frame.model.eq(model_id)].set_index("coordinate_id").loc[reference.index]
        for stratum, mask in source_masks(reference.reset_index()):
            if stratum not in ("all", "validation_cohort", "nonvalidation_nondose"):
                continue
            y = reference.measured.to_numpy()[mask]
            p0, p1 = reference.prediction.to_numpy()[mask], challenger.prediction.to_numpy()[mask]
            draws = rng.integers(0, len(y), size=(BOOTSTRAPS, len(y)))
            chosen0 = np.argmin(p0[draws], axis=1)
            chosen1 = np.argmin(p1[draws], axis=1)
            loss0, loss1 = y[draws[np.arange(BOOTSTRAPS), chosen0]], y[draws[np.arange(BOOTSTRAPS), chosen1]]
            differences = loss1 - loss0
            rmse_delta = np.sqrt(np.mean((p1[draws] - y[draws]) ** 2, axis=1)) - np.sqrt(
                np.mean((p0[draws] - y[draws]) ** 2, axis=1)
            )
            rows.append(
                {
                    "model": model_id,
                    "stratum": stratum,
                    "n": len(y),
                    "bootstrap_kind": "paired_coordinate_menu",
                    "regret_delta_mean": differences.mean(),
                    "regret_delta_p025": np.quantile(differences, 0.025),
                    "regret_delta_p975": np.quantile(differences, 0.975),
                    "win_fraction": (differences < -1e-12).mean(),
                    "tie_fraction": (np.abs(differences) <= 1e-12).mean(),
                    "rmse_delta_mean": rmse_delta.mean(),
                    "rmse_delta_p025": np.quantile(rmse_delta, 0.025),
                    "rmse_delta_p975": np.quantile(rmse_delta, 0.975),
                }
            )
    return rows


def report_banks(output: Path) -> tuple[pd.DataFrame, list[dict]]:
    _coordinates, components, runs = bank_tables()
    fits, prediction_frames, input_checks, label_checks = [], [], [], []
    for panel_name in bench.THIRTY_NINE_BUCKET_PANELS:
        panel = bench.load_panel(panel_name)
        for target in ("uncheatable", "table9"):
            group = panel.group(target)
            bank, query = bank_query(panel, target)
            distance = np.max(np.abs(query.exposures[:, None, :] - panel.features.exposures[None, :, :]), axis=2).min(
                axis=1
            )
            assert np.all(distance > 1e-8)
            table = components[components.panel.eq(panel_name) & components.target.eq(target)]
            assert not table.duplicated(["coordinate_id", "component"]).any()
            atomic = table.pivot(index="coordinate_id", columns="component", values="bpb_mean").reindex(
                index=bank.coordinate_id, columns=group.components
            )
            complete = np.isfinite(atomic.to_numpy()).all(axis=1)
            measured = atomic.to_numpy() @ group.aggregation_weights
            historic = bank[bench.HELDOUT_TARGET_COLUMNS[target][1]].to_numpy(float)
            label_table = bank[["panel", "coordinate_id", "sources"]].copy()
            label_table["target"], label_table["complete_components"] = target, complete
            label_table["historical_aggregate"], label_table["current_weight_aggregate"] = historic, measured
            label_table["difference"] = measured - historic
            label_checks.append(label_table)
            input_checks.append(
                {
                    "panel": panel_name,
                    "target": target,
                    "n": len(bank),
                    "complete_components": int(complete.sum()),
                    "minimum_fit_exposure_distance": distance.min(),
                    "aggregation_disagreements_gt1e5": int(np.sum(np.abs(measured - historic) > 1e-5)),
                }
            )
            for model_id in MODEL_IDS:
                matrix = []
                for index, component in enumerate(group.components):
                    path = fit_path(output, model_id, panel_name, target, index)
                    saved = bench.load_shard(path.with_suffix(".npz"))
                    assert saved is not None and str(saved["fit_hash"].item()) == sha256(path)
                    assert np.array_equal(saved["coordinate_id"], bank.coordinate_id.to_numpy(str))
                    matrix.append(saved["prediction"])
                    fits.append(
                        {
                            "model": model_id,
                            "panel": panel_name,
                            "target": target,
                            "component": component,
                            "component_index": index,
                            "fit_path": str(path),
                            "fit_hash": sha256(path),
                            "shape": saved["shape_json"].item(),
                            "ridge": float(saved["ridge"]),
                            "diagnostics": saved["diagnostics_json"].item(),
                        }
                    )
                predicted = np.stack(matrix, axis=1) @ group.aggregation_weights
                assert np.isfinite(predicted).all()
                frame = bank[["panel", "coordinate_id", "sources", "run_count"]].copy()
                frame["model"], frame["target"] = model_id, target
                frame["measured"], frame["prediction"] = measured, predicted
                frame["historical_aggregate"], frame["complete_components"] = historic, complete
                prediction_frames.append(frame)
    prediction = pd.concat(prediction_frames, ignore_index=True)
    pd.DataFrame(fits).to_csv(output / "fits.csv", index=False)
    pd.concat(label_checks, ignore_index=True).to_csv(output / "label_audit.csv", index=False)
    prediction.to_csv(output / "bank_predictions.csv", index=False)
    metrics, bootstrap = [], []
    for (panel_name, target), panel_frame in prediction.groupby(["panel", "target"]):
        primary = panel_frame[panel_frame.complete_components]
        bootstrap.extend({"panel": panel_name, "target": target, **row} for row in paired_bootstrap(primary))
        for model_id, model_frame in panel_frame.groupby("model"):
            for labels in ("current_weights_complete", "historical_aggregate"):
                selected = (
                    model_frame[model_frame.complete_components].copy()
                    if labels == "current_weights_complete"
                    else model_frame.assign(measured=model_frame.historical_aggregate)
                )
                for stratum, mask in source_masks(selected):
                    metrics.append(
                        {
                            "panel": panel_name,
                            "target": target,
                            "model": model_id,
                            "labels": labels,
                            "stratum": stratum,
                            **selection_metrics(selected.iloc[np.flatnonzero(mask)]),
                        }
                    )
    pd.DataFrame(metrics).to_csv(output / "bank_metrics.csv", index=False)
    pd.DataFrame(bootstrap).to_csv(output / "bank_bootstrap.csv", index=False)
    # Retain exactly which archived runs contributed each coordinate, including repeated seeds.
    keys = prediction[["panel", "coordinate_id"]].drop_duplicates()
    runs[runs.eligible.astype(bool)].merge(keys, on=["panel", "coordinate_id"], validate="many_to_one").to_csv(
        output / "bank_run_provenance.csv", index=False
    )
    return prediction, input_checks


def report_run_noise(prediction: pd.DataFrame, output: Path) -> None:
    _, _, runs = bank_tables()
    rows = []
    for (panel, target), frame in prediction.groupby(["panel", "target"]):
        column = bench.HELDOUT_RUN_COLUMNS[target]
        eligible = runs[runs.panel.eq(panel) & runs.eligible.astype(bool) & runs[column].notna()]
        reference = frame[frame.model.eq(MODEL_IDS[0])].sort_values("coordinate_id")
        values = eligible.groupby("coordinate_id")[column].apply(lambda x: x.to_numpy(float)).to_dict()
        rng = np.random.default_rng(RANDOM_SEED)
        draws = np.stack([rng.choice(values[c], BOOTSTRAPS) for c in reference.coordinate_id], axis=1)
        base_selected = np.argmin(reference.prediction.to_numpy())
        for model in MODEL_IDS:
            own = frame[frame.model.eq(model)].set_index("coordinate_id").loc[reference.coordinate_id]
            selected = np.argmin(own.prediction.to_numpy())
            regret = draws[:, selected] - draws.min(axis=1)
            delta = draws[:, selected] - draws[:, base_selected]
            rows.append(
                {
                    "panel": panel,
                    "target": target,
                    "model": model,
                    "n": len(reference),
                    "repeated_coordinates": sum(len(values[c]) > 1 for c in reference.coordinate_id),
                    "regret_mean": regret.mean(),
                    "regret_p025": np.quantile(regret, 0.025),
                    "regret_p975": np.quantile(regret, 0.975),
                    "delta_mean": delta.mean(),
                    "delta_p025": np.quantile(delta, 0.025),
                    "delta_p975": np.quantile(delta, 0.975),
                    "labels": "historical_run_aggregates; single-run coordinates fixed",
                }
            )
    pd.DataFrame(rows).to_csv(output / "run_label_sensitivity.csv", index=False)


def report_dose(output: Path) -> None:
    _, atomic_bank, runs = bank_tables()
    panel = bench.load_panel(DOSE_PANEL)
    manifest, query = dose_query()
    results = pd.read_csv(builder.EPOCH_DOSE_RESULTS)
    meta = manifest.merge(results, on=["run_order", "run_name"], validate="one_to_one")
    source_rows = runs[runs.source.eq("conditional_epoch_dose_response")][["source_row_id", "coordinate_id"]]
    assert len(source_rows) == 237
    meta = meta.merge(source_rows, left_on="run_name", right_on="source_row_id", how="left", validate="one_to_one")
    distance = np.max(np.abs(query.weights[:, None, :] - panel.features.weights[None, :, :]), axis=2).min(axis=1)
    meta["in_fit_panel"] = distance <= 1e-8
    assert meta.in_fit_panel.sum() == 40 and meta.coordinate_id.notna().sum() == 237
    frames, curve_metrics = [], []
    anchor = int(np.flatnonzero(meta.run_name.eq("p000_proportional_anchor"))[0])
    for target in ("uncheatable", "table9"):
        group = panel.group(target)
        source_path = (
            builder.EPOCH_DOSE_UNCHEATABLE_COMPONENTS
            if target == "uncheatable"
            else builder.EPOCH_DOSE_TABLE9_COMPONENTS
        )
        atomic = pd.read_csv(source_path)
        observed = atomic.pivot(index="run_name", columns="component", values="bpb").reindex(
            index=meta.run_name, columns=group.components
        )
        measured = observed.to_numpy() @ group.aggregation_weights
        # Exact bank records are authoritative for nonoverlapping rows; retain complete source controls.
        bank_atomic = (
            atomic_bank[atomic_bank.panel.eq(DOSE_PANEL) & atomic_bank.target.eq(target)]
            .pivot(index="coordinate_id", columns="component", values="bpb_mean")
            .reindex(columns=group.components)
        )
        valid = meta.coordinate_id.notna().to_numpy()
        measured[valid] = bank_atomic.loc[meta.loc[valid, "coordinate_id"]].to_numpy() @ group.aggregation_weights
        for model_id in MODEL_IDS:
            matrix = [
                bench.load_shard(fit_path(output, model_id, DOSE_PANEL, target, i).with_suffix(".npz"))[
                    "dose_prediction"
                ]
                for i in range(len(group.components))
            ]
            predicted = np.stack(matrix, axis=1) @ group.aggregation_weights
            frame = meta[
                [
                    "run_name",
                    "coordinate_id",
                    "focal_domain",
                    "epoch_multiplier",
                    "target_simulated_epochs",
                    "focal_weight",
                    "complement_scale",
                    "in_fit_panel",
                    "trainer_seed",
                    "data_seed",
                    "training_wandb_run_id",
                ]
            ].copy()
            frame["model"], frame["target"] = model_id, target
            frame["measured"], frame["prediction"] = measured, predicted
            frame["measured_delta"], frame["predicted_delta"] = (
                measured - measured[anchor],
                predicted - predicted[anchor],
            )
            for i, bucket in enumerate(panel.buckets):
                frame[f"weight::{bucket}"] = query.weights[:, i]
            frames.append(frame)
            for bucket, curve in frame[~frame.in_fit_panel].groupby("focal_domain"):
                curve = curve.sort_values(["target_simulated_epochs", "run_name"])
                # Pool repeated training seeds at the same epoch coordinate before shape diagnostics.
                curve = curve.groupby("target_simulated_epochs", as_index=False).agg(
                    measured=("measured", "mean"),
                    prediction=("prediction", "mean"),
                    measured_delta=("measured_delta", "mean"),
                    predicted_delta=("predicted_delta", "mean"),
                )
                x, y, p = (curve[c].to_numpy() for c in ("target_simulated_epochs", "measured", "prediction"))
                assert np.isfinite(y).all()
                measured_min, predicted_min = np.argmin(y), np.argmin(p)
                tail = np.arange(len(x)) >= len(x) // 2
                curve_metrics.append(
                    {
                        "model": model_id,
                        "target": target,
                        "focal_domain": bucket,
                        "n": len(x),
                        "rmse": np.sqrt(np.mean((p - y) ** 2)),
                        "spearman": safe_rank(y, p),
                        "change_rmse": np.sqrt(np.mean((np.diff(p) - np.diff(y)) ** 2)),
                        "adjacent_sign_agreement": np.mean(np.sign(np.diff(p)) == np.sign(np.diff(y))),
                        "relative_rmse": np.sqrt(np.mean((curve.predicted_delta - curve.measured_delta) ** 2)),
                        "measured_min_epoch": x[measured_min],
                        "predicted_min_epoch": x[predicted_min],
                        "minimum_grid_distance": abs(int(measured_min) - int(predicted_min)),
                        "minimum_in_one_neighbor": abs(int(measured_min) - int(predicted_min)) <= 1,
                        "regret": y[predicted_min] - y.min(),
                        "tail_bias": np.mean(p[tail] - y[tail]),
                        "tail_rmse": np.sqrt(np.mean((p[tail] - y[tail]) ** 2)),
                        "measured_tail_change": y[-1] - y[measured_min],
                        "predicted_tail_change": p[-1] - p[measured_min],
                    }
                )
    pd.concat(frames, ignore_index=True).to_csv(output / "dose_predictions.csv", index=False)
    curve_table = pd.DataFrame(curve_metrics)
    curve_table.to_csv(output / "dose_curve_metrics.csv", index=False)
    curve_table.groupby(["model", "target"]).mean(numeric_only=True).reset_index().to_csv(
        output / "dose_summary.csv", index=False
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("Use at most four single-threaded workers")
    prepare(args.output)
    tasks = [
        (model, panel_name, group.name, index, args.output)
        for panel_name in bench.THIRTY_NINE_BUCKET_PANELS
        for group in bench.load_panel(panel_name).groups
        for index in range(len(group.components))
        for model in MODEL_IDS
    ]
    if not args.report_only:
        module = importlib.import_module(
            "experiments.domain_phase_mix.exploratory.two_phase_many.validate_mariner_policy_banks_20260917"
        )
        started = time.monotonic()
        with joblib.parallel_config(backend="loky", inner_max_num_threads=1):
            results = joblib.Parallel(n_jobs=args.workers, return_as="generator_unordered")(
                joblib.delayed(module.fit_component)(*task) for task in tasks
            )
            for index, status in enumerate(results, 1):
                if index % 12 == 0 or index == len(tasks):
                    print(
                        json.dumps(
                            {
                                "completed": index,
                                "total": len(tasks),
                                "elapsed": round(time.monotonic() - started),
                                "status": status,
                            }
                        ),
                        flush=True,
                    )
    prediction, checks = report_banks(args.output)
    report_run_noise(prediction, args.output)
    report_dose(args.output)
    json_write(
        args.output / "verification.json",
        {
            "expected_fits": len(tasks),
            "actual_fits": len(pd.read_csv(args.output / "fits.csv")),
            "all_finite": True,
            "bank_inputs": checks,
            "source_identity": verify_sources(args.output),
            "complete": True,
        },
    )
    print(json.dumps({"complete": True, "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
