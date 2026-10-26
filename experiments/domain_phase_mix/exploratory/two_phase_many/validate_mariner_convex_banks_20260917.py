# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "joblib", "matplotlib"]
# ///
"""Full canonical refits and archived-bank replay of preselected convex finalists."""

import argparse
import dataclasses
import functools
import importlib
import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    analyze_mariner_policy_banks_20260917 as analysis,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    mariner_convex_candidates_20260917 as candidates,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    validate_mariner_policy_banks_20260917 as validation,
)

OUTPUT = validation.OUTPUT / "convex"


def make_model(model_id: str, features):
    return candidates.build_model(
        next(spec for spec in candidates.candidate_specs() if spec.model_id == model_id), features
    )


def prepare(output: Path, model_ids: tuple[str, ...]) -> dict:
    validation.verify_sources(validation.OUTPUT)
    paths = (Path(__file__), Path(candidates.__file__), Path(analysis.__file__))
    manifest = {
        "version": 1,
        "models": list(model_ids),
        "canonical_protocol_hash": validation.sha256(validation.OUTPUT / "manifest.json"),
        "sources": {str(path): validation.sha256(path) for path in paths},
        "fitting": "canonical full fit/heldout_inner_folds; frozen finalist selection excludes bank labels",
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Convex finalist protocol changed")
        return manifest
    validation.json_write(path, manifest)
    for source in paths:
        destination = output / "source_snapshot" / source.name
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, destination)
    (output / "PROTOCOL.md").write_text(
        "# Convex finalist bank replay\n\nFrozen IDs: " + ", ".join(model_ids) + ".\n\n"
        "The independent convex-screen agent selected these IDs before seeing bank results. Each finalist is refit "
        "on all 58 components of each of 3 panels (174 fits/model), using the canonical heldout_inner_folds and "
        "fitted-floor protocol. No bank labels select hyperparameters. Fits cache Fitted objects, heads, full "
        "shape/ridge CV tables, bank and complete 277 dose predictions.\n\n"
        "Use identical coordinates and primary current-weight complete atomic labels as the original three-model "
        "validation, with historical aggregate sensitivity separately reported. In particular, the strict Llama160 "
        "suite has 56 coordinates versus 173 with historical aggregates. Dose evidence uses 237 Uncheatable heldout "
        "points and the measured subset for the suite; 40 in-fit controls are marked.\n"
    )
    return manifest


@functools.cache
def identity(output: Path) -> str:
    validation.verify_sources(validation.OUTPUT)
    manifest = json.loads((output / "manifest.json").read_text())
    for name, digest in manifest["sources"].items():
        if validation.sha256(Path(name)) != digest:
            raise ValueError(f"Source changed {name}")
    return validation.sha256(output / "manifest.json")


def fit_component(model_id: str, panel_name: str, target: str, index: int, output: Path) -> str:
    protocol_hash = identity(output)
    panel = validation.bench.load_panel(panel_name)
    group = panel.group(target)
    component = group.components[index]
    path = validation.fit_path(output, model_id, panel_name, target, index)
    evidence = path.with_suffix(".npz")
    if path.exists() and evidence.exists():
        saved = validation.bench.load_shard(evidence)
        if str(saved["manifest_hash"].item()) != protocol_hash or str(saved["fit_hash"].item()) != validation.sha256(
            path
        ):
            raise ValueError(f"Cached finalist mismatch {path}")
        return "cached"
    features = dataclasses.replace(panel.features, component=component)
    bank, query = validation.bank_query(panel, target)
    query = dataclasses.replace(query, component=component)
    model = make_model(model_id, features)
    started = time.monotonic()
    fitted = model.fit(
        features,
        group.outcomes[:, index],
        np.arange(panel.rows),
        validation.bench.heldout_inner_folds(panel),
        validation.bench._seed(validation.bench.FitTask(model_id, panel_name, target, index, component, 0, 0)),
    )
    assert fitted.head.converged
    predicted = model.predict(fitted, query, np.arange(query.rows))
    assert np.isfinite(predicted).all()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    joblib.dump(fitted, temporary)
    temporary.replace(path)
    np.testing.assert_allclose(
        model.predict(joblib.load(path), query, np.arange(query.rows)), predicted, rtol=1e-13, atol=1e-13
    )
    saved = {
        "manifest_hash": protocol_hash,
        "fit_hash": validation.sha256(path),
        "model": model_id,
        "panel": panel_name,
        "target": target,
        "component": component,
        "component_index": index,
        "prediction": predicted,
        "coordinate_id": bank.coordinate_id.to_numpy(str),
        "shape_json": json.dumps(fitted.shape, sort_keys=True),
        "ridge": fitted.ridge,
        "cv_table": fitted.cv_table,
        "diagnostics_json": json.dumps(fitted.diagnostics, sort_keys=True),
        "coefficients": fitted.head.coefficients,
        "intercept": fitted.head.intercept,
        "floor": fitted.head.floor,
        "fit_prediction": model.predict(fitted, features, np.arange(panel.rows)),
        "elapsed": time.monotonic() - started,
    }
    if panel_name == validation.DOSE_PANEL:
        _, dose = validation.dose_query()
        saved["dose_prediction"] = model.predict(
            fitted, dataclasses.replace(dose, component=component), np.arange(dose.rows)
        )
    validation.bench.atomic_save(evidence, saved)
    return "fitted"


def report(output: Path, model_ids: tuple[str, ...]) -> None:
    template = pd.read_csv(validation.OUTPUT / "bank_predictions.csv")
    template = template[template.model.eq(validation.MODEL_IDS[0])]
    dose_template = pd.read_csv(validation.OUTPUT / "dose_predictions.csv")
    dose_template = dose_template[dose_template.model.eq(validation.MODEL_IDS[0])]
    predictions, fits, doses = [], [], []
    for panel_name in validation.bench.THIRTY_NINE_BUCKET_PANELS:
        panel = validation.bench.load_panel(panel_name)
        for target in ("uncheatable", "table9"):
            group = panel.group(target)
            bank, query = validation.bank_query(panel, target)
            for model_id in model_ids:
                matrix, dose_matrix = [], []
                for index, component in enumerate(group.components):
                    path = validation.fit_path(output, model_id, panel_name, target, index)
                    saved = validation.bench.load_shard(path.with_suffix(".npz"))
                    assert saved is not None and str(saved["fit_hash"].item()) == validation.sha256(path)
                    assert np.array_equal(saved["coordinate_id"], bank.coordinate_id.to_numpy(str))
                    fitted = joblib.load(path)
                    features = dataclasses.replace(panel.features, component=component)
                    model = make_model(model_id, features)
                    design = model.base.design(dataclasses.replace(query, component=component), fitted.shape)
                    spec = model._spec(
                        fitted.shape,
                        candidates.base.LinkKind(str(fitted.diagnostics["link"])),
                        float(fitted.diagnostics["kappa"]),
                    )
                    direct = candidates.base.predict_head(fitted.head, design.values, spec)
                    np.testing.assert_allclose(direct, saved["prediction"], rtol=1e-12, atol=1e-12)
                    winner = np.unravel_index(np.argmin(fitted.cv_table), fitted.cv_table.shape)
                    assert (
                        fitted.shape == model.base.shapes[winner[0]] and fitted.ridge == model.base.ridge_grid[winner[1]]
                    )
                    matrix.append(saved["prediction"])
                    if panel_name == validation.DOSE_PANEL:
                        dose_matrix.append(saved["dose_prediction"])
                    fits.append(
                        {
                            "model": model_id,
                            "panel": panel_name,
                            "target": target,
                            "component": component,
                            "component_index": index,
                            "fit_path": str(path),
                            "fit_hash": validation.sha256(path),
                            "shape": saved["shape_json"].item(),
                            "ridge": float(saved["ridge"]),
                            "diagnostics": saved["diagnostics_json"].item(),
                            "direct_prediction_and_cv_argmin_verified": True,
                        }
                    )
                frame = (
                    template[template.panel.eq(panel_name) & template.target.eq(target)]
                    .set_index("coordinate_id")
                    .loc[bank.coordinate_id]
                    .reset_index()
                )
                frame["model"] = model_id
                frame["prediction"] = np.stack(matrix, axis=1) @ group.aggregation_weights
                predictions.append(frame)
                if dose_matrix:
                    frame = dose_template[dose_template.target.eq(target)].copy()
                    frame["model"] = model_id
                    frame["prediction"] = np.stack(dose_matrix, axis=1) @ group.aggregation_weights
                    frame["predicted_delta"] = (
                        frame.prediction - frame.loc[frame.run_name.eq("p000_proportional_anchor"), "prediction"].iloc[0]
                    )
                    doses.append(frame)
    prediction = pd.concat(predictions, ignore_index=True)
    prediction.to_csv(output / "bank_predictions.csv", index=False)
    pd.DataFrame(fits).to_csv(output / "fits.csv", index=False)
    metrics = []
    for (model, panel, target), own in prediction.groupby(["model", "panel", "target"]):
        for labels in ("current_weights_complete", "historical_aggregate"):
            selected = (
                own[own.complete_components].copy()
                if labels == "current_weights_complete"
                else own.assign(measured=own.historical_aggregate)
            )
            for stratum, mask in validation.source_masks(selected):
                metrics.append(
                    {
                        "model": model,
                        "panel": panel,
                        "target": target,
                        "labels": labels,
                        "stratum": stratum,
                        **validation.selection_metrics(selected.iloc[np.flatnonzero(mask)]),
                    }
                )
    pd.DataFrame(metrics).to_csv(output / "bank_metrics.csv", index=False)
    dose = pd.concat(doses, ignore_index=True)
    dose.to_csv(output / "dose_predictions.csv", index=False)
    analysis.dose_metrics(dose, output)
    analysis.affine_curves(dose).to_csv(output / "dose_affine_curves.csv", index=False)
    validation.json_write(
        output / "verification.json",
        {
            "fits": len(fits),
            "expected_fits": 174 * len(model_ids),
            "same_bank_support_as_reference": True,
            "direct_prediction_and_cv_argmin_verified": True,
            "source_identity": identity(output),
            "complete": True,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("Use at most4 single-threaded workers")
    ids = tuple(args.models)
    prepare(OUTPUT, ids)
    tasks = [
        (model, panel_name, group.name, index, OUTPUT)
        for panel_name in validation.bench.THIRTY_NINE_BUCKET_PANELS
        for group in validation.bench.load_panel(panel_name).groups
        for index in range(len(group.components))
        for model in ids
    ]
    if not args.report_only:
        module = importlib.import_module(
            "experiments.domain_phase_mix.exploratory.two_phase_many.validate_mariner_convex_banks_20260917"
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
    report(OUTPUT, ids)
    print(json.dumps({"complete": True, "output": str(OUTPUT)}), flush=True)


if __name__ == "__main__":
    main()
