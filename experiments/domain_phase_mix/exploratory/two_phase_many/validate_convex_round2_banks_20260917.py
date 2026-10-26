# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "joblib"]
# ///
"""Replay preselected round-two convex candidates on archived policy banks."""

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
    mariner_convex_envelopes_round2_20260917 as envelopes,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    mariner_convex_fitting_round2_20260917 as fitting,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    mariner_convex_round2_models_20260917 as geometry,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_observatory_models_20260902 as base,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    validate_mariner_policy_banks_20260917 as prior,
)

OUTPUT = prior.OUTPUT.parent / "convex_round2_banks"
ENSEMBLE_ID = "round2_convex_equal_average"
MODEL_MODULES = {"cv2": geometry, "cf2": fitting, "ce2": envelopes}


def make_model(model_id: str, features: base.Features):
    """Build exactly one registered equation or fitting candidate."""
    module = MODEL_MODULES[model_id.split("_", 1)[0]]
    found = [spec for spec in module.candidate_specs() if spec.model_id == model_id]
    if len(found) != 1:
        raise ValueError(f"Expected one registered model, found {len(found)}: {model_id}")
    return module.build_model(found[0], features)


def prepare(output: Path, model_ids: tuple[str, ...], members: tuple[str, ...]) -> None:
    """Freeze model identities and reference inputs before reading bank scores."""
    prior.verify_sources(prior.OUTPUT)
    if set(members) - set(model_ids) or len(set(members)) != len(members):
        raise ValueError("Ensemble members must be unique fitted models")
    sources = [Path(__file__), Path(analysis.__file__)]
    sources.extend(dict.fromkeys(Path(MODEL_MODULES[model.split("_", 1)[0]].__file__) for model in model_ids))
    manifest = {
        "models": model_ids,
        "equal_weight_ensemble": members,
        "canonical_manifest_hash": prior.sha256(prior.OUTPUT / "manifest.json"),
        "sources": {str(path): prior.sha256(path) for path in sources},
        "fitting": "Full canonical swarm, heldout_inner_folds; no bank labels in fitting or model weights",
        "labels": "Current-weight complete components primary; historical aggregates separate",
        "interpretation": "Already inspected archives; adaptive development evidence, not prospective validation",
    }
    manifest = json.loads(json.dumps(manifest))
    path = output / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Round-two bank protocol changed; preserve existing outputs")
        return
    prior.json_write(path, manifest)
    for source in sources:
        destination = output / "source_snapshot" / source.name
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, destination)


@functools.cache
def identity(output: Path) -> str:
    prior.verify_sources(prior.OUTPUT)
    manifest = json.loads((output / "manifest.json").read_text())
    for path, digest in manifest["sources"].items():
        if prior.sha256(Path(path)) != digest:
            raise ValueError(f"Frozen round-two source changed: {path}")
    return prior.sha256(output / "manifest.json")


def fit_component(model_id: str, panel_name: str, target: str, index: int, output: Path) -> str:
    """Fit only training rows, then save prediction evidence atomically."""
    protocol_hash = identity(output)
    panel = prior.bench.load_panel(panel_name)
    group = panel.group(target)
    component = group.components[index]
    path = prior.fit_path(output, model_id, panel_name, target, index)
    evidence = path.with_suffix(".npz")
    if path.exists() and evidence.exists():
        saved = prior.bench.load_shard(evidence)
        assert saved is not None
        if str(saved["manifest_hash"].item()) != protocol_hash or str(saved["fit_hash"].item()) != prior.sha256(path):
            raise ValueError(f"Cached full-fit identity mismatch: {path}")
        return "cached"
    features = dataclasses.replace(panel.features, component=component)
    bank, query = prior.bank_query(panel, target)
    query = dataclasses.replace(query, component=component)
    model = make_model(model_id, features)
    started = time.monotonic()
    fitted = model.fit(
        features,
        group.outcomes[:, index],
        np.arange(panel.rows),
        prior.bench.heldout_inner_folds(panel),
        prior.bench._seed(prior.bench.FitTask(model_id, panel_name, target, index, component, 0, 0)),
    )
    if not fitted.head.converged:
        raise ValueError(f"Unconverged full fit: {model_id}/{panel_name}/{target}/{index}")
    prediction = model.predict(fitted, query, np.arange(query.rows))
    if not np.isfinite(prediction).all():
        raise ValueError(f"Nonfinite bank predictions: {model_id}/{panel_name}/{target}/{index}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    joblib.dump(fitted, temporary)
    temporary.replace(path)
    np.testing.assert_allclose(
        model.predict(joblib.load(path), query, np.arange(query.rows)), prediction, rtol=1e-13, atol=1e-13
    )
    payload = {
        "manifest_hash": protocol_hash,
        "fit_hash": prior.sha256(path),
        "model": model_id,
        "panel": panel_name,
        "target": target,
        "component": component,
        "prediction": prediction,
        "coordinate_id": bank.coordinate_id.to_numpy(str),
        "shape_json": json.dumps(fitted.shape, sort_keys=True),
        "ridge": fitted.ridge,
        "diagnostics_json": json.dumps(fitted.diagnostics, sort_keys=True),
        "cv_table": fitted.cv_table,
        "fit_prediction": model.predict(fitted, features, np.arange(panel.rows)),
        "elapsed": time.monotonic() - started,
    }
    if panel_name == prior.DOSE_PANEL:
        _, dose = prior.dose_query()
        payload["dose_prediction"] = model.predict(
            fitted, dataclasses.replace(dose, component=component), np.arange(dose.rows)
        )
    prior.bench.atomic_save(evidence, payload)
    return "fitted"


def equal_average(frame: pd.DataFrame, members: tuple[str, ...], keys: list[str]) -> pd.DataFrame:
    """Average aligned predictions with equal nonnegative model weights."""
    selected = [frame[frame.model.eq(member)].set_index(keys).sort_index() for member in members]
    reference = selected[0]
    if reference.index.has_duplicates:
        raise ValueError("Duplicate ensemble prediction keys")
    for item in selected[1:]:
        if not item.index.equals(reference.index):
            raise ValueError("Ensemble members must have identical prediction support")
        np.testing.assert_allclose(item.measured, reference.measured, rtol=0, atol=0, equal_nan=True)
    result = reference.copy()
    result["model"] = ENSEMBLE_ID
    for column in ("prediction", "predicted_delta"):
        if column in result:
            result[column] = np.mean([item[column].to_numpy() for item in selected], axis=0)
    return result.reset_index()


def report(output: Path, model_ids: tuple[str, ...], members: tuple[str, ...]) -> None:
    """Rebuild both complete and sparse-label readouts on the original support."""
    template = pd.read_csv(prior.OUTPUT / "bank_predictions.csv")
    template = template[template.model.eq("hs17_mariner")]
    dose_template = pd.read_csv(prior.OUTPUT / "dose_predictions.csv")
    dose_template = dose_template[dose_template.model.eq("hs17_mariner")]
    bank_frames, dose_frames, audit = [], [], []
    for panel_name in prior.bench.THIRTY_NINE_BUCKET_PANELS:
        panel = prior.bench.load_panel(panel_name)
        for target in ("uncheatable", "table9"):
            group = panel.group(target)
            bank, query = prior.bank_query(panel, target)
            for model_id in model_ids:
                matrix, doses = [], []
                for index, component in enumerate(group.components):
                    path = prior.fit_path(output, model_id, panel_name, target, index)
                    saved = prior.bench.load_shard(path.with_suffix(".npz"))
                    assert saved is not None
                    assert str(saved["fit_hash"].item()) == prior.sha256(path)
                    assert str(saved["manifest_hash"].item()) == identity(output)
                    np.testing.assert_array_equal(saved["coordinate_id"], bank.coordinate_id.to_numpy(str))
                    model = make_model(model_id, dataclasses.replace(panel.features, component=component))
                    fitted = joblib.load(path)
                    replay = model.predict(
                        fitted, dataclasses.replace(query, component=component), np.arange(query.rows)
                    )
                    np.testing.assert_allclose(replay, saved["prediction"], rtol=1e-13, atol=1e-13)
                    matrix.append(saved["prediction"])
                    if panel_name == prior.DOSE_PANEL:
                        doses.append(saved["dose_prediction"])
                    audit.append(
                        {
                            "model": model_id,
                            "panel": panel_name,
                            "target": target,
                            "component": component,
                            "fit_hash": prior.sha256(path),
                            "prediction_replay_max_error": float(np.max(np.abs(replay - saved["prediction"]))),
                        }
                    )
                own = template[template.panel.eq(panel_name) & template.target.eq(target)]
                frame = own.set_index("coordinate_id").loc[bank.coordinate_id].reset_index()
                frame["model"] = model_id
                frame["prediction"] = np.column_stack(matrix) @ group.aggregation_weights
                bank_frames.append(frame)
                if doses:
                    frame = dose_template[dose_template.target.eq(target)].copy()
                    frame["model"] = model_id
                    frame["prediction"] = np.column_stack(doses) @ group.aggregation_weights
                    frame["predicted_delta"] = (
                        frame.prediction - frame.loc[frame.run_name.eq("p000_proportional_anchor"), "prediction"].iloc[0]
                    )
                    dose_frames.append(frame)
    banks, dose = pd.concat(bank_frames, ignore_index=True), pd.concat(dose_frames, ignore_index=True)
    if members:
        banks = pd.concat(
            [banks, equal_average(banks, members, ["panel", "target", "coordinate_id"])], ignore_index=True
        )
        dose = pd.concat([dose, equal_average(dose, members, ["target", "run_name"])], ignore_index=True)
    banks = pd.concat([template, banks], ignore_index=True)
    dose = pd.concat([dose_template, dose], ignore_index=True)
    banks.to_csv(output / "bank_predictions.csv", index=False)
    dose.to_csv(output / "dose_predictions.csv", index=False)
    metrics = []
    for (model, panel, target), own in banks.groupby(["model", "panel", "target"]):
        for labels in ("current_weights_complete", "historical_aggregate"):
            selected = (
                own[own.complete_components].copy()
                if labels == "current_weights_complete"
                else own.assign(measured=own.historical_aggregate)
            )
            for stratum, mask in prior.source_masks(selected):
                metrics.append(
                    {
                        "model": model,
                        "panel": panel,
                        "target": target,
                        "labels": labels,
                        "stratum": stratum,
                        **prior.selection_metrics(selected.iloc[np.flatnonzero(mask)]),
                    }
                )
    pd.DataFrame(metrics).to_csv(output / "bank_metrics.csv", index=False)
    pd.DataFrame(audit).to_csv(output / "prediction_replay_audit.csv", index=False)
    analysis.dose_metrics(dose, output)
    prior.json_write(
        output / "verification.json",
        {
            "fits": len(audit),
            "expected_fits": 174 * len(model_ids),
            "shared_support": True,
            "saved_prediction_replay": True,
            "manifest_hash": identity(output),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--ensemble-members", nargs="*", default=[])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    model_ids, members = tuple(args.models), tuple(args.ensemble_members)
    if not 1 <= args.workers <= 8:
        raise ValueError("Expected between one and eight single-thread workers")
    prepare(args.output, model_ids, members)
    tasks = [
        (model, panel, group.name, index, args.output)
        for panel in prior.bench.THIRTY_NINE_BUCKET_PANELS
        for group in prior.bench.load_panel(panel).groups
        for index in range(len(group.components))
        for model in model_ids
    ]
    if not args.report_only:
        module = importlib.import_module(
            "experiments.domain_phase_mix.exploratory.two_phase_many.validate_convex_round2_banks_20260917"
        )
        with joblib.parallel_config(backend="loky", inner_max_num_threads=1):
            results = joblib.Parallel(n_jobs=args.workers, return_as="generator_unordered")(
                joblib.delayed(module.fit_component)(*task) for task in tasks
            )
            for count, status in enumerate(results, 1):
                if count % 12 == 0 or count == len(tasks):
                    print(json.dumps({"completed": count, "total": len(tasks), "status": status}), flush=True)
    report(args.output, model_ids, members)
    print(json.dumps({"complete": True, "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
