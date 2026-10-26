# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["joblib", "numpy", "pandas", "scipy", "tabulate"]
# ///

"""Frozen anchor-first screen of changes to convex MARINER fitting procedures."""

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
    benchmark_single_phase_observatory_20260902 as bench,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    mariner_convex_fitting_round2_20260917 as candidates,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    screen_mariner_convex_20260917 as previous,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_observatory_models_20260902 as models,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    single_phase_observatory_registry_20260902 as registry,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    validate_mariner_refit_robustness_20260917 as portable,
)

DEFAULT_OUTPUT = previous.DEFAULT_OUTPUT.parent / "convex_fitting_round2"
CANONICAL_OUTPUT = previous.DEFAULT_OUTPUT


def register_candidates() -> None:
    previous.register_candidates()
    for spec in candidates.candidate_specs():
        registry.ENTRY_BY_ID[spec.model_id] = dataclasses.replace(
            registry.ENTRY_BY_ID[spec.family_id],
            model_id=spec.model_id,
            parent=spec.family_id,
            build=functools.partial(candidates.build_model, spec),
            note=f"Fixed equation, fitting procedure {spec.procedure}",
        )


def prepare(output: Path) -> dict:
    protocol = previous.protocol()
    protocol["sources"].update(
        {
            str(path.resolve()): previous.previous.digest(path)
            for path in (Path(__file__), Path(candidates.__file__), Path(portable.__file__))
        }
    )
    protocol["candidates"] = [dataclasses.asdict(spec) for spec in candidates.candidate_specs()]
    protocol["canonical_controls"] = candidates.FAMILY_IDS
    protocol["screen"] = (
        "Repeat zero, three panels, six anchors per panel, five grouped outer folds; 810 new + 270 reused."
    )
    protocol["procedures"] = {
        "bounded_rms": "Per head training set, multiply coefficient penalties by clip(mean(column^2), .25, 4).",
        "selection": "Every shape/ridge/floor score is pooled BPB RMSE + .5 * mean inner-fold selected-point regret.",
        "joint": "Canonical FittedFloorModel joint_rounds=1: reselect shape/ridge at fitted kappa, then refit floor.",
    }
    protocol["advancement"] = {
        "eligible": "No failed fits; balanced anchor mean regret <= canonical family mean regret + .001 BPB.",
        "ranking": "Mean regret ascending, then component Spearman descending, then lexical model ID.",
        "budget": "At most two convex procedure variants, plus each corresponding MARINER procedure control.",
        "balancing": (
            "Within each panel/objective average components and folds, then equally average six panel/objectives."
        ),
        "full": "All 58 components, three panels, five outer folds, repeat zero; freeze IDs before full fits.",
        "data_boundary": "No bank observations enter fitting or advancement. Archived development data only.",
    }
    protocol = json.loads(json.dumps(protocol))
    output.mkdir(parents=True, exist_ok=True)
    path = output / "screen_manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError("Frozen fitting protocol changed; use a new output identity")
        return protocol
    old = json.loads((CANONICAL_OUTPUT / "screen_manifest.json").read_text())
    for source, checksum in old["sources"].items():
        if previous.previous.digest(Path(source)) != checksum:
            raise ValueError(f"Canonical reuse source changed: {source}")
    if old["split_hash"] != protocol["split_hash"] or old["panel_inputs"] != protocol["panel_inputs"]:
        raise ValueError("Canonical reuse inputs/splits changed")
    path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for source in protocol["sources"]:
        shutil.copyfile(source, snapshot / Path(source).name)
    (output / "canonical_reuse.json").write_text(
        json.dumps(
            {
                "source": str(CANONICAL_OUTPUT),
                "manifest_sha256": previous.previous.digest(CANONICAL_OUTPUT / "screen_manifest.json"),
                "sources_inputs_splits_match": True,
            },
            indent=2,
        )
        + "\n"
    )
    return protocol


@functools.cache
def initialize_worker(output: Path) -> None:
    protocol = json.loads((output / "screen_manifest.json").read_text())
    for source, checksum in protocol["sources"].items():
        if previous.previous.digest(Path(source)) != checksum:
            raise ValueError(f"Frozen fitting source changed: {source}")
    register_candidates()


def augment(task: bench.FitTask, output: Path, saved: dict) -> None:
    """Reconstruct the final head and independently replay the returned inner score."""
    panel = bench.load_panel(task.panel)
    features = dataclasses.replace(panel.features, component=task.component)
    model = registry.ENTRY_BY_ID[task.model_id].build(features)
    split = next(s for s in bench.panel_splits(panel, 1) if s.fold == task.fold)
    response = panel.group(task.target).outcomes[:, task.component_index]
    shape = json.loads(str(saved["shape_json"]))
    diagnostics = json.loads(str(saved["diagnostics_json"]))
    if isinstance(model, candidates.FittingModel):
        grid = model.base
        spec = model.head_spec(shape, float(diagnostics["kappa"]))
        procedure = model.procedure
    elif isinstance(model, models.FittedFloorModel) and isinstance(model.base, models.GridModel):
        grid = model.base
        spec = model._spec(shape, models.LinkKind(diagnostics["link"]), float(diagnostics["kappa"]))
        procedure = candidates.Procedure.CANONICAL
    else:
        raise TypeError("Unexpected fitting model")
    design = grid.design(features, shape)
    training_design = models.Design(design.values[split.train], design.ridge, design.names)
    head = candidates.fit_head(training_design, response[split.train], float(saved["ridge"]), spec, procedure)
    fitted = models.Fitted(shape, float(saved["ridge"]), head, diagnostics)
    np.testing.assert_allclose(model.predict(fitted, features, split.test), saved["prediction"], rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(
        model.predict(fitted, features, split.train), saved["train_prediction"], rtol=1e-9, atol=1e-10
    )
    observations, predictions = [], []
    for train, validation in split.inner:
        inner_design = models.Design(design.values[train], design.ridge, design.names)
        inner_head = candidates.fit_head(inner_design, response[train], fitted.ridge, spec, procedure)
        observations.append(response[validation])
        predictions.append(models.predict_head(inner_head, design.values[validation], spec))
    residual = np.concatenate(predictions) - np.concatenate(observations)
    rmse = float(np.sqrt(np.dot(residual, residual) / len(residual)))
    regret = float(np.mean([float(y[np.argmin(p)] - np.min(y)) for y, p in zip(observations, predictions, strict=True)]))
    objective = rmse + (0.5 * regret if procedure is candidates.Procedure.SELECTION else 0.0)
    np.testing.assert_allclose(rmse, diagnostics["inner_cv_rmse"], rtol=1e-10, atol=1e-12)
    if "inner_cv_objective" in diagnostics:
        np.testing.assert_allclose(objective, diagnostics["inner_cv_objective"], rtol=1e-10, atol=1e-12)
    vertex_features = dataclasses.replace(
        features,
        weights=np.eye(features.buckets),
        exposures=np.diag(features.inventory),
        label=features.label + "|cf2_vertices",
    )
    logits = grid.design(vertex_features, shape).values @ head.coefficients + head.intercept
    rms_squared = np.mean(training_design.values**2, axis=0)
    payload = {f"head_{key}": value for key, value in dataclasses.asdict(head).items()}
    payload.update(
        shape_json=saved["shape_json"],
        ridge=saved["ridge"],
        diagnostics_json=saved["diagnostics_json"],
        prediction_shard_sha256=previous.previous.digest(bench.shard_path(output, task)),
        train=split.train,
        test=split.test,
        replay_rmse=rmse,
        replay_regret=regret,
        replay_objective=objective,
        training_ridge_multipliers=candidates.training_penalty(training_design, procedure),
        fraction_rms_below_floor=float(np.mean(rms_squared <= candidates.PENALTY_BOUNDS[0])),
        fraction_rms_above_ceiling=float(np.mean(rms_squared >= candidates.PENALTY_BOUNDS[1])),
        vertex_logits=logits,
        upper_clip_active_at_vertices=bool(np.any(logits > min(models.LINK_CLIP, head.cap))),
    )
    bench.atomic_save(portable.task_head_path(output, task), payload)


def fit_task(task: bench.FitTask, output: Path, split_hash: str) -> str:
    initialize_worker(output)
    path = bench.shard_path(output, task)
    if task.model_id in candidates.FAMILY_IDS and not path.exists():
        source = bench.shard_path(CANONICAL_OUTPUT, task)
        original = bench.load_shard(source)
        if original is None:
            raise FileNotFoundError(source)
        panel = bench.load_panel(task.panel)
        hashes = bench.task_protocol_hashes(
            task, registry.ENTRY_BY_ID[task.model_id], split_hash, panel, CANONICAL_OUTPUT
        )
        split = next(s for s in bench.panel_splits(panel, 1) if s.fold == task.fold)
        if not bench.valid_shard(source, hashes, task.component, split.test):
            raise ValueError(f"Canonical reuse shard identity mismatch: {source}")
        bench.atomic_save(path, original)
    existing = bench.load_shard(path)
    if existing is not None and str(existing["status"]) != "ok":
        return "failed"
    status = bench.fit_one(task, output, split_hash)
    saved = bench.load_shard(path)
    if saved is not None and str(saved["status"]) == "ok":
        head = bench.load_shard(portable.task_head_path(output, task))
        if head is None or str(head["prediction_shard_sha256"]) != previous.previous.digest(path):
            augment(task, output, saved)
    return status


def planned_tasks(phase: str, output: Path) -> list[bench.FitTask]:
    ids = (*candidates.FAMILY_IDS, *(spec.model_id for spec in candidates.candidate_specs()))
    if phase == "full":
        ids = tuple(json.loads((output / "advancement.json").read_text())["full_model_ids"])
    tier = "screen" if phase == "anchors" else "certify"
    plan = dataclasses.replace(bench.tier_plan(tier, panels=bench.THIRTY_NINE_BUCKET_PANELS), curves=())
    tasks = bench.plan_tasks(plan, ids)
    return sorted(tasks, key=lambda task: (task.fold, task.component_index, task.panel, task.target, task.model_id))


def advancement(output: Path) -> None:
    destination = output / "report_anchors"
    folded = pd.read_csv(destination / "component_fold_metrics.csv")
    pooled = pd.read_csv(destination / "component_metrics.csv")
    balanced_regret = folded.groupby(["model", "panel", "target"]).regret_at_1.mean().groupby("model").mean()
    balanced_rho = pooled.groupby(["model", "panel", "target"]).spearman.mean().groupby("model").mean()
    results = pd.DataFrame({"mean_regret": balanced_regret, "spearman": balanced_rho})
    specs = {spec.model_id: spec for spec in candidates.candidate_specs()}
    for model_id, spec in specs.items():
        matched_id = f"cf2_mariner__{spec.procedure}"
        results.loc[model_id, "delta_regret_vs_canonical_family"] = float(results.at[model_id, "mean_regret"]) - float(
            results.at[spec.family_id, "mean_regret"]
        )
        results.loc[model_id, "delta_regret_vs_matched_mariner"] = float(results.at[model_id, "mean_regret"]) - float(
            results.at[matched_id, "mean_regret"]
        )
        results.loc[model_id, "delta_rho_vs_matched_mariner"] = float(results.at[model_id, "spearman"]) - float(
            results.at[matched_id, "spearman"]
        )
    results.to_csv(output / "anchor_balanced_summary.csv")
    eligible = results.loc[[model_id for model_id, spec in specs.items() if spec.family_id != "cv17_mariner"]]
    eligible = eligible[eligible.delta_regret_vs_canonical_family <= 0.001].reset_index()
    selected = (
        eligible.sort_values(["mean_regret", "spearman", "model"], ascending=[True, False, True]).head(2).model.tolist()
    )
    controls = sorted({f"cf2_mariner__{specs[model_id].procedure}" for model_id in selected})
    result = {
        "selected_convex": selected,
        "matched_controls": controls,
        "full_model_ids": selected + controls,
        "screen_sha256": previous.previous.digest(output / "anchor_balanced_summary.csv"),
        "criterion": json.loads((output / "screen_manifest.json").read_text())["advancement"],
    }
    path = output / "advancement.json"
    if path.exists() and json.loads(path.read_text()) != result:
        raise ValueError("Full-objective advancement is already frozen and differs")
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def report(tasks: list[bench.FitTask], output: Path, split_hash: str, phase: str) -> None:
    predictions, fits = bench.collect_shards(tasks, output, split_hash)
    destination = output / f"report_{phase}"
    destination.mkdir(exist_ok=True)
    fits.to_csv(destination / "fits.csv", index=False)
    failures = fits[fits.status.ne("ok")]
    failures.to_csv(destination / "failures.csv", index=False)
    if len(failures):
        raise ValueError(f"{len(failures)} failed records retained; advancement is blocked")
    previous.previous.report(tasks, output, split_hash, phase)
    heads = []
    for task in tasks:
        head = bench.load_shard(portable.task_head_path(output, task))
        if head is None:
            raise FileNotFoundError(portable.task_head_path(output, task))
        heads.append(
            dataclasses.asdict(task)
            | {
                key: float(head[key])
                for key in (
                    "replay_rmse",
                    "replay_regret",
                    "replay_objective",
                    "fraction_rms_below_floor",
                    "fraction_rms_above_ceiling",
                    "upper_clip_active_at_vertices",
                )
            }
        )
    pd.DataFrame(heads).to_csv(destination / "head_audit.csv", index=False)
    if phase == "anchors":
        advancement(output)
    else:
        aggregate = pd.read_csv(destination / "aggregate_predictions.csv")
        independent = portable.independent_aggregate(predictions)
        keys = portable.KEYS
        np.testing.assert_allclose(
            aggregate.set_index(keys).sort_index()[["prediction", "observed"]],
            independent.set_index(keys).sort_index()[["prediction", "observed"]],
            rtol=1e-12,
            atol=1e-12,
        )
        (destination / "independent_aggregation.json").write_text(
            json.dumps({"passed": True, "rows": len(aggregate), "heads": len(heads)}) + "\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--phase", choices=("anchors", "full"), default="anchors")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("Use one to four single-thread workers")
    register_candidates()
    manifest = prepare(args.output)
    if args.prepare_only:
        return
    tasks = planned_tasks(args.phase, args.output)
    if not args.report_only:
        module = importlib.import_module(
            "experiments.domain_phase_mix.exploratory.two_phase_many.screen_mariner_convex_fitting_round2_20260917"
        )
        started = time.monotonic()
        counts: dict[str, int] = {}
        with joblib.parallel_config(backend="loky", inner_max_num_threads=1):
            results = joblib.Parallel(n_jobs=args.workers, return_as="generator_unordered", batch_size=1)(
                joblib.delayed(module.fit_task)(task, args.output, manifest["split_hash"]) for task in tasks
            )
            for index, status in enumerate(results, 1):
                counts[status] = counts.get(status, 0) + 1
                if index % 24 == 0 or index == len(tasks) or status == "failed":
                    print(
                        json.dumps(
                            {
                                "completed": index,
                                "total": len(tasks),
                                "elapsed": round(time.monotonic() - started),
                                **counts,
                            }
                        ),
                        flush=True,
                    )
    report(tasks, args.output, manifest["split_hash"], args.phase)


if __name__ == "__main__":
    main()
