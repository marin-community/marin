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
"""Calibrate existing phase surrogates using a frozen local Delphi 3e18 inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import analyze_tpp40_frontier_gap_20260909 as previous
import numpy as np
import pandas as pd
from fit_two_phase_link_spines_20260907 import load_module, write_json_atomic
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits

BASE = Path(__file__).resolve().parent
REFERENCE = BASE / "reference_outputs"
OUTPUT = REFERENCE / "delphi_phase_frontier_calibration_20260910"
STANDALONE = BASE.parents[4] / "mixture-selection"
CANONICAL = REFERENCE / "two_phase_surrogate_collaborator_packet_20260721/data/canonical"
ARCHIVE = REFERENCE / "delphi_3e18_append_only_heldouts_20260714"
ALPHA = 2400 / 3007
SEED = 20260910
HPR = previous.HPR
TARGETS = {"uncheatable": "uncheatable_bpb", "table9": "table9_macro_bpb"}
EXPANDED_SOURCES = (
    "delphi_3e18_aggressive_phase_asymmetry_20260722",
    "delphi_3e18_frontier_phase_fiber_20260719",
    "delphi_3e18_frontier_random_phase_population_20260720",
    "delphi_3e18_adversarial_stress_panel_20260716",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(array: np.ndarray) -> str:
    return hashlib.sha256(np.round(array, 9).astype("<f8").tobytes()).hexdigest()[:20]


def prepare() -> None:
    if (OUTPUT / "snapshot.json").exists():
        snapshot = json.loads((OUTPUT / "snapshot.json").read_text())
        for name, digest in snapshot["frozen_hashes"].items():
            assert sha(OUTPUT / name) == digest, name
        print("reuse frozen snapshot", flush=True)
        return
    inputs = OUTPUT / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    sources = {}

    def read(path: Path) -> pd.DataFrame:
        sources[str(path)] = sha(path)
        return pd.read_csv(path, low_memory=False)

    for name in (
        "swarm_weights",
        "swarm_outcomes",
        "buckets",
        "objectives",
        "anchors",
        "splits",
        "reference_task_fits",
        "reference_swarm_predictions_uncheatable",
        "reference_swarm_predictions_table9",
    ):
        path = STANDALONE / "data" / f"{name}.csv"
        sources[str(path)] = sha(path)
        shutil.copyfile(path, inputs / path.name)
    source = STANDALONE / "mixture_selection.py"
    sources[str(source)] = sha(source)
    shutil.copyfile(source, inputs / "mariner.py")
    buckets = tuple(read(inputs / "buckets.csv").bucket)
    inventory = read(inputs / "buckets.csv").epochs_per_unit_weight.to_numpy()
    records, policies, exclusions = [], [], []
    seen = set()

    def add(record: dict, weights: np.ndarray) -> None:
        assert isinstance(record["run_id"], str) and record["run_id"]
        assert weights.shape == (2, len(buckets))
        assert np.isfinite(weights).all() and weights.min() >= -1e-12
        assert np.max(np.abs(weights.sum(axis=1) - 1)) < 1e-6
        weights = np.maximum(weights, 0)
        weights /= weights.sum(axis=1, keepdims=True)
        if record["run_id"] in seen:
            return
        seen.add(record["run_id"])
        aggregate = ALPHA * weights[0] + (1 - ALPHA) * weights[1]
        record.update(
            policy_id=key(weights),
            aggregate_id=key(aggregate),
            tied=bool(np.max(np.abs(weights[0] - weights[1])) < 1e-10),
        )
        policies.append(weights)
        records.append(record)

    archive = read(ARCHIVE / "heldout_current.csv")
    assert not archive.training_series.str.contains("targeted_pairwise", case=False, na=False).any()
    for row in archive.to_dict("records"):
        if not np.isclose(row["phase_0_fraction"], ALPHA, atol=1e-10):
            exclusions.append({"run_id": row["heldout_id"], "reason": "different phase fraction"})
            continue
        assert row["training_state"] == "finished" and row["global_step"] == 3006
        weights = np.array([[json.loads(row[f"phase_{p}_weights_json"])[b] for b in buckets] for p in range(2)])
        add(
            dict(
                run_id=row["wandb_run_id"],
                archive_id=row["heldout_id"],
                source=row["training_series"],
                cohort="archive",
                candidate=row["wandb_run_name"],
                data_seed=row["data_seed.1"],
                trainer_seed=row["trainer_seed.1"],
                target=row["objective"],
                table9_source=row["table9_metric_source"],
                **{t: row[c] for t, c in TARGETS.items()},
            ),
            weights,
        )
    registry = read(REFERENCE / "single_phase_heldout_benchmark_20260902/heldout_runs.csv")
    registry = registry[registry.panel.eq("delphi_3e18_39bucket")]
    for row in registry.to_dict("records"):
        if max(abs(row[f"pool_fraction::{b}"] - 1) for b in buckets) > 1e-10:
            exclusions.append({"run_id": row["row_id"], "reason": "different pool fractions"})
            continue
        vector = np.array([row[f"weight::{b}"] for b in buckets])
        add(
            dict(
                run_id=row["training_wandb_run_id"],
                archive_id="",
                source=row["source"],
                cohort="one_phase_registry",
                candidate=row["source_row_id"],
                data_seed=row["data_seed"],
                trainer_seed=row["trainer_seed"],
                target=row["proposal_target"],
                table9_source="component_audited_registry",
                **{t: row[c] for t, c in TARGETS.items()},
            ),
            np.stack([vector, vector]),
        )
    for path in sorted(REFERENCE.glob("delphi*/measured_results.csv")):
        if " " in path.parent.name or ("3e18" not in path.parent.name and "frontier_factorial" not in path.parent.name):
            continue
        table = read(path)
        if "status" not in table:
            continue  # Older batches already enter through the component-audited registry.
        weights_path = path.parent / "runtime_materialization/candidate_weights.csv"
        if not weights_path.exists():
            weights_path = path.parent / "candidate_weights.csv"
        if not weights_path.exists():
            raise FileNotFoundError(weights_path)
        weight_table = read(weights_path)
        if path.parent.name == "delphi_frontier_factorial_design_20260906":
            weight_table = pd.concat([weight_table, read(path.parent / "candidate_weights_replicate.csv")])
        for row in table.to_dict("records"):
            if row["status"] != "measured":
                exclusions.append(
                    {
                        "run_id": f"{path.parent.name}:{row['candidate_id']}:{row.get('group', '')}",
                        "reason": row["status"],
                    }
                )
                continue
            selected = weight_table[weight_table.candidate_id.eq(row["candidate_id"])]
            assert not selected.domain.duplicated().any()
            vector = selected.set_index("domain").weight.reindex(buckets).fillna(0).to_numpy()
            uri = row["eval_metrics_uri"]
            add(
                dict(
                    run_id=uri,
                    archive_id="",
                    source=path.parent.name,
                    cohort="recent_validation",
                    candidate=row["candidate_id"],
                    data_seed=row.get("data_seed", np.nan),
                    trainer_seed=row.get("trainer_seed", np.nan),
                    target=row["target"],
                    table9_source="native_validation_collector",
                    **{t: row[c] for t, c in TARGETS.items()},
                ),
                np.stack([vector, vector]),
            )
    external_n = len(records)
    for phase in (1, 2):
        table = read(CANONICAL / f"delphi_3e18_{'one' if phase == 1 else 'two'}_phase_fit.csv")
        table.to_csv(inputs / f"canonical_{phase}p.csv", index=False)
        for row in table.to_dict("records"):
            weights = np.array([[row[f"phase_{p}_weight::{b}"] for b in buckets] for p in range(2)])
            add(
                dict(
                    run_id=f"canonical_{phase}p:{row['row_id']}",
                    archive_id="",
                    source=f"canonical_{phase}p",
                    cohort="canonical",
                    candidate=row["row_id"],
                    data_seed=np.nan,
                    trainer_seed=np.nan,
                    target="",
                    table9_source="canonical",
                    **{t: row[c] for t, c in TARGETS.items()},
                ),
                weights,
            )
    frame = pd.DataFrame(records)
    weights = np.stack(policies)
    aggregate = ALPHA * weights[:, 0] + (1 - ALPHA) * weights[:, 1]
    canonical = frame.cohort.eq("canonical").to_numpy()
    distance = cdist(
        weights.reshape(len(weights), -1), weights[canonical].reshape(canonical.sum(), -1), "chebyshev"
    ).min(axis=1)
    frame["training_overlap"] = distance < 1e-9
    frame["external"] = ~canonical
    frame["nearest_training_policy_tv"] = (
        cdist(
            (weights * np.array([ALPHA, 1 - ALPHA])[None, :, None]).reshape(len(weights), -1),
            (weights[canonical] * np.array([ALPHA, 1 - ALPHA])[None, :, None]).reshape(canonical.sum(), -1),
            "cityblock",
        ).min(axis=1)
        / 2
    )
    # Match old frontier-cell IDs from the archived pairing audit, independently of rounded hash conventions.
    meta_path = REFERENCE / "two_phase_hpr_transfer_20260907/identification/archive_large_cell_run_metadata.csv"
    pairs_path = REFERENCE / "two_phase_hpr_transfer_20260907/identification/archive_mirror_pairs.csv"
    read(meta_path).to_csv(inputs / "frontier_metadata.csv", index=False)
    read(pairs_path).to_csv(inputs / "mirror_pairs.csv", index=False)
    template_path = (
        REFERENCE / "two_phase_link_transfer_20260907/controls/hierarchical_phase_replay/uncheatable/full/model.pkl"
    )
    sources[str(template_path)] = sha(template_path)
    shutil.copyfile(template_path, inputs / "hpr_template.pkl")
    frame.to_csv(inputs / "observations.csv", index=False)
    np.savez_compressed(inputs / "coordinates.npz", weights=weights, aggregate=aggregate, inventory=inventory)
    pd.DataFrame(exclusions).to_csv(inputs / "exclusions.csv", index=False)
    write_json_atomic(
        OUTPUT / "snapshot.json",
        {
            "source_hashes": sources,
            "external_identity_rows": external_n,
            "rows_with_canonical_aliases": len(frame),
            "unique_policy_coordinates": frame.policy_id.nunique(),
            "canonical_policy_coordinates": frame.loc[canonical, "policy_id"].nunique(),
            "external_heldout_rows": int((frame.external & ~frame.training_overlap).sum()),
            "excluded_rows": len(exclusions),
            "alpha": ALPHA,
            "frozen_hashes": {str(p.relative_to(OUTPUT)): sha(p) for p in inputs.iterdir() if p.is_file()},
        },
    )
    print(frame.groupby(["cohort", "tied", "training_overlap"]).size().to_string(), flush=True)


def data() -> tuple[pd.DataFrame, dict, Any, Any]:
    frame = pd.read_csv(OUTPUT / "inputs/observations.csv")
    with np.load(OUTPUT / "inputs/coordinates.npz") as arrays:
        coordinates = {k: arrays[k] for k in arrays.files}
    module = load_module(OUTPUT / "inputs/mariner.py", "delphi_calibration_mariner")
    with (OUTPUT / "inputs/hpr_template.pkl").open("rb") as handle:
        template = pickle.load(handle).dataset
    return frame, coordinates, module, template


def reconstruct_mariner(module: Any, objective: str) -> Any:
    inputs = OUTPUT / "inputs"
    swarm = module.read_swarm(inputs / "swarm_weights.csv", inputs / "swarm_outcomes.csv", inputs / "buckets.csv")
    definition = module.read_objectives(inputs / "objectives.csv")[objective]
    anchors = module.read_anchors(inputs / "anchors.csv", objective)
    references = pd.read_csv(inputs / "reference_task_fits.csv").set_index(["objective", "component"])
    tasks = []
    for component in definition.components:
        row = references.loc[(objective, component)]
        anchor = anchors[component]
        shape = {k: float(row[k]) for k in ("rate", "power", "threshold")}
        spec = module.FloorSpec(anchor.proportional, anchor.repeat_sd, row.kappa)
        head = module.fit_head(
            module.design_matrix(swarm.exposures, shape), swarm.outcomes[component].to_numpy(), row.ridge, spec
        )
        assert abs(head.floor - row.floor) < 1e-12
        tasks.append(module.TaskFit(component, shape, row.ridge, row.kappa, False, row.inner_cv_rmse, head))
    fitted = module.ObjectiveFit(objective, swarm.buckets, swarm.inventory, definition.weights, tuple(tasks))
    reference = pd.read_csv(inputs / f"reference_swarm_predictions_{objective}.csv")
    assert list(reference.run) == list(swarm.runs)
    error = float(
        np.max(np.abs(fitted.predict_tasks(swarm.weights) - reference[list(definition.components)].to_numpy()))
    )
    assert error < 1e-9, error
    write_json_atomic(OUTPUT / "fits" / f"mariner_{objective}.json", fitted.to_json())
    write_json_atomic(
        OUTPUT / "fits" / f"mariner_{objective}_parity.json",
        {
            "max_abs_error": error,
            "rows": len(swarm.runs),
            "components": len(tasks),
            "reconstruction": "shipped selected hyperparameters; refit coefficients; no heldout labels",
        },
    )
    return fitted


def fit_hpr(frame: pd.DataFrame, arrays: dict, template: Any, rows: np.ndarray, target: str) -> tuple[Any, dict]:
    selected = frame.iloc[rows].copy()
    selected["position"] = rows
    represented = set(selected.loc[selected.external, "policy_id"])
    selected = selected[~(selected.cohort.eq("canonical") & selected.policy_id.isin(represented))]
    # The two canonical panels share 42 training observations. Identical canonical
    # coordinates and outcomes represent those aliases, not extra replicates.
    canonical = selected[selected.cohort.eq("canonical")].drop_duplicates(["policy_id", target])
    selected = pd.concat([selected[~selected.cohort.eq("canonical")], canonical])
    grouped = selected.groupby("policy_id", sort=False).agg(position=("position", "first"), measured=(target, "mean"))
    indices = grouped.position.to_numpy(int)
    weights = arrays["weights"][indices]
    aggregate = arrays["aggregate"][indices]
    natural = 1 / arrays["inventory"]
    natural /= natural.sum()
    calibration = np.max(np.abs(aggregate - natural), axis=1) < 1e-9
    # Only the tied proportional coordinate is calibration; its repeats enter its mean once.
    calibration &= np.max(np.abs(weights[:, 0] - weights[:, 1]), axis=1) < 1e-9
    assert calibration.sum() == 1, int(calibration.sum())
    labels = previous.labels_for(aggregate, calibration, SEED)
    dataset = replace(
        template,
        frame=frame.iloc[indices].reset_index(drop=True),
        target=grouped.measured.to_numpy(),
        weights=weights,
        c0=ALPHA * arrays["inventory"],
        c1=(1 - ALPHA) * arrays["inventory"],
    )
    fitted, selection = previous.fitted_hpr(dataset, labels)
    selection["training_run_ids"] = selected.run_id.tolist()
    selection["training_policy_ids"] = grouped.index.tolist()
    selection["training_aggregate_ids"] = frame.iloc[indices].aggregate_id.tolist()
    return fitted, selection


def fit(context: str) -> None:
    frame, arrays, module, template = data()
    destination = OUTPUT / "fits" / context
    identity = {
        str(p): sha(p)
        for p in (
            Path(__file__),
            OUTPUT / "PROTOCOL.md",
            OUTPUT / "snapshot.json",
            Path(HPR.__file__),
            Path(HPR.family_grp.__file__),
            Path(previous.__file__),
        )
    }
    if (destination / "complete.json").exists():
        assert json.loads((destination / "complete.json").read_text())["identity"] == identity
        print(f"reuse {context}", flush=True)
        return
    destination.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    for objective in TARGETS:
        mariner = reconstruct_mariner(module, objective)
        if context == "canonical":
            train = frame.source.eq("canonical_2p").to_numpy()
            test = frame.external.to_numpy() & ~frame.training_overlap.to_numpy()
        else:
            source = EXPANDED_SOURCES[int(context[-1])]
            test = frame.source.eq(source).to_numpy() & ~frame.training_overlap.to_numpy()
            forbidden = set(frame.loc[test, "aggregate_id"])
            train = ~frame.source.eq(source).to_numpy() & ~frame.aggregate_id.isin(forbidden).to_numpy()
            assert not set(frame.loc[test, "policy_id"]) & set(frame.loc[train, "policy_id"])
            assert not set(frame.loc[test, "aggregate_id"]) & set(frame.loc[train, "aggregate_id"])
        train &= frame[objective].notna().to_numpy()
        fitted, selection = fit_hpr(frame, arrays, template, np.flatnonzero(train), objective)
        prediction = fitted.predict(arrays["weights"])
        tied = fitted.predict(np.repeat(arrays["aggregate"][:, None, :], 2, axis=1))
        aggregate = mariner.predict(arrays["aggregate"])
        table = frame.copy()
        table["measured"] = table[objective]
        table["hpr"], table["hpr_tied"], table["mariner"] = prediction, tied, aggregate
        table["replacement"] = aggregate + prediction - tied
        table["phase_prediction"] = prediction - tied
        table["in_train"], table["in_test"] = train, test & frame[objective].notna().to_numpy()
        assert np.max(np.abs(table.loc[table.tied, "replacement"] - table.loc[table.tied, "mariner"])) < 1e-12
        table.to_csv(destination / f"{objective}_predictions.csv", index=False)
        (destination / f"{objective}_hpr.pkl").write_bytes(pickle.dumps(fitted))
        write_json_atomic(destination / f"{objective}_selection.json", selection)
        print(
            f"{context} {objective}: {train.sum()} training rows, "
            f"{len(selection['training_policy_ids'])} coordinates, {table.in_test.sum()} held out; "
            f"{time.monotonic() - started:.1f}s",
            flush=True,
        )
    write_json_atomic(
        destination / "complete.json",
        {
            "identity": identity,
            "seconds": time.monotonic() - started,
            "hashes": {p.name: sha(p) for p in destination.iterdir() if p.is_file()},
        },
    )


def summary(group: pd.DataFrame, model: str) -> dict:
    order = group.sort_values([model, "policy_id"], kind="stable")
    y, p = group.measured.to_numpy(), group[model].to_numpy()
    residual = y - p
    best = float(y.min())
    return dict(
        n=len(group),
        rmse=float(np.sqrt(np.mean(residual**2))),
        optimism=float(residual.mean()),
        spearman=float(spearmanr(y, p).statistic) if np.std(p) > 1e-12 and len(group) > 2 else np.nan,
        regret1=float(order.measured.iloc[0] - best),
        regret5=float(order.measured.iloc[:5].min() - best),
        regret10=float(order.measured.iloc[:10].min() - best),
        selected_optimism=float(order.measured.iloc[0] - order[model].iloc[0]),
        selected_measured=float(order.measured.iloc[0]),
        selected_predicted=float(order[model].iloc[0]),
        selected_policy=order.policy_id.iloc[0],
        observed_best=best,
    )


def score() -> None:
    records, predictions = [], []
    for directory in sorted((OUTPUT / "fits").glob("*")):
        if not directory.is_dir():
            continue
        for objective in TARGETS:
            table = pd.read_csv(directory / f"{objective}_predictions.csv")
            table = table[table.in_test & table.measured.notna()]
            # Equal weight per policy, keeping source-specific copies only in source reports.
            for scope, population in [("all_sources", table), *list(table.groupby("source"))]:
                for tied, group in population.groupby("tied"):
                    group = group.groupby("policy_id", as_index=False).agg(
                        measured=("measured", "mean"),
                        mariner=("mariner", "first"),
                        hpr=("hpr", "first"),
                        replacement=("replacement", "first"),
                        aggregate_id=("aggregate_id", "first"),
                        nearest_training_policy_tv=("nearest_training_policy_tv", "first"),
                    )
                    for model in ("mariner", "hpr", "replacement"):
                        for band, selected in [
                            ("all", group),
                            ("predicted_frontier20", group.nsmallest(max(1, int(np.ceil(0.2 * len(group)))), model)),
                        ]:
                            records.append(
                                dict(
                                    context=directory.name,
                                    objective=objective,
                                    scope=scope,
                                    tied=tied,
                                    model=model,
                                    band=band,
                                    **summary(selected, model),
                                )
                            )
            table["context"], table["objective"] = directory.name, objective
            predictions.append(table)
    pd.DataFrame(records).to_csv(OUTPUT / "calibration_metrics.csv", index=False)
    pd.concat(predictions).to_csv(OUTPUT / "heldout_predictions.csv", index=False)
    print(
        pd.DataFrame(records).query("context == 'canonical' and scope == 'all_sources'").round(5).to_string(index=False),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "fit", "score"))
    parser.add_argument(
        "--context", default="canonical", choices=("canonical", "expanded0", "expanded1", "expanded2", "expanded3")
    )
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.action == "prepare":
            prepare()
        elif args.action == "fit":
            fit(args.context)
        else:
            score()


if __name__ == "__main__":
    main()
