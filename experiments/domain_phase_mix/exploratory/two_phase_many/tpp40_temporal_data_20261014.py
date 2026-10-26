# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Complete local TPP40 panel and outcome-independent temporal-screen folds.

This adapter leaves the September benchmark unchanged. Boundary measurements
are supplied for explicitly state-conditioned diagnostics, never as open-loop
policy inputs. Tied endpoints never enter either fitting partition.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.domain_phase_mix.exploratory.two_phase_many import two_phase_benchmark_data_20260921 as benchmark

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "reference_outputs/tpp40_complete_20261014"
OUTPUT = BASE / "reference_outputs/tpp40_temporal_round_20261014/data"
OUTER_FOLDS = 5
INNER_FOLDS = 3
SPLIT_SEED = 20261014
INPUT_NAMES = (
    "policies.csv",
    "policy_weights.csv",
    "bucket_axis.csv",
    "objectives.csv",
    "outcomes.csv",
    "phase0_native.csv",
)


@dataclass(frozen=True)
class TemporalData:
    """Aligned final/boundary observations with frozen policy geometry and folds."""

    panel: benchmark.BenchPanel
    task_names: tuple[str, ...]
    final_targets: np.ndarray
    boundary_targets: np.ndarray
    objective_weights: np.ndarray
    boundary_atom_names: tuple[str, ...]
    boundary_atoms: np.ndarray
    folds: pd.DataFrame
    full_data_inner_labels: np.ndarray
    audit: dict

    @property
    def rows(self) -> pd.DataFrame:
        return self.panel.rows


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_inputs(source: Path) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    snapshot_path = source / "snapshot.json"
    snapshot = json.loads(snapshot_path.read_text())
    hashes = {str(snapshot_path): _sha256(snapshot_path)}
    frames = {}
    for name in INPUT_NAMES:
        path = source / "inputs" / name
        actual = _sha256(path)
        if actual != snapshot["sha256"][f"inputs/{name}"]:
            raise ValueError(f"Source differs from the complete frozen snapshot: {path}")
        hashes[str(path)] = actual
        frames[name] = pd.read_csv(path, float_precision="round_trip")
    return frames, hashes


def _alias_groups(rows: pd.DataFrame) -> np.ndarray:
    """Join policies sharing an aggregate, phase prefix, or full policy."""
    edges = []
    for name in ("aggregate_id", "realized_aggregate_id", "prefix_id", "policy_id"):
        for positions in rows.groupby(name, sort=True).indices.values():
            if len(positions) > 1:
                edges.extend((int(positions[0]), int(other)) for other in positions[1:])
    n = len(rows)
    left, right = zip(*edges, strict=True) if edges else ([], [])
    graph = csr_matrix((np.ones(len(edges)), (left, right)), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    group_ids = {label: min(rows.aggregate_id.iloc[np.flatnonzero(labels == label)]) for label in np.unique(labels)}
    return np.asarray([group_ids[label] for label in labels])


def _alias_audit(rows: pd.DataFrame, folds: pd.DataFrame) -> dict:
    joined = folds.merge(rows, on="row_id", validate="many_to_one")
    summaries: dict[str, object] = {}
    for name in ("aggregate_id", "realized_aggregate_id", "prefix_id", "policy_id", "group_id"):
        counts = rows[name].value_counts()
        summaries[name] = {
            "groups": len(counts),
            "repeated_groups": int((counts > 1).sum()),
            "rows_in_repeated_groups": int(counts[counts > 1].sum()),
            "maximum_group_size": int(counts.max()),
        }
        scored = joined.loc[joined.role.ne("train")]
        if not scored.groupby(name).outer_fold.nunique().eq(1).all():
            raise ValueError(f"Alias group crosses outer scoring folds: {name}")
        for outer in range(OUTER_FOLDS):
            block = joined.loc[joined.outer_fold.eq(outer)]
            train = block.loc[block.role.eq("train")]
            if set(train[name]) & set(block.loc[block.role.ne("train"), name]):
                raise ValueError(f"Alias group crosses train/test: {name}, outer {outer}")
            if not train.groupby(name).inner_fold.nunique().eq(1).all():
                raise ValueError(f"Alias group crosses inner folds: {name}, outer {outer}")
    summaries["matched_asymmetric_tied_aggregate_groups"] = int(rows.groupby("aggregate_id").tied.nunique().eq(2).sum())
    summaries["matched_asymmetric_tied_realized_aggregate_groups"] = int(
        rows.groupby("realized_aggregate_id").tied.nunique().eq(2).sum()
    )
    return summaries


def load_complete_tpp40(source: Path = SOURCE) -> TemporalData:
    """Read the frozen complete panel and derive nested splits from policies only."""
    frames, hashes = _read_inputs(source)
    policies = frames["policies.csv"].sort_values("order").reset_index(drop=True)
    if len(policies) != 280 or policies.order.tolist() != list(range(280)) or policies.run_name.duplicated().any():
        raise ValueError("The complete TPP40 panel must contain exactly 280 unique ordered policies")
    axis = frames["bucket_axis.csv"].sort_values("bucket_index")
    bucket_names = tuple(axis.bucket.astype(str))
    if len(bucket_names) != 39 or axis.bucket_index.tolist() != list(range(39)):
        raise ValueError("Unexpected TPP40 bucket axis")
    weights = frames["policy_weights.csv"]
    if weights.duplicated(["order", "bucket_index"]).any() or len(weights) != 280 * 39:
        raise ValueError("Incomplete or duplicated policy weights")
    phase0, phase1 = (
        weights.pivot(index="order", columns="bucket_index", values=f"phase{phase}_weight")
        .loc[policies.order, axis.bucket_index]
        .to_numpy(float)
        for phase in (0, 1)
    )
    if policies.alpha_nominal.nunique() != 1 or policies.alpha_realized.nunique() != 1:
        raise ValueError("TPP40 has inconsistent phase fractions")
    nominal_alpha = float(policies.alpha_nominal.iloc[0])
    alpha = float(policies.alpha_realized.iloc[0])
    if abs(alpha - 21856 / 27336) > 1e-14 or abs(nominal_alpha - 0.8) > 1e-14:
        raise ValueError("Unexpected TPP40 exposure fractions")
    if policies.simulated_epoch_target_budget.nunique() != 1:
        raise ValueError("TPP40 has inconsistent simulated exposure budgets")
    frozen_inventory = axis.epochs_per_unit_weight.to_numpy(float)
    pool_tokens = axis.pool_tokens.to_numpy(float)
    exposure_budget = float(policies.simulated_epoch_target_budget.iloc[0])
    inventory = exposure_budget / pool_tokens
    supplied_epochs = (
        weights.pivot(index="order", columns="bucket_index", values="exposure_epochs_nominal")
        .loc[policies.order, axis.bucket_index]
        .to_numpy(float)
    )
    if not np.allclose(
        supplied_epochs, (nominal_alpha * phase0 + (1 - nominal_alpha) * phase1) * inventory, rtol=1e-13, atol=1e-13
    ):
        raise ValueError("Configured exposure inventory does not reproduce the assembled physical epoch table")
    rows = benchmark._rows(
        "tpp40_complete",
        phase0,
        phase1,
        nominal_alpha,
        policies.run_name.to_numpy(str),
        policies.source.to_numpy(str),
        policies.region.to_numpy(str),
    )
    rows["aggregate_id"] = rows.group_id
    rows["realized_aggregate_id"] = [benchmark._coordinate_key(row) for row in alpha * phase0 + (1 - alpha) * phase1]
    rows["prefix_id"] = [benchmark._coordinate_key(row) for row in phase0]
    rows["group_id"] = _alias_groups(rows)
    rows["row_index"] = np.arange(len(rows))
    for name in policies.columns:
        if name not in rows:
            rows[name] = policies[name].to_numpy()
    exact_tied = np.all(phase0 == phase1, axis=1)
    if (
        not np.array_equal(exact_tied, policies.tied)
        or not np.array_equal(exact_tied, rows.tied)
        or exact_tied.sum() != 42
    ):
        raise ValueError("Tied status disagrees between exact, historical, and manifest definitions")
    objectives = frames["objectives.csv"].query("objective == 'uncheatable'")
    task_names = tuple(objectives.component.astype(str))
    objective_weights = objectives.weight.to_numpy(float)
    if len(task_names) != 7 or not np.all(objective_weights > 0):
        raise ValueError("Expected seven positively weighted Uncheatable components")
    endpoint = frames["outcomes.csv"].set_index("run_name", verify_integrity=True).loc[rows.row_id]
    boundary = frames["phase0_native.csv"].set_index("run_name", verify_integrity=True).loc[rows.row_id]
    final_targets = endpoint[list(task_names)].to_numpy(float)
    boundary_targets = boundary[list(task_names)].to_numpy(float)
    paloma = tuple(
        sorted(
            key
            for key in boundary.columns.astype(str)
            if key.startswith("eval/paloma/") and key.endswith("/bpb") and key.count("/") == 3
        )
    )
    boundary_atom_names = (*task_names, *paloma)
    boundary_atoms = boundary[list(boundary_atom_names)].to_numpy(float)
    if len(boundary_atom_names) != 23 or not all(
        np.isfinite(values).all() for values in (final_targets, boundary_targets, boundary_atoms)
    ):
        raise ValueError("Incomplete Uncheatable or boundary atomic observations")
    errors = {}
    for role, frame, targets in (("endpoint", endpoint, final_targets), ("boundary", boundary, boundary_targets)):
        errors[role] = float(
            np.max(np.abs(targets @ objective_weights - frame["eval/uncheatable_eval/bpb"].to_numpy(float)))
        )
        if errors[role] > 3e-6:
            raise ValueError(f"Frozen objective weights do not reconstruct {role} native Uncheatable")
    metadata = {
        "grouping_alpha": nominal_alpha,
        "group_rounding_decimals": benchmark.GROUP_DECIMALS,
        "native_constant_exposure_adapter": True,
        "scope": "280 complete TPP40 policies; 238 asymmetric fit-eligible, 42 tied eval-only",
        "snapshot": str(source / "snapshot.json"),
        "exposure_convention": "realized phase durations times simulated epoch inventory; no phase-boundary reset",
        "boundary_contract": "measured-boundary inputs are state-conditioned diagnostics only",
        "pool_tokens": pool_tokens.tolist(),
        "simulated_epoch_target_budget": exposure_budget,
        "frozen_mariner_epochs_per_unit_weight": frozen_inventory.tolist(),
        "frozen_inventory_relative_exposure_difference": float(np.max(np.abs(frozen_inventory / inventory - 1))),
        "objective_components": {
            "uncheatable_bpb": {"components": list(task_names), "weights": objective_weights.tolist()}
        },
    }
    panel = benchmark._validate(
        benchmark.BenchPanel(
            "tpp40_complete",
            bucket_names,
            phase0,
            phase1,
            alpha,
            alpha * inventory,
            (1 - alpha) * inventory,
            {
                **{task: final_targets[:, index] for index, task in enumerate(task_names)},
                "uncheatable_bpb": final_targets @ objective_weights,
            },
            rows,
            hashes,
            metadata,
        )
    )
    folds = benchmark.frozen_folds(panel, OUTER_FOLDS, INNER_FOLDS, SPLIT_SEED)
    folds = folds.merge(rows[["row_id", "row_index"]], on="row_id", validate="many_to_one")
    asymmetric = np.flatnonzero(~panel.tied)
    full_inner = np.full(len(rows), -1, dtype=int)
    full_inner[asymmetric] = benchmark._neighborhood_labels(panel, asymmetric, INNER_FOLDS, SPLIT_SEED + OUTER_FOLDS + 1)
    audit = benchmark.split_coverage_audit(
        panel,
        folds,
        outer_folds=OUTER_FOLDS,
        inner_folds=INNER_FOLDS,
        minimum_asymmetric_test_rows=2,
        minimum_asymmetric_test_groups=2,
        expected_group_count=280,
    )
    if audit["status"] != "sufficient":
        raise ValueError(f"Incomplete TPP40 fold challenge: {audit}")
    audit["alias_groups"] = _alias_audit(rows, folds)
    audit["alias_grouping_recipe"] = (
        "Union closure of nominal aggregate, realized aggregate, exact phase0 prefix, and full-policy aliases "
        "after rounding coordinates to 10 decimals; group key is the lexicographically smallest nominal "
        "aggregate hash in each connected component."
    )
    audit["full_data_inner_seed"] = SPLIT_SEED + OUTER_FOLDS + 1
    audit["full_data_inner_counts"] = {str(fold): int(np.sum(full_inner == fold)) for fold in range(INNER_FOLDS)}
    audit["aggregate_reconstruction_max_error"] = errors
    audit["objective_weight_sum_not_renormalized"] = float(objective_weights.sum())
    audit["limitations"] = [
        "No exact aggregate-matched tied/asymmetric policies or repeated prefixes exist in this design.",
        "Spatial folds are not region-disjoint; region/source counts are diagnostic, not independent replication.",
        "All asymmetric policies have qsplit_signal source; source-disjoint asymmetric validation is unavailable.",
        "Measured boundary state is inadmissible to the open-loop candidate and belongs to a separate diagnostic.",
        "All observations are development evidence after inspection; no prospective confirmation is claimed.",
    ]
    audit["region_source_counts"] = (
        folds.merge(rows[["row_id", "region", "source"]], on="row_id", validate="many_to_one")
        .groupby(["outer_fold", "role", "region", "source"])
        .size()
        .reset_index(name="rows")
        .to_dict(orient="records")
    )
    return TemporalData(
        panel,
        task_names,
        final_targets,
        boundary_targets,
        objective_weights,
        boundary_atom_names,
        boundary_atoms,
        folds,
        full_inner,
        audit,
    )


def build_snapshot(output_dir: Path = OUTPUT, source: Path = SOURCE) -> dict:
    """Write reproducible local arrays, fold assignments, and provenance hashes."""
    data = load_complete_tpp40(source)
    arrays = io.BytesIO()
    np.savez_compressed(
        arrays,
        phase0=data.panel.phase0,
        phase1=data.panel.phase1,
        alpha=np.asarray(data.panel.alpha),
        nominal_alpha=np.asarray(data.panel.metadata["grouping_alpha"]),
        phase0_epochs=data.panel.phase0_epochs,
        phase1_epochs=data.panel.phase1_epochs,
        aggregate=data.panel.aggregate,
        nominal_aggregate=data.panel.grouping_aggregate,
        contrast=data.panel.contrast,
        bucket_names=np.asarray(data.panel.buckets),
        c0=data.panel.c0,
        c1=data.panel.c1,
        pool_tokens=np.asarray(data.panel.metadata["pool_tokens"]),
        frozen_mariner_epochs_per_unit_weight=np.asarray(data.panel.metadata["frozen_mariner_epochs_per_unit_weight"]),
        final_targets=data.final_targets,
        boundary_targets=data.boundary_targets,
        task_names=np.asarray(data.task_names),
        objective_weights=data.objective_weights,
        boundary_atoms=data.boundary_atoms,
        boundary_atom_names=np.asarray(data.boundary_atom_names),
        tied=data.panel.tied,
        row_ids=data.rows.row_id.to_numpy(str),
        full_data_inner_labels=data.full_data_inner_labels,
    )
    contents = {
        "rows.csv": data.rows.to_csv(index=False, float_format="%.17g").encode(),
        "folds.csv": data.folds.to_csv(index=False).encode(),
        "arrays.npz": arrays.getvalue(),
        "split_audit.json": (json.dumps(data.audit, indent=2, sort_keys=True) + "\n").encode(),
    }
    manifest = {
        "dataset": "tpp40_temporal_round_20261014",
        "rows": len(data.rows),
        "asymmetric": int((~data.panel.tied).sum()),
        "tied_eval_only": int(data.panel.tied.sum()),
        "primary_components": list(data.task_names),
        "boundary_atoms": list(data.boundary_atom_names),
        "metadata": data.panel.metadata,
        "split": {
            "seed": SPLIT_SEED,
            "outer_folds": OUTER_FOLDS,
            "inner_folds": INNER_FOLDS,
            "outcome_independent": True,
        },
        "source_hashes": data.panel.input_hashes,
        "implementation_hashes": {
            str(Path(__file__)): _sha256(Path(__file__)),
            str(Path(benchmark.__file__)): _sha256(Path(benchmark.__file__)),
        },
        "package_versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
        "files": {name: hashlib.sha256(payload).hexdigest() for name, payload in contents.items()},
    }
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Refusing to replace a different frozen data manifest; use a new output directory")
    for name, payload in contents.items():
        path = output_dir / name
        if path.exists() and path.read_bytes() != payload:
            raise ValueError(f"Refusing to overwrite a different frozen data artifact: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in contents.items():
        (output_dir / name).write_bytes(payload)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    manifest = build_snapshot(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "rows": manifest["rows"],
                "asymmetric": manifest["asymmetric"],
                "tied_eval_only": manifest["tied_eval_only"],
                "files": manifest["files"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
