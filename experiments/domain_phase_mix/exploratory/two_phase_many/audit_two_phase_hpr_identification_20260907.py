# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy==2.3.5", "pandas==2.2.2", "scipy==1.17.0", "scikit-learn==1.7.2",
#   "cvxpy==1.7.5", "fsspec==2026.1.0", "gcsfs==2026.1.0", "plotly==6.5.1",
#   "tabulate==0.9.0", "threadpoolctl==3.6.0",
# ]
# ///
"""Audit frozen two-phase policy design without loading outcome arrays."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_hierarchical_coverage_grp_20260715 as hpr,
)

INPUT = SCRIPT_DIR / "reference_outputs/two_phase_link_transfer_20260907/inputs"
OUTPUT = SCRIPT_DIR / "reference_outputs/two_phase_hpr_transfer_20260907/identification"
POLICY_KEYS = (
    "weights",
    "aggregate",
    "c0",
    "c1",
    "buckets",
    "family_index",
    "physical_tied",
    "pair_asymmetric_rows",
    "pair_tied_rows",
    "runs",
    "groups",
    "outer_fold",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diagnostics(train: np.ndarray, test: np.ndarray) -> dict[str, float | int]:
    """Measure column-normalized rank, leverage, and test projection novelty."""
    norms = np.linalg.norm(train, axis=0)
    active = norms > max(float(norms.max()), 1.0) * 1e-12
    scale = np.where(active, norms, 1.0)
    scaled = train / scale
    u, singular, vt = np.linalg.svd(scaled, full_matrices=False)
    keep = singular > max(float(singular[0]), 1.0) * 1e-10
    basis = vt[keep]
    leverage = np.sum(u[:, keep] ** 2, axis=1)
    result = {
        "train_pairs": len(train),
        "test_pairs": len(test),
        "columns": train.shape[1],
        "active_columns": int(active.sum()),
        "rank": int(keep.sum()),
        "condition_nonzero": float(singular[keep][0] / singular[keep][-1]),
        "max_train_leverage": float(leverage.max()),
        "median_train_leverage": float(np.median(leverage)),
        "rows_per_rank": float(len(train) / keep.sum()),
    }
    if len(test):
        target = test / scale
        residual = target - target @ basis.T @ basis
        relative = np.linalg.norm(residual, axis=1) / np.maximum(np.linalg.norm(target, axis=1), 1e-12)
        extrapolation = np.sum((target @ basis.T / singular[keep]) ** 2, axis=1)
        result.update(
            {
                "test_span_residual_max": float(relative.max()),
                "test_span_residual_median": float(np.median(relative)),
                "test_prediction_variance_factor_max": float(extrapolation.max()),
                "test_prediction_variance_factor_median": float(np.median(extrapolation)),
            }
        )
    return result


def policy_keys(values: np.ndarray) -> list[tuple[float, ...]]:
    return [tuple(row) for row in np.round(values, 10)]


def audit_archive() -> None:
    """Inspect existing policy metadata; outcomes are excluded by usecols."""
    source = SCRIPT_DIR / "reference_outputs/delphi_3e18_append_only_heldouts_20260714/heldout_current.csv"
    columns = [
        "heldout_id",
        "phase_0_fraction",
        "phase_0_weights_json",
        "phase_1_weights_json",
        "training_series",
        "parameter_count",
        "wandb_run_id",
        "wandb_run_name",
        "objective",
        "data_seed",
        "trainer_seed",
        "phase_boundary_step",
        "num_train_steps",
        "global_step",
        "training_state",
        "anchor_id",
        "direction_id",
        "radius_fraction",
        "seed_block",
        "candidate_kind",
        "table9_metric_source",
        "table9_eval_state",
        "table9_eval_failed_count",
    ]
    frame = pd.read_csv(source, usecols=columns)
    buckets = sorted(json.loads(frame.iloc[0].phase_0_weights_json))
    weights = np.array(
        [
            [[json.loads(row[f"phase_{phase}_weights_json"]).get(bucket, 0) for bucket in buckets] for phase in range(2)]
            for _, row in frame.iterrows()
        ]
    )
    aggregate = (
        frame.phase_0_fraction.to_numpy()[:, None] * weights[:, 0]
        + (1 - frame.phase_0_fraction.to_numpy()[:, None]) * weights[:, 1]
    )
    groups = defaultdict(list)
    for row, key in enumerate(policy_keys(np.column_stack([frame.phase_0_fraction.to_numpy(), aggregate]))):
        groups[key].append(row)
    records = []
    mirrors = []
    for rows in groups.values():
        _, unique_indices = np.unique(np.round(weights[rows].reshape(len(rows), -1), 10), axis=0, return_index=True)
        unique = weights[np.asarray(rows)[unique_indices]]
        if len(unique) < 2:
            continue
        contrast = unique[:, 1] - unique[:, 0]
        norms = np.linalg.norm(contrast, axis=1)
        nonzero = contrast[norms > 1e-9]
        unit = nonzero / np.linalg.norm(nonzero, axis=1)[:, None]
        opposite_direction_pairs = int(np.sum(np.triu(unit @ unit.T < -1 + 1e-8, 1)))
        mirror_pairs = int(
            sum(np.max(np.abs(nonzero[i] + other)) < 1e-8 for i in range(len(nonzero)) for other in nonzero[i + 1 :])
        )
        cell_id = hashlib.sha256(np.round(aggregate[rows[0]], 10).tobytes()).hexdigest()[:12]
        for first in np.flatnonzero(norms > 1e-9):
            for second in np.flatnonzero(norms > 1e-9):
                if second <= first:
                    continue
                error = np.max(np.abs(contrast[first] + contrast[second]))
                if error < 1e-8:
                    left = np.asarray(rows)[np.max(np.abs(weights[rows] - unique[first]), axis=(1, 2)) < 1e-10]
                    right = np.asarray(rows)[np.max(np.abs(weights[rows] - unique[second]), axis=(1, 2)) < 1e-10]
                    mirrors.append(
                        {
                            "cell_id": cell_id,
                            "left_row_ids_json": json.dumps(frame.heldout_id.iloc[left].tolist()),
                            "right_row_ids_json": json.dumps(frame.heldout_id.iloc[right].tolist()),
                            "phase_contrast_norm": float(norms[first]),
                            "mirror_max_abs_error": float(error),
                        }
                    )
        singular = np.linalg.svd(nonzero, compute_uv=False)
        rank = int(np.sum(singular > max(singular[0], 1.0) * 1e-10))
        records.append(
            {
                "cell_id": cell_id,
                "phase_0_fraction": float(frame.phase_0_fraction.iloc[rows[0]]),
                "rows": len(rows),
                "unique_policies": len(unique),
                "tied_unique": int(np.sum(norms <= 1e-9)),
                "contrast_rank": rank,
                "opposite_direction_pairs": opposite_direction_pairs,
                "equal_amplitude_mirror_pairs": mirror_pairs,
                "aggregate_max_abs_spread": float(np.max(np.ptp(aggregate[rows], axis=0))),
                "row_ids_json": json.dumps(frame.heldout_id.iloc[np.asarray(rows)].tolist()),
                "training_series_json": json.dumps(sorted(set(frame.training_series.iloc[np.asarray(rows)]))),
            }
        )
    pd.DataFrame(records).sort_values("rows", ascending=False).to_csv(
        OUTPUT / "archive_fixed_aggregate_cells.csv", index=False
    )
    pd.DataFrame(mirrors).to_csv(OUTPUT / "archive_mirror_pairs.csv", index=False)
    cells = pd.DataFrame(records).sort_values("rows", ascending=False).head(2)
    large_ids = [row_id for packed in cells.row_ids_json for row_id in json.loads(packed)]
    metadata = (
        frame.set_index("heldout_id").loc[large_ids].drop(columns=["phase_0_weights_json", "phase_1_weights_json"])
    )
    centers = pd.Series(np.max(np.abs(weights[:, 0] - weights[:, 1]), axis=1) < 1e-10, index=frame.heldout_id)
    metadata["center_policy"] = centers.loc[metadata.index]
    spec_sources = []
    specs_by_name = {}
    for name in ["delphi_3e18_frontier_phase_fiber_20260719", "delphi_3e18_aggressive_phase_asymmetry_20260722"]:
        spec_path = SCRIPT_DIR / "reference_outputs" / name / "launch_dry_run/run_specs.json"
        spec_sources.append({"path": str(spec_path.relative_to(REPO_ROOT)), "sha256": sha256(spec_path)})
        specs_by_name.update({row["run_name"]: row for row in json.loads(spec_path.read_text())})
    metadata["dry_run_data_seed"] = [
        specs_by_name.get(name.rsplit("-", 1)[0], {}).get("data_seed") for name in metadata.wandb_run_name
    ]
    metadata["dry_run_trainer_seed"] = [
        specs_by_name.get(name.rsplit("-", 1)[0], {}).get("trainer_seed") for name in metadata.wandb_run_name
    ]
    metadata["dry_run_source_run_name"] = [
        specs_by_name.get(name.rsplit("-", 1)[0], {}).get("source_run_name") for name in metadata.wandb_run_name
    ]
    metadata.to_csv(OUTPUT / "archive_large_cell_run_metadata.csv")
    graphs = []
    for alpha in sorted(set(frame.phase_0_fraction)):
        local = weights[frame.phase_0_fraction.eq(alpha)]
        prefixes, continuations = defaultdict(set), defaultdict(set)
        for first, second in zip(policy_keys(local[:, 0]), policy_keys(local[:, 1]), strict=True):
            prefixes[first].add(second)
            continuations[second].add(first)
        rectangles = 0
        keys = list(prefixes)
        for i, first in enumerate(keys):
            for second in keys[i + 1 :]:
                common = len(prefixes[first] & prefixes[second])
                rectangles += common * (common - 1) // 2
        graphs.append(
            {
                "phase_0_fraction": float(alpha),
                "rows": len(local),
                "prefixes": len(prefixes),
                "continuations": len(continuations),
                "prefixes_with_multiple_continuations": sum(len(x) > 1 for x in prefixes.values()),
                "continuations_with_multiple_prefixes": sum(len(x) > 1 for x in continuations.values()),
                "complete_2_by_2_rectangles": rectangles,
            }
        )
    summary = {
        "source": str(source.relative_to(REPO_ROOT)),
        "sha256": sha256(source),
        "loaded_columns": columns,
        "rows": len(frame),
        "sealed_targeted_pairwise_rows_in_inspected_archive": int(
            frame.training_series.astype(str).str.contains("targeted_pairwise", case=False).sum()
        ),
        "sealed_boundary": "July24 audit excludes targeted_pairwise. No separate targeted_pairwise files were read.",
        "run_spec_sources": spec_sources,
        "dry_run_budget_metadata": {
            key: sorted({row[key] for row in specs_by_name.values()})
            for key in [
                "target_flops",
                "realized_train_tokens",
                "batch_size",
                "train_steps",
                "model_hidden_dim",
                "model_layers",
                "trainer_seed",
            ]
        },
        "physical_aggregate_cells_with_multiple_policies": len(records),
        "phase_fraction_groups": graphs,
        "group_tolerance": (
            "Policy and physical-aggregate coordinates rounded to 10 decimals; "
            "reported aggregate spread validates cells."
        ),
        "status": (
            "Preexisting development archive, not prospective data and not interchangeable with the 300M protocol. "
            "Parameter count is 358306688 on 1966 rows and missing on 3."
        ),
    }
    (OUTPUT / "archive_metadata_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with np.load(INPUT / "panel.npz") as data:
        panel = {key: data[key] for key in POLICY_KEYS}
    with np.load(INPUT / "splits.npz") as data:
        splits = {key: data[key] for key in data.files if key.startswith("outer") and "inner" not in key}
    families = tuple(np.flatnonzero(panel["family_index"] == family) for family in np.unique(panel["family_index"]))
    dataset = hpr.family_grp.Dataset(
        frame=pd.DataFrame(index=np.arange(len(panel["weights"]))),
        target=np.zeros(len(panel["weights"])),
        weights=panel["weights"],
        c0=panel["c0"],
        c1=panel["c1"],
        domains=tuple(panel["buckets"]),
        family_names=tuple(f"family{index}" for index in range(len(families))),
        family_members=families,
        quality=np.ones(len(panel["buckets"])),
    )
    aggregate = (panel["c0"] * panel["weights"][:, 0] + panel["c1"] * panel["weights"][:, 1]) / (
        panel["c0"] + panel["c1"]
    )
    tied_dataset = replace(dataset, weights=np.stack([aggregate, aggregate], axis=1))
    asym = panel["pair_asymmetric_rows"]
    tied = panel["pair_tied_rows"]
    assert np.max(np.abs(aggregate[asym] - aggregate[tied])) < 1e-12
    shapes = hpr.family_grp.shape_candidates(hpr.family_grp.Variant.BUCKET_RESOLVED, 12)
    bases = {"linear_bucket_contrast": panel["weights"][:, 1] - panel["weights"][:, 0]}
    names = {}
    for index, shape in enumerate(shapes):
        config = hpr.Config(hpr.Variant.HIERARCHICAL_PHASE_BUCKET_REPLAY, index, shape, 1.0, 1.0, 1.0, 1.0)
        actual = hpr.build_design(dataset, config)
        counterfactual = hpr.build_design(tied_dataset, config)
        contrast = actual.values - counterfactual.values
        assert np.max(np.abs(contrast[panel["physical_tied"]])) < 1e-12
        bases[f"hpr_shape{index}"] = contrast
        names[f"hpr_shape{index}"] = {"shape": asdict(shape), "columns": actual.names}
    records = []
    for name, values in bases.items():
        pair_values = values[asym]
        records.append({"basis": name, "fold": "full", **diagnostics(pair_values, np.empty((0, values.shape[1])))})
        for fold in range(3):
            train = np.isin(asym, splits[f"outer{fold}_train"])
            test = np.isin(asym, splits[f"outer{fold}_test"])
            assert np.all(train | test)
            records.append({"basis": name, "fold": str(fold), **diagnostics(pair_values[train], pair_values[test])})
    pd.DataFrame(records).to_csv(OUTPUT / "feature_geometry.csv", index=False)
    phase0, phase1 = policy_keys(panel["weights"][:, 0]), policy_keys(panel["weights"][:, 1])
    unique_edges = set(zip(phase0, phase1, strict=True))
    prefix_continuations = {key: {second for first, second in unique_edges if first == key} for key in set(phase0)}
    continuation_prefixes = {key: {first for first, second in unique_edges if second == key} for key in set(phase1)}
    rectangles = 0
    prefix_keys = list(prefix_continuations)
    for i, first in enumerate(prefix_keys):
        for second in prefix_keys[i + 1 :]:
            common = len(prefix_continuations[first] & prefix_continuations[second])
            rectangles += common * (common - 1) // 2
    design = {
        "rows": len(phase0),
        "unique_prefixes": len(prefix_continuations),
        "unique_continuations": len(continuation_prefixes),
        "unique_policy_edges": len(unique_edges),
        "prefixes_with_multiple_continuations": sum(len(x) > 1 for x in prefix_continuations.values()),
        "continuations_with_multiple_prefixes": sum(len(x) > 1 for x in continuation_prefixes.values()),
        "complete_2_by_2_rectangles": rectangles,
        "physical_tied_rows": int(panel["physical_tied"].sum()),
        "exact_matched_pairs": len(asym),
        "aggregate_counterpart_max_error": float(np.max(np.abs(aggregate[asym] - aggregate[tied]))),
        "claims": (
            "Policy design only. Rank is an upper bound on coefficient identifiability; "
            "no outcome, covariance, nonnegative active-set, or model-selection claim."
        ),
        "span_definition": (
            "Column L2 scaling uses training pairs only. Row-span residual is the fraction of each held-out feature "
            "row outside training row span; zero does not mean interpolation. Prediction variance factor is "
            "x_test (X_train^T X_train)^+ x_test^T with unit independent pair noise."
        ),
        "outcome_access": (
            "Only POLICY_KEYS are extracted from panel.npz; no outcome arrays, fit parameters, "
            "or observed-score files are read."
        ),
    }
    (OUTPUT / "design_summary.json").write_text(json.dumps(design, indent=2) + "\n")
    (OUTPUT / "feature_definitions.json").write_text(json.dumps(names, indent=2) + "\n")
    sources = [
        Path(__file__),
        INPUT / "panel.npz",
        INPUT / "splits.npz",
        Path(hpr.__file__),
        Path(hpr.family_grp.__file__),
    ]
    manifest = {
        "source_sha256": {str(path.relative_to(REPO_ROOT)): sha256(path) for path in sources},
        "policy_keys": POLICY_KEYS,
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    audit_archive()
    print(json.dumps(design, indent=2))


if __name__ == "__main__":
    main()
