# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy==2.3.5",
#   "pandas==2.2.2",
#   "scipy==1.17.0",
#   "scikit-learn==1.7.2",
# ]
# ///
"""Snapshot the 520-row component panel and nested aggregate-neighborhood folds.

No response model is fitted. The proportional correspondence group supplies
calibration anchors and is always training-only. Neighborhoods are KMeans
clusters of square-root aggregate weights, with every counterpart held together.
Inner clusters use only their outer training groups. Completed snapshots are
reused only when source and output hashes still match.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_aggregate_conditioned_replay_control_20260730 as expanded,
)

REFERENCE = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT = REFERENCE / "two_phase_link_transfer_20260907" / "inputs"
CANONICAL = expanded.swarm39.CANONICAL
COMPONENT_SOURCE = (
    REFERENCE / "single_phase_componentwise_canonical_dsp_20260902" / "input" / "300m_uncheatable_components.csv"
)
OBJECTIVE_PROTOCOL = REFERENCE / "single_phase_observatory_final_model_20260907" / "protocol.json"
ANCHOR_SOURCE = REFERENCE / "delphi_floor_anchors_20260907" / "anchors.csv"
HISTORICAL_FOLDS = REFERENCE / "expanded_300m_pareto_baseline_20260731" / "rows_uncheatable.csv"
SINGLE_PHASE_SOURCE = REPO_ROOT.parent / "mixture-selection" / "mixture_selection.py"
PROTOCOL_VERSION = "two-phase-link-transfer-inputs-v1"
PANEL_NAME = "300m_39bucket"
CALIBRATION_GROUP = "baseline_proportional"
FOLDS = 3
SEED = 20260907
FINAL_INNER_SEED = SEED + FOLDS + 1
PARITY_TOLERANCE = 3e-6
WEIGHT_TOLERANCE = 1e-9


@dataclass(frozen=True)
class Prepared:
    """Arrays, readable records, and outcome-free split identities for a snapshot."""

    panel: dict[str, np.ndarray]
    splits: dict[str, np.ndarray]
    rows: pd.DataFrame
    table: pd.DataFrame
    anchors: pd.DataFrame
    folds: pd.DataFrame
    pairs: pd.DataFrame
    audit: dict[str, object]


def source_hashes() -> dict[str, str]:
    paths = (
        Path(__file__),
        Path(expanded.__file__),
        Path(expanded.swarm39.__file__),
        expanded.PACKET,
        expanded.ONE_PHASE_SOURCE,
        expanded.swarm39.CATALOG,
        CANONICAL / "300m_one_phase_fit.csv",
        COMPONENT_SOURCE,
        OBJECTIVE_PROTOCOL,
        ANCHOR_SOURCE,
        HISTORICAL_FOLDS,
    )
    hashes = {str(path.relative_to(REPO_ROOT)): expanded.swarm39.sha256_of(path) for path in paths}
    hashes[str(SINGLE_PHASE_SOURCE)] = expanded.swarm39.sha256_of(SINGLE_PHASE_SOURCE)
    return hashes


def neighborhood_splits(
    groups: np.ndarray,
    aggregate: np.ndarray,
    calibration: np.ndarray,
    eligible: np.ndarray,
    seed: int,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Partition eligible noncalibration groups using only their aggregate geometry."""
    scoreable = eligible[~calibration[eligible]]
    unique = np.unique(groups[scoreable])
    representatives = np.asarray([scoreable[np.flatnonzero(groups[scoreable] == group)[0]] for group in unique])
    labels = KMeans(n_clusters=FOLDS, n_init=20, random_state=seed, algorithm="lloyd").fit_predict(
        np.sqrt(aggregate[representatives])
    )
    # Label numbers have no geometry semantics; order by each cluster's first group
    # so exported identities do not inherit arbitrary KMeans centroid numbering.
    cluster_order = sorted(np.unique(labels), key=lambda label: str(unique[labels == label].min()))
    result = []
    for label in cluster_order:
        test_groups = unique[labels == label]
        test = eligible[np.isin(groups[eligible], test_groups)]
        train = eligible[~np.isin(groups[eligible], test_groups)]
        assert len(test) and len(train), "empty aggregate-neighborhood fold"
        assert not np.intersect1d(groups[train], groups[test]).size, "counterpart split across train and test"
        assert np.all(np.isin(np.flatnonzero(calibration), train)), "calibration rows must always be training"
        assert not calibration[test].any(), "calibration outcomes must never be scored"
        result.append((train, test))
    assert np.array_equal(np.sort(np.concatenate([test for _, test in result])), scoreable)
    return tuple(result)


def build_inputs() -> Prepared:
    """Reconstruct all 58 atomic tasks and make calibration-safe nested folds."""
    data = expanded.load_300m("uncheatable")
    frame = data.frame.copy()
    protocol = json.loads(OBJECTIVE_PROTOCOL.read_text())
    components = protocol["components"][PANEL_NAME]
    uncheatable_components = tuple(components["uncheatable"])
    table9_components = tuple(components["table9"])
    assert (len(uncheatable_components), len(table9_components)) == (7, 51)
    assert len(set(uncheatable_components + table9_components)) == 58
    uncheatable_weights = np.asarray(protocol["uncheatable_aggregation_weights"]["legacy_60m_300m"], dtype=float)
    table9_weights = np.full(51, 1.0 / 51)
    # Preserve the canonical rounded byte weights; their sum is 1.0000000302.
    assert abs(float(uncheatable_weights.sum()) - 1.0) < 1e-6
    single = frame["policy_family"].eq("single_phase").to_numpy()
    runs = frame["run_name"].astype(str).to_numpy()
    groups = frame["phase_correspondence_key"].astype(str).to_numpy()
    assert len(np.unique(runs)) == len(data.y) == 520
    assert len(np.unique(groups)) == 280
    tied_components = pd.read_csv(COMPONENT_SOURCE).set_index("row_id", verify_integrity=True)
    canonical = pd.read_csv(CANONICAL / "300m_one_phase_fit.csv").set_index("row_id", verify_integrity=True)
    uncheatable = np.column_stack([frame[column.replace("/", "_")].to_numpy(float) for column in uncheatable_components])
    uncheatable[single] = tied_components.loc[runs[single], list(uncheatable_components)].to_numpy(float)
    canonical_weights = canonical.loc[runs[single], [f"phase_0_weight::{name}" for name in data.domain_names]]
    recovery_weight_error = float(np.abs(canonical_weights.to_numpy(float) - data.weights[single, 0]).max())
    assert recovery_weight_error < WEIGHT_TOLERANCE, "recovered tied tasks belong to different policies"
    table9 = frame.loc[:, table9_components].to_numpy(float)
    assert np.isfinite(uncheatable).all() and np.isfinite(table9).all(), "missing atomic targets"
    uncheatable_aggregate = uncheatable @ uncheatable_weights
    table9_aggregate = table9 @ table9_weights
    uncheatable_parity = float(np.abs(uncheatable_aggregate - data.y).max())
    table9_parity = float(np.abs(table9_aggregate - frame[expanded.TARGETS["table9"]].to_numpy(float)).max())
    assert max(uncheatable_parity, table9_parity) <= PARITY_TOLERANCE, "canonical objective reconstruction failed"
    alpha = float(np.median(data.c0 / (data.c0 + data.c1)))
    aggregate = alpha * data.weights[:, 0] + (1.0 - alpha) * data.weights[:, 1]
    epochs = data.c0 * data.weights[:, 0] + data.c1 * data.weights[:, 1]
    tied = np.max(np.abs(data.weights[:, 0] - data.weights[:, 1]), axis=1) < WEIGHT_TOLERANCE
    assert (int(tied.sum()), int((~tied).sum())) == (282, 238)
    assert np.allclose(data.weights.sum(axis=2), 1.0, atol=1e-12, rtol=0)
    assert np.all(data.weights >= 0)
    calibration = groups == CALIBRATION_GROUP
    assert calibration.sum() == 2 and tied[calibration].all()

    pair_records = []
    aggregate_keys: dict[tuple[float, ...], str] = {}
    pair_weight_error = 0.0
    for group in np.unique(groups):
        members = np.flatnonzero(groups == group)
        error = float(np.abs(aggregate[members] - aggregate[members[0]]).max())
        pair_weight_error = max(pair_weight_error, error)
        assert error < WEIGHT_TOLERANCE, f"aggregate mismatch for {group}"
        key = tuple(np.round(aggregate[members[0]], 12))
        assert key not in aggregate_keys, f"distinct correspondence groups share aggregate: {group}"
        aggregate_keys[key] = str(group)
        asymmetric = members[~tied[members]]
        if len(asymmetric):
            tied_members = members[tied[members]]
            assert len(asymmetric) == len(tied_members) == 1
            pair_records.append(
                {"group": str(group), "tied_row": int(tied_members[0]), "asymmetric_row": int(asymmetric[0])}
            )
    pairs = pd.DataFrame(pair_records)
    assert len(pairs) == 238 and pairs["tied_row"].nunique() == 238

    anchors = pd.read_csv(ANCHOR_SOURCE)
    anchors = anchors[anchors["panel"].eq(PANEL_NAME)].set_index("component", verify_integrity=True)
    ordered_components = uncheatable_components + table9_components
    anchors = anchors.loc[list(ordered_components)].reset_index()
    anchors.insert(0, "component_index", np.arange(58))
    anchors["sd_is_aggregate_approximation"] = anchors["target"].eq("uncheatable")
    assert np.isfinite(anchors[["proportional_bpb", "repeat_sd"]].to_numpy(float)).all()
    assert (anchors["repeat_sd"] > 0).all()
    anchors["calibration_group"] = CALIBRATION_GROUP

    outer = neighborhood_splits(groups, aggregate, calibration, np.arange(data.n), SEED)
    splits: dict[str, np.ndarray] = {}
    fold_records = []
    outer_labels = np.full(data.n, -1, dtype=int)
    for outer_id, (train, test) in enumerate(outer):
        outer_labels[test] = outer_id
        splits[f"outer{outer_id}_train"] = train
        splits[f"outer{outer_id}_test"] = test
        fold_records.append((f"outer{outer_id}", SEED, train, test))
        inner_seed = SEED + outer_id + 1
        for inner_id, (inner_train, inner_test) in enumerate(
            neighborhood_splits(groups, aggregate, calibration, train, inner_seed)
        ):
            assert not np.intersect1d(test, np.concatenate([inner_train, inner_test])).size
            name = f"outer{outer_id}_inner{inner_id}"
            splits[f"{name}_train"] = inner_train
            splits[f"{name}_test"] = inner_test
            fold_records.append((name, inner_seed, inner_train, inner_test))
            sub_seed = SEED + 100 + FOLDS * outer_id + inner_id
            for sub_id, (sub_train, sub_test) in enumerate(
                neighborhood_splits(groups, aggregate, calibration, inner_train, sub_seed)
            ):
                assert not np.intersect1d(inner_test, np.concatenate([sub_train, sub_test])).size
                sub_name = f"{name}_sub{sub_id}"
                splits[f"{sub_name}_train"] = sub_train
                splits[f"{sub_name}_test"] = sub_test
                fold_records.append((sub_name, sub_seed, sub_train, sub_test))
    for inner_id, (train, test) in enumerate(
        neighborhood_splits(groups, aggregate, calibration, np.arange(data.n), FINAL_INNER_SEED)
    ):
        name = f"final_inner{inner_id}"
        splits[f"{name}_train"] = train
        splits[f"{name}_test"] = test
        fold_records.append((name, FINAL_INNER_SEED, train, test))
        sub_seed = SEED + 200 + inner_id
        for sub_id, (sub_train, sub_test) in enumerate(
            neighborhood_splits(groups, aggregate, calibration, train, sub_seed)
        ):
            assert not np.intersect1d(test, np.concatenate([sub_train, sub_test])).size
            sub_name = f"{name}_sub{sub_id}"
            splits[f"{sub_name}_train"] = sub_train
            splits[f"{sub_name}_test"] = sub_test
            fold_records.append((sub_name, sub_seed, sub_train, sub_test))
    folds = pd.DataFrame(
        [
            {
                "split": name,
                "seed": seed,
                "train_rows": len(train),
                "test_rows": len(test),
                "train_groups": len(np.unique(groups[train])),
                "test_groups": len(np.unique(groups[test])),
                "train_tied_rows": int(tied[train].sum()),
                "test_tied_rows": int(tied[test].sum()),
                "train_pairs": int((~tied[train]).sum()),
                "test_pairs": int((~tied[test]).sum()),
            }
            for name, seed, train, test in fold_records
        ]
    )
    historical = pd.read_csv(HISTORICAL_FOLDS).set_index("run_name", verify_integrity=True)
    rows = frame[
        [
            "run_name",
            "phase_correspondence_key",
            "phase_pair_group",
            "policy_family",
            "source_experiment",
            "source_panel",
            "panel_source",
            "packet_row_id",
            "paired_single_phase_run_name",
            "paired_two_phase_run_name",
        ]
    ].copy()
    rows.insert(0, "row_index", np.arange(data.n))
    rows["physical_tied"] = tied
    rows["calibration_only"] = calibration
    rows["outer_fold"] = outer_labels
    rows["historical_outer_fold"] = historical.loc[runs, "outer_fold"].to_numpy(int)
    rows["uncheatable_component_source"] = np.where(
        single, str(COMPONENT_SOURCE.relative_to(REPO_ROOT)), str(expanded.PACKET.relative_to(REPO_ROOT))
    )
    for name, _, _, test in fold_records:
        rows[name] = np.isin(np.arange(data.n), test)
    table = rows.copy()
    values = {
        **{
            f"phase_{phase}_weight::{bucket}": data.weights[:, phase, index]
            for phase in (0, 1)
            for index, bucket in enumerate(data.domain_names)
        },
        **{column: uncheatable[:, index] for index, column in enumerate(uncheatable_components)},
        **{column: table9[:, index] for index, column in enumerate(table9_components)},
        "uncheatable_bpb": uncheatable_aggregate,
        "table9_macro_bpb": table9_aggregate,
        "published_uncheatable_bpb": data.y,
        "published_table9_macro_bpb": frame[expanded.TARGETS["table9"]].to_numpy(float),
    }
    table = pd.concat([table, pd.DataFrame(values)], axis=1)
    panel = {
        "weights": data.weights,
        "phase0": data.weights[:, 0],
        "phase1": data.weights[:, 1],
        "aggregate": aggregate,
        "epochs": epochs,
        "c0": data.c0,
        "c1": data.c1,
        "inventory": data.c0 + data.c1,
        "alpha": np.asarray(alpha),
        "buckets": np.asarray(data.domain_names),
        "runs": runs.astype(str),
        "groups": groups.astype(str),
        "family_index": data.family_index,
        "calibration_mask": calibration,
        "physical_tied": tied,
        "outer_fold": outer_labels,
        "uncheatable_outcomes": uncheatable,
        "table9_outcomes": table9,
        "uncheatable_aggregate": uncheatable_aggregate,
        "table9_aggregate": table9_aggregate,
        "uncheatable_components": np.asarray(uncheatable_components),
        "table9_components": np.asarray(table9_components),
        "uncheatable_aggregation_weights": uncheatable_weights,
        "table9_aggregation_weights": table9_weights,
        "anchor_components": np.asarray(ordered_components),
        "anchor_proportional_bpb": anchors["proportional_bpb"].to_numpy(float),
        "anchor_repeat_sd": anchors["repeat_sd"].to_numpy(float),
        "anchor_sd_is_aggregate_approximation": anchors["sd_is_aggregate_approximation"].to_numpy(bool),
        "pair_tied_rows": pairs["tied_row"].to_numpy(int),
        "pair_asymmetric_rows": pairs["asymmetric_row"].to_numpy(int),
    }
    audit = {
        "rows": data.n,
        "groups": len(aggregate_keys),
        "scoreable_rows": int((~calibration).sum()),
        "scoreable_groups": len(np.unique(groups[~calibration])),
        "tied_rows": int(tied.sum()),
        "pairs": len(pairs),
        "calibration_rows": np.flatnonzero(calibration).tolist(),
        "component_counts": {"uncheatable": 7, "table9": 51},
        "maximum_uncheatable_aggregate_error": uncheatable_parity,
        "maximum_table9_aggregate_error": table9_parity,
        "maximum_recovery_weight_error": recovery_weight_error,
        "maximum_counterpart_aggregate_error": pair_weight_error,
        "source_experiment_counts": frame["source_experiment"].value_counts().to_dict(),
        "all_assertions_passed": True,
    }
    return Prepared(panel, splits, rows, table, anchors, folds, pairs, audit)


def prepare(output: Path) -> dict[str, object]:
    """Write a complete snapshot, or verify and reuse an identical durable output."""
    sources = source_hashes()
    manifest_path = output / "manifest.json"
    if output.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["source_sha256"] != sources:
            raise ValueError(f"source hashes changed; choose a new output directory instead of replacing {output}")
        for name, digest in manifest["output_sha256"].items():
            assert expanded.swarm39.sha256_of(output / name) == digest, f"modified output: {name}"
        return manifest
    prepared = build_inputs()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".inputs-", dir=output.parent) as temporary:
        stage = Path(temporary) / "snapshot"
        stage.mkdir()
        np.savez_compressed(stage / "panel.npz", **prepared.panel)
        np.savez_compressed(stage / "splits.npz", **prepared.splits)
        (stage / "single_phase.py").write_bytes(SINGLE_PHASE_SOURCE.read_bytes())
        for name, table in (
            ("rows", prepared.rows),
            ("panel", prepared.table),
            ("anchors", prepared.anchors),
            ("folds", prepared.folds),
            ("pairs", prepared.pairs),
        ):
            table.to_csv(stage / f"{name}.csv", index=False)
        (stage / "audit.json").write_text(json.dumps(prepared.audit, indent=2, sort_keys=True) + "\n")
        manifest = {
            "protocol_version": PROTOCOL_VERSION,
            "source_sha256": sources,
            "output_sha256": {path.name: expanded.swarm39.sha256_of(path) for path in sorted(stage.iterdir())},
            "panel_arrays": {name: list(value.shape) for name, value in prepared.panel.items()},
            "split_arrays": {name: list(value.shape) for name, value in prepared.splits.items()},
            "split_index_space": (
                "Global row_index in panel.csv/panel.npz; inner indices are not local to outer training arrays."
            ),
            "fold_geometry": (
                "KMeans on sqrt aggregate weights of distinct correspondence groups; no outcome-dependent construction."
            ),
            "seeds": {
                "outer": SEED,
                "outer_inner": [SEED + fold + 1 for fold in range(FOLDS)],
                "final_inner": FINAL_INNER_SEED,
            },
            "calibration": (
                "Both baseline_proportional rows always train, never score. "
                "The other 518 rows form 279 scoreable groups."
            ),
            "objective_rule": (
                "Uncheatable uses fixed legacy byte weights; Table-9 is the mean of 51 fixed components. "
                "Saved aggregates are reconstructed from atomic values."
            ),
            "anchor_noise": (
                "300M Table-9 uses per-task SD over eleven proportional runs. Each 300M Uncheatable task instead "
                "uses its panel macro repeat SD as an approximation, with proportional BPB from its own one-phase run."
            ),
            "exposure_units": (
                "c0/c1 are epochs per unit phase weight, rescaled by the existing loader to proportional exposure "
                "0.905353. inventory=c0+c1; epochs=c0*w0+c1*w1."
            ),
            "source_limit": (
                "480 qsplit, 39 deletion, and one stratified row do not support source-disjoint phase fitting; "
                "the frozen splits block aggregate neighborhoods and exact counterparts."
            ),
            "claim_scope": (
                "Development data only; no fitted models, newly held-out training outcomes, or prospective validation."
            ),
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(stage, output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = prepare(args.output_dir)
    print(
        json.dumps(
            {
                "output": str(args.output_dir),
                "protocol": manifest["protocol_version"],
                "files": sorted(manifest["output_sha256"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
