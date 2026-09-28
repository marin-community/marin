# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["fsspec==2026.1.0", "numpy==2.3.5", "pandas==2.2.2", "scipy==1.17.0"]
# ///
"""Prepare already-open fixed-checkpoint branch outcomes without fitting a model."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WORK = Path("/Users/calvinxu/Projects/Work/Marin")
ARCHIVES = WORK / "worktree-archives"
REL = Path("experiments/domain_phase_mix/exploratory/two_phase_many")
OUTPUT = Path(__file__).resolve().parent / "reference_outputs/fixed_checkpoint_branch_wspu_20260907/data"
DIRTY = ARCHIVES / "20260905-dirty-worktrees"
CLEAN = ARCHIVES / "20260905-clean-worktrees"
AUDIT = DIRTY / "marin-delphi-branch-response-audit-20260826/files" / REL
LATER = DIRTY / "marin-delphi-y0-y1-surrogate-20260827/files" / REL / "reference_outputs"
COMPONENTS = (
    "ao3_english",
    "arxiv_computer_science",
    "arxiv_physics",
    "bbc_news",
    "github_cpp",
    "github_python",
    "wikipedia_english",
)
CONFIRMATORY_STATES = (
    "shared_bounded_ensemble_kl0p05",
    "shared_bounded_ensemble_kl0p2",
    "shared_bounded_ensemble_kl0p5",
    "cap4_shared_bounded_ensemble_kl0",
    "cap4_shared_bounded_ensemble_kl0p05",
    "cap4_shared_bounded_ensemble_kl0p2",
)
INPUTS: dict[str, str] = {}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def track(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    INPUTS[str(path)] = sha256(path)
    return path


def archived(root: Path, worktree: str, output: str, filename: str) -> Path:
    return track(root / worktree / "files" / REL / "reference_outputs" / output / filename)


def load_audit():
    source = track(AUDIT / "audit_delphi_phase1_branch_response_20260826.py")
    spec = importlib.util.spec_from_file_location("fixed_branch_historical_audit", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def historical_panels(audit) -> pd.DataFrame:
    proportional_dir = "delphi_phase1_proportional_prefix_combined_results_20260826"
    common = archived(
        CLEAN, "marin-delphi-prefix-candidates", "delphi_phase1_common_branches_20260824", "continuation_weights.csv"
    )
    phase0 = archived(
        CLEAN, "marin-delphi-prefix-candidates", "delphi_phase0_prefix_candidates_20260824", "candidate_weights.csv"
    )
    specs = [
        audit.PanelSpec(
            name="proportional",
            prefix="proportional_control",
            results=archived(
                CLEAN, "marin-delphi-proportional-prefix-branches-20260825", proportional_dir, "branch_results.csv"
            ),
            weights=(
                archived(
                    CLEAN,
                    "marin-delphi-proportional-prefix-branches-20260825",
                    proportional_dir,
                    "continuation_weights.csv",
                ),
            ),
            target="bpb",
            role="role",
        ),
        audit.PanelSpec(
            name="cap10_kl0p05_broad",
            prefix="shared_bounded_ensemble_kl0p05",
            results=archived(
                CLEAN,
                "marin-delphi-phase1-wave2-acquisition-20260825",
                "delphi_phase1_kl0p05_wave1_results_20260825",
                "branch_results.csv",
            ),
            weights=(common,),
            target="uncheatable_bpb",
            role="continuation_role",
            wave_column="wave",
            phase0_weights=phase0,
        ),
        audit.PanelSpec(
            name="cap10_kl0p05_local",
            prefix="shared_bounded_ensemble_kl0p05",
            results=archived(
                DIRTY,
                "marin-delphi-phase1-local-model-20260825",
                "delphi_phase1_kl0p05_local_wave2b_results_20260825",
                "partial_branch_results.csv",
            ),
            weights=(
                archived(
                    DIRTY,
                    "marin-delphi-phase1-local-wave2b-20260825",
                    "delphi_phase1_kl0p05_local_wave2b_20260825",
                    "continuation_weights.csv",
                ),
            ),
            target="uncheatable_bpb",
            role="continuation_role",
            phase0_weights=phase0,
        ),
    ]
    frames = []
    for spec in specs:
        if spec.name == "proportional":
            coverage = json.loads(track(spec.results.parent / "coverage.json").read_text())
            assert coverage["visible_result_rows"] == 174 and coverage["sealed_referee_rows"] == 8
            assert coverage["referee_outcomes_opened"] is False
        frame = audit.load_panel(spec)
        raw = pd.read_csv(spec.results).set_index("provenance_sha256").reindex(frame.provenance_sha256)
        frame["checkpoint_uri"] = raw.checkpoint_uri.to_numpy() if "checkpoint_uri" in raw else ""
        frame["trainer_seed"] = raw.trainer_seed.to_numpy()
        frame["output_root"] = raw.output_root.to_numpy() if "output_root" in raw else ""
        frame["row_id"] = frame.panel + "::" + frame.run_id
        frame["state_id"] = frame.prefix + "::prefix_seed_" + frame.prefix_repeat_seed.astype(str)
        frame["action_id"] = frame.continuation_id
        frame["historical_confirmatory_state"] = False
        frame["phase_fraction_source"] = "historical_materialized_weight_artifacts"
        frames.append(frame)
    return audit.add_anchors(pd.concat(frames, ignore_index=True))


def crossed_panels(audit, buckets: list[str]) -> pd.DataFrame:
    boundary_path = track(
        LATER / "delphi_3e18_phase0_prefix_replay_20260820/materialized_boundary_metrics/prefix_boundary_fit_matrix.csv"
    )
    assert sha256(boundary_path) == "43032db14ae6ab0ee83eaeee24cf8ecab58740db0eae3d938534ff9088b1a646"
    # Read only the exposure columns. Boundary outcomes are not procedure inputs.
    boundary = pd.read_csv(
        boundary_path,
        usecols=[f"phase_0_{kind}::{bucket}" for kind in ("weight", "materialized_epochs") for bucket in buckets],
    )
    rates0 = []
    for bucket in buckets:
        weight = boundary[f"phase_0_weight::{bucket}"].to_numpy(float)
        exposure = boundary[f"phase_0_materialized_epochs::{bucket}"].to_numpy(float)
        ratios = exposure[weight > 1e-12] / weight[weight > 1e-12]
        rate = float(np.median(ratios))
        assert np.allclose(ratios, rate, rtol=1e-9, atol=1e-9)
        rates0.append(rate)
    rates0 = np.asarray(rates0)
    rates1 = rates0 * (607.0 / 2400.0)
    specs = [
        (
            "crossed_broad",
            CLEAN
            / "marin-delphi-crossed-prefix-panel-20260827/files"
            / REL
            / "reference_outputs/delphi_phase1_crossed_prefix_panel_v3_20260827",
            LATER / "delphi_crossed_prefix_results_20260827",
            "crossed_results.csv",
            477,
        ),
        (
            "crossed_local",
            DIRTY
            / "marin-delphi-local-crossed-prefix-20260828/files"
            / REL
            / "reference_outputs/delphi_phase1_local_crossed_prefix_fit079_20260828",
            LATER / "delphi_phase1_local_crossed_prefix_fit079_screen10_results_20260829",
            "local_screen_results.csv",
            117,
        ),
    ]
    frames = []
    for name, design, result_dir, filename, count in specs:
        results_path = track(result_dir / filename)
        coverage = json.loads(track(result_dir / "coverage.json").read_text())
        assert coverage["status"] == "complete" and coverage["observed_rows"] == count
        assert coverage["results_sha256"] == sha256(results_path)
        raw = pd.read_csv(results_path)
        rows = pd.read_csv(track(design / "panel_rows.csv"))
        weights = pd.read_csv(track(design / "panel_weights.csv"))
        registry = json.loads(track(design / "prefix_registry.json").read_text())
        track(design / "manifest.json")
        assert len(raw) == count and raw.provenance_sha256.nunique() == count
        assert set(raw.row_id) == set(rows.row_id)
        for column in (
            "run_order",
            "run_id",
            "prefix_state_id",
            "action_id",
            "role",
            "data_seed",
            "trainer_seed",
            "fit_budget",
        ):
            assert raw.set_index("row_id")[column].equals(rows.set_index("row_id")[column])
        prefix = {row["state_id"]: row for row in registry["prefixes"]}
        w0 = np.asarray(
            [
                [prefix[state]["run_spec"]["phase_weights"]["phase_0"].get(bucket, 0.0) for bucket in buckets]
                for state in raw.prefix_state_id
            ]
        )
        w1 = (
            weights.pivot(index="row_id", columns="bucket", values="phase_1_weight")
            .reindex(index=raw.row_id, columns=buckets)
            .to_numpy(float)
        )
        assert np.isfinite(w0).all() and np.isfinite(w1).all()
        frame = pd.DataFrame(
            {
                "panel": name,
                "prefix": raw.prefix_candidate_id,
                "run_id": raw.run_id.astype(str),
                "continuation_id": raw.continuation_id,
                "role": raw.role,
                "fit_budget": raw.fit_budget,
                "provenance_sha256": raw.provenance_sha256,
                "target": raw.uncheatable_bpb,
                "data_seed": raw.data_seed,
                "trainer_seed": raw.trainer_seed,
                "prefix_repeat_seed": raw.prefix_repeat_seed,
                "source": str(results_path),
                "phase0_epoch_source": "boundary_exposure_rates_times_frozen_prefix_weights",
                "anchor_panel": name + "::" + raw.prefix_state_id,
                "confirmation_candidate": "",
                "wave": "",
                "row_id": name + "::" + raw.row_id,
                "state_id": raw.prefix_state_id,
                "action_id": raw.action_id,
                "checkpoint_uri": [
                    prefix[state]["checkpoint_uri"].format(code_commit=coverage["code_commit"])
                    for state in raw.prefix_state_id
                ],
                "prefix_provenance_sha256": [prefix[state]["provenance_sha256"] for state in raw.prefix_state_id],
                "output_root": raw.output_root,
                "historical_confirmatory_state": raw.prefix_state_id.isin(CONFIRMATORY_STATES),
                "phase_fraction_source": "realized_2400_of_3007_updates",
            }
        )
        columns = [f"{kind}::{bucket}" for kind in ("w", "c", "p0e", "p1e", "cp1e") for bucket in buckets]
        frame = pd.concat(
            [frame, pd.DataFrame(np.column_stack([w1, w0, w0 * rates0, w1 * rates1, w0 * rates1]), columns=columns)],
            axis=1,
        )
        for component in COMPONENTS:
            frame[f"atomic::{component}::bpb"] = raw[f"uncheatable::{component}::bpb"]
        frames.append(frame)
    return audit.add_anchors(pd.concat(frames, ignore_index=True))


def identify_checkpoints(frame: pd.DataFrame) -> pd.DataFrame:
    """Distinguish terminal locations from fixed prefix identities in old exports."""
    frame = frame.copy()
    historical = ~frame.panel.str.startswith("crossed")
    frame["terminal_checkpoint_uri"] = [
        uri if is_historical and uri else (f"{root}/checkpoints/step-3006" if root else "")
        for is_historical, uri, root in zip(historical, frame.checkpoint_uri, frame.output_root, strict=True)
    ]
    frame["prefix_checkpoint_uri"] = np.where(historical, "", frame.checkpoint_uri)
    known_prefixes = frame[~historical & ~frame.state_id.str.contains("bridge")].drop_duplicates(
        ["prefix", "prefix_repeat_seed"]
    )
    for row in frame[historical].itertuples():
        index = int(row.Index)
        matched = known_prefixes[
            known_prefixes.prefix.eq(row.prefix) & known_prefixes.prefix_repeat_seed.eq(row.prefix_repeat_seed)
        ]
        if len(matched) == 1:
            frame.loc[index, "prefix_checkpoint_uri"] = matched.iloc[0].prefix_checkpoint_uri
    frame["checkpoint_uri"] = frame.prefix_checkpoint_uri
    frame["checkpoint_identity_source"] = np.where(
        frame.prefix_checkpoint_uri.ne(""),
        "frozen_crossed_prefix_registry",
        "historical_prefix_candidate_and_repeat_seed_only",
    )
    return frame


def load_inputs(output: Path = OUTPUT) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Read the prepared package without accessing an archived path."""
    frame = pd.read_csv(output / "rows.csv", keep_default_na=False)
    with np.load(output / "arrays.npz", allow_pickle=False) as packed:
        arrays = {key: packed[key] for key in packed.files}
    assert len(frame) == len(arrays["target"]) and np.array_equal(frame.row, np.arange(len(frame)))
    return frame, arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        source_matches = existing["source_sha256"] == sha256(Path(__file__))
        inputs_match = all(
            Path(path).is_file() and sha256(Path(path)) == digest for path, digest in existing["inputs"].items()
        )
        outputs_match = all(
            (output / path).is_file() and sha256(output / path) == digest for path, digest in existing["outputs"].items()
        )
        if source_matches and inputs_match and outputs_match:
            print("Prepared data already complete; all source and output hashes verified.")
            return
    audit = load_audit()
    history = historical_panels(audit)
    buckets = [column.removeprefix("w::") for column in audit.action_columns(history, "w::")]
    assert buckets == sorted(buckets) and len(buckets) == 39
    crossed = crossed_panels(audit, buckets)
    frame = identify_checkpoints(pd.concat([history, crossed], ignore_index=True))
    frame.insert(0, "row", np.arange(len(frame)))
    assert frame.row_id.is_unique and frame.provenance_sha256.is_unique
    w0 = frame[audit.action_columns(frame, "c::")].to_numpy(float)
    w1 = frame[audit.action_columns(frame, "w::")].to_numpy(float)
    assert np.allclose(w0.sum(axis=1), 1, atol=5e-4) and np.allclose(w1.sum(axis=1), 1, atol=5e-4)
    frame["coordinate_hash"] = audit.coordinate_hashes(w1)
    frame["is_tied_control"] = frame.role.str.contains("tied", case=False)
    assert np.max(np.abs((w1 - w0)[frame.is_tied_control])) < 5e-4
    frame["benchmark_role"] = "control"
    role_map = {
        "fixed_prefix_response_fit": "primary_train",
        "outcome_blind_coverage_fit": "primary_test",
        "adaptive_model_fit": "adaptive_descriptive",
    }
    for role, label in role_map.items():
        frame.loc[frame.panel.eq("proportional") & frame.role.eq(role), "benchmark_role"] = label
    for panel, label in (
        ("cap10_kl0p05_broad", "broad_train"),
        ("cap10_kl0p05_local", "local_test"),
        ("crossed_broad", "crossed_broad_action_cv"),
        ("crossed_local", "crossed_local_test"),
    ):
        frame.loc[frame.panel.eq(panel) & frame.fit_budget, "benchmark_role"] = label
    atomic_columns = [f"atomic::{component}::bpb" for component in COMPONENTS]
    component_bpb = frame[atomic_columns].to_numpy(float)
    component_effect = frame[[f"effect::{column}" for column in atomic_columns]].to_numpy(float)
    component_anchor = component_bpb - component_effect
    frame["component_available"] = np.asarray(np.isfinite(component_bpb).all(axis=1))
    weight_path = track(
        AUDIT / "reference_outputs/delphi_phase1_branch_response_audit_20260826/uncheatable_atomic_weights.csv"
    )
    historical_weights = (
        pd.read_csv(weight_path)
        .set_index("atomic_objective")
        .reindex([f"{component}::bpb" for component in COMPONENTS])
        .aggregate_weight.to_numpy(float)
    )
    canonical_source = track(Path(__file__).parent / "benchmark_observed_capability_relaxation_20260724.py")
    weight_node = next(
        node
        for node in ast.parse(canonical_source.read_text()).body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "UNCHEATABLE_COMPONENT_WEIGHTS" for target in node.targets)
    )
    canonical_weights = ast.literal_eval(weight_node.value)
    fixed_weights = np.asarray([canonical_weights[component] for component in COMPONENTS])
    available = frame.component_available.to_numpy(bool)
    aggregate_error = float(
        np.max(np.abs(component_bpb[available] @ fixed_weights - frame.target.to_numpy(float)[available]))
    )
    assert aggregate_error < 5e-7
    historical_weight_error = float(
        np.max(np.abs(component_bpb[available] @ historical_weights - frame.target.to_numpy(float)[available]))
    )
    root = np.sqrt(w1) - np.sqrt(w0)
    tangent = root - np.sum(root * np.sqrt(w0), axis=1)[:, None] * np.sqrt(w0)
    sqrt_h2 = np.column_stack([tangent, 0.5 * np.sum(root**2, axis=1)])
    reference_features, _ = audit.feature_matrix(frame, audit.ModelSpec("sqrt_h2", "sqrt_h2", False))
    assert np.array_equal(sqrt_h2, reference_features)
    component_sd = np.full_like(component_bpb, np.nan)
    anchor_ids = []
    for row in frame.itertuples():
        index = int(row.row)
        selected = (
            frame.anchor_panel.eq(row.anchor_panel)
            & frame.prefix.eq(row.prefix)
            & frame.prefix_repeat_seed.eq(row.prefix_repeat_seed)
            & frame.is_tied_control
        )
        controls = frame.loc[selected]
        anchor_ids.append(json.dumps(controls.row_id.tolist()))
        if len(controls) > 1 and frame.loc[index, "component_available"]:
            component_sd[index] = np.nanstd(component_bpb[selected], axis=0, ddof=1)
    frame["anchor_row_ids_json"] = anchor_ids
    train_anchor = frame.anchor.to_numpy(float).copy()
    train_component_anchor = component_anchor.copy()
    train_component_sd = component_sd.copy()
    train_anchor_sd = frame.anchor_std_bpb.to_numpy(float).copy()
    training_anchor_ids = anchor_ids.copy()
    for test_panel, train_panel in (("cap10_kl0p05_local", "cap10_kl0p05_broad"), ("crossed_local", "crossed_broad")):
        for row in frame[frame.panel.eq(test_panel)].itertuples():
            index = int(row.row)
            candidates = frame[frame.panel.eq(train_panel) & frame.state_id.eq(row.state_id)]
            if candidates.empty:
                train_anchor[index] = np.nan
                train_component_anchor[index] = np.nan
                train_component_sd[index] = np.nan
                train_anchor_sd[index] = np.nan
                training_anchor_ids[index] = "[]"
                continue
            source_index = int(candidates.iloc[0].row)
            train_anchor[index] = frame.loc[source_index, "anchor"]
            train_component_anchor[index] = component_anchor[source_index]
            train_component_sd[index] = component_sd[source_index]
            train_anchor_sd[index] = frame.loc[source_index, "anchor_std_bpb"]
            training_anchor_ids[index] = anchor_ids[source_index]
    frame["training_anchor"] = train_anchor
    frame["training_anchor_sd"] = train_anchor_sd
    frame["training_anchor_row_ids_json"] = training_anchor_ids
    arrays = {
        "bucket_names": np.asarray(buckets),
        "component_names": np.asarray(COMPONENTS),
        "component_weights": fixed_weights,
        "phase0_weight": w0,
        "phase1_weight": w1,
        "historical_component_weights": historical_weights,
        "phase0_epochs": frame[audit.action_columns(frame, "p0e::")].to_numpy(float),
        "phase1_epochs": frame[audit.action_columns(frame, "p1e::")].to_numpy(float),
        "tied_phase1_epochs": frame[audit.action_columns(frame, "cp1e::")].to_numpy(float),
        "component_bpb": component_bpb,
        "component_anchor": component_anchor,
        "component_anchor_sd": component_sd,
        "training_anchor": train_anchor,
        "training_anchor_sd": train_anchor_sd,
        "training_component_anchor": train_component_anchor,
        "training_component_anchor_sd": train_component_sd,
        "target": frame.target.to_numpy(float),
        "anchor": frame.anchor.to_numpy(float),
        "sqrt_h2": sqrt_h2,
    }
    np.savez_compressed(output / "arrays.npz", **arrays)
    metadata = [str(column) for column in frame if "::" not in str(column)]
    frame[metadata].to_csv(output / "rows.csv", index=False)
    # This copy preserves the historical fitting API exactly; no original script is invoked.
    frame.to_csv(output / "audit_frame.csv", index=False)
    frame.to_csv(output / "legacy_frame.csv", index=False)
    pd.DataFrame(
        {
            "bucket": buckets,
            "historic_phase0_rate": arrays["phase0_epochs"][0] / w0[0],
            "historic_phase1_rate": arrays["tied_phase1_epochs"][0] / w0[0],
        }
    ).to_csv(output / "epoch_rates.csv", index=False)
    fold_records = []
    for label in ("primary_train", "broad_train"):
        selected = frame[frame.benchmark_role.eq(label)]
        labels = audit.geometric_folds(selected)
        for row, fold in zip(selected.row, labels, strict=True):
            fold_records.append({"split": f"{label}_historical_inner", "row": int(row), "fold": int(fold)})
    pd.DataFrame(fold_records).to_csv(output / "historical_inner_folds.csv", index=False)
    inventory = (
        frame.groupby(["panel", "benchmark_role"], sort=False)
        .agg(
            rows=("row", "size"),
            states=("state_id", "nunique"),
            actions=("coordinate_hash", "nunique"),
            components=("component_available", "sum"),
        )
        .reset_index()
    )
    inventory.to_csv(output / "inventory.csv", index=False)
    state_summary = (
        frame.groupby(["panel", "state_id"], sort=False)
        .agg(
            rows=("row", "size"),
            fit_rows=("fit_budget", "sum"),
            tied_rows=("is_tied_control", "sum"),
            anchor=("anchor", "first"),
            anchor_sd=("anchor_std_bpb", "first"),
            checkpoint_uri=("checkpoint_uri", "first"),
        )
        .reset_index()
    )
    state_summary.to_csv(output / "states.csv", index=False)
    aliases = (
        frame.groupby("coordinate_hash", sort=False)
        .agg(
            rows=("row", "size"),
            panels=("panel", lambda values: "|".join(sorted(set(values)))),
            action_ids=("action_id", lambda values: "|".join(sorted(set(values)))),
        )
        .reset_index()
    )
    aliases.to_csv(output / "action_aliases.csv", index=False)
    audit_source = AUDIT / "audit_delphi_phase1_branch_response_20260826.py"
    shutil.copyfile(audit_source, output / "historical_audit_source.py")
    checks = {
        "row_count": len(frame),
        "component_rows": int(available.sum()),
        "component_aggregate_max_absolute_error": aggregate_error,
        "sqrt_h2_feature_replay_max_error": 0.0,
        "historical_component_weights_max_error_across_all_component_rows": historical_weight_error,
        "primary_train_rows": int(frame.benchmark_role.eq("primary_train").sum()),
        "primary_test_rows": int(frame.benchmark_role.eq("primary_test").sum()),
        "no_referee_source_opened": True,
        "boundary_outcomes_used": False,
        "arrays": {key: list(value.shape) for key, value in arrays.items()},
    }
    (output / "validation.json").write_text(json.dumps(checks, indent=2) + "\n")
    (output / "README.md").write_text(
        """# Fixed-checkpoint branch inputs

This package contains only previously materialized, already-open outcomes. All indices refer to the same order
in `rows.csv` and `arrays.npz`. `audit_frame.csv` also retains the archived Hellinger benchmark column
interface. `historical_audit_source.py` is copied for exact baseline reproduction; its network-enabled main
must not be called.

Primary comparison: proportional `primary_train` (80 actions) versus `primary_test` (40 outcome-blind
acquisition actions). The 40 `adaptive_descriptive` actions were chosen using earlier outcomes and are
descriptive. The historic cap-10 comparison uses 100 broad training actions and 80 local test actions. Their
native calibration anchors are grouped by panel, prefix, and prefix repeat seed, matching the published audit.
They have macro BPB but lack the full seven component outcomes.

Crossed broad contains 50 actions crossed with 9 frozen checkpoint states and 27 controls. Crossed local
contains 10 local actions crossed with those states and 27 controls. Action holdouts must keep all checkpoint
copies, aliases, and repeats of an action together. `coordinate_hash` and `action_aliases.csv` identify exact
duplicate mixture coordinates, including local anchors reused from historical panels. State holdouts must use
checkpoint identity; related hardware bridge or repeated prefixes require grouping if the question is
unseen-prefix transfer. Six historically confirmatory states are labelled for comparison with the earlier
protocol; all data is now development evidence.

`is_tied_control` outcomes are calibration data, not scored proposals. `anchor_row_ids_json` records native
panel controls. Use `training_anchor` and `training_component_anchor` for the new comparison: local test rows
inherit broad-training controls from the same state. Local-panel controls remain available only for secondary
calibration-sensitivity diagnostics. `training_anchor_row_ids_json` records those exact control IDs.
`component_anchor_sd` and scalar `anchor_std_bpb` are observed repeat SDs, NaN where unavailable; NaN does not
mean zero noise. Geometric folds for historical primary/broad training are frozen in
`historical_inner_folds.csv` and use no outcomes. Later crossed folds should be constructed only from actions
and must keep duplicate coordinates together. This package does not select new models or inspect held-out
outcome scores.

Historical epochs use the exact materialized artifacts. Crossed epochs use the archived boundary exposure
rates and the realized 2400/3007 prefix and 607/3007 continuation duration. Metadata declares 0.8 but realized
step counts govern this coordinate. Only boundary exposure columns were read, never boundary outcomes. No live
storage was accessed, and no sealed referee table was opened.

`prefix_checkpoint_uri` and `terminal_checkpoint_uri` are distinct. The historical cap-10 result export's
original `checkpoint_uri` names the terminal checkpoint; the prefix URI is recovered from the frozen crossed
registry. Crossed bridge URI templates are resolved with their materialization commit. The proportional combined
export does not retain an exact prefix URI, so its provenance is explicitly limited to the recorded prefix
candidate, repeat seed, geometry, run identity and source provenance hash. No checkpoint URI is invented there.

Component order and fixed aggregate weights are stored explicitly. Canonical constants come from the
pre-existing July 24 component benchmark and reconstruct all available macro BPB within 1.97e-7. The old
branch audit's numerically inferred weights are separately preserved as `historical_component_weights`; their
maximum error on the newer crossed rows is 5.31e-7. No weights were fitted here. `sqrt_h2` reproduces the
archived tangent-plus-Hellinger feature matrix bit-for-bit. The proportional endpoint array and historical
folds therefore provide exact inputs for reproducing its 0.003174 BPB test RMSE; that model fitting belongs to
the comparison, not this preparation stage.
"""
    )
    files = {
        str(path.relative_to(output)): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "manifest.json" and "__pycache__" not in path.parts
    }
    (output / "manifest.json").write_text(
        json.dumps(
            {"source_sha256": sha256(Path(__file__)), "inputs": INPUTS, "outputs": files, "contract": checks}, indent=2
        )
        + "\n"
    )
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
