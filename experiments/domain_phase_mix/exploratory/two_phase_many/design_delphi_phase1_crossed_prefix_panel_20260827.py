# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["fsspec[gcs]", "numpy", "pandas"]
# ///
"""Freeze a nine-state by 50-common-branch Delphi continuation panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import fsspec
import numpy as np
import pandas as pd

from experiments.domain_phase_mix import launch_delphi_3e18_phase0_harsh_cap_candidates as harsh
from experiments.domain_phase_mix import launch_delphi_3e18_phase0_prefix_candidates as cap10
from experiments.domain_phase_mix import launch_delphi_3e18_phase0_prefix_replay as replay
from experiments.domain_phase_mix import launch_delphi_augmented_swarm_3e18 as base
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    design_delphi_phase1_common_branches_20260824 as common_design,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_CAP10_WEIGHTS = REFERENCE_OUTPUTS / "delphi_phase0_prefix_candidates_20260824" / "candidate_weights.csv"
DEFAULT_HARSH_WEIGHTS = (
    REFERENCE_OUTPUTS / "delphi_phase0_harsh_cap_candidates_20260825" / "training_candidate_weights.csv"
)
DEFAULT_COMMON_WEIGHTS = REFERENCE_OUTPUTS / "delphi_phase1_common_branches_20260824" / "continuation_weights.csv"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "delphi_phase1_crossed_prefix_panel_v3_20260827"

CAP10_WEIGHTS_SHA256 = "fef07d4188ef05f4df4a43d1eda6a12f7d2daf69a1ae1eb777863fd20db732b6"
HARSH_WEIGHTS_SHA256 = "ec2814449c1e2ff2c7561cfa72cd419f1d62afc67082074ea405e8176b3bb2d1"
COMMON_WEIGHTS_SHA256 = "9305b5c1598c9eb11e7f898f709bfb193f37802efaba40a43fbecd0d52c12355"
HARSH_ALIASES_SHA256 = "a518d943eb1784524e96353010811d3951e077848e907f908c754d788727e299"
CAP10_REPLAY_COMMIT = "2659c1bf8e7dbb0830b4476bb763a90a35d71837"
HARSH_REPLAY_COMMIT = "62ecbfec3c2e59a647b103f4eb9953667cbeffb0"
RUNTIME_CODE_COMMIT = "__RUNTIME_CODE_COMMIT__"
MIXTURE_BLOCK_SIZE = replay.MIXTURE_BLOCK_SIZE
FIT_BRANCH_COUNT = 50
FIT_DATA_SEED = 970_000
LOW_SENTINEL_DATA_SEED = 930_001
HIGH_SENTINEL_DATA_SEED = 930_002
LOW_SENTINEL_ACTION = "fit_maximin_00"
HIGH_SENTINEL_ACTION = "fit_maximin_26"
BRANCH_RUN_ID_BASE = 990_000
BRIDGE_RUN_ID = 999_900
BRIDGE_STATE_ID = "shared_bounded_ensemble_kl0p05__v6e_bridge"
SEED_REPLICATE_STATE_ID = "cap4_shared_bounded_ensemble_kl0__seed1"
BRIDGE_OUTPUT_TEMPLATE = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
    "delphi_3e18_phase0_cap10_v6e_bridge_20260827/{code_commit}/prefix_shared_bounded_ensemble_kl0p05_seed0"
)

CAP10_STATES = (
    "observed_cap10_best",
    "shared_bounded_ensemble_kl0p05",
    "shared_bounded_ensemble_kl0p2",
    "shared_bounded_ensemble_kl0p5",
)
CAP4_STATES = (
    "cap4_shared_bounded_ensemble_kl0",
    "cap4_shared_bounded_ensemble_kl0p05",
    "cap4_shared_bounded_ensemble_kl0p2",
)
STATE_IDS = (*CAP10_STATES, *CAP4_STATES, SEED_REPLICATE_STATE_ID, BRIDGE_STATE_ID)

CAP10_CHECKPOINTS = {
    "observed_cap10_best": (
        "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_3e18_phase0_prefix_candidates_20260824/"
        "prefix_observed_cap10_best_seed0-c3f2d8/checkpoints/step-2399",
        "ed0404837bba22e436c53001ca136c2456ff73e6f279674fcfa278d2f6f5448f",
    ),
    "shared_bounded_ensemble_kl0p05": (
        "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_3e18_phase0_prefix_candidates_20260824/"
        "prefix_shared_bounded_ensemble_kl0p05_seed0-543050/checkpoints/step-2399",
        "5b81b4eba6f2f42a98caf2472ac0c3008be7b591f2db2a2537cdcec67b59af75",
    ),
    "shared_bounded_ensemble_kl0p2": (
        "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_3e18_phase0_prefix_candidates_20260824/"
        "prefix_shared_bounded_ensemble_kl0p2_seed0-a2f1a1/checkpoints/step-2399",
        "fdcf6c3c15d8225a04fdd3991e158acc465c1bc6639e5b9d772d9f9e1a9c719d",
    ),
    "shared_bounded_ensemble_kl0p5": (
        "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_3e18_phase0_prefix_candidates_20260824/"
        "prefix_shared_bounded_ensemble_kl0p5_seed0-d724b9/checkpoints/step-2399",
        "e6c8e854d5b6c40782456a0ed71f2c2f1f085328699830c6a76d30f190b962e3",
    ),
}

HARSH_ROOT = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/" "delphi_3e18_phase0_harsh_cap_candidates_v6e_r2_20260825"
)
HARSH_CHECKPOINTS = {
    "cap4_shared_bounded_ensemble_kl0": (
        "prefix_cap4_shared_bounded_ensemble_kl0_seed0-856708",
        "c946e9d1d6eab50cc1bf43c67eeeaff0fad6fd4390ca4d76978aee3e21e8e18e",
    ),
    "cap4_shared_bounded_ensemble_kl0p05": (
        "prefix_cap4_shared_bounded_ensemble_kl0p05_seed0-9c322c",
        "f3c8aa1602d62217c3a66c226f30f0230dff38ac08fb408cf6a5d3ac863dd67b",
    ),
    "cap4_shared_bounded_ensemble_kl0p2": (
        "prefix_cap4_shared_bounded_ensemble_kl0p2_seed0-9e1cb1",
        "a4373da3a3140d9ed935d4da1d19c05ce8d77cdce967137807f33e967830c244",
    ),
    SEED_REPLICATE_STATE_ID: (
        "prefix_cap4_shared_bounded_ensemble_kl0_seed1-32684b",
        "91c89caa41f857586ec4f222e4105e2549b33a3f57364b3e8357a2a2111d345b",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap10-weights", type=Path, default=DEFAULT_CAP10_WEIGHTS)
    parser.add_argument("--harsh-weights", type=Path, default=DEFAULT_HARSH_WEIGHTS)
    parser.add_argument("--common-weights", type=Path, default=DEFAULT_COMMON_WEIGHTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-checkpoint-audit", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_exact(path: Path, payload: bytes) -> None:
    if path.exists() and path.read_bytes() != payload:
        raise ValueError(f"Refusing to replace different frozen artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def runtime_counts(weights: np.ndarray) -> np.ndarray:
    scaled = weights * MIXTURE_BLOCK_SIZE
    counts = np.floor(scaled).astype(int)
    remainder = MIXTURE_BLOCK_SIZE - int(counts.sum())
    order = np.lexsort((np.arange(len(weights)), -(scaled - counts)))
    counts[order[:remainder]] += 1
    if int(counts.sum()) != MIXTURE_BLOCK_SIZE:
        raise ValueError("Runtime count projection failed")
    return counts


def common_branch_ids(common_weights: pd.DataFrame) -> tuple[tuple[str, ...], dict[str, float | int]]:
    fit_ids = tuple(common_weights.loc[common_weights.fit_budget, "continuation_id"].drop_duplicates())
    if fit_ids != tuple(f"fit_maximin_{position:02d}" for position in range(FIT_BRANCH_COUNT)):
        raise ValueError("Frozen common-branch identities changed")
    buckets = tuple(common_weights.loc[common_weights.continuation_id.eq(fit_ids[0]), "bucket"])
    matrix = np.stack(
        [
            common_weights.loc[common_weights.continuation_id.eq(continuation_id)]
            .set_index("bucket")
            .loc[list(buckets), "phase_1_weight"]
            .to_numpy(dtype=float)
            for continuation_id in fit_ids
        ]
    )
    projection = np.eye(len(buckets)) - np.ones((len(buckets), len(buckets))) / len(buckets)
    basis = np.linalg.svd(projection)[0][:, : len(buckets) - 1]
    centered_features = (matrix - matrix.mean(axis=0)) @ basis
    singular = np.linalg.svd(centered_features, compute_uv=False)
    rank = int(np.linalg.matrix_rank(centered_features, tol=1e-12))
    if rank != len(buckets) - 1:
        raise ValueError(f"Centered branch design has tangent rank {rank}, expected {len(buckets) - 1}")
    residual_degrees_of_freedom = len(fit_ids) - rank - 1
    if residual_degrees_of_freedom != 11:
        raise ValueError(f"Unexpected residual degrees of freedom: {residual_degrees_of_freedom}")
    return fit_ids, {
        "centered_tangent_rank": rank,
        "tangent_degrees_of_freedom": len(buckets) - 1,
        "residual_degrees_of_freedom_per_state": residual_degrees_of_freedom,
        "centered_condition_number": float(singular[0] / singular[-1]),
        "centered_minimum_singular_value": float(singular[-1]),
    }


def source_specs(cap10_weights: Path, harsh_weights: Path) -> dict[str, tuple[base.DelphiSwarmRunSpec, dict[str, Any]]]:
    cap10_specs, _ = cap10.candidate_specs(
        candidate_weights_path=cap10_weights,
        expected_sha256=CAP10_WEIGHTS_SHA256,
        analysis_output_path=base.DEFAULT_ANALYSIS_OUTPUT_PATH,
        tpu_region="us-east5",
        tpu_zone="us-east5-a",
    )
    harsh_specs, _ = harsh.candidate_specs(
        candidate_weights_path=harsh_weights,
        expected_sha256=HARSH_WEIGHTS_SHA256,
        analysis_output_path=base.DEFAULT_ANALYSIS_OUTPUT_PATH,
        tpu_type="v6e-8",
        tpu_region="us-east5",
        tpu_zone="us-east5-b",
    )
    cap10_by_id = {cap10.candidate_id_for_spec(spec): spec for spec in cap10_specs if spec.trainer_seed == 0}
    harsh_by_identity = {
        (harsh.candidate_id_for_spec(spec), spec.trainer_seed): spec
        for spec in harsh_specs
        if harsh.candidate_id_for_spec(spec) in CAP4_STATES
    }
    rows: dict[str, tuple[base.DelphiSwarmRunSpec, dict[str, Any]]] = {}
    for state_id in CAP10_STATES:
        rows[state_id] = (
            cap10_by_id[state_id],
            {
                "candidate_id": state_id,
                "repeat_seed": 0,
                "source_family": "cap10_v5p",
                "source_weights_sha256": CAP10_WEIGHTS_SHA256,
                "source_aliases_sha256": None,
                "prefix_replay_code_commit": CAP10_REPLAY_COMMIT,
                "checkpoint_ready_at_design_time": True,
            },
        )
    for state_id in CAP4_STATES:
        rows[state_id] = (
            harsh_by_identity[(state_id, 0)],
            {
                "candidate_id": state_id,
                "repeat_seed": 0,
                "source_family": "cap4_v6e",
                "source_weights_sha256": HARSH_WEIGHTS_SHA256,
                "source_aliases_sha256": HARSH_ALIASES_SHA256,
                "prefix_replay_code_commit": HARSH_REPLAY_COMMIT,
                "checkpoint_ready_at_design_time": True,
            },
        )
    replicate_spec = harsh_by_identity[("cap4_shared_bounded_ensemble_kl0", 1)]
    rows[SEED_REPLICATE_STATE_ID] = (
        replicate_spec,
        {
            "candidate_id": "cap4_shared_bounded_ensemble_kl0",
            "repeat_seed": 1,
            "source_family": "cap4_v6e_seed_replicate",
            "source_weights_sha256": HARSH_WEIGHTS_SHA256,
            "source_aliases_sha256": HARSH_ALIASES_SHA256,
            "prefix_replay_code_commit": HARSH_REPLAY_COMMIT,
            "checkpoint_ready_at_design_time": True,
        },
    )
    bridge_spec = replace(
        cap10_by_id["shared_bounded_ensemble_kl0p05"],
        run_order=BRIDGE_RUN_ID,
        run_id=BRIDGE_RUN_ID,
        source_experiment="pinlin_calvin_xu/data_mixture/delphi_3e18_phase0_cap10_v6e_bridge_20260827",
        panel_source="cap10_v6e_hardware_bridge",
        tpu_type="v6e-8",
        tpu_region="us-east5",
        tpu_zone="us-east5-b",
        tensor_parallel_size=1,
    )
    rows[BRIDGE_STATE_ID] = (
        bridge_spec,
        {
            "candidate_id": "shared_bounded_ensemble_kl0p05",
            "repeat_seed": 0,
            "source_family": "cap10_v6e_hardware_bridge",
            "source_weights_sha256": CAP10_WEIGHTS_SHA256,
            "source_aliases_sha256": None,
            "prefix_replay_code_commit": RUNTIME_CODE_COMMIT,
            "checkpoint_ready_at_design_time": False,
        },
    )
    if tuple(rows) != STATE_IDS:
        raise ValueError("Prefix state ordering changed")
    return rows


def checkpoint_identity(state_id: str) -> tuple[str, str]:
    if state_id in CAP10_CHECKPOINTS:
        return CAP10_CHECKPOINTS[state_id]
    if state_id == BRIDGE_STATE_ID:
        return (
            f"{BRIDGE_OUTPUT_TEMPLATE}/checkpoints/step-{replay.EXPECTED_PREFIX_HF_STEP}",
            RUNTIME_CODE_COMMIT,
        )
    directory, provenance_sha256 = HARSH_CHECKPOINTS[state_id]
    return f"{HARSH_ROOT}/{directory}/checkpoints/step-{replay.EXPECTED_PREFIX_HF_STEP}", provenance_sha256


def audit_checkpoint(checkpoint_uri: str, provenance_sha256: str) -> None:
    fs, checkpoint_path = fsspec.core.url_to_fs(checkpoint_uri)
    with fs.open(os.path.join(checkpoint_path, "metadata.json")) as handle:
        metadata = json.load(handle)
    if metadata.get("step") != replay.EXPECTED_PREFIX_HF_STEP or metadata.get("is_temporary") is not False:
        raise ValueError(f"Checkpoint is not the permanent phase boundary: {checkpoint_uri}")
    output_root = checkpoint_uri.rsplit("/checkpoints/", maxsplit=1)[0]
    provenance_bytes = fsspec.open(f"{output_root}/{harsh.CANDIDATE_PROVENANCE_FILENAME}", "rb").open().read()
    if bytes_sha256(provenance_bytes) != provenance_sha256:
        raise ValueError(f"Prefix provenance changed: {checkpoint_uri}")


def support_audit(
    prefix_rows: list[dict[str, Any]],
    branch_ids: tuple[str, ...],
    common_by_id: dict[str, pd.DataFrame],
    buckets: tuple[str, ...],
) -> dict[str, Any]:
    geometry = common_design.load_canonical_panel_geometry()
    if geometry.buckets != buckets:
        raise ValueError("Common-branch bucket order no longer matches the canonical panel")
    rows = []
    for prefix in prefix_rows:
        run_spec_payload = cast(dict[str, Any], prefix["run_spec"])
        phase_weights = cast(dict[str, dict[str, float]], run_spec_payload["phase_weights"])
        prefix_weights = np.asarray([phase_weights["phase_0"][bucket] for bucket in buckets])
        phase0_exposure = prefix_weights * geometry.c0
        total_violations = 0
        phase1_violations = 0
        worst_total_excess = -np.inf
        worst_total_bucket = ""
        worst_total_action = ""
        for continuation_id in branch_ids:
            group = common_by_id[continuation_id]
            continuation = group.phase_1_weight.to_numpy(dtype=float)
            phase1_exposure = continuation * geometry.c1
            phase1_cap = group.historical_phase_1_bucket_epoch_cap.to_numpy(dtype=float)
            total_cap = group.historical_total_bucket_epoch_cap.to_numpy(dtype=float)
            phase1_violations += int(np.any(phase1_exposure > phase1_cap + 1e-12))
            total_excess = phase0_exposure + phase1_exposure - total_cap
            if np.any(total_excess > 1e-12):
                total_violations += 1
            position = int(np.argmax(total_excess))
            if float(total_excess[position]) > worst_total_excess:
                worst_total_excess = float(total_excess[position])
                worst_total_bucket = buckets[position]
                worst_total_action = continuation_id
        rows.append(
            {
                "state_id": prefix["state_id"],
                "phase1_support_violation_actions": phase1_violations,
                "total_support_violation_actions": total_violations,
                "prefix_only_total_support_violation_buckets": int(
                    np.sum(
                        phase0_exposure
                        > common_by_id[branch_ids[0]].historical_total_bucket_epoch_cap.to_numpy() + 1e-12
                    )
                ),
                "worst_total_support_excess_epochs": worst_total_excess,
                "worst_total_support_excess_bucket": worst_total_bucket,
                "worst_total_support_excess_action": worst_total_action,
            }
        )
    if any(row["phase1_support_violation_actions"] for row in rows):
        raise ValueError("A frozen common action exceeds its original phase-1 support cap")
    return {
        "per_state": rows,
        "interpretation": (
            "The frozen phase-1 actions remain inside their original phase-1 support. Total-exposure violations are "
            "reported rather than silently filtered because the cap-4 KL=0 prefix itself lies outside the cap-10 "
            "prefix envelope; replacing actions by state would destroy the common-action estimand."
        ),
    }


def build_design(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    for path, expected in (
        (args.cap10_weights, CAP10_WEIGHTS_SHA256),
        (args.harsh_weights, HARSH_WEIGHTS_SHA256),
        (args.common_weights, COMMON_WEIGHTS_SHA256),
    ):
        if file_sha256(path) != expected:
            raise ValueError(f"Frozen input changed: {path}")
    common_weights = pd.read_csv(args.common_weights)
    branch_ids, rank_audit = common_branch_ids(common_weights)
    specs = source_specs(args.cap10_weights, args.harsh_weights)

    prefix_rows = []
    for state_id, (spec, source) in specs.items():
        checkpoint_uri, provenance_sha256 = checkpoint_identity(state_id)
        ready = bool(source["checkpoint_ready_at_design_time"])
        if ready and not args.skip_checkpoint_audit:
            audit_checkpoint(checkpoint_uri, provenance_sha256)
        prefix_rows.append(
            {
                "state_id": state_id,
                "checkpoint_uri": checkpoint_uri,
                "provenance_sha256": provenance_sha256,
                **source,
                "run_spec": asdict(spec),
            }
        )

    buckets = tuple(common_weights.loc[common_weights.continuation_id.eq(branch_ids[0]), "bucket"].astype(str))
    common_by_id = {
        continuation_id: (
            common_weights.loc[common_weights.continuation_id.eq(continuation_id)].set_index("bucket").loc[list(buckets)]
        )
        for continuation_id in branch_ids
    }
    support = support_audit(prefix_rows, branch_ids, common_by_id, buckets)
    rows = []
    long_rows = []
    run_order = 0
    for prefix_position, prefix in enumerate(prefix_rows):
        state_id = str(prefix["state_id"])
        run_spec_payload = cast(dict[str, Any], prefix["run_spec"])
        phase_weights = cast(dict[str, dict[str, float]], run_spec_payload["phase_weights"])
        prefix_weights = np.asarray([phase_weights["phase_0"][bucket] for bucket in buckets])
        actions: list[tuple[str, str, bool, int, np.ndarray, str]] = []
        for continuation_id in branch_ids:
            weights = common_by_id[continuation_id].phase_1_weight.to_numpy(dtype=float)
            actions.append((continuation_id, "common_fit", True, FIT_DATA_SEED, weights, continuation_id))
        tied_weights = runtime_counts(prefix_weights) / MIXTURE_BLOCK_SIZE
        actions.append(("control_tied", "prefix_tied_control", False, FIT_DATA_SEED, tied_weights, "control_tied"))
        for action_id, data_seed, label in (
            (LOW_SENTINEL_ACTION, LOW_SENTINEL_DATA_SEED, "low"),
            (HIGH_SENTINEL_ACTION, HIGH_SENTINEL_DATA_SEED, "high"),
        ):
            weights = common_by_id[action_id].phase_1_weight.to_numpy(dtype=float)
            actions.append(
                (
                    f"sentinel_{label}_{action_id}",
                    f"common_action_{label}_exposure_noise_control",
                    False,
                    data_seed,
                    weights,
                    action_id,
                )
            )
        for continuation_id, role, fit_budget, data_seed, weights, action_id in actions:
            counts = runtime_counts(weights)
            exact_weights = counts / MIXTURE_BLOCK_SIZE
            if not np.array_equal(exact_weights, weights):
                raise ValueError(f"Action is not runtime-exact: {state_id}/{continuation_id}")
            row_id = f"p{prefix_position:02d}__{continuation_id}"
            rows.append(
                {
                    "run_order": run_order,
                    "run_id": BRANCH_RUN_ID_BASE + run_order,
                    "row_id": row_id,
                    "prefix_state_id": state_id,
                    "prefix_candidate_id": prefix["candidate_id"],
                    "prefix_repeat_seed": prefix["repeat_seed"],
                    "continuation_id": continuation_id,
                    "action_id": action_id,
                    "role": role,
                    "fit_budget": fit_budget,
                    "data_seed": data_seed,
                    "trainer_seed": 0,
                }
            )
            for bucket, count, weight in zip(buckets, counts, exact_weights, strict=True):
                long_rows.append(
                    {
                        "row_id": row_id,
                        "bucket": bucket,
                        "phase_1_count": int(count),
                        "phase_1_weight": float(weight),
                    }
                )
            run_order += 1
    panel_rows = pd.DataFrame(rows)
    panel_weights = pd.DataFrame(long_rows)
    controls_per_state = 3
    expected_rows = len(STATE_IDS) * (FIT_BRANCH_COUNT + controls_per_state)
    if len(panel_rows) != expected_rows or panel_rows.row_id.nunique() != expected_rows:
        raise ValueError("Crossed panel row identity changed")
    prefix_registry = {
        "prefixes": prefix_rows,
        "prefix_count": len(prefix_rows),
        "state_ids": list(STATE_IDS),
        "phase_boundary_completed_updates": replay.EXPECTED_PREFIX_TRAIN_STEPS,
        "phase_boundary_checkpoint_step": replay.EXPECTED_PREFIX_HF_STEP,
    }
    manifest: dict[str, Any] = {
        "contract_version": "delphi_phase1_crossed_prefix_panel_20260827_v3",
        "estimand": "prefix-conditioned response to the same absolute phase-1 action",
        "primary_model_contract": (
            "Compare prefix-shared and prefix-varying parameters in a bounded-shape DSP branch-response model. "
            "The 38-dimensional tangent linear fit is descriptive and a lack-of-fit diagnostic, not the primary "
            "mechanistic claim."
        ),
        "primary_inference": {
            "target": "Uncheatable BPB",
            "comparison": (
                "action-blocked cross-validated error of prefix-shared versus prefix-varying bounded-shape DSP"
            ),
            "resampling_unit": "the 50 common phase-1 actions, kept intact across prefix states",
            "uncertainty": "paired bootstrap over common actions with prefix-seed controls reported separately",
            "multiplicity": (
                "Uncheatable is the sole confirmatory target; components and other endpoints are exploratory"
            ),
            "confirmatory_state_ids": [
                "shared_bounded_ensemble_kl0p05",
                "shared_bounded_ensemble_kl0p2",
                "shared_bounded_ensemble_kl0p5",
                "cap4_shared_bounded_ensemble_kl0",
                "cap4_shared_bounded_ensemble_kl0p05",
                "cap4_shared_bounded_ensemble_kl0p2",
            ],
            "required_sensitivity": (
                "repeat the primary comparison without cap4_shared_bounded_ensemble_kl0, whose prefix lies outside "
                "the historical total-exposure envelope"
            ),
        },
        "prefix_count": len(STATE_IDS),
        "fit_branches_per_prefix": FIT_BRANCH_COUNT,
        "controls_per_prefix": controls_per_state,
        "total_rows": len(panel_rows),
        "fit_rows": int(panel_rows.fit_budget.sum()),
        "new_fit_rows": int(panel_rows.fit_budget.sum()),
        "reused_fit_rows": 0,
        "new_control_rows": int((~panel_rows.fit_budget).sum()),
        "common_fit_branch_ids": list(branch_ids),
        "selection_rule": "use the complete frozen 50-action bank without endpoint-label-dependent reselection",
        "rank_audit": rank_audit,
        "support_audit": support,
        "common_random_numbers": {
            "fit_data_seed": FIT_DATA_SEED,
            "continuation_trainer_seed": 0,
            "low_exposure_sentinel": {
                "action_id": LOW_SENTINEL_ACTION,
                "data_seed": LOW_SENTINEL_DATA_SEED,
            },
            "high_exposure_sentinel": {
                "action_id": HIGH_SENTINEL_ACTION,
                "data_seed": HIGH_SENTINEL_DATA_SEED,
            },
        },
        "controls": {
            "prefix_seed_single_draw_diagnostic_state": SEED_REPLICATE_STATE_ID,
            "prefix_seed_diagnostic_limitation": (
                "one paired seed contrast is not a null distribution and both cap-4 KL=0 states are outside the "
                "historical total-exposure envelope"
            ),
            "phase0_hardware_bridge_state": BRIDGE_STATE_ID,
            "phase0_hardware_bridge_interpretation": (
                "the bridge detects a composite phase-0 hardware-by-code offset at one cap-10 weight point; it does "
                "not identify a hardware-only correction and must not be used as one"
            ),
            "branch_code_commit_policy": "rerun every panel cell under one exact launch commit; reuse zero old cells",
        },
        "state_roles": {
            "observed_cap10_best": (
                "descriptive-only outcome-selected incumbent; excluded from the confirmatory prefix-shared versus "
                "prefix-varying comparison"
            ),
            BRIDGE_STATE_ID: "hardware-by-code diagnostic; excluded from the confirmatory model comparison",
            SEED_REPLICATE_STATE_ID: (
                "single-draw prefix-seed diagnostic; excluded from the confirmatory model comparison"
            ),
        },
        "missing_cell_policy": (
            "Idempotently rerun the exact frozen row until complete. Do not replace missing actions or states. "
            "Any irrecoverable missing fit row blocks the confirmatory interaction comparison; incomplete data may "
            "only support explicitly labeled descriptive regularized fits."
        ),
        "endpoint_labels_used_for_branch_selection": False,
        "endpoint_labels_used_for_state_selection": {
            "observed_cap10_best": True,
            "all_other_states": False,
        },
        "exposure_policy": (
            "This panel intentionally drops the later harsh-cap experiment's 10-total-epoch gate and instead keeps "
            "the complete frozen action bank within its original historical support envelope. The bounded-shape DSP "
            "must therefore cover the observed 0.245-to-53.7 phase-1 materialized-epoch range."
        ),
        "source_artifacts": {
            "cap10_weights_sha256": CAP10_WEIGHTS_SHA256,
            "harsh_weights_sha256": HARSH_WEIGHTS_SHA256,
            "common_weights_sha256": COMMON_WEIGHTS_SHA256,
        },
    }
    return panel_rows, panel_weights, prefix_registry, manifest


def write_design(args: argparse.Namespace) -> dict[str, Any]:
    panel_rows, panel_weights, prefix_registry, manifest = build_design(args)
    rows_bytes = panel_rows.to_csv(index=False).encode()
    weights_bytes = panel_weights.to_csv(index=False).encode()
    registry_bytes = (json.dumps(prefix_registry, indent=2, sort_keys=True) + "\n").encode()
    rows_path = args.output_dir / "panel_rows.csv"
    weights_path = args.output_dir / "panel_weights.csv"
    registry_path = args.output_dir / "prefix_registry.json"
    write_exact(rows_path, rows_bytes)
    write_exact(weights_path, weights_bytes)
    write_exact(registry_path, registry_bytes)
    manifest["panel_rows_sha256"] = bytes_sha256(rows_bytes)
    manifest["panel_weights_sha256"] = bytes_sha256(weights_bytes)
    manifest["prefix_registry_sha256"] = bytes_sha256(registry_bytes)
    manifest_path = args.output_dir / "manifest.json"
    write_exact(manifest_path, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
    return manifest


def main() -> None:
    manifest = write_design(parse_args())
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
