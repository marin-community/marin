# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["fsspec[gcs]", "numpy", "pandas"]
# ///
"""Freeze a local crossed-prefix panel around the validated fit079 action."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    design_delphi_phase1_common_branches_20260824 as common_design,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    design_delphi_phase1_crossed_prefix_panel_20260827 as crossed_design,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_FRONTIER_CONTRACT = (
    REFERENCE_OUTPUTS / "delphi_phase1_proportional_prefix_wave1_20260825" / "validated_frontier_contract.json"
)
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "delphi_phase1_local_crossed_prefix_fit079_20260828"

FRONTIER_CONTRACT_SHA256 = "898b8e5fdab2a9695808acfa31137918f5cf7e720674fb51e43bbc961b0aed27"
BRIDGE_CODE_COMMIT = "6f2bb6c226d936c4882715c45635caced44edcca"
BRIDGE_CHECKPOINT_URI = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
    "delphi_3e18_phase0_cap10_v6e_bridge_20260827/"
    f"{BRIDGE_CODE_COMMIT}/prefix_shared_bounded_ensemble_kl0p05_seed0/checkpoints/step-2399"
)
BRIDGE_PROVENANCE_SHA256 = "8662825536a631fe379c56e2cc31d4316767f1025a700e384dfade2d3a636100"

MIXTURE_BLOCK_SIZE = common_design.MIXTURE_BLOCK_SIZE
FIT_ACTION_COUNT = 10
BOUNDARY_DIRECTION_COUNT = 3
PAIRED_INTERIOR_DIRECTION_COUNT = 3
FORWARD_DIRECTION_COUNT = BOUNDARY_DIRECTION_COUNT + PAIRED_INTERIOR_DIRECTION_COUNT
REVERSE_DIRECTION_COUNT = PAIRED_INTERIOR_DIRECTION_COUNT
FIT_DATA_SEED = 974_000
ANCHOR_REPEAT_DATA_SEED = 974_001
SENTINEL_REPEAT_DATA_SEED = 974_002
BRANCH_RUN_ID_BASE = 1_010_000
TARGET_PAIRED_HELLINGER = 0.02
MINIMUM_LOCAL_HELLINGER = 0.01
BOUNDARY_TRANSFER_COUNT = 2
MAX_LOCAL_HELLINGER = 0.05
MAX_ADDED_PHASE1_EPOCHS = 0.20
MAX_SCREEN_CONDITION_NUMBER = 50.0
CONTRACT_VERSION = "delphi_phase1_local_crossed_prefix_fit079_20260828_v5"
PANEL_SOURCE = "delphi_phase1_local_crossed_prefix_fit079_screen10"
EXPERIMENT_NAME = "pinlin_calvin_xu/data_mixture/delphi_3e18_phase1_local_crossed_prefix_fit079_screen10_20260828"
CONFIRMATORY_STATE_IDS = (
    "shared_bounded_ensemble_kl0p05",
    "shared_bounded_ensemble_kl0p2",
    "shared_bounded_ensemble_kl0p5",
    "cap4_shared_bounded_ensemble_kl0p05",
    "cap4_shared_bounded_ensemble_kl0p2",
)
WITHIN_EXPOSURE_SENSITIVITY_STATE_IDS = (
    "shared_bounded_ensemble_kl0p5",
    "cap4_shared_bounded_ensemble_kl0p05",
    "cap4_shared_bounded_ensemble_kl0p2",
)
UNIQUE_WEIGHT_SENSITIVITY_STATE_IDS = (
    "observed_cap10_best",
    "shared_bounded_ensemble_kl0p05",
    "shared_bounded_ensemble_kl0p2",
    "shared_bounded_ensemble_kl0p5",
    "cap4_shared_bounded_ensemble_kl0",
    "cap4_shared_bounded_ensemble_kl0p05",
    "cap4_shared_bounded_ensemble_kl0p2",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier-contract", type=Path, default=DEFAULT_FRONTIER_CONTRACT)
    parser.add_argument("--cap10-weights", type=Path, default=crossed_design.DEFAULT_CAP10_WEIGHTS)
    parser.add_argument("--harsh-weights", type=Path, default=crossed_design.DEFAULT_HARSH_WEIGHTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-checkpoint-audit", action="store_true")
    return parser.parse_args()


def frontier_counts(path: Path, buckets: tuple[str, ...]) -> np.ndarray:
    if crossed_design.file_sha256(path) != FRONTIER_CONTRACT_SHA256:
        raise ValueError(f"Validated frontier contract changed: {path}")
    contract = cast(dict[str, Any], json.loads(path.read_text()))
    if contract.get("continuation_id") != "pred_scalar42_h2p5":
        raise ValueError("Validated frontier action identity changed")
    counts_by_bucket = cast(dict[str, int], contract.get("runtime_counts", {}))
    if set(counts_by_bucket) != set(buckets):
        raise ValueError("Validated frontier bucket identities changed")
    counts = np.asarray([counts_by_bucket[bucket] for bucket in buckets], dtype=np.int64)
    if int(counts.sum()) != MIXTURE_BLOCK_SIZE or int(counts.min()) < 0:
        raise ValueError("Validated frontier counts are not a runtime mixture block")
    return counts


def paired_transfer_count(
    anchor_counts: np.ndarray,
    donor: int,
    target: int,
    phase1_scales: np.ndarray,
) -> int | None:
    anchor_weights = anchor_counts / MIXTURE_BLOCK_SIZE
    for count in range(1, int(anchor_counts[target]) + 1):
        forward = anchor_counts.copy()
        forward[donor] -= count
        forward[target] += count
        reverse = anchor_counts.copy()
        reverse[donor] += count
        reverse[target] -= count
        hellinger = (
            common_design.hellinger(forward / MIXTURE_BLOCK_SIZE, anchor_weights),
            common_design.hellinger(reverse / MIXTURE_BLOCK_SIZE, anchor_weights),
        )
        maximum_added_epoch = max(
            count * float(phase1_scales[target]) / MIXTURE_BLOCK_SIZE,
            count * float(phase1_scales[donor]) / MIXTURE_BLOCK_SIZE,
        )
        if max(hellinger) > MAX_LOCAL_HELLINGER or maximum_added_epoch > MAX_ADDED_PHASE1_EPOCHS:
            return None
        if min(hellinger) >= TARGET_PAIRED_HELLINGER:
            return count
    return None


def exposure_stratified_targets(
    candidates: list[int],
    phase1_scales: np.ndarray,
    count: int,
) -> tuple[int, ...]:
    if len(candidates) < count:
        raise ValueError(f"Cannot select {count} directions from {len(candidates)} candidates")
    ordered = sorted(candidates, key=lambda target: (float(phase1_scales[target]), target))
    positions = np.rint(np.linspace(0, len(ordered) - 1, count)).astype(int)
    selected = tuple(ordered[position] for position in positions)
    if len(set(selected)) != count:
        raise ValueError("Exposure-stratified direction selection produced duplicates")
    return selected


def sparse_screen_audit(action_counts: np.ndarray) -> dict[str, float | int]:
    weights = action_counts / MIXTURE_BLOCK_SIZE
    projection = np.eye(weights.shape[1]) - np.ones((weights.shape[1], weights.shape[1])) / weights.shape[1]
    basis = np.linalg.svd(projection)[0][:, : weights.shape[1] - 1]
    offsets = (weights - weights[0]) @ basis
    singular = np.linalg.svd(offsets, compute_uv=False)
    rank = int(np.linalg.matrix_rank(offsets, tol=1e-12))
    expected_rank = BOUNDARY_DIRECTION_COUNT + PAIRED_INTERIOR_DIRECTION_COUNT
    residual_degrees_of_freedom = len(weights) - rank - 1
    condition_number = float(singular[0] / singular[rank - 1])
    if rank != expected_rank:
        raise ValueError(f"Sparse local screen has tangent rank {rank}, expected {expected_rank}")
    if residual_degrees_of_freedom != REVERSE_DIRECTION_COUNT:
        raise ValueError(f"Unexpected residual degrees of freedom: {residual_degrees_of_freedom}")
    if condition_number > MAX_SCREEN_CONDITION_NUMBER:
        raise ValueError(f"Sparse local screen condition number is too high: {condition_number}")
    return {
        "anchor_tangent_rank": rank,
        "tangent_degrees_of_freedom": weights.shape[1] - 1,
        "unresolved_tangent_dimensions": weights.shape[1] - 1 - rank,
        "residual_degrees_of_freedom_per_state": residual_degrees_of_freedom,
        "anchor_condition_number": condition_number,
        "anchor_minimum_nonzero_singular_value": float(singular[rank - 1]),
    }


def local_action_bank(
    anchor_counts: np.ndarray,
    buckets: tuple[str, ...],
    phase1_scales: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    donor = int(np.argmax(anchor_counts))
    boundary_transfer_counts = {
        target: BOUNDARY_TRANSFER_COUNT
        for target in range(len(buckets))
        if target != donor and anchor_counts[target] == 0
    }
    boundary_targets = exposure_stratified_targets(
        list(boundary_transfer_counts),
        phase1_scales,
        BOUNDARY_DIRECTION_COUNT,
    )
    paired_transfer_counts = {
        target: count
        for target in range(len(buckets))
        if target != donor and anchor_counts[target] > 0
        if (count := paired_transfer_count(anchor_counts, donor, target, phase1_scales)) is not None
    }
    interior_targets = exposure_stratified_targets(
        list(paired_transfer_counts),
        phase1_scales,
        PAIRED_INTERIOR_DIRECTION_COUNT,
    )
    actions: list[dict[str, Any]] = [
        {
            "action_id": "local_anchor_fit079",
            "role": "validated_frontier_anchor",
            "target_bucket": None,
            "signed_transfer_count": 0,
            "counts": anchor_counts.copy(),
        }
    ]
    for target in interior_targets:
        count = paired_transfer_counts[target]
        moved = anchor_counts.copy()
        moved[donor] -= count
        moved[target] += count
        actions.append(
            {
                "action_id": f"local_plus_{target:02d}",
                "role": "local_interior_forward",
                "target_bucket": buckets[target],
                "signed_transfer_count": count,
                "counts": moved,
            }
        )
        moved = anchor_counts.copy()
        moved[donor] += count
        moved[target] -= count
        actions.append(
            {
                "action_id": f"local_minus_{target:02d}",
                "role": "local_interior_reverse",
                "target_bucket": buckets[target],
                "signed_transfer_count": -count,
                "counts": moved,
            }
        )
    for target in boundary_targets:
        count = boundary_transfer_counts[target]
        moved = anchor_counts.copy()
        moved[donor] -= count
        moved[target] += count
        actions.append(
            {
                "action_id": f"local_plus_{target:02d}",
                "role": "local_boundary_activation",
                "target_bucket": buckets[target],
                "signed_transfer_count": count,
                "counts": moved,
            }
        )
    if len(actions) != FIT_ACTION_COUNT:
        raise ValueError(f"Expected {FIT_ACTION_COUNT} local actions; found {len(actions)}")
    count_matrix = np.stack([cast(np.ndarray, action["counts"]) for action in actions])
    if int(count_matrix.min()) < 0 or np.any(count_matrix.sum(axis=1) != MIXTURE_BLOCK_SIZE):
        raise ValueError("A local action is not a valid runtime mixture block")
    if len({tuple(row) for row in count_matrix}) != FIT_ACTION_COUNT:
        raise ValueError("Local action bank contains duplicate runtime mixtures")
    anchor_weights = anchor_counts / MIXTURE_BLOCK_SIZE
    hellinger = np.asarray([common_design.hellinger(row / MIXTURE_BLOCK_SIZE, anchor_weights) for row in count_matrix])
    if np.any((hellinger[1:] < MINIMUM_LOCAL_HELLINGER) | (hellinger[1:] > MAX_LOCAL_HELLINGER)):
        raise ValueError(f"A local action is outside the detectable Hellinger annulus: {hellinger}")
    if float(hellinger.max()) > MAX_LOCAL_HELLINGER:
        raise ValueError(f"Local action exceeds Hellinger radius: {hellinger.max()}")
    added_epochs = np.maximum(
        (count_matrix - anchor_counts[None, :]) * phase1_scales[None, :] / MIXTURE_BLOCK_SIZE,
        0.0,
    )
    if float(added_epochs.max()) > MAX_ADDED_PHASE1_EPOCHS:
        raise ValueError(f"Local action adds too many materialized phase-1 epochs: {added_epochs.max()}")
    audit = sparse_screen_audit(count_matrix)
    audit.update(
        {
            "forward_directions": FORWARD_DIRECTION_COUNT,
            "boundary_activation_forward_directions": len(boundary_targets),
            "interior_forward_directions": len(interior_targets),
            "reverse_directions": len(interior_targets),
            "boundary_target_buckets": [buckets[target] for target in boundary_targets],
            "paired_interior_target_buckets": [buckets[target] for target in interior_targets],
            "boundary_candidate_count": len(boundary_transfer_counts),
            "paired_interior_eligible_candidate_count": len(paired_transfer_counts),
            "paired_interior_excluded_candidate_count": int(np.sum(anchor_counts > 0)) - len(paired_transfer_counts) - 1,
            "boundary_target_phase1_scales": [float(phase1_scales[target]) for target in boundary_targets],
            "paired_interior_target_phase1_scales": [float(phase1_scales[target]) for target in interior_targets],
            "donor_bucket": buckets[donor],
            "maximum_hellinger_to_anchor": float(hellinger.max()),
            "minimum_nonzero_hellinger_to_anchor": float(hellinger[hellinger > 0.0].min()),
            "median_hellinger_to_anchor": float(np.median(hellinger)),
            "maximum_added_phase1_materialized_epochs": float(added_epochs.max()),
            "minimum_positive_added_phase1_materialized_epochs": float(added_epochs[added_epochs > 0.0].min()),
            "action_doses": [
                {
                    "action_id": str(action["action_id"]),
                    "hellinger_to_anchor": float(action_hellinger),
                    "maximum_added_phase1_materialized_epoch": float(action_added_epochs.max()),
                    "signed_transfer_count": int(action["signed_transfer_count"]),
                }
                for action, action_hellinger, action_added_epochs in zip(actions, hellinger, added_epochs, strict=True)
            ],
        }
    )
    return actions, audit


def prefix_registry(args: argparse.Namespace) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    specs = crossed_design.source_specs(args.cap10_weights, args.harsh_weights)
    prefix_rows: list[dict[str, Any]] = []
    for state_id, (spec, source) in specs.items():
        checkpoint_uri, provenance_sha256 = crossed_design.checkpoint_identity(state_id)
        source = dict(source)
        if state_id == crossed_design.BRIDGE_STATE_ID:
            checkpoint_uri = BRIDGE_CHECKPOINT_URI
            provenance_sha256 = BRIDGE_PROVENANCE_SHA256
            source["prefix_replay_code_commit"] = BRIDGE_CODE_COMMIT
            source["checkpoint_ready_at_design_time"] = True
        if not args.skip_checkpoint_audit:
            crossed_design.audit_checkpoint(checkpoint_uri, provenance_sha256)
        prefix_rows.append(
            {
                "state_id": state_id,
                "checkpoint_uri": checkpoint_uri,
                "provenance_sha256": provenance_sha256,
                **source,
                "run_spec": asdict(spec),
            }
        )
    state_ids = tuple(str(row["state_id"]) for row in prefix_rows)
    if state_ids != crossed_design.STATE_IDS:
        raise ValueError("Prefix state ordering changed")
    return prefix_rows, state_ids


def exposure_audit(
    prefix_rows: list[dict[str, Any]],
    actions: list[dict[str, Any]],
    anchor_counts: np.ndarray,
    buckets: tuple[str, ...],
    phase0_scales: np.ndarray,
    phase1_scales: np.ndarray,
) -> dict[str, Any]:
    action_counts = np.stack([cast(np.ndarray, action["counts"]) for action in actions])
    anchor_phase1 = anchor_counts * phase1_scales / MIXTURE_BLOCK_SIZE
    action_phase1 = action_counts * phase1_scales[None, :] / MIXTURE_BLOCK_SIZE
    rows = []
    for prefix in prefix_rows:
        run_spec = cast(dict[str, Any], prefix["run_spec"])
        phase_weights = cast(dict[str, dict[str, float]], run_spec["phase_weights"])
        phase0 = np.asarray([phase_weights["phase_0"][bucket] for bucket in buckets]) * phase0_scales
        anchor_total = phase0 + anchor_phase1
        panel_total = phase0[None, :] + action_phase1
        rows.append(
            {
                "state_id": prefix["state_id"],
                "anchor_max_total_materialized_epoch": float(anchor_total.max()),
                "panel_max_total_materialized_epoch": float(panel_total.max()),
                "actions_above_10_total_epochs": int(np.sum(np.any(panel_total > 10.0 + 1e-12, axis=1))),
            }
        )
    return {
        "per_state": rows,
        "interpretation": (
            "The action neighborhood is constrained by its incremental phase-1 exposure, not by state-specific "
            "filtering. Two cap-10 weight points, represented by three states including the v6e bridge, already "
            "exceed the historical 10-total-epoch heuristic under the validated anchor; retaining them preserves "
            "the common-action estimand."
        ),
    }


def build_design(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    for path, expected in (
        (args.cap10_weights, crossed_design.CAP10_WEIGHTS_SHA256),
        (args.harsh_weights, crossed_design.HARSH_WEIGHTS_SHA256),
    ):
        if crossed_design.file_sha256(path) != expected:
            raise ValueError(f"Frozen prefix input changed: {path}")
    geometry = common_design.load_canonical_panel_geometry()
    anchor_counts = frontier_counts(args.frontier_contract, geometry.buckets)
    actions, rank_audit = local_action_bank(anchor_counts, geometry.buckets, geometry.c1)
    prefix_rows, state_ids = prefix_registry(args)
    support = exposure_audit(
        prefix_rows,
        actions,
        anchor_counts,
        geometry.buckets,
        geometry.c0,
        geometry.c1,
    )
    action_by_id = {str(action["action_id"]): action for action in actions}
    anchor_weights = anchor_counts / MIXTURE_BLOCK_SIZE

    def sentinel_key(action: dict[str, Any]) -> tuple[float, float, str]:
        counts = cast(np.ndarray, action["counts"])
        maximum_added_epoch = float(np.maximum((counts - anchor_counts) * geometry.c1 / MIXTURE_BLOCK_SIZE, 0.0).max())
        return (
            maximum_added_epoch,
            common_design.hellinger(counts / MIXTURE_BLOCK_SIZE, anchor_weights),
            str(action["action_id"]),
        )

    sentinel_action = max(
        actions,
        key=sentinel_key,
    )
    rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    run_order = 0
    for prefix_position, prefix in enumerate(prefix_rows):
        run_spec = cast(dict[str, Any], prefix["run_spec"])
        phase_weights = cast(dict[str, dict[str, float]], run_spec["phase_weights"])
        prefix_counts = crossed_design.runtime_counts(
            np.asarray([phase_weights["phase_0"][bucket] for bucket in geometry.buckets])
        )
        row_actions: list[tuple[str, str, bool, int, np.ndarray, str]] = [
            (
                str(action["action_id"]),
                str(action["role"]),
                True,
                FIT_DATA_SEED,
                cast(np.ndarray, action["counts"]),
                str(action["action_id"]),
            )
            for action in actions
        ]
        row_actions.extend(
            (
                (
                    "control_tied",
                    "prefix_tied_control",
                    False,
                    FIT_DATA_SEED,
                    prefix_counts,
                    "control_tied",
                ),
                (
                    "control_anchor_repeat",
                    "anchor_exposure_noise_control",
                    False,
                    ANCHOR_REPEAT_DATA_SEED,
                    anchor_counts.copy(),
                    "local_anchor_fit079",
                ),
                (
                    "control_local_sentinel_repeat",
                    "local_action_exposure_noise_control",
                    False,
                    SENTINEL_REPEAT_DATA_SEED,
                    cast(np.ndarray, sentinel_action["counts"]).copy(),
                    str(sentinel_action["action_id"]),
                ),
            )
        )
        for continuation_id, role, fit_budget, data_seed, counts, action_id in row_actions:
            if action_id in action_by_id and not np.array_equal(counts, action_by_id[action_id]["counts"]):
                raise ValueError(f"Local action identity changed: {action_id}")
            row_id = f"screen10__p{prefix_position:02d}__{continuation_id}"
            rows.append(
                {
                    "run_order": run_order,
                    "run_id": BRANCH_RUN_ID_BASE + run_order,
                    "row_id": row_id,
                    "prefix_state_id": prefix["state_id"],
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
            weights = counts / MIXTURE_BLOCK_SIZE
            for bucket, count, weight in zip(geometry.buckets, counts, weights, strict=True):
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
    expected_rows = len(state_ids) * (FIT_ACTION_COUNT + 3)
    if len(panel_rows) != expected_rows or panel_rows.row_id.nunique() != expected_rows:
        raise ValueError("Local crossed-panel row identity changed")
    for selected_states in (
        CONFIRMATORY_STATE_IDS,
        WITHIN_EXPOSURE_SENSITIVITY_STATE_IDS,
        UNIQUE_WEIGHT_SENSITIVITY_STATE_IDS,
    ):
        if not set(selected_states).issubset(state_ids):
            raise ValueError(f"Analysis state identity changed: {selected_states}")
    registry = {
        "prefixes": prefix_rows,
        "prefix_count": len(prefix_rows),
        "state_ids": list(state_ids),
        "phase_boundary_completed_updates": crossed_design.replay.EXPECTED_PREFIX_TRAIN_STEPS,
        "phase_boundary_checkpoint_step": crossed_design.replay.EXPECTED_PREFIX_HF_STEP,
    }
    manifest: dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "panel_source": PANEL_SOURCE,
        "experiment_name": EXPERIMENT_NAME,
        "estimand": "cross-prefix transfer of the validated fit079 action and a sparse local mechanism screen",
        "research_question": (
            "Does the validated fit079 action improve over tied continuation on average across five independent, "
            "outcome-unselected prefix-weight states? As a secondary question, do any of three paired interior "
            "directions have a nonzero pooled odd response?"
        ),
        "prefix_count": len(state_ids),
        "fit_branches_per_prefix": FIT_ACTION_COUNT,
        "controls_per_prefix": 3,
        "fit_rows": int(panel_rows.fit_budget.sum()),
        "reused_fit_rows": 0,
        "new_fit_rows": int(panel_rows.fit_budget.sum()),
        "new_control_rows": int((~panel_rows.fit_budget).sum()),
        "total_rows": len(panel_rows),
        "common_fit_branch_ids": [str(action["action_id"]) for action in actions],
        "selection_rule": (
            "Start from the previously validated fit079 action; select three paired interior directions and three "
            "one-sided boundary activations at low, middle, and high phase-1 materialized-exposure scales without "
            "using endpoint labels."
        ),
        "rank_audit": rank_audit,
        "support_audit": support,
        "local_design": {
            "anchor_action_id": "local_anchor_fit079",
            "forward_direction_count": FORWARD_DIRECTION_COUNT,
            "boundary_activation_forward_directions": rank_audit["boundary_activation_forward_directions"],
            "interior_forward_directions": rank_audit["interior_forward_directions"],
            "reverse_direction_count": REVERSE_DIRECTION_COUNT,
            "target_paired_hellinger": TARGET_PAIRED_HELLINGER,
            "minimum_local_hellinger": MINIMUM_LOCAL_HELLINGER,
            "boundary_transfer_count": BOUNDARY_TRANSFER_COUNT,
            "maximum_hellinger_to_anchor": MAX_LOCAL_HELLINGER,
            "maximum_added_phase1_materialized_epochs": MAX_ADDED_PHASE1_EPOCHS,
            "dose_interpretation": (
                "Paired interior moves use the smallest symmetric count transfer reaching Hellinger 0.02 in both "
                "directions while adding at most 0.20 materialized phase-1 epoch. Each boundary activation moves "
                "two runtime counts, matching Hellinger radius but not total variation or materialized exposure to "
                "the paired interior rays. Realized distances and exposure doses are recorded per action in "
                "rank_audit.action_doses."
            ),
            "direction_interpretation": (
                "The three paired interior directions support prespecified pooled odd contrasts and descriptive "
                "even-curvature checks. The three one-sided boundary activations are exploratory because matching "
                "their Hellinger radius does not match their count, total-variation, or exposure dose to the interior "
                "moves. This sparse screen does not identify the other 32 tangent dimensions. Every direction "
                f"exchanges mass with the single reference donor {rank_audit['donor_bucket']}; target and donor "
                "marginal effects are not separately identified."
            ),
            "selection_interpretation": (
                "Low, middle, and high refer to deterministic rank quantiles of phase-1 materialized-exposure "
                "scale within each eligible boundary or paired-interior candidate pool. Eligibility and selected "
                "scales are frozen in rank_audit."
            ),
            "reverse_direction_selection": (
                "matched negatives for the three active-bucket targets selected at the low, middle, and high "
                "phase-1 exposure scales"
            ),
            "residual_degrees_interpretation": (
                "The three residual degrees of freedom arise from paired interior curvature checks, not a noise "
                "estimate; noise is measured by the repeated anchor and sentinel controls."
            ),
        },
        "common_random_numbers": {
            "fit_data_seed": FIT_DATA_SEED,
            "continuation_trainer_seed": 0,
            "anchor_repeat_data_seed": ANCHOR_REPEAT_DATA_SEED,
            "local_sentinel_repeat_data_seed": SENTINEL_REPEAT_DATA_SEED,
            "local_sentinel_action_id": sentinel_action["action_id"],
        },
        "primary_inference": {
            "target": "Uncheatable BPB",
            "confirmatory_state_ids": list(CONFIRMATORY_STATE_IDS),
            "state_count": len(CONFIRMATORY_STATE_IDS),
            "resampling_unit": "the five distinct confirmatory prefix-weight states",
            "comparison": (
                "For each confirmatory state s, delta_s = UncheatableBPB(s, local_anchor_fit079, data_seed=974000) "
                "- UncheatableBPB(s, control_tied, data_seed=974000). The primary estimate is mean_s(delta_s); "
                "negative is better."
            ),
            "uncertainty": (
                "Report a two-sided 95% t interval over the five state contrasts and the leave-one-state-out range. "
                "Claim pooled improvement only if the interval upper bound is below zero. Anchor and max-exposure "
                "sentinel repeats calibrate data-order noise; tied has no independent repeat, so the interval does "
                "not separately identify tied heteroscedasticity."
            ),
            "power_scope": (
                "Using prior contrast SD 0.0004-0.0006 BPB as a planning reference, five states give an approximate "
                "two-sided 80%-power MDE of 0.00065-0.00097 BPB. State heterogeneity can make the realized MDE larger. "
                "The panel is not powered for individual-prefix superiority or state-by-action interactions."
            ),
            "multiplicity": "Uncheatable anchor-versus-tied is the sole primary hypothesis; no primary correction.",
        },
        "secondary_inference": {
            "target": "Uncheatable BPB",
            "confirmatory_state_ids": list(CONFIRMATORY_STATE_IDS),
            "interior_odd_contrasts": [
                {
                    "ray_id": "interior_37",
                    "formula": "0.5 * (BPB(local_plus_37) - BPB(local_minus_37))",
                },
                {
                    "ray_id": "interior_32",
                    "formula": "0.5 * (BPB(local_plus_32) - BPB(local_minus_32))",
                },
                {
                    "ray_id": "interior_24",
                    "formula": "0.5 * (BPB(local_plus_24) - BPB(local_minus_24))",
                },
            ],
            "estimator": "mean across the five confirmatory states for each named odd contrast",
            "multiplicity": "Holm-adjust the three two-sided interior-odd tests at familywise alpha 0.05.",
            "scope": "Secondary mechanism screen only; no frontier or promotion claim.",
        },
        "descriptive_analysis": {
            "interior_even_contrasts": (
                "For each paired interior ray j and state s, 0.5 * (BPB(plus_j) + BPB(minus_j)) "
                "- BPB(local_anchor_fit079)."
            ),
            "boundary_contrasts": [
                "BPB(local_plus_30) - BPB(local_anchor_fit079)",
                "BPB(local_plus_10) - BPB(local_anchor_fit079)",
                "BPB(local_plus_20) - BPB(local_anchor_fit079)",
            ],
            "boundary_limitation": (
                "Boundary moves are Hellinger-matched to interior rays but differ by 10-16x in total variation and "
                "up to roughly 200x in materialized exposure. Null or cross-family differences are descriptive and "
                "must not be interpreted as a boundary mechanism test."
            ),
            "state_by_action_interactions": "descriptive; one run is available per action-state cell",
        },
        "state_analysis_contract": {
            "confirmatory_state_ids": list(CONFIRMATORY_STATE_IDS),
            "state_roles": {
                "observed_cap10_best": "outcome-selected incumbent; descriptive only",
                "cap4_shared_bounded_ensemble_kl0": (
                    "prefix state used to develop and validate the anchor action; descriptive positive control"
                ),
                "cap4_shared_bounded_ensemble_kl0__seed1": (
                    "trainer-seed diagnostic at the same prefix weight point; not an independent state"
                ),
                "shared_bounded_ensemble_kl0p05__v6e_bridge": (
                    "hardware-by-code diagnostic at the same prefix weight point; not an independent state"
                ),
            },
            "required_sensitivities": {
                "seven_unique_weight_points": list(UNIQUE_WEIGHT_SENSITIVITY_STATE_IDS),
                "within_10_epoch_confirmatory_subset": list(WITHIN_EXPOSURE_SENSITIVITY_STATE_IDS),
                "all_nine_states": "report descriptively without treating duplicate weight points as exchangeable",
            },
        },
        "outcome_use": {
            "anchor": (
                "outcome-selected and independently validated on cap4_shared_bounded_ensemble_kl0 before this panel"
            ),
            "local_directions": "constructed without endpoint labels",
            "prefix_states": (
                "five distinct, outcome-unselected weight points are confirmatory; four outcome-dependent or "
                "duplicate diagnostic states are assigned descriptive roles in state_analysis_contract"
            ),
        },
        "missing_cell_policy": (
            "Idempotently rerun the exact frozen row until complete. Do not replace a missing local direction or "
            "prefix state; incomplete data supports only explicitly labeled descriptive fits."
        ),
        "source_artifacts": {
            "frontier_contract_sha256": FRONTIER_CONTRACT_SHA256,
            "cap10_weights_sha256": crossed_design.CAP10_WEIGHTS_SHA256,
            "harsh_weights_sha256": crossed_design.HARSH_WEIGHTS_SHA256,
            "bridge_provenance_sha256": BRIDGE_PROVENANCE_SHA256,
            "bridge_code_commit": BRIDGE_CODE_COMMIT,
        },
    }
    return panel_rows, panel_weights, registry, manifest


def write_design(args: argparse.Namespace) -> dict[str, Any]:
    panel_rows, panel_weights, registry, manifest = build_design(args)
    rows_bytes = panel_rows.to_csv(index=False, lineterminator="\n").encode()
    weights_bytes = panel_weights.to_csv(index=False, lineterminator="\n").encode()
    registry_bytes = (json.dumps(registry, indent=2, sort_keys=True) + "\n").encode()
    crossed_design.write_exact(args.output_dir / "panel_rows.csv", rows_bytes)
    crossed_design.write_exact(args.output_dir / "panel_weights.csv", weights_bytes)
    crossed_design.write_exact(args.output_dir / "prefix_registry.json", registry_bytes)
    manifest["panel_rows_sha256"] = crossed_design.bytes_sha256(rows_bytes)
    manifest["panel_weights_sha256"] = crossed_design.bytes_sha256(weights_bytes)
    manifest["prefix_registry_sha256"] = crossed_design.bytes_sha256(registry_bytes)
    crossed_design.write_exact(
        args.output_dir / "manifest.json",
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
    )
    return manifest


def main() -> None:
    print(json.dumps(write_design(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
