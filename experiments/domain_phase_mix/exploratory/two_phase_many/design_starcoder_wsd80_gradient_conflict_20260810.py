# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

"""Materialize the unsubmitted WSD80 gradient-dynamics experiment design."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
SOURCE_DESIGN = REPO_ROOT / "experiments/domain_phase_mix/starcoder_wsd80_dense_support_surface_design_20260808.json"
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/starcoder_wsd80_gradient_conflict_design_20260810"

DESIGN_VERSION = "2026-08-10-review-v5"
PRIMARY_POOL_SEED = 2_026_081_101
REPLICATION_POOL_SEED = 2_026_081_102
P1_SEEDS = tuple(range(2_026_081_000, 2_026_081_024))
H4_CALIBRATION_SEEDS = tuple(range(2_026_083_000, 2_026_083_008))
P3_SEEDS = tuple(range(2_026_082_000, 2_026_082_012))
P2_SEEDS = P3_SEEDS[:8]
BOUNDARY_SEEDS = tuple(range(2_026_084_000, 2_026_084_016))

COMMON_TIED_WEIGHT = 0.35
BOUNDARY_AGGREGATE = 0.18
BOUNDARY_CONTRAST_P0_MINUS_P1 = -0.4
BOUNDARY_FRACTIONS = (0.60, 0.70, 0.80, 0.85, 0.90)
STEP_ALIGNMENT = 16
CANARY_EXACT_FORK_SEEDS = P1_SEEDS[:2]
CANARY_EXACT_FORK_SOURCE_STEP = 2_826
CANARY_EXACT_FORK_UPDATES = 16
CANARY_EXACT_FORK_REFERENCE_STEP = CANARY_EXACT_FORK_SOURCE_STEP + CANARY_EXACT_FORK_UPDATES
CANARY_EXACT_FORK_REFERENCE_LABEL = "canary_exact_fork_reference"


@dataclass(frozen=True)
class Trajectory:
    trajectory_id: str
    arm: str
    cell_id: str
    support_id: str
    support_pool_seed: int | None
    training_seed: int
    policy_role: str
    phase_0_fraction: float
    phase_1_fraction: float
    phase_0_starcoder: float | None
    phase_1_starcoder: float | None
    aggregate_starcoder: float | None
    phase_contrast_p0_minus_p1: float | None
    upstream_phase_contrast_p1_minus_p0: float | None
    coordinate_selection_rule: str
    total_steps: int
    boundary_step: int
    optimizer_decay_step: int
    primary_inference: bool


@dataclass(frozen=True)
class Checkpoint:
    trajectory_id: str
    checkpoint_label: str
    checkpoint_step: int
    total_steps: int
    forced_final: bool


CORE_DISTRIBUTIONS = (
    ("starcoder_on_policy", "training_source_on_policy"),
    ("starcoder_support_reference", "training_source_fixed_support_reference"),
    ("starcoder_excluded_global", "training_source_global_holdout"),
    ("nemotron_aggregate", "training_source_frozen_leaf_aggregate"),
    ("paloma_programming_languages", "primary_code_target"),
    ("uncheatable_github_python", "code_consistency_target"),
    ("uncheatable_wikipedia_english", "natural_language_robustness_target"),
    ("paloma_c4_en", "primary_temporal_reference"),
)
NEMOTRON_LEAVES = (
    "hq_actual",
    "hq_synth",
    "medium_high",
    "medium",
    "medium_low",
    "low_actual",
)

TEMPORAL_PRIMARY_LABELS = {
    "fraction_0p40",
    "fraction_0p55",
    "decay_minus_256",
    "decay_minus_64",
    "decay_plus_64",
    "decay_plus_256",
    "fraction_0p90",
}


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _cells() -> dict[str, dict[str, Any]]:
    payload = json.loads(SOURCE_DESIGN.read_text())
    cells = {row["cell_id"]: row for row in payload["cells"]}
    expected = {
        "r0_shared_h0640_s03820",
        "r1_increase_d_h0640_s07320",
        "r2_increase_d_h0640_s14960",
        "r3_increase_d_h0640_s28260",
    }
    if set(cells) != expected:
        raise ValueError(f"Fixed-N cells drifted: {sorted(cells)}")
    return cells


def _aligned_boundary(total_steps: int, fraction: float) -> int:
    return (int(total_steps * fraction) // STEP_ALIGNMENT) * STEP_ALIGNMENT


def _trajectory_id(arm: str, cell: dict[str, Any], support: str, policy: str, seed: int) -> str:
    slug = cell["cell_slug"]
    return f"gcf_{arm}_{slug}_{support}_{policy}_s{seed}"


def _append_fixed_policy(
    rows: list[Trajectory],
    *,
    arm: str,
    cell: dict[str, Any],
    support_id: str,
    support_pool_seed: int | None,
    seeds: tuple[int, ...],
    policy_role: str,
    p0: float | None,
    p1: float | None,
    selection_rule: str,
    primary_inference: bool = False,
) -> None:
    boundary = int(cell["boundary_step"])
    beta = boundary / int(cell["total_steps"])
    aggregate = None if p0 is None or p1 is None else beta * p0 + (1.0 - beta) * p1
    contrast = None if p0 is None or p1 is None else p0 - p1
    for seed in seeds:
        policy_slug = policy_role.replace("_", "-")
        rows.append(
            Trajectory(
                trajectory_id=_trajectory_id(arm, cell, support_id, policy_slug, seed),
                arm=arm,
                cell_id=cell["cell_id"],
                support_id=support_id,
                support_pool_seed=support_pool_seed,
                training_seed=seed,
                policy_role=policy_role,
                phase_0_fraction=beta,
                phase_1_fraction=1.0 - beta,
                phase_0_starcoder=p0,
                phase_1_starcoder=p1,
                aggregate_starcoder=aggregate,
                phase_contrast_p0_minus_p1=contrast,
                upstream_phase_contrast_p1_minus_p0=None if contrast is None else -contrast,
                coordinate_selection_rule=selection_rule,
                total_steps=int(cell["total_steps"]),
                boundary_step=boundary,
                optimizer_decay_step=boundary,
                primary_inference=primary_inference,
            )
        )


def build_trajectories(cells: dict[str, dict[str, Any]]) -> list[Trajectory]:
    rows: list[Trajectory] = []
    ordered_cells = sorted(cells.values(), key=lambda row: row["rung"])
    r3_id = "r3_increase_d_h0640_s28260"

    # P1: common tied trajectory. The r3 m100/full pair carries primary inference;
    # m100b is a second independently selected finite-support pool.
    for cell in ordered_cells:
        if cell["cell_id"] == r3_id:
            support_specs = (
                ("m100a", PRIMARY_POOL_SEED, P1_SEEDS),
                ("m100b", REPLICATION_POOL_SEED, P1_SEEDS[:8]),
                ("full", None, P1_SEEDS),
            )
        else:
            support_specs = (
                ("m100a", PRIMARY_POOL_SEED, P1_SEEDS[:8]),
                ("full", None, P1_SEEDS[:8]),
            )
        for support_id, pool_seed, seeds in support_specs:
            _append_fixed_policy(
                rows,
                arm="p1",
                cell=cell,
                support_id=support_id,
                support_pool_seed=pool_seed,
                seeds=seeds,
                policy_role="common_tied_035",
                p0=COMMON_TIED_WEIGHT,
                p1=COMMON_TIED_WEIGHT,
                selection_rule="fixed_coordinate",
                primary_inference=cell["cell_id"] == r3_id and support_id == "m100a",
            )
        if cell["cell_id"] == r3_id:
            _append_fixed_policy(
                rows,
                arm="p1",
                cell=cell,
                support_id="m100a",
                support_pool_seed=PRIMARY_POOL_SEED,
                seeds=H4_CALIBRATION_SEEDS,
                policy_role="common_tied_035_h4_calibration",
                p0=COMMON_TIED_WEIGHT,
                p1=COMMON_TIED_WEIGHT,
                selection_rule="fixed_coordinate_independent_h4_calibration",
            )

    # P3: fitted-surface argmins are intentionally unresolved. A launcher must
    # refuse these rows until a frozen surface materializer fills p0/p1.
    for cell in ordered_cells:
        seeds = P3_SEEDS if cell["cell_id"] == r3_id else P3_SEEDS[:4]
        for support_id, pool_seed in (("m100a", PRIMARY_POOL_SEED), ("full", None)):
            _append_fixed_policy(
                rows,
                arm="p3",
                cell=cell,
                support_id=support_id,
                support_pool_seed=pool_seed,
                seeds=seeds,
                policy_role="fitted_surface_best_two_phase",
                p0=None,
                p1=None,
                selection_rule=(
                    "frozen_degree4_spatial_cv_ridge_argmin_on_closed_empirical_hull_then_snap_to_nearest_design_coordinate"
                ),
            )

    # P2: best tied controls only where the fitted tied optimum is materially
    # separated from p=0.35. Coordinates remain unresolved for the same reason.
    for cell_id, seeds in (
        ("r1_increase_d_h0640_s07320", P2_SEEDS[:4]),
        (r3_id, P2_SEEDS),
    ):
        _append_fixed_policy(
            rows,
            arm="p2",
            cell=cells[cell_id],
            support_id="m100a",
            support_pool_seed=PRIMARY_POOL_SEED,
            seeds=seeds,
            policy_role="fitted_surface_best_tied",
            p0=None,
            p1=None,
            selection_rule="same_frozen_surface_restricted_to_diagonal_then_snap_to_nearest_design_coordinate",
        )

    # B: switch-time intervention at exact aggregate and contrast. The data
    # switch moves while the optimizer decay onset remains fixed at 0.8T.
    cell = cells[r3_id]
    total_steps = int(cell["total_steps"])
    for fraction in BOUNDARY_FRACTIONS:
        boundary = _aligned_boundary(total_steps, fraction)
        beta = boundary / total_steps
        p0 = BOUNDARY_AGGREGATE + (1.0 - beta) * BOUNDARY_CONTRAST_P0_MINUS_P1
        p1 = BOUNDARY_AGGREGATE - beta * BOUNDARY_CONTRAST_P0_MINUS_P1
        if not 0.0 <= p0 <= 1.0 or not 0.0 <= p1 <= 1.0:
            raise ValueError(f"Infeasible boundary arm beta={beta}: {(p0, p1)}")
        for seed in BOUNDARY_SEEDS:
            role = f"boundary_beta_{fraction:.2f}".replace(".", "p")
            rows.append(
                Trajectory(
                    trajectory_id=_trajectory_id("b", cell, "m100a", role, seed),
                    arm="b",
                    cell_id=cell["cell_id"],
                    support_id="m100a",
                    support_pool_seed=PRIMARY_POOL_SEED,
                    training_seed=seed,
                    policy_role=role,
                    phase_0_fraction=beta,
                    phase_1_fraction=1.0 - beta,
                    phase_0_starcoder=p0,
                    phase_1_starcoder=p1,
                    aggregate_starcoder=beta * p0 + (1.0 - beta) * p1,
                    phase_contrast_p0_minus_p1=p0 - p1,
                    upstream_phase_contrast_p1_minus_p0=p1 - p0,
                    coordinate_selection_rule="fixed_aggregate_fixed_contrast_realized_boundary",
                    total_steps=total_steps,
                    boundary_step=boundary,
                    optimizer_decay_step=int(cell["boundary_step"]),
                    primary_inference=False,
                )
            )
    _append_fixed_policy(
        rows,
        arm="b",
        cell=cell,
        support_id="m100a",
        support_pool_seed=PRIMARY_POOL_SEED,
        seeds=BOUNDARY_SEEDS,
        policy_role="boundary_tied_018",
        p0=BOUNDARY_AGGREGATE,
        p1=BOUNDARY_AGGREGATE,
        selection_rule="fixed_coordinate_contrast_zero_anchor",
    )

    if len(rows) != 268:
        raise ValueError(f"Trajectory count drifted: {len(rows)} != 268")
    if len({row.trajectory_id for row in rows}) != len(rows):
        raise ValueError("Trajectory IDs are not unique")
    return rows


def _normal_checkpoint_steps(row: Trajectory) -> list[tuple[str, int]]:
    total = row.total_steps
    boundary = row.boundary_step
    if row.arm == "p1":
        requested = [
            ("fraction_0p10", int(total * 0.10)),
            ("fraction_0p25", int(total * 0.25)),
            ("fraction_0p40", int(total * 0.40)),
            ("fraction_0p55", int(total * 0.55)),
            ("fraction_0p70", int(total * 0.70)),
            ("decay_minus_256", boundary - 256),
            ("decay_minus_64", boundary - 64),
            ("decay_onset", boundary),
            ("decay_plus_64", boundary + 64),
            ("decay_plus_256", boundary + 256),
            ("fraction_0p90", int(total * 0.90)),
            ("final", total - 1),
        ]
        if (
            row.cell_id == "r3_increase_d_h0640_s28260"
            and row.support_id == "m100a"
            and row.policy_role == "common_tied_035"
            and row.training_seed in CANARY_EXACT_FORK_SEEDS
        ):
            requested.append((CANARY_EXACT_FORK_REFERENCE_LABEL, CANARY_EXACT_FORK_REFERENCE_STEP))
    elif row.arm == "p3":
        requested = [
            ("fraction_0p25", int(total * 0.25)),
            ("fraction_0p55", int(total * 0.55)),
            ("fraction_0p70", int(total * 0.70)),
            ("decay_minus_64", boundary - 64),
            ("decay_plus_64", boundary + 64),
            ("decay_plus_256", boundary + 256),
            ("fraction_0p90", int(total * 0.90)),
            ("final", total - 1),
        ]
    elif row.arm == "p2":
        requested = [("final", total - 1)]
    else:
        optimizer_decay = row.optimizer_decay_step
        requested = [
            ("fraction_0p55", int(total * 0.55)),
            ("data_switch_minus_64", boundary - 64),
            ("data_switch", boundary),
            ("data_switch_plus_64", boundary + 64),
            ("optimizer_decay_minus_64", optimizer_decay - 64),
            ("optimizer_decay_onset", optimizer_decay),
            ("optimizer_decay_plus_64", optimizer_decay + 64),
            ("fraction_0p90", int(total * 0.90)),
            ("final", total - 1),
        ]
        if row.policy_role == "boundary_tied_018":
            requested.extend(
                [
                    ("fraction_0p40", int(total * 0.40)),
                    ("optimizer_decay_minus_256", optimizer_decay - 256),
                ]
            )
    labels_by_step: dict[int, list[str]] = {}
    for label, step in requested:
        labels_by_step.setdefault(step, []).append(label)
    return [("|".join(labels), step) for step, labels in sorted(labels_by_step.items())]


def build_checkpoints(trajectories: list[Trajectory]) -> list[Checkpoint]:
    rows: list[Checkpoint] = []
    for trajectory in trajectories:
        for label, step in _normal_checkpoint_steps(trajectory):
            rows.append(
                Checkpoint(
                    trajectory_id=trajectory.trajectory_id,
                    checkpoint_label=label,
                    checkpoint_step=step,
                    total_steps=trajectory.total_steps,
                    forced_final=step == trajectory.total_steps - 1,
                )
            )
    if len(rows) != 2_542:
        raise ValueError(f"Checkpoint count drifted: {len(rows)} != 2542")
    return rows


def build_checkpointer_manifest(trajectories: list[Trajectory], checkpoints: list[Checkpoint]) -> list[dict[str, Any]]:
    checkpoints_by_trajectory: dict[str, list[Checkpoint]] = {}
    for checkpoint in checkpoints:
        checkpoints_by_trajectory.setdefault(checkpoint.trajectory_id, []).append(checkpoint)

    rows: list[dict[str, Any]] = []
    for trajectory in trajectories:
        trajectory_checkpoints = checkpoints_by_trajectory[trajectory.trajectory_id]
        steps = [row.checkpoint_step for row in trajectory_checkpoints]
        keep = [{"every": step, "until": None if step == steps[-1] else step} for step in steps]
        realized = _enumerate_permanent_checkpoint_steps(trajectory.total_steps, keep)
        if realized != steps:
            raise ValueError(f"Checkpoint policy drifted for {trajectory.trajectory_id}: {realized} != {steps}")
        rows.append(
            {
                "trajectory_id": trajectory.trajectory_id,
                "save_interval": "15_minutes_temporary_only",
                "keep_last_temporary_checkpoints": 1,
                "keep_json": json.dumps(keep, separators=(",", ":")),
                "expected_checkpoint_steps": "|".join(str(step) for step in steps),
                "expected_checkpoint_count": len(steps),
            }
        )
    return rows


def _enumerate_permanent_checkpoint_steps(total_steps: int, keep: list[dict[str, int | None]]) -> list[int]:
    realized: list[int] = []
    # StepInfo labels the completed update, so T updates produce labels 0..T-1.
    for step in range(1, total_steps):
        policy = next(row for row in keep if row["until"] is None or int(row["until"]) >= step)
        if step % int(policy["every"]) == 0:
            realized.append(step)
    return realized


def build_probe_manifest(trajectories: list[Trajectory], checkpoints: list[Checkpoint]) -> list[dict[str, Any]]:
    trajectory_by_id = {row.trajectory_id: row for row in trajectories}
    rows: list[dict[str, Any]] = []
    for checkpoint in checkpoints:
        trajectory = trajectory_by_id[checkpoint.trajectory_id]
        labels = set(checkpoint.checkpoint_label.split("|"))
        if CANARY_EXACT_FORK_REFERENCE_LABEL in labels:
            continue
        if trajectory.arm == "p2":
            continue
        if trajectory.arm == "b" and trajectory.policy_role not in {
            "boundary_beta_0p60",
            "boundary_beta_0p85",
            "boundary_tied_018",
        }:
            continue
        primary_state = (
            trajectory.arm == "p1"
            and trajectory.cell_id == "r3_increase_d_h0640_s28260"
            and bool(labels & TEMPORAL_PRIMARY_LABELS)
        )
        distributions = list(CORE_DISTRIBUTIONS)
        if primary_state and trajectory.support_id == "m100a":
            distributions.extend((f"nemotron_{leaf}", "nemotron_leaf_geometry_calibration") for leaf in NEMOTRON_LEAVES)
        for distribution_id, role in distributions:
            is_leaf = role == "nemotron_leaf_geometry_calibration"
            if is_leaf:
                replicate_blocks = 16
            elif primary_state and trajectory.support_id == "m100a":
                replicate_blocks = 64
            elif primary_state and trajectory.support_id in {"m100b", "full"}:
                replicate_blocks = 32
            elif (
                trajectory.arm == "b"
                and trajectory.policy_role
                in {
                    "boundary_beta_0p60",
                    "boundary_beta_0p85",
                }
                and labels
                & {
                    "data_switch_minus_64",
                    "data_switch",
                    "data_switch_plus_64",
                    "optimizer_decay_minus_64",
                    "optimizer_decay_onset",
                    "optimizer_decay_plus_64",
                }
            ):
                replicate_blocks = 64
            elif (
                trajectory.arm == "b"
                and trajectory.policy_role == "boundary_tied_018"
                and labels
                & {
                    "fraction_0p40",
                    "fraction_0p55",
                    "optimizer_decay_minus_256",
                    "optimizer_decay_minus_64",
                    "optimizer_decay_onset",
                    "optimizer_decay_plus_64",
                }
            ):
                replicate_blocks = 64
            else:
                replicate_blocks = 16
            if role == "training_source_on_policy":
                probe_sequence_set_id = f"dynamic_exposure:{trajectory.trajectory_id}:{distribution_id}"
            elif role == "training_source_fixed_support_reference":
                probe_sequence_set_id = f"frozen:s{trajectory.training_seed}:{trajectory.support_id}:{distribution_id}"
            else:
                probe_sequence_set_id = f"frozen:s{trajectory.training_seed}:{distribution_id}"
            if trajectory.arm == "p1" and trajectory.primary_inference and primary_state:
                analysis_role = "h2_primary"
            elif trajectory.policy_role == "common_tied_035_h4_calibration" and primary_state:
                analysis_role = "h4_independent_calibration"
            elif trajectory.arm == "p1" and trajectory.support_id == "full" and primary_state:
                analysis_role = "h3_full_support_pair"
            elif trajectory.arm == "p1" and trajectory.support_id == "m100b" and primary_state:
                analysis_role = "h3_second_pool_sensitivity"
            elif (
                trajectory.arm == "b"
                and trajectory.policy_role == "boundary_tied_018"
                and labels
                & {
                    "fraction_0p40",
                    "fraction_0p55",
                    "optimizer_decay_minus_256",
                    "optimizer_decay_minus_64",
                }
            ):
                analysis_role = "h2_aggregate_matched"
            elif trajectory.arm == "b":
                analysis_role = "h5_event_localization"
            else:
                analysis_role = "descriptive_trajectory"
            rows.append(
                {
                    "trajectory_id": trajectory.trajectory_id,
                    "checkpoint_label": checkpoint.checkpoint_label,
                    "checkpoint_step": checkpoint.checkpoint_step,
                    "distribution_id": distribution_id,
                    "distribution_role": role,
                    "replicate_blocks": replicate_blocks,
                    "sequences_per_block": 64,
                    "tokens_per_sequence": 2_048,
                    "probe_sequence_set_id": probe_sequence_set_id,
                    "optimizer_superbatch_blocks": 2,
                    "optimizer_update_draw_count": replicate_blocks // 2,
                    "training_sequences_per_update": 128,
                    "two_pass_sufficient_statistics": True,
                    "optimizer_update_enabled": checkpoint.checkpoint_step < checkpoint.total_steps - 1,
                    "primary_state": primary_state,
                    "analysis_role": analysis_role,
                }
            )
    return rows


def build_rollout_manifest(trajectories: list[Trajectory]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    q_values = (0.0, 0.25, 0.35, 0.45, 0.55, 0.75, 1.0)

    def add(parent: Trajectory, checkpoint_label: str, orders: tuple[int, ...], role: str) -> None:
        for order_seed in orders:
            for q in q_values:
                policy = f"q_{q:.2f}".replace(".", "p")
                rows.append(
                    {
                        "rollout_id": f"roll_{parent.trajectory_id}_{checkpoint_label}_{policy}_o{order_seed}",
                        "parent_trajectory_id": parent.trajectory_id,
                        "parent_checkpoint_label": checkpoint_label,
                        "starcoder_weight": q,
                        "rollout_control": "data_update",
                        "source_support_id": parent.support_id,
                        "source_stream_rule": "continue_parent_support_with_frozen_per_source_order",
                        "predicted_update_transform": "exact_optimizer_on_weighted_training_batch_gradient",
                        "primary_q_range": 0.25 <= q <= 0.55,
                        "rollout_order_seed": order_seed,
                        "updates": 512,
                        "readout_steps": "128|256|512",
                        "permanent_checkpoints": 0,
                        "analysis_role": role,
                    }
                )

    r3 = "r3_increase_d_h0640_s28260"
    calibration_parents = [
        row
        for row in trajectories
        if row.arm == "p1"
        and row.cell_id == r3
        and row.support_id == "m100a"
        and row.training_seed in H4_CALIBRATION_SEEDS
    ]
    for row in calibration_parents:
        add(row, "decay_minus_64", (0,), "h4_independent_calibration")

    p1_m100 = [
        row
        for row in trajectories
        if row.arm == "p1" and row.cell_id == r3 and row.support_id == "m100a" and row.training_seed in P1_SEEDS[:16]
    ]
    for row in p1_m100:
        add(row, "decay_minus_64", (0,), "h4_primary_validation")
    for row in p1_m100[:4]:
        add(row, "decay_minus_64", (1,), "h4_order_variance_sensitivity")

    for support in ("m100a", "full"):
        parents = [
            row
            for row in trajectories
            if row.arm == "p1" and row.cell_id == r3 and row.support_id == support and row.training_seed in P1_SEEDS[:4]
        ]
        for parent in parents:
            for checkpoint_label in ("fraction_0p55", "fraction_0p90"):
                add(parent, checkpoint_label, (0,), "h4_time_support_sensitivity")

    for support in ("m100a", "full"):
        parents = [
            row
            for row in trajectories
            if row.arm == "p3" and row.cell_id == r3 and row.support_id == support and row.training_seed in P3_SEEDS[:4]
        ]
        for parent in parents:
            add(parent, "decay_plus_64", (0,), "h4_policy_transport_sensitivity")

    if len(rows) != 364:
        raise ValueError(f"Rollout count drifted: {len(rows)} != 364")
    return rows


def build_optimizer_transform_manifest(rollouts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_key: dict[tuple[str, str, float], dict[str, Any]] = {}
    for rollout in rollouts:
        key = (
            str(rollout["parent_trajectory_id"]),
            str(rollout["parent_checkpoint_label"]),
            float(rollout["starcoder_weight"]),
        )
        rows_by_key[key] = {
            "parent_trajectory_id": key[0],
            "parent_checkpoint_label": key[1],
            "starcoder_weight": key[2],
            "nemotron_weight": 1.0 - key[2],
            "source_gradient_blocks_per_draw": 2,
            "sequences_per_training_scale_draw": 128,
            "transform": "exact_optimizer_on_weighted_training_batch_gradient",
            "include_one_step_no_data_update": True,
        }
    return [rows_by_key[key] for key in sorted(rows_by_key)]


def build_probe_preflight_manifest(trajectories: list[Trajectory], probes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    eligible_trajectories = {
        row.trajectory_id
        for row in trajectories
        if row.arm == "p1"
        and row.cell_id == "r3_increase_d_h0640_s28260"
        and row.support_id == "m100a"
        and row.training_seed in P1_SEEDS[:2]
    }
    return [
        row
        for row in probes
        if row["trajectory_id"] in eligible_trajectories
        and row["primary_state"]
        and row["distribution_role"] != "nemotron_leaf_geometry_calibration"
    ]


def main() -> None:
    cells = _cells()
    trajectories = build_trajectories(cells)
    checkpoints = build_checkpoints(trajectories)
    checkpointer_rows = build_checkpointer_manifest(trajectories, checkpoints)
    probes = build_probe_manifest(trajectories, checkpoints)
    rollouts = build_rollout_manifest(trajectories)
    optimizer_transforms = build_optimizer_transform_manifest(rollouts)
    probe_preflight = build_probe_preflight_manifest(trajectories, probes)

    if len(probes) != 18_496:
        raise ValueError(f"Gradient probe count drifted: {len(probes)} != 18496")
    if len(probe_preflight) != 112:
        raise ValueError(f"Probe preflight count drifted: {len(probe_preflight)} != 112")
    if len(optimizer_transforms) != 336:
        raise ValueError(f"Optimizer transform count drifted: {len(optimizer_transforms)} != 336")
    if any(row["trajectory_id"].startswith("gcf_p2_") for row in probes):
        raise ValueError("P2 trajectories must not receive gradient probes")
    if any(row["rollout_control"] != "data_update" for row in rollouts):
        raise ValueError("Long no-data rollouts are forbidden")
    if sum(row.primary_inference for row in trajectories) != len(P1_SEEDS):
        raise ValueError("Primary inference must contain exactly the 24 r3/m100a P1 seeds")
    for row in trajectories:
        if row.phase_contrast_p0_minus_p1 is None:
            continue
        if abs(row.phase_contrast_p0_minus_p1 + float(row.upstream_phase_contrast_p1_minus_p0)) > 1e-12:
            raise ValueError(f"Contrast conventions disagree for {row.trajectory_id}")

    seed_sets = {
        "p1": set(P1_SEEDS),
        "h4_calibration": set(H4_CALIBRATION_SEEDS),
        "p3": set(P3_SEEDS),
        "boundary": set(BOUNDARY_SEEDS),
    }
    for left_name, left in seed_sets.items():
        for right_name, right in seed_sets.items():
            if left_name < right_name and left & right:
                raise ValueError(f"Inference seed namespaces overlap: {left_name}, {right_name}")
    if set(P2_SEEDS) != set(P3_SEEDS[:8]):
        raise ValueError("P2 seeds must pair with the first eight P3 seeds")
    if PRIMARY_POOL_SEED == REPLICATION_POOL_SEED:
        raise ValueError("Primary and replication support pools must use distinct seeds")

    trajectory_rows = [asdict(row) for row in trajectories]
    checkpoint_rows = [asdict(row) for row in checkpoints]
    unresolved = sum(row.phase_0_starcoder is None or row.phase_1_starcoder is None for row in trajectories)
    if unresolved != 60 or any(
        row.arm not in {"p2", "p3"} and (row.phase_0_starcoder is None or row.phase_1_starcoder is None)
        for row in trajectories
    ):
        raise ValueError("Only the 60 frozen-surface P2/P3 trajectories may remain unresolved")
    materialized_tokens_by_cell = {cell_id: int(cell["materialized_tokens"]) for cell_id, cell in cells.items()}
    training_materialized_tokens = sum(materialized_tokens_by_cell[row.cell_id] for row in trajectories)
    gradient_probe_tokens = sum(
        int(row["replicate_blocks"]) * int(row["sequences_per_block"]) * int(row["tokens_per_sequence"])
        for row in probes
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    artifact_rows = {
        "trajectory_manifest.csv": trajectory_rows,
        "checkpoint_manifest.csv": checkpoint_rows,
        "checkpointer_manifest.csv": checkpointer_rows,
        "gradient_probe_manifest.csv": probes,
        "probe_preflight_manifest.csv": probe_preflight,
        "rollout_manifest.csv": rollouts,
        "optimizer_transform_manifest.csv": optimizer_transforms,
    }
    for filename, rows in artifact_rows.items():
        _write_csv(OUTPUT_DIR / filename, rows)

    artifact_sha256 = {filename: _file_sha256(OUTPUT_DIR / filename) for filename in sorted(artifact_rows)}
    for filename in ("design.md", "review_checklist.md"):
        artifact_sha256[filename] = _file_sha256(OUTPUT_DIR / filename)
    artifact_sha256[str(SOURCE_DESIGN.relative_to(REPO_ROOT))] = _file_sha256(SOURCE_DESIGN)
    artifact_sha256[str(Path(__file__).resolve().relative_to(REPO_ROOT))] = _file_sha256(Path(__file__).resolve())
    payload = {
        "design_version": DESIGN_VERSION,
        "description": "Unsubmitted review design for WSD80 source-gradient, optimizer-update, and rollout probes.",
        "source_design": str(SOURCE_DESIGN.relative_to(REPO_ROOT)),
        "trajectory_count": len(trajectories),
        "checkpoint_count": len(checkpoints),
        "gradient_probe_row_count": len(probes),
        "probe_preflight_row_count": len(probe_preflight),
        "rollout_count": len(rollouts),
        "optimizer_transform_count": len(optimizer_transforms),
        "unresolved_trajectory_count": unresolved,
        "training_materialized_token_count": training_materialized_tokens,
        "gradient_probe_token_count": gradient_probe_tokens,
        "required_region": "us-central1",
        "required_zone": "us-central1-a",
        "required_bucket_prefix": "gs://marin-us-central1",
        "launch_allowed": False,
        "launch_blockers": [
            "obtain explicit user approval of the scientific design",
            "materialize all fitted-surface P2/P3 coordinates from frozen dense-support surfaces",
            "implement independent support-pool selection and exact on-policy/support-reference/global-holdout ledgers",
            "prove that target and reference examples do not overlap training support",
            "implement and test independent data-switch and optimizer-decay schedules",
            "implement exact full-state probe and rollout runner without resetting optimizer count or parent step",
            "freeze the statistical analysis plan and exact H5 power calculation",
            "complete independent launcher-code and regional launch reviews before any run",
            "pass the us-central1 launch-safety validator on the exact future command before any run",
            "complete the early-checkpoint serialization and exact-fork canary before the remaining training fanout",
            "run the two-seed measurement-reliability preflight before full probe and rollout fanout",
            "freeze numeric reliability thresholds and H2 decision rules",
            "run the disjoint H4 calibration cohort and freeze the utility-to-loss mapping before H2 unblinding",
            (
                "measure one full-state checkpoint and provision temporary and permanent retention before full training "
                "fanout"
            ),
            (
                "complete independent mechanistic, statistical, probe-code, and rollout-code reviews before scientific "
                "fanout"
            ),
        ],
        "primary_hypothesis": (
            "On the common tied r3 m100a trajectory, normalized MuonH-trunk target-choice alignment becomes "
            "more code-specific from mid-training to immediately before WSD decay, with full-update utility "
            "and exact short rollouts required for material relevance."
        ),
        "artifact_sha256": artifact_sha256,
        "design_sha256": "",
    }
    payload["design_sha256"] = _canonical_hash({**payload, "design_sha256": ""})
    (OUTPUT_DIR / "design_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
