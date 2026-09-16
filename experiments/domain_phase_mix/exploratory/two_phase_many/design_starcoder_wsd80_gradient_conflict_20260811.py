# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["jax==0.11.0", "numpy==2.3.5"]
# ///

"""Freeze the review-v9 WSD80 gradient-conflict training and probe design."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from experiments.domain_phase_mix import launch_starcoder_wsd_80_20_surface as base
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    design_starcoder_wsd80_dense_support_surfaces_20260808 as dense_design,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    design_starcoder_wsd80_gradient_conflict_20260810 as v5,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
V8_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/starcoder_wsd80_gradient_conflict_design_20260811"
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/starcoder_wsd80_gradient_conflict_design_20260811_v9"
SOURCE_DESIGN = REPO_ROOT / "experiments/domain_phase_mix/starcoder_wsd80_dense_support_surface_design_20260808.json"
CONFIRMATION_MANIFEST = (
    SCRIPT_DIR
    / "reference_outputs/starcoder_wsd80_matched_nd_stage1_20260731/confirmation_design_20260801/design_manifest.json"
)
CONFIRMATION_RESULTS = (
    SCRIPT_DIR
    / "reference_outputs/starcoder_wsd80_matched_nd_stage1_20260731/confirmation_results_20260801"
    / "cell_confirmation_summary.csv"
)
V5_MANIFEST = SCRIPT_DIR / "reference_outputs/starcoder_wsd80_gradient_conflict_design_20260810/design_manifest.json"
V8_SUPPORT_AUDIT = V8_OUTPUT_DIR / "support_partition_audit.json"
V8_SUPPORT_POSITION_HISTOGRAM = V8_OUTPUT_DIR / "support_position_histogram.csv"
SUPPORT_AUDIT = OUTPUT_DIR / V8_SUPPORT_AUDIT.name
SUPPORT_POSITION_HISTOGRAM = OUTPUT_DIR / V8_SUPPORT_POSITION_HISTOGRAM.name
SUPPORT_AUDIT_SOURCE = SCRIPT_DIR / "audit_starcoder_wsd80_gradient_conflict_support_20260811.py"

DESIGN_VERSION = "2026-08-11-review-v9"
PRIMARY_POOL_SEED = v5.PRIMARY_POOL_SEED
FINITE_SUPPORT_BATCHES = 1_068
SECOND_SUPPORT_START_BATCH = FINITE_SUPPORT_BATCHES
TRAIN_HOLDOUT_SEQUENCES_PER_COMPONENT = 4_096
TRAIN_HOLDOUT_SEED = 2_026_081_102
TRAIN_HOLDOUT_PARTITION = "random_sparse_swap"
P1_SEEDS = v5.P1_SEEDS
H4_CALIBRATION_SEEDS = v5.H4_CALIBRATION_SEEDS
POLICY_PAIR_SEEDS = v5.P2_SEEDS
BOUNDARY_SEEDS = v5.BOUNDARY_SEEDS

CONFIRMED_TWO_PHASE = (0.02, 0.82)
CONFIRMED_TIED = (0.70, 0.70)
CONFIRMED_MEAN_GAIN_BPB = 0.006101168692111969
CONFIRMED_GAIN_CI95 = (0.005269144432774185, 0.006933192951449753)

# H5 uses 1,760 complete mixture blocks. Its realized StarCoder count is
# exactly 368 per block in aggregate, with a fixed -820 count contrast.
H5_TOTAL_STEPS = 28_160
H5_TOTAL_BLOCKS = H5_TOTAL_STEPS * base.BATCH_SIZE // base.MIXTURE_BLOCK_SIZE
H5_DECAY_STEP = int(H5_TOTAL_STEPS * 0.8)
H5_CELL_ID = "h5_fixed_aggregate_h0640_s28160"
H5_CELL_SLUG = "h5d28160"
H5_SWITCH_FRACTIONS = (0.60, 0.70, 0.80, 0.85, 0.90)
H5_AGGREGATE_COUNT_PER_BLOCK = 368
H5_CONTRAST_COUNT_P0_MINUS_P1 = -820
NEAREST_AGGREGATE_TIED_COUNT_PER_BLOCK = 368

TEMPORAL_POLICY_LABELS = {
    "fraction_0p40",
    "fraction_0p55",
    "decay_minus_256",
    "decay_minus_64",
    "decay_plus_64",
    "decay_plus_256",
    "fraction_0p90",
}


def _confirmation_evidence() -> dict[str, Any]:
    manifest = json.loads(CONFIRMATION_MANIFEST.read_text())
    coordinates_by_role: dict[str, set[tuple[float, float]]] = {}
    for row in manifest["runs"]:
        coordinates_by_role.setdefault(row["role"], set()).add(
            (float(row["phase_0_starcoder"]), float(row["phase_1_starcoder"]))
        )
    expected_coordinates = {
        "untied_candidate": {CONFIRMED_TWO_PHASE},
        "tied_comparator": {CONFIRMED_TIED},
    }
    if coordinates_by_role != expected_coordinates:
        raise ValueError(f"Confirmed policy coordinates drifted: {coordinates_by_role} != {expected_coordinates}")

    with CONFIRMATION_RESULTS.open(newline="") as handle:
        summaries = list(csv.DictReader(handle))
    if len(summaries) != 1:
        raise ValueError(f"Expected one confirmed-cell summary, got {len(summaries)}")
    summary = summaries[0]
    observed_statistics = (
        float(summary["mean_gain_bpb"]),
        float(summary["ci95_low"]),
        float(summary["ci95_high"]),
    )
    expected_statistics = (CONFIRMED_MEAN_GAIN_BPB, *CONFIRMED_GAIN_CI95)
    if observed_statistics != expected_statistics or summary["confirmed"] != "True":
        raise ValueError(f"Confirmed policy statistics drifted: {observed_statistics} != {expected_statistics}")

    return {
        "cell_id": summary["cell_id"],
        "pair_count": int(summary["pair_count"]),
        "two_phase": {"phase_0_starcoder": CONFIRMED_TWO_PHASE[0], "phase_1_starcoder": CONFIRMED_TWO_PHASE[1]},
        "tied": {"phase_0_starcoder": CONFIRMED_TIED[0], "phase_1_starcoder": CONFIRMED_TIED[1]},
        "mean_gain_bpb": CONFIRMED_MEAN_GAIN_BPB,
        "gain_ci95": list(CONFIRMED_GAIN_CI95),
    }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _phase_component_weights(starcoder_weight: float) -> tuple[float, ...]:
    total_nemotron_tokens = sum(base.NEMOTRON_TOKEN_COUNTS.values())
    broad_weight = 1.0 - starcoder_weight
    broad = tuple(
        broad_weight * token_count / total_nemotron_tokens for token_count in base.NEMOTRON_TOKEN_COUNTS.values()
    )
    return (*broad, starcoder_weight)


def _realized_starcoder_count_per_block(starcoder_weight: float) -> int:
    weights = np.asarray(_phase_component_weights(starcoder_weight), dtype=np.float64)
    weights /= weights.sum()
    counts = np.asarray(weights * base.MIXTURE_BLOCK_SIZE, dtype=np.int32)
    counts[int(np.argmax(counts))] += base.MIXTURE_BLOCK_SIZE - int(counts.sum())
    return int(counts[-1])


def _nominal_weight_for_realized_count(target_count: int) -> float:
    if not 0 < target_count < base.MIXTURE_BLOCK_SIZE:
        raise ValueError(f"Realized StarCoder count must be interior: {target_count}")

    def lower_bound(count: int) -> float:
        low, high = 0.0, 1.0
        for _ in range(80):
            midpoint = (low + high) / 2.0
            if _realized_starcoder_count_per_block(midpoint) >= count:
                high = midpoint
            else:
                low = midpoint
        return high

    lower = lower_bound(target_count)
    upper = lower_bound(target_count + 1)
    nominal = (lower + upper) / 2.0
    observed = _realized_starcoder_count_per_block(nominal)
    if observed != target_count:
        raise ValueError(f"Could not realize StarCoder count {target_count}; got {observed}")
    return nominal


def _append_policy_pair(rows: list[v5.Trajectory], cells: dict[str, dict[str, Any]]) -> None:
    cell = cells["r3_increase_d_h0640_s28260"]
    for arm, role, coordinate in (
        ("p2", "confirmed_tied_comparator", CONFIRMED_TIED),
        ("p3", "confirmed_two_phase_winner", CONFIRMED_TWO_PHASE),
    ):
        for support_id, pool_seed in (("m100a", PRIMARY_POOL_SEED), ("full", None)):
            v5._append_fixed_policy(
                rows,
                arm=arm,
                cell=cell,
                support_id=support_id,
                support_pool_seed=pool_seed,
                seeds=POLICY_PAIR_SEEDS,
                policy_role=role,
                p0=coordinate[0],
                p1=coordinate[1],
                selection_rule="fresh_seed_confirmed_discrete_pair_20260801",
            )


def _append_aggregate_matched_tied(rows: list[v5.Trajectory], cells: dict[str, dict[str, Any]]) -> None:
    """Add the nearest tied control to the confirmed policy's realized aggregate."""
    cell = cells["r3_increase_d_h0640_s28260"]
    tied = _nominal_weight_for_realized_count(NEAREST_AGGREGATE_TIED_COUNT_PER_BLOCK)
    for support_id, pool_seed in (("m100a", PRIMARY_POOL_SEED), ("full", None)):
        v5._append_fixed_policy(
            rows,
            arm="p4",
            cell=cell,
            support_id=support_id,
            support_pool_seed=pool_seed,
            seeds=POLICY_PAIR_SEEDS,
            policy_role="aggregate_nearest_tied_018",
            p0=tied,
            p1=tied,
            selection_rule="nearest_integer_block_to_confirmed_two_phase_aggregate",
        )


def _append_h5_cell(cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    source = cells["r3_increase_d_h0640_s28260"]
    h5_cell = {
        **source,
        "cell_id": H5_CELL_ID,
        "cell_slug": H5_CELL_SLUG,
        "materialized_tokens": H5_TOTAL_STEPS * base.BATCH_SIZE * base.SEQ_LEN,
        "total_steps": H5_TOTAL_STEPS,
        "boundary_step": H5_DECAY_STEP,
    }
    cells[H5_CELL_ID] = h5_cell
    return h5_cell


def _append_exact_h5(rows: list[v5.Trajectory], cells: dict[str, dict[str, Any]]) -> None:
    cell = cells[H5_CELL_ID]
    if H5_TOTAL_BLOCKS != 1_760 or H5_DECAY_STEP % v5.STEP_ALIGNMENT != 0:
        raise ValueError("H5 horizon must contain 1,760 complete blocks and an aligned 80% decay onset")

    for fraction in H5_SWITCH_FRACTIONS:
        boundary = int(H5_TOTAL_STEPS * fraction)
        if boundary % v5.STEP_ALIGNMENT != 0:
            raise ValueError(f"H5 boundary is not mixture-block aligned: {fraction}")
        beta = boundary / H5_TOTAL_STEPS
        p0_count = H5_AGGREGATE_COUNT_PER_BLOCK + round((1.0 - beta) * H5_CONTRAST_COUNT_P0_MINUS_P1)
        p1_count = p0_count - H5_CONTRAST_COUNT_P0_MINUS_P1
        p0 = _nominal_weight_for_realized_count(p0_count)
        p1 = _nominal_weight_for_realized_count(p1_count)
        phase_0_blocks = boundary * base.BATCH_SIZE // base.MIXTURE_BLOCK_SIZE
        phase_1_blocks = H5_TOTAL_BLOCKS - phase_0_blocks
        total_count = phase_0_blocks * p0_count + phase_1_blocks * p1_count
        if total_count != H5_TOTAL_BLOCKS * H5_AGGREGATE_COUNT_PER_BLOCK:
            raise ValueError(f"H5 aggregate count drifted at beta={fraction}: {total_count}")
        if p0_count - p1_count != H5_CONTRAST_COUNT_P0_MINUS_P1:
            raise ValueError(f"H5 contrast count drifted at beta={fraction}")
        role = f"boundary_beta_{fraction:.2f}".replace(".", "p")
        for seed in BOUNDARY_SEEDS:
            rows.append(
                v5.Trajectory(
                    trajectory_id=v5._trajectory_id("b", cell, "m100a", role, seed),
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
                    aggregate_starcoder=H5_AGGREGATE_COUNT_PER_BLOCK / base.MIXTURE_BLOCK_SIZE,
                    phase_contrast_p0_minus_p1=H5_CONTRAST_COUNT_P0_MINUS_P1 / base.MIXTURE_BLOCK_SIZE,
                    upstream_phase_contrast_p1_minus_p0=-H5_CONTRAST_COUNT_P0_MINUS_P1 / base.MIXTURE_BLOCK_SIZE,
                    coordinate_selection_rule="integer_block_exact_aggregate_and_contrast",
                    total_steps=H5_TOTAL_STEPS,
                    boundary_step=boundary,
                    optimizer_decay_step=H5_DECAY_STEP,
                    primary_inference=False,
                )
            )

    tied = _nominal_weight_for_realized_count(H5_AGGREGATE_COUNT_PER_BLOCK)
    v5._append_fixed_policy(
        rows,
        arm="b",
        cell=cell,
        support_id="m100a",
        support_pool_seed=PRIMARY_POOL_SEED,
        seeds=BOUNDARY_SEEDS,
        policy_role="boundary_tied_018",
        p0=tied,
        p1=tied,
        selection_rule="integer_block_exact_tied_anchor",
    )


def build_trajectories(cells: dict[str, dict[str, Any]]) -> list[v5.Trajectory]:
    rows = [row for row in v5.build_trajectories(cells) if row.arm == "p1"]
    rows = [replace(row, support_pool_seed=PRIMARY_POOL_SEED) if row.support_id == "m100b" else row for row in rows]
    _append_h5_cell(cells)
    _append_policy_pair(rows, cells)
    _append_aggregate_matched_tied(rows, cells)
    _append_exact_h5(rows, cells)
    if len(rows) != 256 or len({row.trajectory_id for row in rows}) != len(rows):
        raise ValueError(f"Review-v9 trajectory identities drifted: {len(rows)}")
    return rows


def _checkpoint_steps(row: v5.Trajectory) -> list[tuple[str, int]]:
    if row.arm in {"p1", "p2", "p3", "p4"}:
        total = row.total_steps
        boundary = row.optimizer_decay_step
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
    else:
        requested = v5._normal_checkpoint_steps(row)
        if row.policy_role in {"boundary_beta_0p60", "boundary_beta_0p85"}:
            requested.extend(
                [
                    ("fraction_0p40", int(row.total_steps * 0.40)),
                    ("optimizer_decay_minus_256", row.optimizer_decay_step - 256),
                ]
            )
        labels_by_step: dict[int, list[str]] = {}
        for label, step in requested:
            labels_by_step.setdefault(step, []).extend(label.split("|"))
        return [("|".join(labels), step) for step, labels in sorted(labels_by_step.items())]
    labels_by_step: dict[int, list[str]] = {}
    for label, step in requested:
        labels_by_step.setdefault(step, []).append(label)
    return [("|".join(labels), step) for step, labels in sorted(labels_by_step.items())]


def build_checkpoints(trajectories: list[v5.Trajectory]) -> list[v5.Checkpoint]:
    return [
        v5.Checkpoint(
            trajectory_id=trajectory.trajectory_id,
            checkpoint_label=label,
            checkpoint_step=step,
            total_steps=trajectory.total_steps,
            forced_final=step == trajectory.total_steps - 1,
        )
        for trajectory in trajectories
        for label, step in _checkpoint_steps(trajectory)
    ]


def _policy_pair_probe_rows(
    trajectories: list[v5.Trajectory],
    checkpoints: list[v5.Checkpoint],
) -> list[dict[str, Any]]:
    probes = [
        row
        for row in v5.build_probe_manifest(trajectories, checkpoints)
        if not row["trajectory_id"].startswith("gcf_p4_")
    ]
    trajectory_by_id = {row.trajectory_id: row for row in trajectories}
    p2_by_key = {(row.support_id, row.training_seed): row for row in trajectories if row.arm == "p2"}
    p4_by_key = {(row.support_id, row.training_seed): row for row in trajectories if row.arm == "p4"}
    additions: list[dict[str, Any]] = []
    for row in probes:
        trajectory = trajectory_by_id[row["trajectory_id"]]
        labels = set(str(row["checkpoint_label"]).split("|"))
        if trajectory.arm == "b" and trajectory.policy_role in {"boundary_beta_0p60", "boundary_beta_0p85"}:
            h5_profile_states = {
                "fraction_0p40",
                "fraction_0p55",
                "data_switch_minus_64",
                "data_switch",
                "data_switch_plus_64",
                "optimizer_decay_minus_256",
                "optimizer_decay_minus_64",
                "optimizer_decay_onset",
                "optimizer_decay_plus_64",
            }
            if labels & h5_profile_states:
                row["replicate_blocks"] = 64
                row["optimizer_update_draw_count"] = 32
                row["analysis_role"] = "h5_preregistered_profile"
        if trajectory.arm != "p3":
            continue
        primary_state = bool(labels & TEMPORAL_POLICY_LABELS)
        if primary_state:
            row["primary_state"] = True
            row["analysis_role"] = "confirmed_policy_pair_mechanism"
            row["replicate_blocks"] = 64 if trajectory.support_id == "m100a" else 32
            row["optimizer_update_draw_count"] = int(row["replicate_blocks"]) // 2
        else:
            continue
        for paired, analysis_role in (
            (p2_by_key[(trajectory.support_id, trajectory.training_seed)], "confirmed_policy_pair_joint_contrast"),
            (p4_by_key[(trajectory.support_id, trajectory.training_seed)], "aggregate_nearest_policy_pair_mechanism"),
        ):
            clone = dict(row)
            clone["trajectory_id"] = paired.trajectory_id
            clone["analysis_role"] = analysis_role
            if str(clone["probe_sequence_set_id"]).startswith("dynamic_exposure:"):
                clone["probe_sequence_set_id"] = str(clone["probe_sequence_set_id"]).replace(
                    trajectory.trajectory_id, paired.trajectory_id
                )
            additions.append(clone)
    probes.extend(additions)
    return probes


def _rollout_rows(trajectories: list[v5.Trajectory]) -> list[dict[str, Any]]:
    rollouts = v5.build_rollout_manifest(trajectories)
    trajectory_by_id = {row.trajectory_id: row for row in trajectories}
    p2_by_key = {(row.support_id, row.training_seed): row for row in trajectories if row.arm == "p2"}
    p4_by_key = {(row.support_id, row.training_seed): row for row in trajectories if row.arm == "p4"}
    additions: list[dict[str, Any]] = []
    for row in rollouts:
        parent = trajectory_by_id[row["parent_trajectory_id"]]
        if parent.arm != "p3":
            continue
        for paired, analysis_role in (
            (p2_by_key[(parent.support_id, parent.training_seed)], "h4_confirmed_policy_pair_joint_transport"),
            (p4_by_key[(parent.support_id, parent.training_seed)], "h4_aggregate_nearest_policy_transport"),
        ):
            clone = dict(row)
            clone["rollout_id"] = str(clone["rollout_id"]).replace(parent.trajectory_id, paired.trajectory_id)
            clone["parent_trajectory_id"] = paired.trajectory_id
            clone["analysis_role"] = analysis_role
            additions.append(clone)
    rollouts.extend(additions)
    return rollouts


def _exact_trajectory_rows(
    trajectories: list[v5.Trajectory],
    cells: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trajectory in trajectories:
        if trajectory.phase_0_starcoder is None or trajectory.phase_1_starcoder is None:
            raise ValueError(f"Review-v8 trajectory is unresolved: {trajectory.trajectory_id}")
        cell = {
            **cells[trajectory.cell_id],
            "total_steps": trajectory.total_steps,
            "boundary_step": trajectory.boundary_step,
        }
        phase_0_sequences, phase_1_sequences = dense_design._realized_starcoder_sequences(
            cell=cell,
            phase_0=trajectory.phase_0_starcoder,
            phase_1=trajectory.phase_1_starcoder,
            data_seed=trajectory.training_seed,
        )
        total_sequences = trajectory.total_steps * base.BATCH_SIZE
        support_start = SECOND_SUPPORT_START_BATCH if trajectory.support_id == "m100b" else 0
        support_batches = FINITE_SUPPORT_BATCHES if trajectory.support_id in {"m100a", "m100b"} else None
        row = {
            **asdict(trajectory),
            "support_start_batches": support_start if support_batches is not None else None,
            "support_batches": support_batches,
            "train_holdout_sequences_per_component": TRAIN_HOLDOUT_SEQUENCES_PER_COMPONENT,
            "train_holdout_seed": TRAIN_HOLDOUT_SEED,
            "train_holdout_partition": TRAIN_HOLDOUT_PARTITION,
            "starcoder_phase_0_sequences": phase_0_sequences,
            "starcoder_phase_1_sequences": phase_1_sequences,
            "starcoder_total_sequences": phase_0_sequences + phase_1_sequences,
            "realized_aggregate_starcoder": (phase_0_sequences + phase_1_sequences) / total_sequences,
            "realized_phase_0_starcoder_per_block": _realized_starcoder_count_per_block(trajectory.phase_0_starcoder),
            "realized_phase_1_starcoder_per_block": _realized_starcoder_count_per_block(trajectory.phase_1_starcoder),
        }
        rows.append(row)
    return rows


def _design_markdown(payload: dict[str, Any]) -> str:
    return f"""# StarCoder WSD80 Gradient-Conflict Experiment

Status: review-v9 candidate; all training, probe, and rollout fanout is fail-closed

Design version: `{DESIGN_VERSION}`

## Purpose

This experiment tests whether the StarCoder WSD80 two-phase gain is explained by a state-dependent change in the
training update that best improves Programming Languages BPB. It separates source-gradient disagreement, target
revaluation, finite-support repetition, local behavioral prediction, and an explicit switch-time intervention. No
single cosine or endpoint contrast is allowed to stand in for the complete mechanism.

The frozen matrix has {payload['trajectory_count']} trajectories, {payload['checkpoint_count']} permanent full-state
checkpoints, {payload['gradient_probe_row_count']} distribution-by-checkpoint probe rows,
{payload['rollout_count']} exact-state short rollouts, and {payload['optimizer_transform_count']} unique optimizer
counterfactuals. Probe and rollout execution remain blocked until the {payload['probe_preflight_row_count']}-row
numerical preflight passes without inspecting hypothesis signs.

## Prior evidence and selection provenance

At the 7.408B-token r3 cell, the selected two-phase policy `(p0,p1)={CONFIRMED_TWO_PHASE}` beat the selected tied
policy `{CONFIRMED_TIED}` by {CONFIRMED_MEAN_GAIN_BPB:.6f} Programming Languages BPB over eight fresh paired seeds;
the 95% interval was [{CONFIRMED_GAIN_CI95[0]:.6f}, {CONFIRMED_GAIN_CI95[1]:.6f}]. The coordinate was selected from
the dense discovery surface. The 0.005-BPB confirmation threshold was specified after Stage 1 discovery, not before
the discovery panel, so this is fixed input evidence for this experiment rather than untouched confirmation.

The `{CONFIRMED_TWO_PHASE}` versus `{CONFIRMED_TIED}` endpoint comparison jointly changes aggregate StarCoder dose,
finite-support repetition, and phase schedule. It cannot by itself identify phase ordering. Review v9 therefore adds
a paired tied policy with the nearest realizable integer-block aggregate,
{NEAREST_AGGREGATE_TIED_COUNT_PER_BLOCK}/{base.MIXTURE_BLOCK_SIZE} StarCoder sequences per block, at the same
28,260-update horizon. The mean paired realized-aggregate mismatch is
{payload['nearest_tied_aggregate_match']['mean_absolute_difference']:.9f} and the maximum is
{payload['nearest_tied_aggregate_match']['maximum_absolute_difference']:.9f}, both below one sequence per
2,048-sequence block.

## Training arms

| Arm | Role | Supports | Seeds | Runs |
| --- | --- | --- | ---: | ---: |
| P1 | tied 0.35 temporal spine over four token horizons | m100a, full; r3 also m100b and H4 calibration | varied | 112 |
| P2 | selected tied 0.70 comparator at r3 | m100a, full | 8 paired | 16 |
| P3 | selected two-phase (0.02, 0.82) policy at r3 | m100a, full | 8 paired | 16 |
| P4 | nearest-aggregate tied 368/2048 policy at r3 | m100a, full | 8 paired | 16 |
| B | five exact fixed-aggregate/fixed-contrast switch times plus tied 0.18 | m100a | 16 paired | 96 |

P3 versus P4 is the primary nearest-aggregate endpoint contrast. P3 versus P2 is retained as the historical joint
policy contrast and may not be described as a pure phase-order effect. H5 has no full-support arm, so its causal
schedule conclusion is conditional on m100a and cannot identify a repetition-by-schedule interaction.

## State, gradients, and optimizer-aware utility

At checkpoint `t` and training seed `s`, let `g(q,s,r,t)` be the parameter gradient of the mean next-token loss on
replicate block `r` from distribution `q`. The fixed source distributions are unseen StarCoder reference examples,
the included-support StarCoder reference, on-policy StarCoder, a frozen Nemotron aggregate, and six diagnostic
Nemotron leaves. The primary target is Paloma Programming Languages BPB; GitHub Python is a code consistency target,
and C4 English plus Wikipedia English are natural-language references.

Let `Delta(q,t)` be the exact one-step parameter change obtained by applying the frozen optimizer state to a
training-scale 128-sequence source-`q` batch, and let `Delta(0,t)` be the no-data optimizer-memory update. Target
utility and normalized source-choice alignment are

```
U_y(q,t) = -g_y(t)^T [Delta(q,t) - Delta(0,t)]
X_y(t)   = U_y(S,t) - U_y(N,t)
A_y(t)   = -<g_y(t), Delta(S,t)-Delta(N,t)>
           / (||g_y(t)|| ||Delta(S,t)-Delta(N,t)||).
```

`A_y` is the primary directional statistic; raw `X_y`, all vector norms, and exact short rollouts are materiality
checks. The linear trunk uses MuonH, so the primary geometry projects each matrix gradient/update onto the model
weight-norm tangent and aggregates matrix dot products before normalization. Full-model, head, embedding, layer,
raw-gradient, and optimizer-update results are reported separately. A full-model-only effect is classified as
lexical/head/embedding evidence, not trunk conflict.

The packed StarCoder cache does not preserve repository/language metadata. Exact source-sequence identities and the
global training holdout are audited, but exact content overlap between the StarCoder training corpus and external
Programming Languages evaluation examples is not proven. Target-alignment results are therefore conditional on this
unresolved overlap risk and must agree in sign on GitHub Python before receiving a code-generalization interpretation.

## Hypotheses and frozen estimands

### H1: source conflict (descriptive)

Report projected raw-gradient and optimizer-update cosines between StarCoder and Nemotron over training. Negative
values are direct conflict; declining positive values are increasing disagreement. H1 has no confirmatory p-value and
cannot determine which source benefits the target.

### H2: temporal target revaluation

On P1 r3/m100a tied 0.35, define mid states `M={{0.40T,0.55T}}` and late pre-decay states
`L-={{0.8T-256,0.8T-64}}`. For each seed,

```
T_H2 = [mean A_PL(L-) - mean A_PL(M)] - [mean A_C4(L-) - mean A_C4(M)].
```

The primary test uses 24 training seeds, a one-sided exact sign-flip test, and a 95% seed bootstrap interval. GitHub
Python must agree in sign; Wikipedia-minus-C4 is a negative control. The post-decay contrast uses
`L+={{0.8T+64,0.8T+256}}` and is secondary because no no-decay control identifies the optimizer-schedule cause.

### H3: repetition interaction

Compare the same H2 statistic between seed-paired P1 m100a and full-support trajectories, always using the same global
StarCoder holdout as the primary source probe. A repetition mechanism predicts an m100a-specific decline in unseen
source utility plus growing included-support versus holdout separation. m100b is sensitivity evidence only: it is a
second fixed finite support, not a random-pool population replicate.

### H4: exact-state behavioral validity

From frozen parent states, run 512 updates at StarCoder shares
`q={{0,0.25,0.35,0.45,0.55,0.75,1}}` with common source order within each parent. Fit the utility-to-BPB mapping only on
eight independent calibration seeds, freeze it, then evaluate 16 preregistered P1 seeds. Those validation parents
reuse H2 trajectories and the H2 decay-minus-64 state, so this is an outcome-held-out mapping check rather than an
independent-trajectory validation set. The primary readout is
Programming Languages BPB after 512 updates over `q in [0.25,0.55]`; update spread must exceed three measurement SEs.
This tests local predictive validity, not full-endpoint mediation. P2 and P4 rollouts are paired with P3 to separate
the historical joint contrast from the nearest-aggregate phase contrast.

### H5: moved data-switch intervention

H5 runs for {H5_TOTAL_STEPS:,} updates ({H5_TOTAL_BLOCKS:,} complete mixture blocks), uses a distinct cell identity
`{H5_CELL_ID}`, and holds realized aggregate and contrast at
`a={H5_AGGREGATE_COUNT_PER_BLOCK}/{base.MIXTURE_BLOCK_SIZE}` and
`delta={H5_CONTRAST_COUNT_P0_MINUS_P1}/{base.MIXTURE_BLOCK_SIZE}`. Data switches occur at
`beta={H5_SWITCH_FRACTIONS}` while optimizer decay remains fixed at 0.8T. The primary endpoint contrast is beta 0.60
versus beta 0.85 on Programming Languages BPB, using 16 paired seeds and a two-sided exact sign-flip test. The
practical-equivalence margin is 0.001 BPB. The planning SD 0.000995 BPB was borrowed from the unmatched P2/P3 policy
contrast; achieved paired uncertainty, not that proxy, governs interpretation.

For the preregistered H5 mechanism profile, beta 0.60 and beta 0.85 both retain 0.40T, 0.55T, decay-256, decay-64,
decay onset, and decay+64 at 64 probe blocks. For each target alignment, define policy differences
`D_mid`, `D_pre`, and `D_post` at means of `{{0.40T,0.55T}}`, `{{decay-256,decay-64}}`, and
`{{decay onset,decay+64}}`. The primary profile estimand is `D_pre-D_mid`; `D_post-D_pre` is secondary. This tests
whether the policies diverge after beta 0.60 has switched while beta 0.85 has not. At beta 0.90, the 0.90T observation
and data switch are the same state; they are one deduplicated event and cannot be treated as independent evidence.
The tied aggregate anchor retains the same states at the default 16 probe blocks for descriptive context only; it is
not part of the preregistered beta0.60-minus-beta0.85 profile estimand.

## Confirmatory families and multiplicity

The five confirmatory families are: P3-versus-P4 endpoint gain, H2 temporal revaluation, H3 support interaction, H4
held-out rollout prediction, and H5 beta0.60-versus-beta0.85 endpoint effect. Their primary p-values receive Holm
familywise correction at alpha 0.05. H1, H2 post-decay, P3-versus-P2, H5 profile and pairwise secondary contrasts,
individual layers, leaves, targets, and other time points are secondary or descriptive and cannot replace a failed
primary result. Report effect sizes and uncertainty regardless of adjusted significance.

## Checkpoint and probe contract

P1-P4 retain 0.10T, 0.25T, 0.40T, 0.55T, 0.70T, decay-256, decay-64, decay, decay+64, decay+256, 0.90T, and final.
H5 retains 0.55T; switch-64/at/+64; decay-64/at/+64; 0.90T; and final. Beta 0.60 and 0.85 additionally retain 0.40T
and decay-256; the tied H5 control carries the same aggregate-matched temporal states. Coincident states are one
physical checkpoint with joined labels. All checkpoints contain model, optimizer, RNG, data-iterator, and step state.

Core primary states use 64 independent 64-sequence blocks, yielding 32 optimizer-scale draws. Full and m100b support
comparisons use 32 blocks; leaf diagnostics use 16. Blocks estimate measurement reliability but never become
inferential degrees of freedom. Seed is the inferential unit. Corrected cosines are not clipped.

## Support, holdout, and placement

- Each source removes {TRAIN_HOLDOUT_SEQUENCES_PER_COMPONENT} sequences selected by seeded Feistel permutation
  `{TRAIN_HOLDOUT_SEED}`; the retained complement is `{TRAIN_HOLDOUT_PARTITION}`.
- m100a and m100b are adjacent, sequence-disjoint support views. They share 512 physical source blocks but zero packed
  sequences and zero global-holdout sequences. Exact digests are pinned in `support_partition_audit.json`.
- All parent and child compute, caches, checkpoints, temporary recovery state, and analysis outputs are constrained to
  `us-central1/us-central1-a` under `gs://marin-us-central1`.
- The support-audit implementation, audit payload, source-position histogram, design source, and every generated
  manifest are SHA-256 pinned.

## Operational evidence and remaining release gates

The original tied canary established exact checkpoint continuation, serialization, throughput, and recovery, but did
not execute a nonzero data switch. Stage 1 established finite/full loader throughput. A separately released three-row
full-length gate must finish the actual `(0.02,0.82)` switch at step 22,608 and the m100b offset branch without reading
endpoint BPB. Review v9 additionally requires a short decoupled-switch canary where the data switch and optimizer
decay occur at different steps.

The full panel may launch only after all of the following are hash-pinned in a new post-gate release:

1. The three-row full-length gate passes terminal checkpoint, boundary throughput, finite/full parity, and recovery
   thresholds without endpoint inspection.
2. The decoupled-switch canary proves the H5 data switch occurs independently of LR decay.
3. All {payload['trajectory_count']} runtime configurations, exact permanent checkpoint sets, support identities,
   output-root inventory, storage projection, and central1 paths pass automated audits.
4. A staged launch schedule has explicit allowed trajectory IDs and concurrency ceilings no greater than 64; no stage
   increases concurrency by more than 2x. The launcher has no permissive default concurrency.
5. Automated row, batch, and global stop rules are frozen. Existing successful output roots are explicitly skipped;
   partial or unexpected roots block rather than silently resume.
6. Independent mechanistic/statistical and launcher/regional reviews return pass on the exact v9 artifacts and command.

Training release does not authorize probes or rollouts. Those remain separately gated on numerical reliability,
target/reference provenance, exact optimizer-transform validation, and independent probe-code review.
"""


def _review_checklist() -> str:
    return """# Gradient-Conflict Review-v9 Prerelease Checklist

This checklist records requirements at design freeze time, not live completion
status. The hash-pinned training release is the authoritative completion record.

## Training fanout

- [x] Verify exact-state continuation across optimizer-decay onset.
- [x] Audit all expected permanent steps and no extras on both tied canary parents.
- [x] Verify terminal serialization is not duplicated.
- [ ] Complete the three-row full-length gate across a nonzero policy switch.
- [ ] Complete the short decoupled-switch H5 canary.
- [ ] Materialize and audit every training runtime config.
- [ ] Pin the sparse-swap holdout algorithm, Feistel permutation, and finite/full block-layer counts in tests.
- [ ] Prove m100a/m100b sequence disjointness and global-holdout exclusion; report physical-block overlap.
- [ ] Record source-position coverage and the unavailable language/repository-composition metadata.
- [ ] Treat paired training seeds, not sequences or physical blocks, as the inferential units.
- [ ] Audit every cache, parent, child, state, and checkpoint path as us-central1-local.
- [ ] Set and audit a separate rolling temporary-checkpoint prefix.
- [ ] Inventory every existing output root and reject partial or unexpected histories.
- [ ] Freeze automated row, batch, and global stop-rule evaluation.
- [ ] Pass independent mechanistic, statistical, launcher, and regional reviews on v9.
- [ ] Issue a new post-gate release with exact stage IDs and explicit concurrency at most 64.
- [ ] Pass the central1 launch-safety validator on the exact Iris command.

## Probe fanout

- [ ] Freeze and hash target/reference cache provenance.
- [ ] Implement the optimizer-aware gradient probe and rollout runners.
- [ ] Pass the 112-row numerical preflight without inspecting H2 signs.
- [ ] Freeze reliability thresholds, H2 SESOI/MDE, and the H4 utility-to-loss mapping.
- [ ] Run independent probe-code and statistical reviews.
"""


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(V8_SUPPORT_AUDIT, SUPPORT_AUDIT)
    shutil.copyfile(V8_SUPPORT_POSITION_HISTOGRAM, SUPPORT_POSITION_HISTOGRAM)
    support_audit = json.loads(SUPPORT_AUDIT.read_text())
    if support_audit.get("audit_version") != "2026-08-11-support-v1":
        raise ValueError("Frozen support audit version drifted")
    if support_audit.get("holdout", {}).get("retained_view") != "rank-paired sparse tail swaps":
        raise ValueError("Frozen support audit no longer identifies the sparse-swap retained view")
    if support_audit.get("cross_support") != {
        "holdout_overlap_sequence_count": {"m100a": 0, "m100b": 0},
        "shared_physical_block_count": 512,
        "shared_sequence_count": 0,
    }:
        raise ValueError("Frozen support overlap contract drifted")
    confirmation_evidence = _confirmation_evidence()
    cells = v5._cells()
    trajectories = build_trajectories(cells)
    checkpoints = build_checkpoints(trajectories)
    checkpointer_rows = v5.build_checkpointer_manifest(trajectories, checkpoints)
    probes = _policy_pair_probe_rows(trajectories, checkpoints)
    probe_preflight = v5.build_probe_preflight_manifest(trajectories, probes)
    rollouts = _rollout_rows(trajectories)
    optimizer_transforms = v5.build_optimizer_transform_manifest(rollouts)
    trajectory_rows = _exact_trajectory_rows(trajectories, cells)
    checkpoint_rows = [asdict(row) for row in checkpoints]

    p3_aggregates = {
        (row["support_id"], row["training_seed"]): row["realized_aggregate_starcoder"]
        for row in trajectory_rows
        if row["arm"] == "p3"
    }
    p4_aggregates = {
        (row["support_id"], row["training_seed"]): row["realized_aggregate_starcoder"]
        for row in trajectory_rows
        if row["arm"] == "p4"
    }
    if set(p3_aggregates) != set(p4_aggregates):
        raise ValueError("P3/P4 aggregate-control pairing drifted")
    aggregate_differences = [abs(p3_aggregates[key] - p4_aggregates[key]) for key in sorted(p3_aggregates)]
    if max(aggregate_differences) >= 1 / base.MIXTURE_BLOCK_SIZE:
        raise ValueError(f"Nearest tied aggregate is more than one sequence per block away: {aggregate_differences}")

    if len(probe_preflight) != 112:
        raise ValueError(f"Probe preflight count drifted: {len(probe_preflight)}")
    if {row["train_holdout_seed"] for row in trajectory_rows} != {TRAIN_HOLDOUT_SEED}:
        raise ValueError("Training rows do not share one global holdout seed")
    if {row["train_holdout_partition"] for row in trajectory_rows} != {TRAIN_HOLDOUT_PARTITION}:
        raise ValueError("Training rows do not share the frozen holdout implementation")
    if any(row["trajectory_id"].startswith("gcf_p2_") for row in probes if not row["primary_state"]):
        raise ValueError("P2 probes must be part of the confirmed policy-pair temporal audit")
    if any(row["trajectory_id"].startswith("gcf_p4_") for row in probes if not row["primary_state"]):
        raise ValueError("P4 probes must be part of the nearest-aggregate temporal audit")
    if {row.cell_id for row in trajectories if row.arm == "b"} != {H5_CELL_ID}:
        raise ValueError("H5 trajectories do not use their distinct fixed-horizon cell")
    m100a = {
        (row.training_seed, row.support_pool_seed, 0, FINITE_SUPPORT_BATCHES)
        for row in trajectories
        if row.support_id == "m100a"
    }
    m100b = {
        (row.training_seed, row.support_pool_seed, SECOND_SUPPORT_START_BATCH, FINITE_SUPPORT_BATCHES)
        for row in trajectories
        if row.support_id == "m100b"
    }
    for training_seed, pool_seed, start, size in m100b:
        if (training_seed, pool_seed, 0, size) not in m100a or start < size:
            raise ValueError("m100a/m100b support slices are not paired and disjoint")

    training_tokens = sum(row.total_steps * base.BATCH_SIZE * base.SEQ_LEN for row in trajectories)
    probe_tokens = sum(
        int(row["replicate_blocks"]) * int(row["sequences_per_block"]) * int(row["tokens_per_sequence"])
        for row in probes
    )
    payload: dict[str, Any] = {
        "design_version": DESIGN_VERSION,
        "description": (
            "Review-v9 WSD80 gradient-conflict design with a nearest-aggregate phase control and exact H5 identity."
        ),
        "confirmed_policy_pair": confirmation_evidence,
        "nearest_tied_aggregate_match": {
            "confirmed_two_phase_realized_aggregate_range": [min(p3_aggregates.values()), max(p3_aggregates.values())],
            "nearest_tied_realized_aggregate_range": [min(p4_aggregates.values()), max(p4_aggregates.values())],
            "mean_absolute_difference": float(np.mean(aggregate_differences)),
            "maximum_absolute_difference": max(aggregate_differences),
            "maximum_allowed_difference": 1 / base.MIXTURE_BLOCK_SIZE,
        },
        "trajectory_count": len(trajectories),
        "checkpoint_count": len(checkpoints),
        "gradient_probe_row_count": len(probes),
        "probe_preflight_row_count": len(probe_preflight),
        "rollout_count": len(rollouts),
        "optimizer_transform_count": len(optimizer_transforms),
        "training_materialized_token_count": training_tokens,
        "gradient_probe_token_count": probe_tokens,
        "canary_expected_permanent_checkpoint_count": 13,
        "train_holdout_seed": TRAIN_HOLDOUT_SEED,
        "train_holdout_partition": TRAIN_HOLDOUT_PARTITION,
        "support_partition_audit": support_audit,
        "required_region": "us-central1",
        "required_zone": "us-central1-a",
        "required_bucket_prefix": "gs://marin-us-central1",
        "training_fanout_allowed": False,
        "probe_fanout_allowed": False,
        "training_launch_blockers": [
            "complete the three-row nonzero-switch full-length operational gate",
            "complete a short decoupled data-switch versus optimizer-decay canary",
            "inventory every output root and freeze skip-or-block handling",
            "freeze automated operational thresholds and staged concurrency release",
            "audit every runtime config, source identity, checkpoint set, storage projection, and regional path",
            "complete post-gate independent mechanistic, statistical, launcher, and regional reviews",
        ],
        "probe_launch_blockers": [
            "freeze target/reference cache provenance",
            "implement probe and rollout runners",
            "pass the 112-row numerical preflight and freeze reliability thresholds",
        ],
        "input_artifact_sha256": {
            str(SOURCE_DESIGN.relative_to(REPO_ROOT)): _file_sha256(SOURCE_DESIGN),
            str(CONFIRMATION_MANIFEST.relative_to(REPO_ROOT)): _file_sha256(CONFIRMATION_MANIFEST),
            str(CONFIRMATION_RESULTS.relative_to(REPO_ROOT)): _file_sha256(CONFIRMATION_RESULTS),
            str(V5_MANIFEST.relative_to(REPO_ROOT)): _file_sha256(V5_MANIFEST),
            str(V8_SUPPORT_AUDIT.relative_to(REPO_ROOT)): _file_sha256(V8_SUPPORT_AUDIT),
            str(V8_SUPPORT_POSITION_HISTOGRAM.relative_to(REPO_ROOT)): _file_sha256(V8_SUPPORT_POSITION_HISTOGRAM),
            str(SUPPORT_AUDIT_SOURCE.relative_to(REPO_ROOT)): _file_sha256(SUPPORT_AUDIT_SOURCE),
        },
        "artifact_sha256": {},
        "design_sha256": "",
    }

    artifacts = {
        "trajectory_manifest.csv": trajectory_rows,
        "checkpoint_manifest.csv": checkpoint_rows,
        "checkpointer_manifest.csv": checkpointer_rows,
        "gradient_probe_manifest.csv": probes,
        "probe_preflight_manifest.csv": probe_preflight,
        "rollout_manifest.csv": rollouts,
        "optimizer_transform_manifest.csv": optimizer_transforms,
    }
    for filename, rows in artifacts.items():
        _write_csv(OUTPUT_DIR / filename, rows)
    (OUTPUT_DIR / "design.md").write_text(_design_markdown(payload))
    (OUTPUT_DIR / "review_checklist.md").write_text(_review_checklist())
    payload["artifact_sha256"] = {
        filename: _file_sha256(OUTPUT_DIR / filename)
        for filename in (
            *sorted(artifacts),
            "design.md",
            "review_checklist.md",
            SUPPORT_AUDIT.name,
            SUPPORT_POSITION_HISTOGRAM.name,
        )
    }
    payload["artifact_sha256"][str(Path(__file__).resolve().relative_to(REPO_ROOT))] = _file_sha256(
        Path(__file__).resolve()
    )
    payload["design_sha256"] = _canonical_hash({**payload, "design_sha256": ""})
    (OUTPUT_DIR / "design_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
