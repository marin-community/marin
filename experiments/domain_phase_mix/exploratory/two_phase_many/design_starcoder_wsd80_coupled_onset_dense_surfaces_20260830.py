# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["jax==0.11.0", "numpy==2.3.5"]
# ///

"""Freeze dense StarCoder surfaces with coupled phase and LR-decay onset."""

from __future__ import annotations

import gzip
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from experiments.domain_phase_mix import launch_starcoder_wsd_80_20_surface as base
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    design_starcoder_wsd80_dense_support_surfaces_20260808 as source_design,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    design_starcoder_wsd80_lr_onset_dense_surfaces_20260825 as lr_only_design,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DOMAIN_PHASE_MIX_DIR = SCRIPT_DIR.parents[1]
REPO_ROOT = DOMAIN_PHASE_MIX_DIR.parents[1]
OUTPUT_PATH = DOMAIN_PHASE_MIX_DIR / "starcoder_wsd80_coupled_onset_dense_surface_design_20260830.json.gz"
ARTIFACT_DIR = SCRIPT_DIR / "reference_outputs" / "starcoder_wsd80_coupled_onset_dense_surface_design_20260830"
REPORT_PATH = ARTIFACT_DIR / "report.md"

DESIGN_VERSION = "2026-08-30-coupled-v1"
PRIMARY_SUPPORT_ID = lr_only_design.PRIMARY_SUPPORT_ID
DISCOVERY_SEED = 20_260_711
CONFIRMATION_SEEDS = tuple(range(20_260_841, 20_260_849))
PRIMARY_METRIC = lr_only_design.PRIMARY_METRIC
SECONDARY_BROAD_METRIC = lr_only_design.SECONDARY_BROAD_METRIC
SOURCE_BOUNDARY_FRACTION = 0.8


@dataclass(frozen=True)
class CoupledArm:
    """One phase-boundary and LR-decay onset intervention."""

    arm_id: str
    requested_onset_fraction: float
    role: str


ARMS = (
    CoupledArm("coupled_0p60", 0.60, "main"),
    CoupledArm("coupled_0p80", 0.80, "reference"),
    CoupledArm("coupled_0p90", 0.90, "main"),
)


def _aligned_boundary_step(total_steps: int, requested_fraction: float) -> int:
    step_alignment = base.MIXTURE_BLOCK_SIZE // base.BATCH_SIZE
    boundary_step = (int(total_steps * requested_fraction) // step_alignment) * step_alignment
    if boundary_step <= 0 or boundary_step >= total_steps:
        raise ValueError(f"Invalid aligned boundary: {boundary_step}/{total_steps}")
    return boundary_step


def _maximum_contrast(aggregate: float, phase_0_fraction: float, sign: int) -> float:
    if sign > 0:
        return min(aggregate / (1.0 - phase_0_fraction), (1.0 - aggregate) / phase_0_fraction)
    return min((1.0 - aggregate) / (1.0 - phase_0_fraction), aggregate / phase_0_fraction)


def _normalized_fiber_position(coordinate: dict[str, Any]) -> float:
    contrast = float(coordinate["phase_contrast"])
    if abs(contrast) <= 1e-15:
        return 0.0
    aggregate = float(coordinate["aggregate_starcoder"])
    limit = _maximum_contrast(aggregate, SOURCE_BOUNDARY_FRACTION, 1 if contrast > 0 else -1)
    normalized = contrast / limit
    if abs(normalized) > 1.0 + 1e-10:
        raise ValueError(f"{coordinate['coordinate_id']}: source coordinate exceeds its fiber")
    return float(np.clip(normalized, -1.0, 1.0))


def _remap_coordinate(coordinate: dict[str, Any], phase_0_fraction: float) -> dict[str, float]:
    """Preserve aggregate exposure and normalized fiber position at a new split."""
    aggregate = float(coordinate["aggregate_starcoder"])
    normalized = _normalized_fiber_position(coordinate)
    if normalized == 0.0:
        contrast = 0.0
    else:
        limit = _maximum_contrast(aggregate, phase_0_fraction, 1 if normalized > 0 else -1)
        contrast = normalized * limit
    phase_0 = aggregate - (1.0 - phase_0_fraction) * contrast
    phase_1 = aggregate + phase_0_fraction * contrast
    if not (-1e-12 <= phase_0 <= 1.0 + 1e-12 and -1e-12 <= phase_1 <= 1.0 + 1e-12):
        raise ValueError(f"{coordinate['coordinate_id']}: remapped phase weights are infeasible")
    phase_0 = float(np.clip(phase_0, 0.0, 1.0))
    phase_1 = float(np.clip(phase_1, 0.0, 1.0))
    observed_aggregate = phase_0_fraction * phase_0 + (1.0 - phase_0_fraction) * phase_1
    if not np.isclose(observed_aggregate, aggregate, atol=1e-12):
        raise ValueError(f"{coordinate['coordinate_id']}: aggregate exposure drifted")
    return {
        "phase_0_starcoder": phase_0,
        "phase_1_starcoder": phase_1,
        "aggregate_starcoder": aggregate,
        "phase_contrast": phase_1 - phase_0,
        "normalized_fiber_position": normalized,
    }


def _optimizer_profile(cell: dict[str, Any], boundary_step: int) -> dict[str, Any]:
    total_steps = int(cell["total_steps"])
    optimizer = base._optimizer(int(cell["materialized_tokens"]))
    configured = replace(optimizer, decay=total_steps - boundary_step, min_lr_ratio=0.0)
    schedule = configured.lr_scheduler(total_steps)(jnp.arange(total_steps))
    return {
        "decay_onset_step": boundary_step,
        "decay_steps": total_steps - boundary_step,
        "min_lr_ratio": 0.0,
        "learning_rate": float(optimizer.learning_rate),
        "adam_lr": float(optimizer.adam_lr),
        "normalized_lr_integral": float(schedule.sum() / float(optimizer.learning_rate)),
        "normalized_phase_1_lr_integral": float(schedule[boundary_step:].sum() / float(optimizer.learning_rate)),
    }


def _run_name(arm: CoupledArm, coordinate_id: str) -> str:
    onset_slug = f"{arm.requested_onset_fraction:.2f}".replace(".", "p")
    return f"lrcd_ds_{PRIMARY_SUPPORT_ID}_{onset_slug}_{coordinate_id}_s{DISCOVERY_SEED % 10000:04d}"


def _run_row(
    *,
    run_order: int,
    cell: dict[str, Any],
    support: dict[str, Any],
    coordinate: dict[str, Any],
    arm: CoupledArm,
) -> dict[str, Any]:
    boundary_step = _aligned_boundary_step(int(cell["total_steps"]), arm.requested_onset_fraction)
    realized_fraction = boundary_step / int(cell["total_steps"])
    remapped = _remap_coordinate(coordinate, realized_fraction)
    arm_cell = {**cell, "boundary_step": boundary_step, "realized_phase_0_fraction": realized_fraction}
    phase_0_sequences, phase_1_sequences = source_design._realized_starcoder_sequences(
        cell=arm_cell,
        phase_0=remapped["phase_0_starcoder"],
        phase_1=remapped["phase_1_starcoder"],
        data_seed=DISCOVERY_SEED,
    )
    support_tokens = int(support["starcoder_realized_support_tokens"])
    identity = {
        "design_version": DESIGN_VERSION,
        "arm_id": arm.arm_id,
        "coordinate_id": coordinate["coordinate_id"],
        "data_seed": DISCOVERY_SEED,
    }
    return {
        "row_id": f"coupled_onset_surface_{lr_only_design.canonical_sha256(identity)[:24]}",
        "run_order": run_order,
        "run_name": _run_name(arm, coordinate["coordinate_id"]),
        "stage": "surface_discovery",
        "cell_id": cell["cell_id"],
        "cell_slug": cell["cell_slug"],
        "rung": cell["rung"],
        "hidden_size": cell["hidden_size"],
        "total_steps": cell["total_steps"],
        "boundary_step": boundary_step,
        "requested_onset_fraction": arm.requested_onset_fraction,
        "realized_onset_fraction": realized_fraction,
        "materialized_tokens": cell["materialized_tokens"],
        "total_parameters": cell["total_parameters"],
        "non_embedding_parameters": cell["non_embedding_parameters"],
        "support_id": support["support_id"],
        "support_role": support["role"],
        "epoch_multiplier": support["epoch_multiplier"],
        "starcoder_support_batches": support["starcoder_support_batches"],
        "starcoder_realized_support_tokens": support_tokens,
        "starcoder_support_fraction": support["starcoder_support_fraction"],
        "coordinate_id": coordinate["coordinate_id"],
        "coordinate_sources": coordinate["sources"],
        "selection_class": lr_only_design._coordinate_role(coordinate),
        **remapped,
        "starcoder_phase_0_sequences": phase_0_sequences,
        "starcoder_phase_1_sequences": phase_1_sequences,
        "starcoder_total_sequences": phase_0_sequences + phase_1_sequences,
        "starcoder_phase_0_epochs": phase_0_sequences * base.SEQ_LEN / support_tokens,
        "starcoder_phase_1_epochs": phase_1_sequences * base.SEQ_LEN / support_tokens,
        "starcoder_support_wraps": (phase_0_sequences + phase_1_sequences) * base.SEQ_LEN > support_tokens,
        "arm_id": arm.arm_id,
        "arm_role": arm.role,
        "decay_onset_fraction": realized_fraction,
        "peak_lr_multiplier": 1.0,
        "optimizer": _optimizer_profile(cell, boundary_step),
        "data_seed": DISCOVERY_SEED,
        "trainer_seed": DISCOVERY_SEED,
    }


def build_payload() -> dict[str, Any]:
    """Build the coupled-onset discovery panel from the audited common grid."""
    source = lr_only_design._load_source_design()
    cell = lr_only_design._cell(source)
    support = lr_only_design._supports(source)[PRIMARY_SUPPORT_ID]
    coordinates = lr_only_design._coordinates(source)
    rows = [
        _run_row(
            run_order=arm_index * len(coordinates) + coordinate_index,
            cell=cell,
            support=support,
            coordinate=coordinates[coordinate_id],
            arm=arm,
        )
        for arm_index, arm in enumerate(ARMS)
        for coordinate_index, coordinate_id in enumerate(sorted(coordinates))
    ]
    if len(rows) != 375 or len({row["row_id"] for row in rows}) != len(rows):
        raise ValueError("Coupled row inventory drifted")
    for row in rows:
        if row["boundary_step"] != row["optimizer"]["decay_onset_step"]:
            raise ValueError(f"{row['run_name']}: phase and LR onset are not coupled")

    environment = {
        "jax_version": jax.__version__,
        "numpy_version": np.__version__,
        "jax_default_prng_impl": jax.config.jax_default_prng_impl,
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "uv_lock_sha256": lr_only_design.file_sha256(REPO_ROOT / "uv.lock"),
    }
    arm_records = []
    for arm in ARMS:
        boundary_step = _aligned_boundary_step(int(cell["total_steps"]), arm.requested_onset_fraction)
        arm_records.append(
            {
                **asdict(arm),
                "boundary_step": boundary_step,
                "realized_onset_fraction": boundary_step / int(cell["total_steps"]),
                "optimizer": _optimizer_profile(cell, boundary_step),
            }
        )
    return {
        "design_version": DESIGN_VERSION,
        "question": (
            "When the phase-2 mixture switch and cosine LR decay begin together, does an earlier onset and longer "
            "second phase produce a larger two-phase endpoint advantage?"
        ),
        "hypothesis": {
            "directional": "Earlier coupled onset produces a larger best-tied minus best-untied BPB gain.",
            "null": "The selected two-phase gain and optimum fiber position do not change with coupled onset.",
        },
        "claim_boundary": (
            "The treatment intentionally couples phase duration and LR-decay duration. The predecessor 2026-08-25 "
            "panel, which varied LR onset at a fixed 0.80T phase boundary, is the orthogonal LR-only control."
        ),
        "source_design_path": str(lr_only_design.SOURCE_DESIGN_PATH.relative_to(REPO_ROOT)),
        "source_design_sha256": lr_only_design.SOURCE_DESIGN_SHA256,
        "source_design_file_sha256": lr_only_design.file_sha256(lr_only_design.SOURCE_DESIGN_PATH),
        "orthogonal_lr_only_control": {
            "design_path": str(lr_only_design.OUTPUT_PATH.relative_to(REPO_ROOT)),
            "design_version": lr_only_design.DESIGN_VERSION,
            "fieldbook_experiment_id": "exp_01m0xqtjba8rr95gxdme079rwz",
        },
        "design_environment": environment,
        "training_environment": environment,
        "runtime_cache_contract": source["runtime_cache_contract"],
        "source_placement": {
            "marin_prefix": lr_only_design.MARIN_PREFIX,
            "tpu_type": lr_only_design.TPU_TYPE,
            "region": lr_only_design.TPU_REGION,
            "zone": lr_only_design.TPU_ZONE,
        },
        "cell": cell,
        "support": support,
        "coordinates": [
            {
                **lr_only_design._coordinate_record(coordinates[key]),
                "normalized_fiber_position": _normalized_fiber_position(coordinates[key]),
            }
            for key in sorted(coordinates)
        ],
        "arms": arm_records,
        "discovery_seed": DISCOVERY_SEED,
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "metrics": {
            "primary": PRIMARY_METRIC,
            "broad_secondary": SECONDARY_BROAD_METRIC,
            "direction": "lower_is_better",
            "endpoint_step": int(cell["total_steps"]) - 1,
        },
        "analysis_contract": {
            "surface_estimand": (
                "For each complete arm, select the minimum raw-grid tied coordinate and minimum raw-grid eligible "
                "untied coordinate; gain is tied BPB minus untied BPB."
            ),
            "two_phaseness": (
                "Report selected gain, absolute phase contrast, and absolute normalized fiber position. The common "
                "coordinate IDs preserve aggregate exposure and normalized fiber position across phase splits."
            ),
            "ordered_test": (
                "After fresh eight-seed confirmation of each arm's selected tied and untied policies, test the "
                "preregistered order gain_0p60 >= gain_0p80 >= gain_0p90."
            ),
            "adaptive_confirmation": (
                "Freeze a second hashed manifest only after all 375 discovery endpoints are complete; use the eight "
                "reserved seeds and never reuse the discovery seed."
            ),
        },
        "completeness_contract": {
            "valid_endpoint": f"one finite {PRIMARY_METRIC} value at exact step {int(cell['total_steps']) - 1}",
            "surface_discovery": "all 125 coordinates in all three arms are required before selection or inference",
            "failure_rule": "retry the exact frozen identity; never drop or replace a failed row",
        },
        "checkpoint_contract": {
            "surface_discovery": "terminal permanent checkpoint only",
            "temporary_recovery": "time-based resumable checkpoints every ten minutes",
        },
        "reuse_contract": {
            "reuse_existing_rows": False,
            "reason": "Run all three arms under one coupled-design identity and W&B group.",
        },
        "stage_counts": {"surface_discovery": 375},
        "expected_run_count": 375,
        "runs": rows,
    }


def _report(payload: dict[str, Any]) -> str:
    arm_lines = [
        (
            f"- {arm['arm_id']}: requested={arm['requested_onset_fraction']:.2f}T, "
            f"realized={arm['realized_onset_fraction']:.6f}T, boundary/decay step={arm['boundary_step']}."
        )
        for arm in payload["arms"]
    ]
    return "\n".join(
        [
            "# StarCoder coupled phase/LR-onset dense-surface design",
            "",
            "## Intervention",
            "",
            "For every arm, phase 2 and cosine LR decay begin at the same mixture-block-aligned update. The 125 "
            "coordinates preserve aggregate StarCoder exposure and normalized fiber position across splits.",
            "",
            *arm_lines,
            "",
            "## Inventory and inference",
            "",
            "- 375 discovery rows: three coupled arms by 125 coordinates.",
            "- No fixed-policy confirmation is spent before observing the complete surfaces.",
            "- After completion, select each arm's best tied and eligible untied raw-grid policies and freeze a fresh "
            "eight-seed confirmation manifest.",
            "- The predecessor fixed-boundary experiment is retained only as an LR-only control.",
            "",
        ]
    )


def main() -> None:
    payload = build_payload()
    payload["design_sha256"] = lr_only_design.canonical_sha256(payload)
    serialized = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    OUTPUT_PATH.write_bytes(gzip.compress(serialized, mtime=0))
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(_report(payload), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH} ({payload['expected_run_count']} rows, {payload['design_sha256']})")


if __name__ == "__main__":
    main()
