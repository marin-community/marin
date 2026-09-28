# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["jax==0.11.0", "numpy==2.3.5"]
# ///

"""Freeze the StarCoder WSD80 LR-onset endpoint-surface intervention."""

from __future__ import annotations

import gzip
import hashlib
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

SCRIPT_DIR = Path(__file__).resolve().parent
DOMAIN_PHASE_MIX_DIR = SCRIPT_DIR.parents[1]
REPO_ROOT = DOMAIN_PHASE_MIX_DIR.parents[1]
SOURCE_DESIGN_PATH = DOMAIN_PHASE_MIX_DIR / "starcoder_wsd80_dense_support_surface_design_20260808.json"
HISTORICAL_CONFIRMATION_PATH = (
    DOMAIN_PHASE_MIX_DIR / "starcoder_wsd80_dense_support_empirical_optimum_confirmation_design_20260811.json.gz"
)
OUTPUT_PATH = DOMAIN_PHASE_MIX_DIR / "starcoder_wsd80_lr_onset_dense_surface_design_20260825.json.gz"
ARTIFACT_DIR = SCRIPT_DIR / "reference_outputs" / "starcoder_wsd80_lr_onset_dense_surface_design_20260825"
REPORT_PATH = ARTIFACT_DIR / "report.md"

DESIGN_VERSION = "2026-08-25-v2"
SOURCE_DESIGN_SHA256 = "d4ffb9079f969af808230c623555315262cb314434a21db6d36e9651b747cd48"
HISTORICAL_CONFIRMATION_FILE_SHA256 = "a2014b02ca8a193a2112aae52c8bd2e62354267571bdf6d08bb20f7e4e5f42e7"
HISTORICAL_CONFIRMATION_DESIGN_SHA256 = "ea116688ba7b0fa38713b5e616fb560f7708e2d385e782b757fc745674beecec"
CELL_ID = "r3_increase_d_h0640_s28260"
PRIMARY_SUPPORT_ID = "m100"
REPLICATION_SUPPORT_ID = "m200"
DISCOVERY_SEED = 20_260_711
HISTORICAL_CONFIRMATION_SEEDS = tuple(range(20_260_821, 20_260_826))
CONFIRMATION_SEEDS = tuple(range(20_260_831, 20_260_839))
PRIMARY_TIED_COORDINATE = "c109"
PRIMARY_UNTIED_COORDINATE = "c020"
REPLICATION_TIED_COORDINATE = "c079"
REPLICATION_UNTIED_COORDINATE = "c011"
MINIMUM_UNTIED_ABSOLUTE_CONTRAST = 0.04
PRIMARY_METRIC = "eval/paloma/dolma_100_programing_languages-llama3/bpb"
SECONDARY_BROAD_METRIC = "eval/paloma/c4_en-llama3/bpb"
MARIN_PREFIX = "gs://marin-us-central1"
TPU_TYPE = "v5p-8"
TPU_REGION = "us-central1"
TPU_ZONE = "us-central1-a"


@dataclass(frozen=True)
class ScheduleArm:
    """One optimizer schedule applied to a fixed data policy."""

    arm_id: str
    decay_onset_fraction: float | None
    peak_lr_multiplier: float
    role: str


MAIN_ARMS = (
    ScheduleArm("decay_0p60", 0.60, 1.0, "main"),
    ScheduleArm("decay_0p80", 0.80, 1.0, "historical_reference"),
    ScheduleArm("decay_0p90", 0.90, 1.0, "main"),
    ScheduleArm("no_decay", None, 1.0, "main"),
)


def canonical_sha256(value: Any) -> str:
    """Return a stable hash for a JSON-compatible value."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_source_design() -> dict[str, Any]:
    payload = json.loads(SOURCE_DESIGN_PATH.read_text(encoding="utf-8"))
    claimed_hash = payload.pop("design_sha256")
    observed_hash = canonical_sha256(payload)
    if claimed_hash != SOURCE_DESIGN_SHA256 or observed_hash != SOURCE_DESIGN_SHA256:
        raise ValueError(f"Source design hash drifted: {observed_hash} != {claimed_hash}")
    payload["design_sha256"] = claimed_hash
    return payload


def _load_historical_confirmation() -> dict[str, Any]:
    if file_sha256(HISTORICAL_CONFIRMATION_PATH) != HISTORICAL_CONFIRMATION_FILE_SHA256:
        raise ValueError("Historical confirmation file hash drifted")
    payload = json.loads(gzip.decompress(HISTORICAL_CONFIRMATION_PATH.read_bytes()))
    if payload.get("design_sha256") != HISTORICAL_CONFIRMATION_DESIGN_SHA256:
        raise ValueError("Historical confirmation design hash drifted")
    if tuple(payload.get("fresh_seeds", ())) != HISTORICAL_CONFIRMATION_SEEDS:
        raise ValueError("Historical confirmation seeds drifted")
    return payload


def _cell(payload: dict[str, Any]) -> dict[str, Any]:
    matches = [row for row in payload["cells"] if row["cell_id"] == CELL_ID]
    if len(matches) != 1:
        raise ValueError(f"Expected one source cell {CELL_ID}, got {len(matches)}")
    return matches[0]


def _supports(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    selected = {
        row["support_id"]: row
        for row in payload["supports"]
        if row["cell_id"] == CELL_ID and row["support_id"] in {PRIMARY_SUPPORT_ID, REPLICATION_SUPPORT_ID}
    }
    if set(selected) != {PRIMARY_SUPPORT_ID, REPLICATION_SUPPORT_ID}:
        raise ValueError(f"Source support inventory drifted: {sorted(selected)}")
    return selected


def _coordinates(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    coordinates = {row["coordinate_id"]: row for row in payload["coordinates"]}
    if len(coordinates) != 125:
        raise ValueError(f"Expected 125 common coordinates, got {len(coordinates)}")
    required = {
        PRIMARY_TIED_COORDINATE,
        PRIMARY_UNTIED_COORDINATE,
        REPLICATION_TIED_COORDINATE,
        REPLICATION_UNTIED_COORDINATE,
    }
    if not required <= set(coordinates):
        raise ValueError(f"Required fixed coordinates are absent: {sorted(required - set(coordinates))}")
    return coordinates


def _historical_fixed_policy_provenance(payload: dict[str, Any]) -> list[dict[str, Any]]:
    expected = {
        (PRIMARY_SUPPORT_ID, "tied"): PRIMARY_TIED_COORDINATE,
        (PRIMARY_SUPPORT_ID, "untied"): PRIMARY_UNTIED_COORDINATE,
        (REPLICATION_SUPPORT_ID, "tied"): REPLICATION_TIED_COORDINATE,
        (REPLICATION_SUPPORT_ID, "untied"): REPLICATION_UNTIED_COORDINATE,
    }
    selected = [
        row
        for row in payload["selected_policies"]
        if row["cell_id"] == CELL_ID and row["support_id"] in {PRIMARY_SUPPORT_ID, REPLICATION_SUPPORT_ID}
    ]
    observed = {(row["support_id"], row["policy_class"]): row["coordinate_id"] for row in selected}
    if observed != expected:
        raise ValueError(f"Historical fixed-policy provenance drifted: {observed}")
    return selected


def _coordinate_record(coordinate: dict[str, Any]) -> dict[str, Any]:
    record = dict(coordinate)
    record["source_geometry_role"] = record.pop("policy_role")
    record["selection_class"] = _coordinate_role(coordinate)
    return record


def _optimizer_profile(cell: dict[str, Any], arm: ScheduleArm) -> dict[str, Any]:
    total_steps = int(cell["total_steps"])
    optimizer = base._optimizer(int(cell["materialized_tokens"]))
    if arm.decay_onset_fraction is None:
        decay_onset_step = None
        decay_steps = int(optimizer.decay)
        min_lr_ratio = 1.0
    else:
        decay_onset_step = round(total_steps * arm.decay_onset_fraction)
        decay_steps = total_steps - decay_onset_step
        min_lr_ratio = 0.0
    learning_rate = float(optimizer.learning_rate) * arm.peak_lr_multiplier
    adam_lr = float(optimizer.adam_lr) * arm.peak_lr_multiplier
    configured = replace(
        optimizer,
        learning_rate=learning_rate,
        adam_lr=adam_lr,
        decay=decay_steps,
        min_lr_ratio=min_lr_ratio,
    )
    schedule = configured.lr_scheduler(total_steps)(jnp.arange(total_steps))
    boundary_step = int(cell["boundary_step"])
    return {
        "decay_onset_step": decay_onset_step,
        "decay_steps": decay_steps,
        "min_lr_ratio": min_lr_ratio,
        "learning_rate": learning_rate,
        "adam_lr": adam_lr,
        "normalized_lr_integral": float(schedule.sum() / float(optimizer.learning_rate)),
        "normalized_phase_1_lr_integral": float(schedule[boundary_step:].sum() / float(optimizer.learning_rate)),
    }


def _area_matched_arm(cell: dict[str, Any]) -> ScheduleArm:
    profiles = {arm.arm_id: _optimizer_profile(cell, arm) for arm in MAIN_ARMS}
    multiplier = profiles["decay_0p60"]["normalized_lr_integral"] / profiles["decay_0p80"]["normalized_lr_integral"]
    return ScheduleArm(
        "decay_0p80_area_match_0p60",
        0.80,
        float(multiplier),
        "lr_integral_sensitivity_only",
    )


def _coordinate_role(coordinate: dict[str, Any]) -> str:
    phase_0 = float(coordinate["phase_0_starcoder"])
    phase_1 = float(coordinate["phase_1_starcoder"])
    if abs(phase_1 - phase_0) <= 1e-12:
        return "tied"
    if abs(phase_1 - phase_0) < MINIMUM_UNTIED_ABSOLUTE_CONTRAST:
        return "ineligible_near_tied"
    return "eligible_untied"


def _run_name(stage: str, support_id: str, arm_id: str, coordinate_id: str, seed: int) -> str:
    stage_slug = {"primary_spine": "sp", "surface_discovery": "ds", "replay_replication": "rp"}[stage]
    arm_slug = arm_id.removeprefix("decay_").replace("no_decay", "nodecay")
    return f"lrod_{stage_slug}_{support_id}_{arm_slug}_{coordinate_id}_s{seed % 10000:04d}"


def _run_row(
    *,
    run_order: int,
    stage: str,
    cell: dict[str, Any],
    support: dict[str, Any],
    coordinate: dict[str, Any],
    arm: ScheduleArm,
    seed: int,
) -> dict[str, Any]:
    phase_0 = float(coordinate["phase_0_starcoder"])
    phase_1 = float(coordinate["phase_1_starcoder"])
    phase_0_sequences, phase_1_sequences = source_design._realized_starcoder_sequences(
        cell=cell,
        phase_0=phase_0,
        phase_1=phase_1,
        data_seed=seed,
    )
    support_tokens = int(support["starcoder_realized_support_tokens"])
    starcoder_total_sequences = phase_0_sequences + phase_1_sequences
    identity = {
        "design_version": DESIGN_VERSION,
        "stage": stage,
        "cell_id": cell["cell_id"],
        "support_id": support["support_id"],
        "arm_id": arm.arm_id,
        "coordinate_id": coordinate["coordinate_id"],
        "data_seed": seed,
    }
    return {
        "row_id": f"lr_onset_surface_{canonical_sha256(identity)[:24]}",
        "run_order": run_order,
        "run_name": _run_name(stage, support["support_id"], arm.arm_id, coordinate["coordinate_id"], seed),
        "stage": stage,
        "cell_id": cell["cell_id"],
        "cell_slug": cell["cell_slug"],
        "rung": cell["rung"],
        "hidden_size": cell["hidden_size"],
        "total_steps": cell["total_steps"],
        "boundary_step": cell["boundary_step"],
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
        "selection_class": _coordinate_role(coordinate),
        "phase_0_starcoder": phase_0,
        "phase_1_starcoder": phase_1,
        "aggregate_starcoder": coordinate["aggregate_starcoder"],
        "phase_contrast": coordinate["phase_contrast"],
        "starcoder_phase_0_sequences": phase_0_sequences,
        "starcoder_phase_1_sequences": phase_1_sequences,
        "starcoder_total_sequences": starcoder_total_sequences,
        "starcoder_phase_0_epochs": phase_0_sequences * base.SEQ_LEN / support_tokens,
        "starcoder_phase_1_epochs": phase_1_sequences * base.SEQ_LEN / support_tokens,
        "starcoder_support_wraps": starcoder_total_sequences * base.SEQ_LEN > support_tokens,
        "arm_id": arm.arm_id,
        "arm_role": arm.role,
        "decay_onset_fraction": arm.decay_onset_fraction,
        "peak_lr_multiplier": arm.peak_lr_multiplier,
        "optimizer": _optimizer_profile(cell, arm),
        "data_seed": seed,
        "trainer_seed": seed,
    }


def build_payload() -> dict[str, Any]:
    """Build the complete frozen design from the audited dense-support panel."""
    source = _load_source_design()
    historical_confirmation = _load_historical_confirmation()
    historical_fixed_policies = _historical_fixed_policy_provenance(historical_confirmation)
    if set(CONFIRMATION_SEEDS) & set(HISTORICAL_CONFIRMATION_SEEDS):
        raise ValueError("New confirmation seeds overlap the historical confirmation")
    cell = _cell(source)
    supports = _supports(source)
    coordinates = _coordinates(source)
    area_arm = _area_matched_arm(cell)
    arms = (*MAIN_ARMS, area_arm)
    rows: list[dict[str, Any]] = []

    def add(stage: str, support_id: str, arm: ScheduleArm, coordinate_id: str, seed: int) -> None:
        rows.append(
            _run_row(
                run_order=len(rows),
                stage=stage,
                cell=cell,
                support=supports[support_id],
                coordinate=coordinates[coordinate_id],
                arm=arm,
                seed=seed,
            )
        )

    for arm in MAIN_ARMS:
        for coordinate_id in sorted(coordinates):
            add("surface_discovery", PRIMARY_SUPPORT_ID, arm, coordinate_id, DISCOVERY_SEED)

    for arm in arms:
        for coordinate_id in (PRIMARY_TIED_COORDINATE, PRIMARY_UNTIED_COORDINATE):
            for seed in CONFIRMATION_SEEDS:
                add("primary_spine", PRIMARY_SUPPORT_ID, arm, coordinate_id, seed)

    for arm in MAIN_ARMS:
        for coordinate_id in (REPLICATION_TIED_COORDINATE, REPLICATION_UNTIED_COORDINATE):
            for seed in CONFIRMATION_SEEDS:
                add("replay_replication", REPLICATION_SUPPORT_ID, arm, coordinate_id, seed)

    identities = {
        (row["stage"], row["support_id"], row["arm_id"], row["coordinate_id"], row["data_seed"]) for row in rows
    }
    if len(rows) != len(identities) or len({row["row_id"] for row in rows}) != len(rows):
        raise ValueError("Run identities are not unique")
    stage_counts = {stage: sum(row["stage"] == stage for row in rows) for stage in {row["stage"] for row in rows}}
    expected_stage_counts = {"surface_discovery": 500, "primary_spine": 80, "replay_replication": 64}
    if stage_counts != expected_stage_counts:
        raise ValueError(f"Stage counts drifted: {stage_counts}")

    runtime_environment = {
        "jax_version": jax.__version__,
        "numpy_version": np.__version__,
        "jax_default_prng_impl": jax.config.jax_default_prng_impl,
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "uv_lock_sha256": file_sha256(REPO_ROOT / "uv.lock"),
    }
    payload = {
        "design_version": DESIGN_VERSION,
        "question": (
            "How does optimizer budget remaining after the fixed 0.80T data-mixture switch change the endpoint "
            "advantage of untied StarCoder/Nemotron schedules?"
        ),
        "hypothesis": {
            "confirmatory": "LR schedule changes tied-minus-untied endpoint gain; test two-sided.",
            "directional_sensitivity": "Earlier decay may increase gain, but the opposite direction is plausible.",
        },
        "claim_boundary": (
            "Main arms jointly change decay onset, cumulative LR, and post-switch optimizer budget. The area-matched "
            "arm is sensitivity-only because matching LR integral also changes peak LR and the prefix trajectory."
        ),
        "source_design_path": str(SOURCE_DESIGN_PATH.relative_to(REPO_ROOT)),
        "source_design_sha256": SOURCE_DESIGN_SHA256,
        "source_design_file_sha256": file_sha256(SOURCE_DESIGN_PATH),
        "historical_confirmation": {
            "path": str(HISTORICAL_CONFIRMATION_PATH.relative_to(REPO_ROOT)),
            "file_sha256": HISTORICAL_CONFIRMATION_FILE_SHA256,
            "design_sha256": HISTORICAL_CONFIRMATION_DESIGN_SHA256,
            "seeds": list(HISTORICAL_CONFIRMATION_SEEDS),
            "fixed_policy_provenance": historical_fixed_policies,
        },
        "design_environment": runtime_environment,
        "training_environment": runtime_environment,
        "runtime_cache_contract": source["runtime_cache_contract"],
        "placement": {
            "marin_prefix": MARIN_PREFIX,
            "tpu_type": TPU_TYPE,
            "region": TPU_REGION,
            "zone": TPU_ZONE,
        },
        "cell": cell,
        "supports": [supports[PRIMARY_SUPPORT_ID], supports[REPLICATION_SUPPORT_ID]],
        "coordinates": [_coordinate_record(coordinates[key]) for key in sorted(coordinates)],
        "arms": [
            {
                **asdict(arm),
                "optimizer": _optimizer_profile(cell, arm),
            }
            for arm in arms
        ],
        "discovery_seed": DISCOVERY_SEED,
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "minimum_untied_absolute_contrast": MINIMUM_UNTIED_ABSOLUTE_CONTRAST,
        "metrics": {
            "primary": PRIMARY_METRIC,
            "broad_secondary": SECONDARY_BROAD_METRIC,
            "direction": "lower_is_better",
            "endpoint_step": cell["total_steps"] - 1,
            "scale_invariant_secondary": (
                "log(BPB_tied / BPB_untied) within seed and schedule arm; positive favors the untied policy"
            ),
        },
        "analysis_contract": {
            "p1_primary": {
                "estimand": "paired fixed-coordinate tied BPB minus untied BPB within seed and schedule arm",
                "scale_invariant_secondary": "paired log(BPB_tied / BPB_untied) within seed and schedule arm",
                "support_id": PRIMARY_SUPPORT_ID,
                "tied_coordinate_id": PRIMARY_TIED_COORDINATE,
                "untied_coordinate_id": PRIMARY_UNTIED_COORDINATE,
                "reference_arm": "decay_0p80",
                "test": "two-sided seed-blocked schedule-by-policy contrasts",
                "scale_disagreement_rule": (
                    "The additive contrast is primary. If its direction disagrees with the log-gain contrast, "
                    "report the effect as BPB-level-dependent and do not claim scale-robust increased two-phaseness."
                ),
                "historical_reproducibility_check": {
                    "historical_gain_ci95": [0.006108, 0.009044],
                    "historical_seeds": list(HISTORICAL_CONFIRMATION_SEEDS),
                    "new_independent_seeds": list(CONFIRMATION_SEEDS),
                    "interpretation": (
                        "This is an independent cross-runtime reproducibility check, not a validity control. If the "
                        "new decay_0p80 estimate is nonpositive or a two-sample seed-bootstrap difference excludes "
                        "zero at 95%, report the bridge as failed and restrict claims to within-panel schedule effects."
                    ),
                },
            },
            "p2_surface_deformation": {
                "estimand": "per-coordinate arm-minus-decay_0p80 BPB over the common 125-coordinate grid",
                "test": "paired coordinate-level deformation versus aggregate weight and phase contrast",
            },
            "p3_schedule_specific_optima": {
                "selection": (
                    "within each complete arm, choose the minimum raw-grid tied coordinate and minimum raw-grid "
                    "eligible untied coordinate; do not use fitted-surface argmins"
                ),
                "confirmation": (
                    "run both selected coordinates on all eight confirmation seeds, deduplicating exact primary-spine "
                    "rows; report selection-inclusive uncertainty and do not pool the discovery seed"
                ),
                "gain": "selected tied BPB minus selected untied BPB; positive favors two-phase",
                "adaptive_inventory": (
                    "freeze a second hashed design after all four discovery arms are complete; it contains at most "
                    "64 new rows (4 arms x 2 selected policies x 8 seeds) before exact primary-spine deduplication"
                ),
            },
            "replication": {
                "support_id": REPLICATION_SUPPORT_ID,
                "tied_coordinate_id": REPLICATION_TIED_COORDINATE,
                "untied_coordinate_id": REPLICATION_UNTIED_COORDINATE,
                "historical_gain_ci95": [0.008333, 0.012641],
                "scope": (
                    "independent-block replication of the schedule-by-policy interaction; support and fixed "
                    "coordinates both change, so this is not a coordinate-matched support contrast"
                ),
            },
            "coordinate_distance": "descriptive_only",
            "multiple_testing": (
                "Holm family-wise correction at alpha=0.05 over the six additive-gain main-arm contrasts versus "
                "decay_0p80 (decay_0p60, decay_0p90, and no_decay in each of m100 and m200). The area-matched arm "
                "and log-gain secondary are sensitivity analyses without confirmatory significance claims."
            ),
        },
        "completeness_contract": {
            "valid_endpoint": (
                f"one finite {PRIMARY_METRIC} value at exact step {cell['total_steps'] - 1} for the frozen row identity"
            ),
            "surface_discovery": (
                "each main arm must contain all 125 unique coordinates before p2 or p3 is computed; missing or "
                "failed rows are retried with the exact frozen identity, never dropped or replaced"
            ),
            "paired_stages": (
                "each arm-policy block must contain all eight paired seeds before inference; no complete-case "
                "substitution or unpaired fallback is permitted"
            ),
            "failure_rule": "defer the affected estimand until complete rather than analyze a partial arm or seed block",
        },
        "checkpoint_contract": {
            "surface_discovery": "terminal permanent checkpoint only",
            "primary_spine_and_replication": "phase-boundary and terminal permanent checkpoints",
            "temporary_recovery": "time-based resumable checkpoints every ten minutes",
        },
        "reuse_contract": {
            "reuse_existing_rows": False,
            "reason": (
                "LR-onset trajectories use a training holdout and different seeds. Historical dense rows used JAX "
                "0.10.1 and a different launcher, artifact, and checkpoint contract. Re-run every row under one "
                "runtime; use historical results only for the independent reproducibility check."
            ),
        },
        "stage_counts": expected_stage_counts,
        "expected_run_count": len(rows),
        "runs": rows,
    }
    return payload


def _report(payload: dict[str, Any]) -> str:
    arm_lines = [
        (
            f"- `{arm['arm_id']}`: onset={arm['decay_onset_fraction']}, peak multiplier="
            f"{arm['peak_lr_multiplier']:.8f}, normalized total LR integral="
            f"{arm['optimizer']['normalized_lr_integral']:.3f}, phase-1 integral="
            f"{arm['optimizer']['normalized_phase_1_lr_integral']:.3f}."
        )
        for arm in payload["arms"]
    ]
    return "\n".join(
        [
            "# StarCoder WSD80 LR-onset dense-surface design",
            "",
            "## Decision",
            "",
            "Use the 7.408B-token, 1x-replay block because its historical paired-gain SD was 0.001182 BPB, versus "
            "0.005565 BPB in the 1.001B full-pool block. The headline test is a fixed-coordinate paired contrast; "
            "the dense surfaces test deformation and schedule-specific optima secondarily.",
            "",
            "## Inventory",
            "",
            f"- {payload['stage_counts']['primary_spine']} primary-spine rows.",
            f"- {payload['stage_counts']['surface_discovery']} dense-discovery rows.",
            f"- {payload['stage_counts']['replay_replication']} 2x-replay replication rows.",
            f"- {payload['expected_run_count']} total independently trained rows.",
            "- Up to 64 adaptive fresh-confirmation rows are frozen only after all four raw grids are complete.",
            "",
            "## Schedule arms",
            "",
            *arm_lines,
            "",
            "The LR-integral-matched arm is sensitivity-only: it changes peak LR and therefore does not isolate onset "
            "duration by itself.",
            "",
            "## Inference",
            "",
            "Positive gain means tied BPB minus untied BPB is positive. The primary is two-sided and uses c109 "
            "(0.70, 0.70) versus c020 (0.01008, 0.85968) on eight paired seeds. Surface optima are selected only "
            "from complete raw grids and require fresh confirmation; fitted-surface argmins and coordinate distance "
            "are not confirmatory.",
            "",
            "The scale-invariant secondary is log(BPB_tied / BPB_untied). The six main-arm additive-gain contrasts "
            "across m100 and m200 use Holm family-wise correction; the LR-integral-matched arm is sensitivity-only.",
            "All 125 endpoint metrics per discovery arm and all eight paired seeds per fixed-policy block are required. "
            "Missing rows are retried with the same identity; partial-arm and complete-case analyses are forbidden.",
            "The eight new confirmation seeds are disjoint from the five historical seeds. The 0.80T comparison is an "
            "independent cross-runtime reproducibility check, not a validity control.",
            "",
        ]
    )


def main() -> None:
    payload = build_payload()
    payload["design_sha256"] = canonical_sha256(payload)
    serialized = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    OUTPUT_PATH.write_bytes(gzip.compress(serialized, mtime=0))
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(_report(payload), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH} ({payload['expected_run_count']} rows, {payload['design_sha256']})")


if __name__ == "__main__":
    main()
