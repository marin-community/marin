# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Probe source-gradient geometry in the StarCoder WSD80 LR-onset intervention."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import fsspec
import jax
from fray.types import ResourceConfig
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, materialized_config, run
from marin.execution.remote import remote
from marin.training.training import TrainLmOnPodConfig

from experiments.domain_phase_mix import launch_starcoder_wsd80_lr_onset_intervention as training
from experiments.domain_phase_mix import starcoder_wsd80_gradient_mechanism_repair as mechanism
from experiments.domain_phase_mix import starcoder_wsd80_gradient_probe as probe

logger = logging.getLogger(__name__)

NAME = "pinlin_calvin_xu/data_mixture/starcoder_wsd80_lr_onset_gradient_probe_v1_20260823"
VERSION = "2026.08.23.1"
RESULT_ROOT = f"{training.MARIN_PREFIX}/analysis/{NAME}"
MAX_CONCURRENT = 64
TASK_IMAGE = "ghcr.io/marin-community/iris-task@sha256:c646ef8b571571edfc96c75fd9c8cc712ad286b61b33781070bdc29ab9f9a6ab"
FULL_LAUNCH_CONFIRMATION = "I_AUTHORIZE_THE_STARCODER_WSD80_LR_ONSET_GRADIENT_PROBES"
SOURCE_IDS = (mechanism.freeze.GLOBAL_STARCODER, mechanism.freeze.NEMOTRON)
HALVES = ("half_a", "half_b")
BLOCKS_PER_HALF = 16
UPDATE_DRAWS_PER_HALF = 4
PANEL_SEQUENCE_COUNT = 2 * BLOCKS_PER_HALF * probe.PROBE_BATCH_SIZE
PROBE_CHECKPOINT_STEPS = (
    training._step(0.55),
    training._step(0.70),
    training._step(0.80),
    training._step(0.90),
    training._step(0.95),
    training.TOTAL_STEPS - 1,
)
PREFLIGHT_CHECKPOINT_STEPS = PROBE_CHECKPOINT_STEPS
if not set(PROBE_CHECKPOINT_STEPS).issubset(training.CHECKPOINT_STEPS):
    raise ValueError("LR-onset probes require checkpoints retained by the training panel")

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).parent / "manifests/starcoder_wsd80_lr_onset_gradient_probe_v1_20260823"
MANIFEST_PATH = OUTPUT_DIR / "probe_manifest.json"
CONTRACT_PATH = OUTPUT_DIR / "analysis_contract.json"
RELEASE_PATH = OUTPUT_DIR / "release.json"
CC_REVIEW_PATH = REPO_ROOT / ".agents/handoffs/starcoder_wsd80_lr_onset_gradient_probe_cc_review_20260823.md"


@dataclass(frozen=True)
class ProbeRow:
    row_id: str
    trajectory_id: str
    arm: str
    training_seed: int
    checkpoint_step: int
    expected_restored_state_step: int
    normalized_time: float
    steps_from_decay_onset: int | None
    checkpoint_uri: str
    train_config_sha256: str
    expected_learning_rate: float
    expected_adam_lr: float
    expected_state_equivalence_class: str
    starcoder_sequence_set_id: str
    nemotron_sequence_set_id: str
    half_a_sequence_offset: int
    half_b_sequence_offset: int


@dataclass(frozen=True)
class ProbeGroupConfig:
    row: dict[str, Any]
    pod_config: TrainLmOnPodConfig
    output_path: str
    release_sha256: str


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256(path.read_bytes())


def _write_remote_json(path: str, payload: dict[str, Any]) -> None:
    encoded = (_canonical_json(payload) + "\n").encode()
    fs, plain_path = fsspec.core.url_to_fs(path)
    fs.makedirs(os.path.dirname(plain_path), exist_ok=True)
    try:
        with fs.open(plain_path, "xb") as handle:
            handle.write(encoded)
    except FileExistsError as error:
        with fs.open(plain_path, "rb") as handle:
            if handle.read() != encoded:
                raise RuntimeError(f"Existing probe output differs: {path}") from error


def _read_remote_json(path: str) -> dict[str, Any] | None:
    fs, plain_path = fsspec.core.url_to_fs(path)
    if not fs.exists(plain_path):
        return None
    with fs.open(plain_path, "rb") as handle:
        document = json.load(handle)
    payload_sha256 = document.get("payload_sha256")
    expected = _sha256(
        _canonical_json({key: value for key, value in document.items() if key != "payload_sha256"}).encode()
    )
    if payload_sha256 != expected:
        raise RuntimeError(f"Existing probe output failed its payload hash: {path}")
    return document


def _row_identity(row: Mapping[str, Any], release_sha256: str) -> str:
    return _sha256(_canonical_json({"row": dict(row), "release_sha256": release_sha256}).encode())


def _sequence_set_id(source: str, training_seed: int) -> str:
    if source == mechanism.freeze.NEMOTRON:
        return f"starcoder-wsd80-lr-onset-fixed-reference-v1/{source}/training-seed-{training_seed}"
    return f"starcoder-wsd80-lr-onset-fixed-reference-v1/{source}/global"


def _expected_state_equivalence_class(trajectory: training.Trajectory, checkpoint_step: int) -> str:
    onset = trajectory.optimizer_decay_step
    if onset is None or checkpoint_step <= onset:
        return "peak_lr_prefix"
    return trajectory.arm


def _expected_learning_rates(trajectory: training.Trajectory, checkpoint_step: int) -> tuple[float, float]:
    arm = training._arm_by_name()[trajectory.arm]
    optimizer = training._optimizer(arm)
    learning_rate = float(optimizer.lr_scheduler(training.TOTAL_STEPS)(checkpoint_step))
    adam_lr = float(optimizer.lr_scheduler(training.TOTAL_STEPS, override_lr=optimizer.adam_lr)(checkpoint_step))
    return learning_rate, adam_lr


def _rows_and_configs() -> tuple[list[ProbeRow], dict[str, TrainLmOnPodConfig]]:
    trajectories, steps = training.build_training_steps()
    training.audit_runtime_configs(trajectories, steps)
    artifact_cache: dict[int, Any] = {}
    rows = []
    configs = {}
    for trajectory, step in zip(trajectories, steps, strict=True):
        pod_config = cast(
            TrainLmOnPodConfig,
            materialized_config(step, training.MARIN_PREFIX, artifact_cache=artifact_cache),
        )
        config_sha256 = probe.freeze._config_identity(pod_config)["full_train_config_sha256"]
        configs[trajectory.trajectory_id] = pod_config
        for checkpoint_step in PROBE_CHECKPOINT_STEPS:
            expected_learning_rate, expected_adam_lr = _expected_learning_rates(trajectory, checkpoint_step)
            row_identity = {
                "trajectory_id": trajectory.trajectory_id,
                "checkpoint_step": checkpoint_step,
                "reference_panel": "fixed_16_plus_16_per_source_v1",
            }
            rows.append(
                ProbeRow(
                    row_id=f"lr_onset_probe_{_sha256(_canonical_json(row_identity).encode())[:24]}",
                    trajectory_id=trajectory.trajectory_id,
                    arm=trajectory.arm,
                    training_seed=trajectory.training_seed,
                    checkpoint_step=checkpoint_step,
                    expected_restored_state_step=probe.freeze.expected_restored_state_step(checkpoint_step),
                    normalized_time=checkpoint_step / training.TOTAL_STEPS,
                    steps_from_decay_onset=(
                        None
                        if trajectory.optimizer_decay_step is None
                        else checkpoint_step - trajectory.optimizer_decay_step
                    ),
                    checkpoint_uri=f"{step.path(training.MARIN_PREFIX)}/checkpoints/step-{checkpoint_step}",
                    train_config_sha256=config_sha256,
                    expected_learning_rate=expected_learning_rate,
                    expected_adam_lr=expected_adam_lr,
                    expected_state_equivalence_class=_expected_state_equivalence_class(trajectory, checkpoint_step),
                    starcoder_sequence_set_id=_sequence_set_id(SOURCE_IDS[0], trajectory.training_seed),
                    nemotron_sequence_set_id=_sequence_set_id(SOURCE_IDS[1], trajectory.training_seed),
                    half_a_sequence_offset=0,
                    half_b_sequence_offset=BLOCKS_PER_HALF * probe.PROBE_BATCH_SIZE,
                )
            )
    if len(rows) != 32 * len(PROBE_CHECKPOINT_STEPS):
        raise ValueError("Probe row inventory drifted")
    if len({row.row_id for row in rows}) != len(rows):
        raise ValueError("Probe row identities are not unique")
    for seed in training.TRAINING_SEEDS:
        seed_hashes = {
            row.train_config_sha256
            for row in rows
            if row.training_seed == seed and row.checkpoint_step == PROBE_CHECKPOINT_STEPS[0]
        }
        if len(seed_hashes) != len(training.ARMS):
            raise ValueError(f"Training seed {seed} does not expose four distinct arm configurations")
    return sorted(rows, key=lambda row: row.row_id), configs


def _analysis_contract() -> dict[str, Any]:
    return {
        "schema_version": "2026-08-23-starcoder-wsd80-lr-onset-probe-v1",
        "question": "Does raw StarCoder-Nemotron gradient-cosine decline follow LR-decay onset?",
        "primary_statistic": (
            "split-half disattenuated cosine between mean heldout StarCoder and frozen Nemotron raw gradients, "
            "restricted to the projected trainable trunk geometry used by the existing onset artifact"
        ),
        "primary_json_path": "noise_corrected_source_gradient_statistics.projected.trunk.disattenuated_cosine",
        "uncorrected_primary_json_path": "combined_source_gradient_statistics.projected.trunk.cosine",
        "unprojected_sensitivity_json_path": "noise_corrected_source_gradient_statistics.raw.trunk.disattenuated_cosine",
        "secondary_statistic": (
            "projected-trunk cosine between corrected optimizer updates, descriptive only when realized LR "
            "is below 1e-3 of peak"
        ),
        "reference_panel": {
            "sources": list(SOURCE_IDS),
            "halves": list(HALVES),
            "blocks_per_half": BLOCKS_PER_HALF,
            "sequences_per_block": probe.PROBE_BATCH_SIZE,
            "optimizer_update_draws_per_half": UPDATE_DRAWS_PER_HALF,
            "panel_sequence_count": PANEL_SEQUENCE_COUNT,
            "half_sequence_offsets": [0, BLOCKS_PER_HALF * probe.PROBE_BATCH_SIZE],
            "halves_are_non_overlapping_within_each_seed": True,
            "sequence_set_ids_shared_across_arms_and_checkpoints": True,
            "paired_by_training_seed": True,
        },
        "primary_contrast": "no_decay_minus_decay_0p60_at_final_checkpoint, paired within training seed",
        "primary_test": "two-sided exact Wilcoxon signed-rank test across 8 paired training seeds at alpha=0.05",
        "secondary_family": [
            "matched_decay_fraction_0p5: decay_0p60@0p80T vs decay_0p80@0p90T vs decay_0p90@0p95T",
            "matched_time_0p90T: all four arms",
            "onset_anchored_0p90: (0p90T-final) vs the same no_decay window",
            "no_decay_stability: 0p55T-final",
        ],
        "multiplicity": "Holm family-wise correction across the four secondary tests only",
        "identification_checks": {
            "split_half_reliability_threshold": 0.5,
            "low_reliability_interpretation": "primary inconclusive",
            "pipeline_health_reliability_gate_arms": ["no_decay"],
            "decay_arm_reliability_is_advisory": True,
            "state_identity_partition_is_frozen_per_seed_checkpoint": True,
            "frozen_learning_rates_are_checked_against_restored_optimizer_state": True,
            "reference_first_batch_hash_is_constant_across_arms_and_checkpoints": True,
        },
        "secondary_time_axes": [
            "normalized_training_time",
            "steps_since_decay_onset",
            "fraction_of_decay_elapsed",
            "restored_learning_rate_ratio",
            "cumulative_learning_rate",
        ],
        "endpoint_metrics_read": False,
    }


def freeze_release() -> dict[str, Any]:
    if not CC_REVIEW_PATH.is_file() or not CC_REVIEW_PATH.read_text().rstrip().endswith("VERDICT: PASS"):
        raise ValueError("Probe release requires a CC review ending in VERDICT: PASS")
    rows, _ = _rows_and_configs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps([asdict(row) for row in rows], indent=2, sort_keys=True) + "\n")
    CONTRACT_PATH.write_text(json.dumps(_analysis_contract(), indent=2, sort_keys=True) + "\n")
    training_release = json.loads(training.RELEASE_PATH.read_text())
    implementation_paths = tuple(
        Path(module.__file__).resolve() for module in (training, mechanism, mechanism.freeze, probe, probe.freeze)
    )
    release = {
        "release_sha256": "",
        "release_version": VERSION,
        "runtime_path": str(Path(__file__).relative_to(REPO_ROOT)),
        "runtime_sha256": _file_sha256(Path(__file__)),
        "implementation_files": {str(path.relative_to(REPO_ROOT)): _file_sha256(path) for path in implementation_paths},
        "manifest_path": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
        "manifest_sha256": _file_sha256(MANIFEST_PATH),
        "manifest_row_count": len(rows),
        "analysis_contract_path": str(CONTRACT_PATH.relative_to(REPO_ROOT)),
        "analysis_contract_sha256": _file_sha256(CONTRACT_PATH),
        "cc_review_path": str(CC_REVIEW_PATH.relative_to(REPO_ROOT)),
        "cc_review_sha256": _file_sha256(CC_REVIEW_PATH),
        "training_release_path": str(training.RELEASE_PATH.relative_to(REPO_ROOT)),
        "training_release_sha256": training_release["release_sha256"],
        "training_release_file_sha256": _file_sha256(training.RELEASE_PATH),
        "result_root": RESULT_ROOT,
        "task_image": TASK_IMAGE,
        "maximum_concurrent": MAX_CONCURRENT,
        "confirmation": FULL_LAUNCH_CONFIRMATION,
        "endpoint_metrics_read": False,
    }
    release["release_sha256"] = _sha256(_canonical_json(release).encode())
    RELEASE_PATH.write_text(json.dumps(release, indent=2, sort_keys=True) + "\n")
    return release


def _load_release() -> dict[str, Any]:
    release = json.loads(RELEASE_PATH.read_text())
    claimed = release["release_sha256"]
    if claimed != _sha256(_canonical_json({**release, "release_sha256": ""}).encode()):
        raise ValueError("Probe release payload hash drifted")
    checks = {
        release["runtime_path"]: release["runtime_sha256"],
        release["manifest_path"]: release["manifest_sha256"],
        release["analysis_contract_path"]: release["analysis_contract_sha256"],
        release["cc_review_path"]: release["cc_review_sha256"],
        release["training_release_path"]: release["training_release_file_sha256"],
        **release["implementation_files"],
    }
    drifted = [path for path, expected in checks.items() if _file_sha256(REPO_ROOT / path) != expected]
    if drifted:
        raise ValueError(f"Probe release files drifted: {drifted}")
    training_release = training._load_release()
    if training_release["release_sha256"] != release["training_release_sha256"]:
        raise ValueError("Training release identity drifted")
    frozen_rows = json.loads((REPO_ROOT / release["manifest_path"]).read_text())
    live_rows, _ = _rows_and_configs()
    if frozen_rows != [asdict(row) for row in live_rows]:
        raise ValueError("Frozen probe manifest differs from the executable row inventory")
    return release


def _source_half_key(source: str, half: str) -> str:
    return f"{source}/{half}"


def _sequence_set_for_row(row: Mapping[str, Any], source: str) -> str:
    prefix = "starcoder" if source == SOURCE_IDS[0] else "nemotron"
    return str(row[f"{prefix}_sequence_set_id"])


def _half_offset(row: Mapping[str, Any], half: str) -> int:
    return int(row[f"{half}_sequence_offset"])


def _mean_pair(left: Any, right: Any) -> Any:
    return probe._tree_scale(probe._tree_add(left, right), 0.5)


def _noise_corrected_statistics(
    combined: Mapping[str, Any],
    left_halves: Mapping[str, Any],
    right_halves: Mapping[str, Any],
) -> dict[str, Any]:
    result = {}
    for geometry in sorted(set(combined) & set(left_halves) & set(right_halves)):
        groups = sorted(set(combined[geometry]) & set(left_halves[geometry]) & set(right_halves[geometry]))
        result[geometry] = {}
        for group in groups:
            combined_group = combined[geometry][group]
            left_group = left_halves[geometry][group]
            right_group = right_halves[geometry][group]
            numerator = float(combined_group["dot"])
            left_signal_sq = float(left_group["dot"])
            right_signal_sq = float(right_group["dot"])
            signals_defined = left_signal_sq > 0.0 and right_signal_sq > 0.0
            denominator = math.sqrt(left_signal_sq * right_signal_sq) if signals_defined else 0.0
            left_mean_sq = float(combined_group["left_norm"]) ** 2
            right_mean_sq = float(combined_group["right_norm"]) ** 2
            left_noise_sq = max(left_mean_sq - left_signal_sq, 0.0)
            right_noise_sq = max(right_mean_sq - right_signal_sq, 0.0)
            left_reliability = left_group["cosine"]
            right_reliability = right_group["cosine"]
            result[geometry][group] = {
                "disattenuated_cosine": numerator / denominator if denominator > 0.0 else None,
                "defined": signals_defined,
                "combined_cross_source_dot": numerator,
                "left_split_half_signal_sq": left_signal_sq,
                "right_split_half_signal_sq": right_signal_sq,
                "left_split_half_reliability": left_reliability,
                "right_split_half_reliability": right_reliability,
                "left_spearman_brown_reliability": (
                    2.0 * float(left_reliability) / (1.0 + float(left_reliability))
                    if left_reliability is not None and float(left_reliability) > -1.0
                    else None
                ),
                "right_spearman_brown_reliability": (
                    2.0 * float(right_reliability) / (1.0 + float(right_reliability))
                    if right_reliability is not None and float(right_reliability) > -1.0
                    else None
                ),
                "left_signal_to_noise": left_signal_sq / left_noise_sq if left_noise_sq > 0.0 else None,
                "right_signal_to_noise": right_signal_sq / right_noise_sq if right_noise_sq > 0.0 else None,
            }
    return result


def _runtime_observation() -> dict[str, Any]:
    return {
        "backend": jax.default_backend(),
        "device_count": len(jax.devices()),
        "local_device_count": jax.local_device_count(),
        "device_kinds": sorted({str(device.device_kind) for device in jax.devices()}),
        "probe_batch_size": probe.PROBE_BATCH_SIZE,
    }


def _learning_rate_tolerance(expected: float) -> float:
    return max(4e-9, 1e-5 * abs(expected))


def run_probe_group(config: ProbeGroupConfig) -> None:
    row = config.row
    row_path = f"{config.output_path}/result.json"
    marker_path = f"{config.output_path}/complete.json"
    identity = _row_identity(row, config.release_sha256)
    if not config.output_path.startswith(RESULT_ROOT):
        raise ValueError(f"Probe output escaped the central1 result root: {config.output_path}")
    existing = _read_remote_json(marker_path)
    if existing is not None:
        result = _read_remote_json(row_path)
        if (
            existing.get("identity_sha256") != identity
            or existing.get("release_sha256") != config.release_sha256
            or result is None
            or result.get("identity_sha256") != identity
            or existing.get("result_payload_sha256") != result.get("payload_sha256")
        ):
            raise RuntimeError("Completed probe marker belongs to another row identity")
        return
    existing_result = _read_remote_json(row_path)
    if existing_result is not None:
        if (
            existing_result.get("identity_sha256") != identity
            or existing_result.get("release_sha256") != config.release_sha256
        ):
            raise RuntimeError("Existing unmarked probe result belongs to another row identity")
        marker = {
            "schema_version": "2026-08-23-starcoder-wsd80-lr-onset-probe-complete-v1",
            "identity_sha256": identity,
            "release_sha256": config.release_sha256,
            "row_id": row["row_id"],
            "result_payload_sha256": existing_result["payload_sha256"],
        }
        marker["payload_sha256"] = _sha256(_canonical_json(marker).encode())
        _write_remote_json(marker_path, marker)
        return
    if probe.freeze._config_identity(config.pod_config)["full_train_config_sha256"] != row["train_config_sha256"]:
        raise ValueError("Probe training configuration drifted")
    metadata = probe._read_checkpoint_metadata(row["checkpoint_uri"], int(row["checkpoint_step"]))
    train_config = probe._prepare_train_config(config.pod_config, row["checkpoint_uri"], row["row_id"])
    trainer, state, Pos, data_key, optimizer_mask = probe._initialize_runtime(train_config)
    try:
        if int(state.step) != int(row["expected_restored_state_step"]):
            raise RuntimeError("Probe restored the wrong checkpoint state")
        learning_rates = training._restored_hyperparameters(state, train_config)
        for field in ("learning_rate", "adam_lr"):
            frozen_expected = float(row[f"expected_{field}"])
            if abs(float(learning_rates["observed"][field]) - frozen_expected) > _learning_rate_tolerance(
                frozen_expected
            ):
                raise RuntimeError(
                    f"Restored {field} differs from the frozen probe manifest: "
                    f"{learning_rates['observed'][field]} != {frozen_expected}"
                )
        source_views, stream_summary = probe._source_views(train_config, Pos, data_key, int(state.step))
        gradient_fn, update_fn, _, _ = probe._gradient_functions(trainer)
        gradients = {}
        updates = {}
        no_data_updates = {}
        summaries = {}
        for source in SOURCE_IDS:
            base_dataset = probe._distribution_dataset(
                source,
                sequence_set_id=_sequence_set_for_row(row, source),
                train_config=train_config,
                Pos=Pos,
                sources=source_views,
            )
            for half in HALVES:
                key = _source_half_key(source, half)
                dataset = probe.ShiftedRestartDataset(
                    base_dataset,
                    start=_half_offset(row, half),
                    length=PANEL_SEQUENCE_COUNT,
                )
                gradient, update, no_data_update, summary = mechanism._mean_gradient_and_updates(
                    trainer=trainer,
                    state=state,
                    dataset=dataset,
                    blocks=BLOCKS_PER_HALF,
                    update_draws=UPDATE_DRAWS_PER_HALF,
                    seed_id=f"{_sequence_set_for_row(row, source)}/{half}",
                    gradient_fn=gradient_fn,
                    update_fn=update_fn,
                )
                gradients[key] = gradient
                updates[key] = update
                no_data_updates[key] = no_data_update
                summaries[key] = summary
        no_data_audit = mechanism._assert_common_no_data_update(no_data_updates, summaries)
        half_statistics = {
            half: mechanism._statistics_bundle(
                gradients[_source_half_key(SOURCE_IDS[0], half)],
                gradients[_source_half_key(SOURCE_IDS[1], half)],
                model=state.model,
                optimizer_mask=optimizer_mask,
            )
            for half in HALVES
        }
        half_update_statistics = {
            half: mechanism._statistics_bundle(
                updates[_source_half_key(SOURCE_IDS[0], half)],
                updates[_source_half_key(SOURCE_IDS[1], half)],
                model=state.model,
                optimizer_mask=optimizer_mask,
            )
            for half in HALVES
        }
        combined_gradients = {
            source: _mean_pair(
                gradients[_source_half_key(source, HALVES[0])],
                gradients[_source_half_key(source, HALVES[1])],
            )
            for source in SOURCE_IDS
        }
        combined_updates = {
            source: _mean_pair(
                updates[_source_half_key(source, HALVES[0])],
                updates[_source_half_key(source, HALVES[1])],
            )
            for source in SOURCE_IDS
        }
        within_source_half_agreement = {
            source: mechanism._statistics_bundle(
                gradients[_source_half_key(source, HALVES[0])],
                gradients[_source_half_key(source, HALVES[1])],
                model=state.model,
                optimizer_mask=optimizer_mask,
            )
            for source in SOURCE_IDS
        }
        within_source_update_half_agreement = {
            source: mechanism._statistics_bundle(
                updates[_source_half_key(source, HALVES[0])],
                updates[_source_half_key(source, HALVES[1])],
                model=state.model,
                optimizer_mask=optimizer_mask,
            )
            for source in SOURCE_IDS
        }
        combined_gradient_statistics = mechanism._statistics_bundle(
            combined_gradients[SOURCE_IDS[0]],
            combined_gradients[SOURCE_IDS[1]],
            model=state.model,
            optimizer_mask=optimizer_mask,
        )
        combined_update_statistics = mechanism._statistics_bundle(
            combined_updates[SOURCE_IDS[0]],
            combined_updates[SOURCE_IDS[1]],
            model=state.model,
            optimizer_mask=optimizer_mask,
        )
        payload = {
            "schema_version": "2026-08-23-starcoder-wsd80-lr-onset-probe-result-v1",
            "identity_sha256": identity,
            "release_sha256": config.release_sha256,
            "row": row,
            "checkpoint_metadata": metadata,
            "restored_state_step": int(state.step),
            "restored_learning_rates": learning_rates,
            "optimizer_schedule": probe._optimizer_schedule_summary(train_config),
            "restored_optimizer": probe._restored_optimizer_summary(
                state,
                int(row["checkpoint_step"]),
                int(row["expected_restored_state_step"]),
                allow_partial_checkpoint=train_config.trainer.allow_partial_checkpoint,
            ),
            "state_fingerprint": {
                "model_sha256": probe._tree_sha256(state.model),
                "optimizer_state_sha256": probe._tree_sha256(state.opt_state),
                "training_key_sha256": probe._tree_sha256(state.training_key),
            },
            "expected_state_equivalence_class": row["expected_state_equivalence_class"],
            "source_stream": stream_summary,
            "numerical_summaries": summaries,
            "no_data_update_invariance": no_data_audit,
            "half_source_gradient_statistics": half_statistics,
            "half_source_optimizer_update_statistics": half_update_statistics,
            "combined_source_gradient_statistics": combined_gradient_statistics,
            "combined_source_optimizer_update_statistics": combined_update_statistics,
            "within_source_gradient_half_agreement": within_source_half_agreement,
            "within_source_optimizer_update_half_agreement": within_source_update_half_agreement,
            "noise_corrected_source_gradient_statistics": _noise_corrected_statistics(
                combined_gradient_statistics,
                within_source_half_agreement[SOURCE_IDS[0]],
                within_source_half_agreement[SOURCE_IDS[1]],
            ),
            "runtime_observation": _runtime_observation(),
            "muon_projection_coverage": probe._runtime_muon_projection_coverage(state.model, optimizer_mask),
            "endpoint_metrics_read": False,
        }
        payload["payload_sha256"] = _sha256(_canonical_json(payload).encode())
        _write_remote_json(row_path, payload)
        marker = {
            "schema_version": "2026-08-23-starcoder-wsd80-lr-onset-probe-complete-v1",
            "identity_sha256": identity,
            "release_sha256": config.release_sha256,
            "row_id": row["row_id"],
            "result_payload_sha256": payload["payload_sha256"],
        }
        marker["payload_sha256"] = _sha256(_canonical_json(marker).encode())
        _write_remote_json(marker_path, marker)
    finally:
        probe._close_runtime(trainer)


def _steps(release: Mapping[str, Any], stage: int) -> list[ArtifactStep[Artifact]]:
    rows, configs = _rows_and_configs()
    if stage == 0:
        rows = [
            row
            for row in rows
            if row.training_seed == training.TRAINING_SEEDS[0] and row.checkpoint_step in PREFLIGHT_CHECKPOINT_STEPS
        ]
    resources = replace(
        ResourceConfig.with_tpu(
            training.TPU_TYPE,
            cpu=training.historical.TPU_HOST_CPU,
            ram=training.historical.TPU_HOST_RAM,
            regions=(training.TPU_REGION,),
            zone=training.TPU_ZONE,
        ),
        image=TASK_IMAGE,
    )
    prefix = f"{training.MARIN_PREFIX}/"
    result = []
    for row in rows:
        row_payload = asdict(row)
        config = ProbeGroupConfig(
            row=row_payload,
            pod_config=configs[row.trajectory_id],
            output_path="",
            release_sha256=str(release["release_sha256"]),
        )
        result.append(
            ArtifactStep(
                name=f"{RESULT_ROOT.removeprefix(prefix)}/{row.row_id}",
                version=VERSION,
                artifact_type=Artifact,
                run=remote(run_probe_group, resources=resources, name=row.row_id),
                build_config=lambda ctx, config=config: replace(config, output_path=ctx.output_path),
            )
        )
    return result


def _selected_rows(stage: int) -> list[ProbeRow]:
    rows, _ = _rows_and_configs()
    if stage == 0:
        return [
            row
            for row in rows
            if row.training_seed == training.TRAINING_SEEDS[0] and row.checkpoint_step in PREFLIGHT_CHECKPOINT_STEPS
        ]
    return rows


def _result_path(row: ProbeRow) -> str:
    return f"{RESULT_ROOT}/{row.row_id}/{VERSION}/result.json"


def audit_outputs(stage: int, release: Mapping[str, Any]) -> dict[str, Any]:
    rows = _selected_rows(stage)
    documents = {}
    missing = []
    for row in rows:
        document = _read_remote_json(_result_path(row))
        if document is None:
            missing.append(row.row_id)
            continue
        if document.get("identity_sha256") != _row_identity(asdict(row), str(release["release_sha256"])):
            raise RuntimeError(f"Probe result identity drifted: {row.row_id}")
        documents[row.row_id] = document

    identity_failures = []
    reference_hashes: dict[tuple[int, str, str], set[str]] = {}
    grouped: dict[tuple[int, int], list[tuple[ProbeRow, dict[str, Any]]]] = {}
    for row in rows:
        document = documents.get(row.row_id)
        if document is None:
            continue
        grouped.setdefault((row.training_seed, row.checkpoint_step), []).append((row, document))
        for source in SOURCE_IDS:
            for half in HALVES:
                summary = document["numerical_summaries"][_source_half_key(source, half)]
                reference_hashes.setdefault((row.training_seed, source, half), set()).add(
                    str(summary["first_batch_sha256"])
                )

    for (seed, checkpoint_step), group in sorted(grouped.items()):
        if len(group) != len(training.ARMS):
            identity_failures.append(f"seed={seed} step={checkpoint_step}: incomplete arm inventory")
            continue
        training_keys = {document["state_fingerprint"]["training_key_sha256"] for _, document in group}
        if len(training_keys) != 1:
            identity_failures.append(f"seed={seed} step={checkpoint_step}: training keys differ across arms")
        expected_classes = {row.expected_state_equivalence_class for row, _ in group}
        for field in ("model_sha256", "optimizer_state_sha256"):
            hashes_by_class: dict[str, set[str]] = {}
            for row, document in group:
                hashes_by_class.setdefault(row.expected_state_equivalence_class, set()).add(
                    str(document["state_fingerprint"][field])
                )
            if any(len(hashes) != 1 for hashes in hashes_by_class.values()):
                identity_failures.append(
                    f"seed={seed} step={checkpoint_step}: {field} differs within an expected identity class"
                )
            class_hashes = {next(iter(hashes)) for hashes in hashes_by_class.values() if hashes}
            if len(class_hashes) != len(expected_classes):
                identity_failures.append(
                    f"seed={seed} step={checkpoint_step}: {field} does not realize the frozen identity partition"
                )

    reference_failures = [
        f"seed={seed} source={source} half={half}: {len(hashes)} first-batch hashes"
        for (seed, source, half), hashes in sorted(reference_hashes.items())
        if len(hashes) != 1
    ]
    final_step = training.TOTAL_STEPS - 1
    reliability_gate_failures = []
    decay_reliability_advisories = []
    for row in rows:
        if row.checkpoint_step != final_step or row.row_id not in documents:
            continue
        statistic = documents[row.row_id]["noise_corrected_source_gradient_statistics"]["projected"]["trunk"]
        for side in ("left", "right"):
            value = statistic[f"{side}_split_half_reliability"]
            if value is None or float(value) < 0.5:
                finding = f"{row.row_id}:{side}={value}"
                if row.arm == "no_decay":
                    reliability_gate_failures.append(finding)
                else:
                    decay_reliability_advisories.append(finding)

    complete = not missing
    passed = complete and not identity_failures and not reference_failures and not reliability_gate_failures
    return {
        "expected_row_count": len(rows),
        "completed_row_count": len(documents),
        "missing_count": len(missing),
        "missing_examples": missing[:16],
        "identity_failures": identity_failures,
        "reference_batch_failures": reference_failures,
        "split_half_reliability_gate_failures": reliability_gate_failures,
        "decay_arm_reliability_advisories": decay_reliability_advisories,
        "complete": complete,
        "passed": passed,
        "endpoint_metrics_read": False,
    }


def audit_readiness(stage: int) -> dict[str, Any]:
    release = _load_release()
    rows = _selected_rows(stage)
    missing = []
    for row in rows:
        try:
            probe._read_checkpoint_metadata(row.checkpoint_uri, row.checkpoint_step)
        except (FileNotFoundError, ValueError):
            missing.append(row.row_id)
    result = {
        "release_sha256": release["release_sha256"],
        "row_count": len(rows),
        "missing_count": len(missing),
        "missing_examples": missing[:16],
        "endpoint_metrics_read": False,
    }
    if stage == 1:
        result["stage0_output_audit"] = audit_outputs(0, release)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--stage", type=int, choices=(0, 1))
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT)
    parser.add_argument("--confirmation")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    if args.freeze:
        release = freeze_release()
        logger.info("Frozen LR-onset probe release %s", release["release_sha256"])
        return
    release = _load_release()
    if args.stage is None:
        raise ValueError("Probe execution requires --stage 0 or --stage 1")
    if args.max_concurrent < 1 or args.max_concurrent > int(release["maximum_concurrent"]):
        raise ValueError("Probe concurrency is outside the frozen release")
    readiness = audit_readiness(args.stage)
    if args.audit:
        print(
            json.dumps(
                {"checkpoint_readiness": readiness, "output_audit": audit_outputs(args.stage, release)},
                indent=2,
                sort_keys=True,
            )
        )
        return
    if readiness["missing_count"]:
        raise RuntimeError(f"Probe checkpoint readiness failed: {readiness}")
    if args.stage == 1 and not readiness["stage0_output_audit"]["passed"]:
        raise RuntimeError(f"Probe stage-0 acceptance failed: {readiness['stage0_output_audit']}")
    if args.confirmation != release["confirmation"]:
        raise ValueError("Probe launch confirmation is missing or incorrect")
    if os.getenv("MARIN_PREFIX", training.MARIN_PREFIX) != training.MARIN_PREFIX:
        raise ValueError("LR-onset probes must remain central1-local")
    os.environ["MARIN_PREFIX"] = training.MARIN_PREFIX
    steps = _steps(release, args.stage)
    run(*steps, max_concurrent=min(args.max_concurrent, len(steps)), force_run_failed=True)


if __name__ == "__main__":
    main()
