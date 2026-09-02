# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the StarCoder WSD80 learning-rate onset intervention in central1."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

import fsspec
import jax
import numpy as np
from fray.types import ResourceConfig
from levanter.main.train_lm import TrainLmConfig
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, lower, materialized_config, run
from marin.execution.remote import remote
from marin.experiment.train import train_lm
from marin.training.training import LevanterCheckpoint, TrainLmOnPodConfig

from experiments.domain_phase_mix import launch_starcoder_wsd80_gradient_conflict_full as historical
from experiments.domain_phase_mix import launch_starcoder_wsd_80_20_surface as base
from experiments.domain_phase_mix import starcoder_wsd80_gradient_probe as gradient_probe
from experiments.llama import llama3_tokenizer_vocab_size
from experiments.scaling_law_sweeps.completed_adamh import CompletedAdamHHeuristic

logger = logging.getLogger(__name__)

NAME = "pinlin_calvin_xu/data_mixture/starcoder_wsd80_lr_onset_intervention_v1_20260823"
VERSION = "2026.08.23.1"
WANDB_GROUP = "starcoder_wsd80_lr_onset_intervention_v1_20260823"
PANEL_TAG = "starcoder_wsd80_lr_onset_intervention"
MARIN_PREFIX = "gs://marin-us-central1"
TPU_TYPE = "v5p-8"
TPU_REGION = "us-central1"
TPU_ZONE = "us-central1-a"
MAX_CONCURRENT = 32
FULL_LAUNCH_CONFIRMATION = "I_AUTHORIZE_THE_STARCODER_WSD80_LR_ONSET_INTERVENTION"

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).parent / "manifests/starcoder_wsd80_lr_onset_intervention_v1_20260823"
TRAJECTORY_MANIFEST_PATH = OUTPUT_DIR / "trajectory_manifest.json"
CHECKPOINT_MANIFEST_PATH = OUTPUT_DIR / "checkpoint_manifest.json"
DESIGN_CONTRACT_PATH = OUTPUT_DIR / "design_contract.json"
RELEASE_PATH = OUTPUT_DIR / "release.json"
CC_REVIEW_PATH = REPO_ROOT / ".agents/handoffs/starcoder_wsd80_lr_onset_intervention_cc_review_20260823.md"
STAGE0_VALIDATION_NAME = f"analysis/{NAME}/stage0_validation"
STAGE0_VALIDATION_VERSION = "2026.08.23.1"

TOTAL_STEPS = 3_820
MATERIALIZED_TOKENS = TOTAL_STEPS * base.BATCH_SIZE * base.SEQ_LEN
HIDDEN_SIZE = 640
EXPECTED_TOTAL_PARAMETERS = 210_052_480
EXPECTED_NON_EMBEDDING_PARAMETERS = 45_884_800
STARCODER_WEIGHT = 0.35
TRAINING_SEEDS = tuple(range(2_026_081_000, 2_026_081_008))
HOLDOUT_SEQUENCES_PER_COMPONENT = 4_096
HOLDOUT_SEED = 2_026_081_102
HOLDOUT_PARTITION = "random_sparse_swap"


@dataclass(frozen=True)
class ScheduleArm:
    arm: str
    decay_onset_fraction: float | None
    description: str


@dataclass(frozen=True)
class Trajectory:
    trajectory_id: str
    arm: str
    training_seed: int
    total_steps: int
    optimizer_decay_step: int | None
    support_pool_seed: int | None
    support_start_batches: int | None
    support_batches: int | None
    train_holdout_sequences_per_component: int
    train_holdout_seed: int
    train_holdout_partition: str


@dataclass(frozen=True)
class Stage0CheckpointValidationConfig:
    arm: str
    checkpoint_uri: str
    checkpoint_step: int
    expected_restored_state_step: int
    pod_config: TrainLmOnPodConfig
    output_path: str
    release_sha256: str


@dataclass(frozen=True)
class Stage0AggregateValidationConfig:
    arm_outputs: dict[str, str]
    expected_arms: tuple[str, ...]
    expected_learning_rate: float
    checkpoint_step: int
    output_path: str
    release_sha256: str


ARMS = (
    ScheduleArm("decay_0p60", 0.60, "Cosine LR decay begins at 0.60T and reaches zero at T."),
    ScheduleArm("decay_0p80", 0.80, "Cosine LR decay begins at 0.80T and reaches zero at T."),
    ScheduleArm("decay_0p90", 0.90, "Cosine LR decay begins at 0.90T and reaches zero at T."),
    ScheduleArm(
        "no_decay",
        None,
        "Use the historical 0.80T schedule boundary but hold peak LR constant through T.",
    ),
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _step(fraction: float) -> int:
    return round(TOTAL_STEPS * fraction)


def _checkpoint_steps() -> tuple[int, ...]:
    fractions = (0.10, 0.25, 0.40, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
    local_offsets = {
        _step(0.60) - 64,
        _step(0.60) + 64,
        _step(0.80) - 64,
        _step(0.80) + 64,
        _step(0.80) + 256,
        _step(0.90) - 64,
        _step(0.90) + 64,
    }
    steps = {*(_step(fraction) for fraction in fractions), *local_offsets, TOTAL_STEPS - 1}
    if min(steps) < 1 or max(steps) >= TOTAL_STEPS:
        raise ValueError("Checkpoint grid lies outside the trajectory")
    return tuple(sorted(steps))


CHECKPOINT_STEPS = _checkpoint_steps()


def _keep_policy() -> list[dict[str, int | None]]:
    return [{"every": step, "until": None if step == TOTAL_STEPS - 1 else step} for step in CHECKPOINT_STEPS]


def _trajectories() -> tuple[Trajectory, ...]:
    rows = []
    for arm in ARMS:
        onset = None if arm.decay_onset_fraction is None else _step(arm.decay_onset_fraction)
        for seed in TRAINING_SEEDS:
            rows.append(
                Trajectory(
                    trajectory_id=f"lr_onset_{arm.arm}_h0640_s03820_s{seed}",
                    arm=arm.arm,
                    training_seed=seed,
                    total_steps=TOTAL_STEPS,
                    optimizer_decay_step=onset,
                    support_pool_seed=None,
                    support_start_batches=None,
                    support_batches=None,
                    train_holdout_sequences_per_component=HOLDOUT_SEQUENCES_PER_COMPONENT,
                    train_holdout_seed=HOLDOUT_SEED,
                    train_holdout_partition=HOLDOUT_PARTITION,
                )
            )
    return tuple(rows)


def _optimizer(arm: ScheduleArm):
    optimizer = base._optimizer(MATERIALIZED_TOKENS)
    if arm.decay_onset_fraction is None:
        return replace(optimizer, min_lr_ratio=1.0)
    onset = _step(arm.decay_onset_fraction)
    return replace(optimizer, decay=TOTAL_STEPS - onset)


def _arm_by_name() -> dict[str, ScheduleArm]:
    return {arm.arm: arm for arm in ARMS}


def _learning_rate(arm: ScheduleArm, step: int) -> float:
    return float(_optimizer(arm).lr_scheduler(TOTAL_STEPS)(step))


def _design_contract() -> dict[str, Any]:
    schedule_values = {
        arm.arm: {
            str(step): _learning_rate(arm, step)
            for step in sorted({0, _step(0.55), _step(0.60), _step(0.70), _step(0.80), _step(0.90), TOTAL_STEPS - 1})
        }
        for arm in ARMS
    }
    return {
        "schema_version": "2026-08-23-starcoder-wsd80-lr-onset-v1",
        "question": "Does StarCoder-Nemotron gradient-cosine decline follow LR-decay onset?",
        "paper_motivation": "Wen et al., River Valley Loss Landscape, arXiv:2410.05192",
        "historical_comparator": {
            "arm": "decay_0p80_historical",
            "cell_id": "r0_shared_h0640_s03820",
            "policy_role": "common_tied_035",
            "support_id": "full",
            "training_seeds": list(TRAINING_SEEDS),
        },
        "arms": [asdict(arm) for arm in ARMS],
        "trajectory_count": len(_trajectories()),
        "training_seeds": list(TRAINING_SEEDS),
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "schedule_values": schedule_values,
        "model": {
            "hidden_size": HIDDEN_SIZE,
            "total_parameters": EXPECTED_TOTAL_PARAMETERS,
            "non_embedding_parameters": EXPECTED_NON_EMBEDDING_PARAMETERS,
        },
        "training": {
            "total_steps": TOTAL_STEPS,
            "materialized_tokens": MATERIALIZED_TOKENS,
            "batch_size": base.BATCH_SIZE,
            "sequence_length": base.SEQ_LEN,
            "starcoder_weight": STARCODER_WEIGHT,
            "data_policy": "tied_static_full_source",
            "holdout_sequences_per_component": HOLDOUT_SEQUENCES_PER_COMPONENT,
            "holdout_seed": HOLDOUT_SEED,
            "holdout_partition": HOLDOUT_PARTITION,
        },
        "placement": {
            "marin_prefix": MARIN_PREFIX,
            "tpu_type": TPU_TYPE,
            "region": TPU_REGION,
            "zone": TPU_ZONE,
        },
        "endpoint_metrics_read_by_probe_or_onset_analysis": False,
    }


def _write_remote_json(path: str, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    fs, plain_path = fsspec.core.url_to_fs(path)
    fs.makedirs(os.path.dirname(plain_path), exist_ok=True)
    try:
        with fs.open(plain_path, "xb") as handle:
            handle.write(encoded)
    except FileExistsError as error:
        with fs.open(plain_path, "rb") as handle:
            if handle.read() != encoded:
                raise RuntimeError(f"Existing validation artifact differs: {path}") from error


def _read_remote_json(path: str) -> dict[str, Any]:
    fs, plain_path = fsspec.core.url_to_fs(path)
    with fs.open(plain_path, "rb") as handle:
        return json.load(handle)


def _restored_hyperparameters(state: Any, train_config: TrainLmConfig) -> dict[str, Any]:
    observed: dict[str, list[float]] = {"learning_rate": [], "adam_lr": []}
    for path, value in jax.tree_util.tree_flatten_with_path(state.opt_state)[0]:
        array = gradient_probe._array(value)
        if not hasattr(array, "ndim") or array.ndim != 0 or not np.issubdtype(array.dtype, np.floating):
            continue
        name = jax.tree_util.keystr(path).lower()
        if "hyperparams_states" in name:
            continue
        for field in observed:
            if field in name:
                observed[field].append(float(jax.device_get(array)))
    unique = {field: sorted(set(values)) for field, values in observed.items()}
    if any(len(values) != 1 for values in unique.values()):
        raise RuntimeError(f"Restored optimizer does not expose exactly one floating hyperparameter: {unique}")
    schedule_step = int(state.step) - 1
    horizon = gradient_probe._effective_optimizer_schedule_num_train_steps(train_config)
    expected = {
        "learning_rate": float(train_config.optimizer.lr_scheduler(horizon)(schedule_step)),
        "adam_lr": float(
            train_config.optimizer.lr_scheduler(horizon, override_lr=train_config.optimizer.adam_lr)(schedule_step)
        ),
    }
    observed_scalar = {field: values[0] for field, values in unique.items()}
    tolerance = 5e-7
    differences = {field: abs(observed_scalar[field] - expected[field]) for field in expected}
    if any(value > tolerance for value in differences.values()):
        raise RuntimeError(
            f"Restored optimizer hyperparameters differ from schedule: observed={observed_scalar}, expected={expected}"
        )
    return {
        "state_step": int(state.step),
        "schedule_step": schedule_step,
        "observed": observed_scalar,
        "expected": expected,
        "absolute_differences": differences,
        "absolute_tolerance": tolerance,
        "matches_expected": True,
    }


def run_stage0_checkpoint_validation(config: Stage0CheckpointValidationConfig) -> None:
    train_config = gradient_probe._prepare_train_config(
        config.pod_config,
        config.checkpoint_uri,
        f"lr-onset-stage0-{config.arm}",
    )
    trainer, state, _, _, _ = gradient_probe._initialize_runtime(train_config)
    try:
        if int(state.step) != config.expected_restored_state_step:
            raise RuntimeError(f"{config.arm}: restored state step drifted")
        learning_rates = _restored_hyperparameters(state, train_config)
        payload = {
            "schema_version": "2026-08-23-lr-onset-stage0-state-v1",
            "arm": config.arm,
            "checkpoint_uri": config.checkpoint_uri,
            "checkpoint_step": config.checkpoint_step,
            "restored_state_step": int(state.step),
            "release_sha256": config.release_sha256,
            "learning_rates": learning_rates,
            "state_fingerprint": {
                "model_sha256": gradient_probe._tree_sha256(state.model),
                "optimizer_state_sha256": gradient_probe._tree_sha256(state.opt_state),
                "training_key_sha256": gradient_probe._tree_sha256(state.training_key),
            },
            "device_count": len(jax.devices()),
            "device_kinds": sorted({str(device.device_kind) for device in jax.devices()}),
        }
        payload["payload_sha256"] = _sha256_bytes(_canonical_json(payload).encode())
        _write_remote_json(f"{config.output_path}/validation.json", payload)
    finally:
        gradient_probe._close_runtime(trainer)


def run_stage0_aggregate_validation(config: Stage0AggregateValidationConfig) -> None:
    documents = {arm: _read_remote_json(f"{path}/validation.json") for arm, path in config.arm_outputs.items()}
    if tuple(sorted(documents)) != tuple(sorted(config.expected_arms)):
        raise RuntimeError("Stage-0 validator arm inventory drifted")
    for arm, document in documents.items():
        payload_sha256 = document.pop("payload_sha256")
        if payload_sha256 != _sha256_bytes(_canonical_json(document).encode()):
            raise RuntimeError(f"{arm}: stage-0 validation payload hash failed")
        document["payload_sha256"] = payload_sha256
        if document["release_sha256"] != config.release_sha256:
            raise RuntimeError(f"{arm}: stage-0 release identity drifted")
        if document["checkpoint_step"] != config.checkpoint_step:
            raise RuntimeError(f"{arm}: stage-0 checkpoint drifted")
        if not document["learning_rates"]["matches_expected"]:
            raise RuntimeError(f"{arm}: stage-0 LR schedule validation failed")
        if abs(document["learning_rates"]["observed"]["learning_rate"] - config.expected_learning_rate) > 5e-7:
            raise RuntimeError(f"{arm}: schedule diverged before the intervention")
    fingerprint_fields = ("model_sha256", "optimizer_state_sha256", "training_key_sha256")
    fingerprint_cardinality = {
        field: len({document["state_fingerprint"][field] for document in documents.values()})
        for field in fingerprint_fields
    }
    if any(value != 1 for value in fingerprint_cardinality.values()):
        raise RuntimeError(f"Stage-0 prefixes are not bitwise identical: {fingerprint_cardinality}")
    payload = {
        "schema_version": "2026-08-23-lr-onset-stage0-aggregate-v1",
        "status": "PASS",
        "release_sha256": config.release_sha256,
        "checkpoint_step": config.checkpoint_step,
        "expected_learning_rate": config.expected_learning_rate,
        "arm_inventory": sorted(documents),
        "fingerprint_cardinality": fingerprint_cardinality,
        "arm_documents": documents,
    }
    payload["payload_sha256"] = _sha256_bytes(_canonical_json(payload).encode())
    _write_remote_json(f"{config.output_path}/validation.json", payload)


def freeze_release() -> dict[str, Any]:
    """Materialize the reviewed, hash-pinned training release."""
    if not CC_REVIEW_PATH.is_file():
        raise FileNotFoundError(f"Required CC review is missing: {CC_REVIEW_PATH}")
    review = CC_REVIEW_PATH.read_text()
    if not review.rstrip().endswith("VERDICT: PASS"):
        raise ValueError("CC review does not end in VERDICT: PASS")
    trajectories = _trajectories()
    if len(trajectories) != 32 or len({row.trajectory_id for row in trajectories}) != 32:
        raise ValueError("Trajectory inventory drifted")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    contract = _design_contract()
    DESIGN_CONTRACT_PATH.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    TRAJECTORY_MANIFEST_PATH.write_text(
        json.dumps([asdict(row) for row in trajectories], indent=2, sort_keys=True) + "\n"
    )
    checkpoint_rows = [
        {
            "trajectory_id": row.trajectory_id,
            "checkpoint_step": step,
            "normalized_time": step / TOTAL_STEPS,
            "steps_from_decay_onset": "" if row.optimizer_decay_step is None else step - row.optimizer_decay_step,
        }
        for row in trajectories
        for step in CHECKPOINT_STEPS
    ]
    CHECKPOINT_MANIFEST_PATH.write_text(json.dumps(checkpoint_rows, indent=2, sort_keys=True) + "\n")
    release = {
        "release_version": VERSION,
        "release_sha256": "",
        "runtime_path": str(Path(__file__).relative_to(REPO_ROOT)),
        "runtime_sha256": _file_sha256(Path(__file__)),
        "design_contract_path": str(DESIGN_CONTRACT_PATH.relative_to(REPO_ROOT)),
        "design_contract_sha256": _file_sha256(DESIGN_CONTRACT_PATH),
        "trajectory_manifest_path": str(TRAJECTORY_MANIFEST_PATH.relative_to(REPO_ROOT)),
        "trajectory_manifest_sha256": _file_sha256(TRAJECTORY_MANIFEST_PATH),
        "checkpoint_manifest_path": str(CHECKPOINT_MANIFEST_PATH.relative_to(REPO_ROOT)),
        "checkpoint_manifest_sha256": _file_sha256(CHECKPOINT_MANIFEST_PATH),
        "cc_review_path": str(CC_REVIEW_PATH.relative_to(REPO_ROOT)),
        "cc_review_sha256": _file_sha256(CC_REVIEW_PATH),
        "cc_review_verdict": "PASS",
        "maximum_concurrent_trajectories": MAX_CONCURRENT,
        "stage_concurrency": {"0": 4, "1": 28},
        "confirmation": FULL_LAUNCH_CONFIRMATION,
    }
    release["release_sha256"] = _sha256_bytes(_canonical_json(release).encode())
    RELEASE_PATH.write_text(json.dumps(release, indent=2, sort_keys=True) + "\n")
    return release


def _load_release() -> dict[str, Any]:
    release = json.loads(RELEASE_PATH.read_text())
    claimed = release["release_sha256"]
    if claimed != _sha256_bytes(_canonical_json({**release, "release_sha256": ""}).encode()):
        raise ValueError("Release payload hash drifted")
    checks = {
        Path(release["runtime_path"]): release["runtime_sha256"],
        Path(release["design_contract_path"]): release["design_contract_sha256"],
        Path(release["trajectory_manifest_path"]): release["trajectory_manifest_sha256"],
        Path(release["checkpoint_manifest_path"]): release["checkpoint_manifest_sha256"],
        Path(release["cc_review_path"]): release["cc_review_sha256"],
    }
    drifted = [str(path) for path, expected in checks.items() if _file_sha256(REPO_ROOT / path) != expected]
    if drifted:
        raise ValueError(f"Frozen release files drifted: {drifted}")
    if not (REPO_ROOT / release["cc_review_path"]).read_text().rstrip().endswith("VERDICT: PASS"):
        raise ValueError("Frozen CC review verdict drifted")
    return release


def _build_training(
    trajectory: Trajectory,
    *,
    model: Any,
    resources: ResourceConfig,
    nemotron: dict[str, Any],
    starcoder: Any,
    training_handles: tuple[Any, ...],
    validation: tuple[Any, ...],
) -> ArtifactStep[LevanterCheckpoint]:
    arm = _arm_by_name()[trajectory.arm]
    component_names = tuple(handle.name for handle in training_handles)
    weights_by_name = base._phase_leaf_weights(
        STARCODER_WEIGHT,
        nemotron=nemotron,
        starcoder=starcoder,
    )
    static_weights = {handle: weights_by_name[handle.name] for handle in training_handles}
    training = train_lm(
        name=f"checkpoints/{NAME}/trajectories/{trajectory.trajectory_id}",
        version=VERSION,
        model=model,
        optimizer=_optimizer(arm),
        datasets=static_weights,
        validation=validation,
        batch_size=base.BATCH_SIZE,
        seq_len=base.SEQ_LEN,
        num_train_steps=TOTAL_STEPS,
        z_loss_weight=None,
        evals=None,
        resources=resources,
        steps_per_eval=1_000,
        wandb_project="marin",
        wandb_group=WANDB_GROUP,
        run_id=trajectory.trajectory_id,
        tags=(PANEL_TAG, trajectory.arm, "full_source", "tied_035", "starcoder", "wsd80_20"),
        env_vars={"HF_ALLOW_CODE_EVAL": "1"},
    )
    return historical._configure_training(
        training,
        trajectory=cast(Any, trajectory),
        phase_weights=[(0, weights_by_name)],
        training_component_names=component_names,
        starcoder_name=training_handles[-1].name,
        keep=_keep_policy(),
    )


def build_training_steps() -> tuple[tuple[Trajectory, ...], tuple[ArtifactStep[LevanterCheckpoint], ...]]:
    """Build and audit the complete set of resumable training artifacts."""
    trajectories = _trajectories()
    nemotron, starcoder, validation = historical._training_data()
    training_handles = tuple([nemotron[split] for split in base.NEMOTRON_TOKEN_COUNTS] + [starcoder])
    if tuple(handle.name for handle in training_handles) != historical.EXPECTED_TRAINING_COMPONENT_NAMES:
        raise ValueError("Training component identities drifted")
    model = CompletedAdamHHeuristic()._build_model_config(HIDDEN_SIZE, seq_len=base.SEQ_LEN)
    if model.total_trainable_params(llama3_tokenizer_vocab_size) != EXPECTED_TOTAL_PARAMETERS:
        raise ValueError("Total parameter count drifted")
    if model.total_trainable_params(0) != EXPECTED_NON_EMBEDDING_PARAMETERS:
        raise ValueError("Non-embedding parameter count drifted")
    resources = ResourceConfig.with_tpu(
        TPU_TYPE,
        cpu=historical.TPU_HOST_CPU,
        ram=historical.TPU_HOST_RAM,
        regions=(TPU_REGION,),
        zone=TPU_ZONE,
    )
    steps = tuple(
        _build_training(
            trajectory,
            model=model,
            resources=resources,
            nemotron=nemotron,
            starcoder=starcoder,
            training_handles=training_handles,
            validation=validation,
        )
        for trajectory in trajectories
    )
    return trajectories, steps


def _stage0_validation_step(
    release: dict[str, Any],
    trajectories: tuple[Trajectory, ...],
    training_steps: tuple[ArtifactStep[LevanterCheckpoint], ...],
) -> ArtifactStep[Artifact]:
    checkpoint_step = _step(0.55)
    resources = ResourceConfig.with_tpu(
        TPU_TYPE,
        cpu=historical.TPU_HOST_CPU,
        ram=historical.TPU_HOST_RAM,
        regions=(TPU_REGION,),
        zone=TPU_ZONE,
    )
    artifact_cache: dict[int, Any] = {}
    arm_steps: dict[str, ArtifactStep[Artifact]] = {}
    for trajectory, training_step in zip(trajectories, training_steps, strict=True):
        if trajectory.training_seed != TRAINING_SEEDS[0]:
            continue
        pod_config = cast(
            TrainLmOnPodConfig,
            materialized_config(training_step, MARIN_PREFIX, artifact_cache=artifact_cache),
        )
        checkpoint_uri = f"{training_step.path(MARIN_PREFIX)}/checkpoints/step-{checkpoint_step}"
        validator_config = Stage0CheckpointValidationConfig(
            arm=trajectory.arm,
            checkpoint_uri=checkpoint_uri,
            checkpoint_step=checkpoint_step,
            expected_restored_state_step=gradient_probe.freeze.expected_restored_state_step(checkpoint_step),
            pod_config=pod_config,
            output_path="",
            release_sha256=release["release_sha256"],
        )
        arm_steps[trajectory.arm] = ArtifactStep(
            name=f"{STAGE0_VALIDATION_NAME}/arms/{trajectory.arm}",
            version=STAGE0_VALIDATION_VERSION,
            artifact_type=Artifact,
            run=remote(run_stage0_checkpoint_validation, resources=resources, name=f"validate-{trajectory.arm}"),
            build_config=lambda ctx, config=validator_config: replace(config, output_path=ctx.output_path),
            deps=(training_step,),
        )
    expected_arms = tuple(sorted(arm.arm for arm in ARMS))
    if tuple(sorted(arm_steps)) != expected_arms:
        raise ValueError("Stage-0 validator does not cover every schedule arm")
    deps = tuple(arm_steps[arm] for arm in expected_arms)
    aggregate_config = Stage0AggregateValidationConfig(
        arm_outputs={},
        expected_arms=expected_arms,
        expected_learning_rate=_learning_rate(ARMS[0], checkpoint_step),
        checkpoint_step=checkpoint_step,
        output_path="",
        release_sha256=release["release_sha256"],
    )
    return ArtifactStep(
        name=STAGE0_VALIDATION_NAME,
        version=STAGE0_VALIDATION_VERSION,
        artifact_type=Artifact,
        run=run_stage0_aggregate_validation,
        build_config=lambda ctx: replace(
            aggregate_config,
            arm_outputs={arm: ctx.artifact_path(arm_steps[arm]) for arm in expected_arms},
            output_path=ctx.output_path,
        ),
        deps=deps,
    )


def _validated_stage0_release(validation_step: ArtifactStep[Artifact], release: dict[str, Any]) -> dict[str, Any]:
    path = f"{validation_step.path(MARIN_PREFIX)}/validation.json"
    validation = _read_remote_json(path)
    payload_sha256 = validation.pop("payload_sha256")
    if payload_sha256 != _sha256_bytes(_canonical_json(validation).encode()):
        raise ValueError("Stage-0 aggregate validation payload hash failed")
    validation["payload_sha256"] = payload_sha256
    if validation.get("release_sha256") != release["release_sha256"] or validation.get("status") != "PASS":
        raise ValueError("Stage-0 prefix and LR validation has not passed")
    return validation


def audit_runtime_configs(
    trajectories: tuple[Trajectory, ...], steps: tuple[ArtifactStep[LevanterCheckpoint], ...]
) -> None:
    """Fail closed on every treatment and shared training property."""
    if len(trajectories) != 32 or len(steps) != 32:
        raise ValueError("Runtime trajectory inventory drifted")
    arm_lookup = _arm_by_name()
    artifact_cache: dict[int, Any] = {}
    for trajectory, step in zip(trajectories, steps, strict=True):
        config = cast(
            TrainLmOnPodConfig,
            materialized_config(step, MARIN_PREFIX, artifact_cache=artifact_cache),
        )
        train_config = cast(TrainLmConfig, config.train_config)
        arm = arm_lookup[trajectory.arm]
        optimizer = train_config.optimizer
        expected_decay = (
            base._optimizer(MATERIALIZED_TOKENS).decay
            if arm.decay_onset_fraction is None
            else TOTAL_STEPS - _step(arm.decay_onset_fraction)
        )
        if optimizer.decay != expected_decay:
            raise ValueError(f"{trajectory.trajectory_id}: decay duration drifted")
        expected_min_ratio = 1.0 if arm.decay_onset_fraction is None else 0.0
        if optimizer.min_lr_ratio != expected_min_ratio:
            raise ValueError(f"{trajectory.trajectory_id}: minimum LR ratio drifted")
        if optimizer.warmup != base._optimizer(MATERIALIZED_TOKENS).warmup:
            raise ValueError(f"{trajectory.trajectory_id}: warmup drifted")
        if train_config.optimizer_schedule_num_train_steps is not None:
            raise ValueError(f"{trajectory.trajectory_id}: optimizer schedule acquired an independent horizon")
        if train_config.trainer.num_train_steps != TOTAL_STEPS:
            raise ValueError(f"{trajectory.trajectory_id}: total steps drifted")
        train_weights = train_config.data.train_weights
        if not isinstance(train_weights, list) or len(train_weights) != 1 or train_weights[0][0] != 0:
            raise ValueError(f"{trajectory.trajectory_id}: data policy is not static")
        if train_config.trainer.seed != trajectory.training_seed or train_config.data_seed != trajectory.training_seed:
            raise ValueError(f"{trajectory.trajectory_id}: paired seed drifted")
        if train_config.data.train_holdout_seed != HOLDOUT_SEED:
            raise ValueError(f"{trajectory.trajectory_id}: holdout seed drifted")
        if train_config.data.train_holdout_partition != HOLDOUT_PARTITION:
            raise ValueError(f"{trajectory.trajectory_id}: holdout partition drifted")
        if train_config.data.max_train_batches is not None:
            raise ValueError(f"{trajectory.trajectory_id}: full-source policy acquired a support cap")
        if train_config.trainer.checkpointer.keep != _keep_policy():
            raise ValueError(f"{trajectory.trajectory_id}: checkpoint retention drifted")
    before_treatment = _step(0.55)
    rates = {_learning_rate(arm, before_treatment) for arm in ARMS}
    if len(rates) != 1:
        raise ValueError(f"Schedule arms differ before the earliest treatment: {rates}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT)
    parser.add_argument("--stage", type=int, choices=(0, 1))
    parser.add_argument("--confirmation")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    if args.freeze:
        release = freeze_release()
        logger.info("Frozen LR-onset release %s", release["release_sha256"])
        return
    release = _load_release()
    if args.max_concurrent < 1 or args.max_concurrent > int(release["maximum_concurrent_trajectories"]):
        raise ValueError("Requested concurrency is outside the frozen release")
    trajectories, steps = build_training_steps()
    audit_runtime_configs(trajectories, steps)
    validation_step = _stage0_validation_step(release, trajectories, steps)
    if args.audit:
        historical.audit_sources(MARIN_PREFIX, cast(Any, trajectories))
        logger.info("Audited %d LR-onset trajectories", len(trajectories))
        return
    if args.dry_run:
        for step in steps:
            lower(step)
        logger.info("Lowered %d LR-onset training graphs", len(steps))
        return
    if args.confirmation != release["confirmation"]:
        raise ValueError("Full launch confirmation is missing or incorrect")
    if args.stage is None:
        raise ValueError("External launch requires explicit --stage 0 or --stage 1")
    if os.getenv("MARIN_PREFIX", MARIN_PREFIX) != MARIN_PREFIX:
        raise ValueError("LR-onset intervention must remain central1-local")
    os.environ["MARIN_PREFIX"] = MARIN_PREFIX
    historical.audit_sources(MARIN_PREFIX, cast(Any, trajectories))
    selected = tuple(
        step
        for trajectory, step in zip(trajectories, steps, strict=True)
        if (trajectory.training_seed == TRAINING_SEEDS[0]) == (args.stage == 0)
    )
    stage_limit = int(release["stage_concurrency"][str(args.stage)])
    if args.max_concurrent > stage_limit:
        raise ValueError(f"Stage {args.stage} permits at most {stage_limit} concurrent trajectories")
    if args.stage == 1:
        _validated_stage0_release(validation_step, release)
    requested = (*selected, validation_step) if args.stage == 0 else selected
    run(*requested, max_concurrent=min(args.max_concurrent, len(selected)), force_run_failed=True)


if __name__ == "__main__":
    main()
