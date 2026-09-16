# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the two-trajectory WSD80 gradient-conflict canary and its exact forks.

Any identity-bearing config change after submission requires a new ``VERSION``;
artifact paths intentionally do not carry an automatic config fingerprint.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

from fray.types import ResourceConfig
from levanter.checkpoint import is_checkpoint_path
from levanter.main.train_lm import TrainLmConfig
from marin.execution.lazy import ArtifactStep, StepContext, lower, materialized_config, run
from marin.experiment.train import train_lm
from marin.processing.tokenize.tokenize import TokenizedCache
from marin.training.training import LevanterCheckpoint, TrainLmOnPodConfig
from rigging.filesystem import prefix_join

from experiments.datasets.dolma import dolma_datasets
from experiments.datasets.nemotron import nemotron_datasets
from experiments.datasets.paloma import paloma_datasets
from experiments.datasets.uncheatable import uncheatable_datasets
from experiments.domain_phase_mix import launch_starcoder_wsd80_dense_support_surfaces as dense
from experiments.domain_phase_mix import launch_starcoder_wsd_80_20_surface as base
from experiments.llama import llama3_tokenizer, llama3_tokenizer_vocab_size
from experiments.scaling_law_sweeps.completed_adamh import CompletedAdamHHeuristic

logger = logging.getLogger(__name__)

NAME = "pinlin_calvin_xu/data_mixture/starcoder_wsd80_gradient_conflict_20260810"
VERSION = "2026.08.10"
WANDB_EXPERIMENT_TAG = "starcoder_wsd80_gradient_conflict_20260810"
CANARY_MANIFEST = Path(__file__).with_name("starcoder_wsd80_gradient_conflict_canary_20260810.json")
REPO_ROOT = Path(__file__).resolve().parents[2]
SCIENTIFIC_DESIGN_DIR = (
    Path(__file__).parent
    / "exploratory/two_phase_many/reference_outputs/starcoder_wsd80_gradient_conflict_design_20260810"
)
SCIENTIFIC_DESIGN_MANIFEST = SCIENTIFIC_DESIGN_DIR / "design_manifest.json"
EXPECTED_CANARY_MANIFEST_SHA256 = "811d8064104454b5f33f4ae7582ae27024dd1dec886fb76ee17a5d9b3bb5b727"
EXPECTED_CANARY_VERSION = "2026-08-10-canary-v3"
EXPECTED_SCIENTIFIC_DESIGN_VERSION = "2026-08-10-review-v5"
EXPECTED_SCIENTIFIC_DESIGN_MANIFEST_SHA256 = "8ca0c9f433ef6fccf02fb7ed597d90e3b0ea3b663c58d12bb63b2a4a61bec0dc"
EXPECTED_TRAJECTORY_IDS = (
    "gcf_p1_r3d28260_m100a_common-tied-035_s2026081000",
    "gcf_p1_r3d28260_m100a_common-tied-035_s2026081001",
)
EXPECTED_TRAINING_COMPONENT_NAMES = (
    "nemotron_cc/hq_actual-llama3",
    "nemotron_cc/hq_synth-llama3",
    "nemotron_cc/medium_high-llama3",
    "nemotron_cc/medium-llama3",
    "nemotron_cc/medium_low-llama3",
    "nemotron_cc/low_actual-llama3",
    "dolma/starcoder",
)
CHECKPOINT_INTERVAL = datetime.timedelta(minutes=15)
DEFAULT_MAX_CONCURRENT = 2
DECAY_FORK_SOURCE_STEP = 22_544
DECAY_FORK_REFERENCE_STEP = 22_672
DECAY_FORK_NUM_UPDATES = DECAY_FORK_REFERENCE_STEP - DECAY_FORK_SOURCE_STEP
DECAY_FORK_TRAINER_NUM_STEPS = DECAY_FORK_REFERENCE_STEP + 1
EXPECTED_OPTIMIZER_CONFIG = {
    "learning_rate": 0.02,
    "weight_decay": 0.1,
    "min_lr_ratio": 0.0,
    "warmup": 282,
    "decay": 5652,
    "rewarmup": 0.0,
    "cooldown": None,
    "cycle_length": None,
    "cycles": None,
    "lr_schedule": "cosine",
    "haps": None,
    "weight_decay_modules": None,
    "default_weight_decay_mask": None,
    "adam_lr": 0.008,
    "momentum": 0.95,
    "nesterov": True,
    "backend_steps": 5,
    "beta1": 0.9,
    "beta2": 0.98,
    "epsilon": 1e-15,
    "muon_epsilon": 1e-5,
    "max_grad_norm": 1.0,
    "coefficient_type": "quintic",
}


@dataclass(frozen=True)
class CanaryTrajectory:
    """One preregistered canary trajectory."""

    trajectory_id: str
    training_seed: int


@dataclass(frozen=True)
class CanaryDesign:
    """Frozen launch decisions shared by the two trajectories."""

    canary_version: str
    scientific_design_version: str
    scientific_design_manifest_sha256: str
    cell_id: str
    model_hidden_size: int
    total_parameters: int
    non_embedding_parameters: int
    total_steps: int
    boundary_step: int
    materialized_tokens: int
    phase_0_starcoder: float
    phase_1_starcoder: float
    support_id: str
    support_pool_seed: int
    starcoder_support_batches: int
    starcoder_support_sequences: int
    starcoder_support_tokens: int
    scientific_checkpoint_steps: tuple[int, ...]
    diagnostic_checkpoint_steps: tuple[int, ...]
    fork_source_step: int
    fork_reference_step: int
    fork_num_updates: int
    fork_trainer_num_train_steps: int
    trajectories: tuple[CanaryTrajectory, ...]


@dataclass(frozen=True)
class _SourceValidationRequest:
    starcoder_support_batches: int
    starcoder_realized_support_tokens: int


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _design_artifact_path(relative_path: str) -> Path:
    if relative_path.startswith("experiments/"):
        return REPO_ROOT / relative_path
    return SCIENTIFIC_DESIGN_DIR / relative_path


def _load_reviewed_design_manifest() -> dict[str, Any]:
    observed_hash = _file_sha256(SCIENTIFIC_DESIGN_MANIFEST)
    if observed_hash != EXPECTED_SCIENTIFIC_DESIGN_MANIFEST_SHA256:
        raise ValueError(
            "Reviewed scientific design manifest drifted: "
            f"{observed_hash} != {EXPECTED_SCIENTIFIC_DESIGN_MANIFEST_SHA256}"
        )
    manifest = json.loads(SCIENTIFIC_DESIGN_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("design_version") != EXPECTED_SCIENTIFIC_DESIGN_VERSION:
        raise ValueError(f"Unexpected reviewed design version: {manifest.get('design_version')!r}")
    expected_placement = {
        "required_bucket_prefix": base.DEFAULT_MARIN_PREFIX,
        "required_region": base.DEFAULT_TPU_REGION,
        "required_zone": base.DEFAULT_TPU_ZONE,
    }
    observed_placement = {key: manifest.get(key) for key in expected_placement}
    if observed_placement != expected_placement:
        raise ValueError(f"Reviewed design placement drifted: {observed_placement} != {expected_placement}")
    claimed_hash = manifest.get("design_sha256")
    observed_canonical_hash = _canonical_sha256({**manifest, "design_sha256": ""})
    if claimed_hash != observed_canonical_hash:
        raise ValueError(f"Reviewed design self-hash drifted: {observed_canonical_hash} != {claimed_hash}")
    for relative_path, expected_hash in manifest["artifact_sha256"].items():
        artifact_path = _design_artifact_path(relative_path)
        observed_artifact_hash = _file_sha256(artifact_path)
        if observed_artifact_hash != expected_hash:
            raise ValueError(
                f"Reviewed design artifact drifted: {relative_path}: {observed_artifact_hash} != {expected_hash}"
            )
    return manifest


def load_canary_design() -> CanaryDesign:
    """Load the immutable canary manifest and fail on any identity drift."""
    observed_hash = _file_sha256(CANARY_MANIFEST)
    if observed_hash != EXPECTED_CANARY_MANIFEST_SHA256:
        raise ValueError(f"Canary manifest drifted: {observed_hash} != {EXPECTED_CANARY_MANIFEST_SHA256}")
    payload = json.loads(CANARY_MANIFEST.read_text(encoding="utf-8"))
    exact_fork = payload.pop("exact_fork")
    trajectories = tuple(CanaryTrajectory(**row) for row in payload.pop("trajectories"))
    scientific_checkpoint_steps = tuple(payload.pop("scientific_checkpoint_steps"))
    diagnostic_checkpoint_steps = tuple(payload.pop("diagnostic_checkpoint_steps"))
    design = CanaryDesign(
        **payload,
        scientific_checkpoint_steps=scientific_checkpoint_steps,
        diagnostic_checkpoint_steps=diagnostic_checkpoint_steps,
        fork_source_step=int(exact_fork["source_step"]),
        fork_reference_step=int(exact_fork["reference_step"]),
        fork_num_updates=int(exact_fork["num_updates"]),
        fork_trainer_num_train_steps=int(exact_fork["trainer_num_train_steps"]),
        trajectories=trajectories,
    )
    if design.canary_version != EXPECTED_CANARY_VERSION:
        raise ValueError(f"Unexpected canary version: {design.canary_version}")
    if design.scientific_design_version != EXPECTED_SCIENTIFIC_DESIGN_VERSION:
        raise ValueError(f"Unexpected scientific design version: {design.scientific_design_version}")
    if design.scientific_design_manifest_sha256 != EXPECTED_SCIENTIFIC_DESIGN_MANIFEST_SHA256:
        raise ValueError("Canary no longer points to the reviewed scientific design")
    if tuple(row.trajectory_id for row in trajectories) != EXPECTED_TRAJECTORY_IDS:
        raise ValueError("Canary trajectory identities drifted")
    if len({row.training_seed for row in trajectories}) != len(trajectories):
        raise ValueError("Canary training seeds must be unique")
    if design.boundary_step * 5 != design.total_steps * 4:
        raise ValueError("Canary phase boundary is not exactly 80/20")
    if design.materialized_tokens != design.total_steps * base.BATCH_SIZE * base.SEQ_LEN:
        raise ValueError("Canary token accounting drifted")
    if design.starcoder_support_sequences != design.starcoder_support_batches * base.BATCH_SIZE:
        raise ValueError("Canary support sequence accounting drifted")
    if design.starcoder_support_tokens != design.starcoder_support_sequences * base.SEQ_LEN:
        raise ValueError("Canary support token accounting drifted")
    checkpoint_steps = (*design.scientific_checkpoint_steps, *design.diagnostic_checkpoint_steps)
    if len(set(checkpoint_steps)) != len(checkpoint_steps):
        raise ValueError("Scientific and diagnostic checkpoint steps overlap")
    if any(step <= 0 or step >= design.total_steps for step in checkpoint_steps):
        raise ValueError("Canary checkpoint lies outside the training horizon")
    if design.fork_reference_step - design.fork_source_step != design.fork_num_updates:
        raise ValueError("Exact-fork update count drifted")
    if design.fork_trainer_num_train_steps != design.fork_reference_step + 1:
        raise ValueError("Exact-fork trainer horizon does not produce the declared reference checkpoint")
    if design.fork_source_step not in design.scientific_checkpoint_steps:
        raise ValueError("Exact fork does not begin at a scientific checkpoint")
    if design.fork_reference_step not in design.diagnostic_checkpoint_steps:
        raise ValueError("Exact-fork reference is not isolated as a diagnostic checkpoint")
    _validate_reviewed_design_contract(design)
    return design


def _model_for_design(design: CanaryDesign) -> Any:
    model = CompletedAdamHHeuristic()._build_model_config(design.model_hidden_size, seq_len=base.SEQ_LEN)
    total_parameters = model.total_trainable_params(llama3_tokenizer_vocab_size)
    non_embedding_parameters = model.total_trainable_params(0)
    if total_parameters != design.total_parameters or non_embedding_parameters != design.non_embedding_parameters:
        raise ValueError(
            "Canary model shape drifted: "
            f"({total_parameters}, {non_embedding_parameters}) != "
            f"({design.total_parameters}, {design.non_embedding_parameters})"
        )
    schedule = base._schedule_summary(design.materialized_tokens)
    if schedule["total_steps"] != design.total_steps or schedule["boundary_step"] != design.boundary_step:
        raise ValueError("Canary runtime schedule drifted")
    return model


def _optimizer_for_design(design: CanaryDesign) -> Any:
    optimizer = base._optimizer(design.materialized_tokens)
    observed = asdict(optimizer)
    if observed != EXPECTED_OPTIMIZER_CONFIG:
        raise ValueError(f"Canary optimizer drifted: {observed} != {EXPECTED_OPTIMIZER_CONFIG}")
    return optimizer


def _checkpoint_keep(
    design: CanaryDesign,
    *,
    fork: bool,
    decay_fork: bool = False,
) -> list[dict[str, int | None]]:
    if fork:
        steps = [DECAY_FORK_REFERENCE_STEP if decay_fork else design.fork_reference_step]
    else:
        steps = sorted((*design.scientific_checkpoint_steps, *design.diagnostic_checkpoint_steps))
    return [{"every": step, "until": None if step == steps[-1] else step} for step in steps]


def _validate_reviewed_design_contract(design: CanaryDesign) -> None:
    manifest = _load_reviewed_design_manifest()
    if manifest["checkpoint_count"] != 2_542 or manifest["gradient_probe_row_count"] != 18_496:
        raise ValueError("Reviewed design counts drifted from the canary contract")

    with (SCIENTIFIC_DESIGN_DIR / "checkpointer_manifest.csv").open(newline="") as handle:
        rows = {row["trajectory_id"]: row for row in csv.DictReader(handle)}
    expected_keep = _checkpoint_keep(design, fork=False)
    for trajectory_id in EXPECTED_TRAJECTORY_IDS:
        row = rows.get(trajectory_id)
        if row is None:
            raise ValueError(f"Reviewed design omits canary trajectory: {trajectory_id}")
        if json.loads(row["keep_json"]) != expected_keep:
            raise ValueError(f"Canary checkpoint policy diverges from reviewed design: {trajectory_id}")
        if int(row["expected_checkpoint_count"]) != len(expected_keep):
            raise ValueError(f"Canary checkpoint count diverges from reviewed design: {trajectory_id}")

    diagnostic_rows: list[dict[str, str]] = []
    with (SCIENTIFIC_DESIGN_DIR / "checkpoint_manifest.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["checkpoint_label"] == "canary_exact_fork_reference":
                diagnostic_rows.append(row)
    diagnostic_identities = {(row["trajectory_id"], int(row["checkpoint_step"])) for row in diagnostic_rows}
    expected_diagnostics = {(trajectory_id, design.fork_reference_step) for trajectory_id in EXPECTED_TRAJECTORY_IDS}
    if diagnostic_identities != expected_diagnostics:
        raise ValueError("Reviewed design's canary exact-fork references drifted")

    with (SCIENTIFIC_DESIGN_DIR / "gradient_probe_manifest.csv").open(newline="") as handle:
        leaked_diagnostics = [
            row
            for row in csv.DictReader(handle)
            if row["trajectory_id"] in EXPECTED_TRAJECTORY_IDS
            and int(row["checkpoint_step"]) == design.fork_reference_step
        ]
    if leaked_diagnostics:
        raise ValueError("Canary diagnostic checkpoint leaked into scientific probe rows")


def _trajectory_checkpoint_path(marin_prefix: str, trajectory_id: str, step: int) -> str:
    trajectory_path = prefix_join(
        marin_prefix,
        f"checkpoints/{NAME}/trajectories/{trajectory_id}/{VERSION}",
    )
    return prefix_join(trajectory_path, f"checkpoints/step-{step}")


def _training_data() -> tuple[
    dict[str, ArtifactStep[TokenizedCache]],
    ArtifactStep[TokenizedCache],
    tuple[ArtifactStep[TokenizedCache], ...],
]:
    nemotron = nemotron_datasets(tokenizer=llama3_tokenizer)
    starcoder = dolma_datasets(tokenizer=llama3_tokenizer)["dolma/starcoder"]
    validation = (
        *paloma_datasets(tokenizer=llama3_tokenizer).values(),
        *uncheatable_datasets(tokenizer=llama3_tokenizer).values(),
    )
    return nemotron, starcoder, validation


def _configure_training(
    training: ArtifactStep[LevanterCheckpoint],
    *,
    design: CanaryDesign,
    trajectory: CanaryTrajectory,
    phase_weights: list[tuple[int, dict[str, float]]],
    starcoder_name: str,
    fork_source_path: str | None,
    fork_reference_step: int | None,
    temporary_base_path: str,
) -> ArtifactStep[LevanterCheckpoint]:
    """Install fixed support, exact seeds, checkpoint policy, and optional exact-state resume."""

    def build_config(ctx: StepContext) -> TrainLmOnPodConfig:
        pod_config = training.build_config(ctx)
        train_config = cast(TrainLmConfig, pod_config.train_config)
        data_config = replace(
            train_config.data,
            train_weights=phase_weights,
            mixture_block_size=base.MIXTURE_BLOCK_SIZE,
            experiment_budget=None,
            target_budget=None,
            simulated_epoch_subset_seed=None,
            max_train_batches={starcoder_name: design.starcoder_support_batches},
            max_train_batches_subset_seed=design.support_pool_seed,
        )
        is_fork = fork_source_path is not None
        if is_fork != (fork_reference_step is not None):
            raise ValueError("Fork source and reference step must be specified together")
        trainer = replace(
            train_config.trainer,
            seed=trajectory.training_seed,
            # The 16-update diagnostic always restarts from its parent. Longer scientific
            # rollouts need a separate resume-aware runner rather than reusing this branch.
            load_checkpoint=False if is_fork else None,
            load_checkpoint_path=None,
            initialize_from=fork_source_path,
            checkpointer=replace(
                train_config.trainer.checkpointer,
                save_interval=CHECKPOINT_INTERVAL,
                keep=(
                    [{"every": fork_reference_step, "until": None}]
                    if fork_reference_step is not None
                    else _checkpoint_keep(design, fork=False)
                ),
                keep_last_temporary_checkpoints=1,
                temporary_base_path=temporary_base_path,
            ),
        )
        train_config = replace(
            train_config,
            data=data_config,
            data_seed=trajectory.training_seed,
            trainer=trainer,
            optimizer_schedule_num_train_steps=design.total_steps if is_fork else None,
        )
        return replace(pod_config, train_config=train_config)

    return replace(training, build_config=build_config)


def build_training_steps(
    design: CanaryDesign,
    *,
    marin_prefix: str,
    tpu_type: str,
    tpu_region: str,
    tpu_zone: str,
    fork: bool,
    decay_fork: bool = False,
) -> tuple[ArtifactStep[LevanterCheckpoint], ...]:
    """Build the two parent trajectories or their 16-update exact-state forks."""
    model = _model_for_design(design)
    optimizer = _optimizer_for_design(design)
    nemotron, starcoder, validation = _training_data()
    phase_0_weights = base._phase_leaf_weights(design.phase_0_starcoder, nemotron=nemotron, starcoder=starcoder)
    phase_1_weights = base._phase_leaf_weights(design.phase_1_starcoder, nemotron=nemotron, starcoder=starcoder)
    training_handles = tuple([nemotron[split] for split in base.NEMOTRON_TOKEN_COUNTS] + [starcoder])
    expected_names = (*tuple(nemotron[split].name for split in base.NEMOTRON_TOKEN_COUNTS), starcoder.name)
    if expected_names != EXPECTED_TRAINING_COMPONENT_NAMES:
        raise ValueError(f"Canary training components drifted: {expected_names}")
    if tuple(handle.name for handle in training_handles) != expected_names:
        raise ValueError("Canary training-cache ordering drifted")
    if len({id(handle) for handle in training_handles}) != len(training_handles):
        raise ValueError("Canary training-cache handles contain duplicates")
    if set(phase_0_weights) != set(expected_names) or set(phase_1_weights) != set(expected_names):
        raise ValueError("Canary phase-weight keys drifted")
    static_weights = {handle: phase_0_weights[handle.name] for handle in training_handles}
    resources = ResourceConfig.with_tpu(tpu_type, regions=(tpu_region,), zone=tpu_zone)

    steps: list[ArtifactStep[LevanterCheckpoint]] = []
    if decay_fork and not fork:
        raise ValueError("decay_fork requires fork=True")
    fork_source_step = DECAY_FORK_SOURCE_STEP if decay_fork else design.fork_source_step
    fork_reference_step = DECAY_FORK_REFERENCE_STEP if decay_fork else design.fork_reference_step
    fork_trainer_num_steps = DECAY_FORK_TRAINER_NUM_STEPS if decay_fork else design.fork_trainer_num_train_steps

    for trajectory in design.trajectories:
        run_id = trajectory.trajectory_id
        output_role = "canary_decay_forks" if decay_fork else "canary_forks" if fork else "trajectories"
        if fork:
            run_id = (
                f"{trajectory.trajectory_id}_fork{fork_reference_step - fork_source_step}"
                f"_from_step{fork_source_step}"
            )
        training = train_lm(
            name=f"checkpoints/{NAME}/{output_role}/{run_id}",
            version=VERSION,
            model=model,
            optimizer=optimizer,
            datasets=static_weights,
            validation=validation,
            batch_size=base.BATCH_SIZE,
            seq_len=base.SEQ_LEN,
            num_train_steps=fork_trainer_num_steps if fork else design.total_steps,
            z_loss_weight=None,
            evals=None,
            resources=resources,
            steps_per_eval=1_000,
            wandb_project="marin",
            wandb_group=NAME,
            run_id=run_id,
            tags=(
                WANDB_EXPERIMENT_TAG,
                design.scientific_design_version,
                design.canary_version,
                design.cell_id,
                design.support_id,
                "exact_fork" if fork else "canary_parent",
                "starcoder",
                "wsd80_20",
            ),
            env_vars={"HF_ALLOW_CODE_EVAL": "1"},
        )
        fork_source_path = (
            _trajectory_checkpoint_path(marin_prefix, trajectory.trajectory_id, fork_source_step) if fork else None
        )
        temporary_base_path = prefix_join(
            marin_prefix,
            f"temporary_checkpoints/{NAME}/{output_role}/{VERSION}",
        )
        steps.append(
            _configure_training(
                training,
                design=design,
                trajectory=trajectory,
                phase_weights=[(0, phase_0_weights), (design.boundary_step, phase_1_weights)],
                starcoder_name=starcoder.name,
                fork_source_path=fork_source_path,
                fork_reference_step=fork_reference_step if fork else None,
                temporary_base_path=temporary_base_path,
            )
        )
    return tuple(steps)


def audit_runtime_configs(
    design: CanaryDesign,
    steps: tuple[ArtifactStep[LevanterCheckpoint], ...],
    *,
    marin_prefix: str,
    fork: bool,
    decay_fork: bool = False,
) -> None:
    """Materialize both configs and verify every scientific identity-bearing field."""
    for trajectory, step in zip(design.trajectories, steps, strict=True):
        pod_config = materialized_config(step, marin_prefix)
        if not isinstance(pod_config, TrainLmOnPodConfig):
            raise TypeError(f"Unexpected canary runtime config: {type(pod_config)}")
        train_config = cast(TrainLmConfig, pod_config.train_config)
        expected_steps = (
            DECAY_FORK_TRAINER_NUM_STEPS
            if decay_fork
            else design.fork_trainer_num_train_steps if fork else design.total_steps
        )
        expected_optimizer_horizon = design.total_steps if fork else None
        if train_config.trainer.num_train_steps != expected_steps:
            raise ValueError(f"{trajectory.trajectory_id}: training horizon drifted")
        if train_config.optimizer_schedule_num_train_steps != expected_optimizer_horizon:
            raise ValueError(f"{trajectory.trajectory_id}: optimizer horizon drifted")
        if train_config.trainer.seed != trajectory.training_seed or train_config.data_seed != trajectory.training_seed:
            raise ValueError(f"{trajectory.trajectory_id}: model/data seed drifted")
        if train_config.data.max_train_batches != {"dolma/starcoder": design.starcoder_support_batches}:
            raise ValueError(f"{trajectory.trajectory_id}: StarCoder support cap drifted")
        if train_config.data.max_train_batches_subset_seed != design.support_pool_seed:
            raise ValueError(f"{trajectory.trajectory_id}: support-pool seed drifted")
        if train_config.data.simulated_epoch_subset_seed is not None:
            raise ValueError(f"{trajectory.trajectory_id}: global simulated subset leaked into canary")
        if train_config.data.experiment_budget is not None or train_config.data.target_budget is not None:
            raise ValueError(f"{trajectory.trajectory_id}: global simulated budget leaked into canary")
        phase_weights = train_config.data.train_weights
        if not isinstance(phase_weights, list) or [step for step, _ in phase_weights] != [0, design.boundary_step]:
            raise ValueError(f"{trajectory.trajectory_id}: phase schedule drifted")
        if train_config.trainer.checkpointer.keep != _checkpoint_keep(
            design,
            fork=fork,
            decay_fork=decay_fork,
        ):
            raise ValueError(f"{trajectory.trajectory_id}: permanent checkpoint policy drifted")
        if train_config.trainer.checkpointer.save_interval != CHECKPOINT_INTERVAL:
            raise ValueError(f"{trajectory.trajectory_id}: temporary checkpoint interval drifted")
        if train_config.trainer.checkpointer.keep_last_temporary_checkpoints != 1:
            raise ValueError(f"{trajectory.trajectory_id}: temporary checkpoint retention drifted")
        expected_temporary_prefix = prefix_join(
            marin_prefix,
            f"temporary_checkpoints/{NAME}/"
            f"{'canary_decay_forks' if decay_fork else 'canary_forks' if fork else 'trajectories'}/{VERSION}",
        )
        if train_config.trainer.checkpointer.temporary_base_path != expected_temporary_prefix:
            raise ValueError(f"{trajectory.trajectory_id}: temporary checkpoint prefix drifted")
        if fork:
            fork_source_step = DECAY_FORK_SOURCE_STEP if decay_fork else design.fork_source_step
            expected_source = _trajectory_checkpoint_path(
                marin_prefix,
                trajectory.trajectory_id,
                fork_source_step,
            )
            if (
                train_config.trainer.initialize_from != expected_source
                or train_config.trainer.load_checkpoint is not False
            ):
                raise ValueError(f"{trajectory.trajectory_id}: exact-state fork source drifted")
        elif train_config.trainer.initialize_from is not None:
            raise ValueError(f"{trajectory.trajectory_id}: parent unexpectedly initializes from another run")


def _validate_source(design: CanaryDesign, marin_prefix: str) -> None:
    request = _SourceValidationRequest(
        starcoder_support_batches=design.starcoder_support_batches,
        starcoder_realized_support_tokens=design.starcoder_support_tokens,
    )
    dense._validate_starcoder_source(
        marin_prefix,
        cast(tuple[dense.SurfaceRun, ...], (request,)),
    )


def _require_fork_sources(
    design: CanaryDesign,
    *,
    marin_prefix: str,
    decay_fork: bool = False,
) -> None:
    source_step = DECAY_FORK_SOURCE_STEP if decay_fork else design.fork_source_step
    reference_step = DECAY_FORK_REFERENCE_STEP if decay_fork else design.fork_reference_step
    missing: list[str] = []
    for trajectory in design.trajectories:
        for step in (source_step, reference_step):
            checkpoint = _trajectory_checkpoint_path(
                marin_prefix,
                trajectory.trajectory_id,
                step,
            )
            if not is_checkpoint_path(checkpoint):
                missing.append(checkpoint)
    if missing:
        raise FileNotFoundError(f"Exact-fork source checkpoints are not complete: {missing}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("canary-train", "canary-fork", "canary-decay-fork"),
        default="canary-train",
    )
    parser.add_argument("--marin-prefix", default=base.DEFAULT_MARIN_PREFIX)
    parser.add_argument("--tpu-type", default=base.DEFAULT_TPU_TYPE)
    parser.add_argument("--tpu-region", default=base.DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=base.DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--audit-runtime-configs", action="store_true")
    parser.add_argument("--audit-source", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    if os.getenv("CI") is not None:
        logger.info("Skipping WSD80 gradient-conflict canary in CI")
        return
    if args.marin_prefix != base.DEFAULT_MARIN_PREFIX:
        raise ValueError(f"Historical StarCoder work must remain central1-local: {args.marin_prefix!r}")
    if args.tpu_type != base.DEFAULT_TPU_TYPE:
        raise ValueError(f"Historical StarCoder accelerator is frozen: {args.tpu_type!r}")
    if args.tpu_region != base.DEFAULT_TPU_REGION or args.tpu_zone != base.DEFAULT_TPU_ZONE:
        raise ValueError(
            "StarCoder child placement must remain central1-local: "
            f"region={args.tpu_region!r}, zone={args.tpu_zone!r}"
        )
    if args.max_concurrent < 1 or args.max_concurrent > len(EXPECTED_TRAJECTORY_IDS):
        raise ValueError(f"Canary concurrency must be in [1, {len(EXPECTED_TRAJECTORY_IDS)}]")

    dense._validate_runtime_scientific_environment()
    design = load_canary_design()
    fork = args.stage in {"canary-fork", "canary-decay-fork"}
    decay_fork = args.stage == "canary-decay-fork"
    steps = build_training_steps(
        design,
        marin_prefix=args.marin_prefix,
        tpu_type=args.tpu_type,
        tpu_region=args.tpu_region,
        tpu_zone=args.tpu_zone,
        fork=fork,
        decay_fork=decay_fork,
    )
    audit_runtime_configs(
        design,
        steps,
        marin_prefix=args.marin_prefix,
        fork=fork,
        decay_fork=decay_fork,
    )
    if args.audit_runtime_configs:
        logger.info("Audited %d canary runtime configs", len(steps))
        return
    if args.dry_run:
        for step in steps:
            lower(step)
        logger.info("Lowered %d canary %s graphs", len(steps), args.stage)
        return

    os.environ["MARIN_PREFIX"] = args.marin_prefix
    _validate_source(design, args.marin_prefix)
    if args.audit_source:
        return
    if fork:
        _require_fork_sources(design, marin_prefix=args.marin_prefix, decay_fork=decay_fork)
    run(*steps, max_concurrent=min(args.max_concurrent, len(steps)))


if __name__ == "__main__":
    main()
