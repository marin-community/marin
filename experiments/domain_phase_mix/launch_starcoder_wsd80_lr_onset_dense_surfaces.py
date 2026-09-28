# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the frozen StarCoder WSD80 LR-onset endpoint surfaces."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import os
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import gcsfs
import jax
import jax.numpy as jnp
import numpy as np
from fray.types import ResourceConfig
from levanter.main.train_lm import TrainLmConfig
from marin.execution.lazy import ArtifactStep, StepContext, lower, materialized_config, run
from marin.experiment.data import mixture
from marin.experiment.train import train_lm
from marin.processing.tokenize.tokenize import TokenizedCache
from marin.training.training import LevanterCheckpoint, TrainLmOnPodConfig

from experiments.datasets.dolma import DOLMA_LLAMA3_OVERRIDES, dolma_datasets
from experiments.datasets.nemotron import nemotron_datasets
from experiments.datasets.paloma import paloma_datasets
from experiments.datasets.uncheatable import uncheatable_datasets
from experiments.domain_phase_mix import launch_starcoder_wsd80_dense_support_surfaces as source_surface
from experiments.domain_phase_mix import launch_starcoder_wsd_80_20_surface as base
from experiments.llama import llama3_tokenizer, llama3_tokenizer_vocab_size
from experiments.scaling_law_sweeps.completed_adamh import CompletedAdamHHeuristic

logger = logging.getLogger(__name__)

DESIGN_PATH = Path(__file__).with_name("starcoder_wsd80_lr_onset_dense_surface_design_20260825.json.gz")
EXPECTED_DESIGN_SHA256 = "99d3af5aff1f62e679c200219e92914df1f12a407e9afe08cd365c0a8079d889"
EXPECTED_RUN_COUNT = 644
EXPECTED_STAGE_COUNTS = {"surface_discovery": 500, "primary_spine": 80, "replay_replication": 64}
EXPECTED_CELL_ID = "r3_increase_d_h0640_s28260"
EXPECTED_SUPPORT_IDS = frozenset({"m100", "m200"})
EXPECTED_MAIN_ARM_IDS = frozenset({"decay_0p60", "decay_0p80", "decay_0p90", "no_decay"})
EXPECTED_ARM_IDS = EXPECTED_MAIN_ARM_IDS | {"decay_0p80_area_match_0p60"}
EXPECTED_TOTAL_STEPS = 28_260
EXPECTED_BOUNDARY_STEP = 22_608
EXPECTED_WARMUP_STEPS = 282
EXPECTED_FIRST_CONFIRMATION_SEED = 20_260_831
EXPECTED_MATERIALIZED_TOKENS = 7_408_189_440
EXPECTED_TOTAL_PARAMETERS = 210_052_480
EXPECTED_NON_EMBEDDING_PARAMETERS = 45_884_800
EXPECTED_TRAINING_ENVIRONMENT = {
    "jax_version": "0.11.0",
    "numpy_version": "2.3.5",
    "jax_default_prng_impl": "threefry2x32",
    "jax_enable_x64": False,
    "uv_lock_sha256": "5edf2440895451ed317d6a1c219b5ce266b10c31c0e486c9ea38bbe25f827566",
}
MAX_CONCURRENT = 128
TEMPORARY_CHECKPOINT_INTERVAL = timedelta(minutes=10)
FULL_LAUNCH_CONFIRMATION = "I_AUTHORIZE_THE_STARCODER_WSD80_LR_ONSET_DENSE_SURFACES"
SOURCE_PLACEMENT = {
    "marin_prefix": "gs://marin-us-central1",
    "tpu_type": "v5p-8",
    "region": "us-central1",
    "zone": "us-central1-a",
}
CENTRAL2_STARCODER_INPUT_IDS_CONTRACT = {
    "object_count": 1_618,
    "total_bytes": 344_101_194_077,
    "metadata_sha256": "fa8d0341b94ce701e8cf115ae695e522a8cb13d99a1ec5d42d49080a514a870c",
}
NON_DATA_CACHE_OBJECTS = frozenset({"shard_ledger.json", "shard_ledger.json.bak"})

REPO_ROOT = Path(__file__).resolve().parents[2]
DESIGN_GENERATOR_PATH = (
    Path(__file__).parent / "exploratory/two_phase_many/design_starcoder_wsd80_lr_onset_dense_surfaces_20260825.py"
)


@dataclass(frozen=True)
class Deployment:
    """Region-local execution identity for the shared frozen scientific design."""

    deployment_id: str
    name: str
    version: str
    wandb_group: str
    panel_tag: str
    run_id_prefix: str
    marin_prefix: str
    tpu_type: str
    tpu_region: str
    tpu_zone: str
    output_dir: Path
    cc_review_path: Path
    allow_empty_starcoder_finished_shards: bool = False

    @property
    def release_path(self) -> Path:
        return self.output_dir / "release.json"

    def release_record(self) -> dict[str, Any]:
        return {
            "deployment_id": self.deployment_id,
            "name": self.name,
            "version": self.version,
            "wandb_group": self.wandb_group,
            "panel_tag": self.panel_tag,
            "run_id_prefix": self.run_id_prefix,
            "marin_prefix": self.marin_prefix,
            "tpu_type": self.tpu_type,
            "tpu_region": self.tpu_region,
            "tpu_zone": self.tpu_zone,
            "allow_empty_starcoder_finished_shards": self.allow_empty_starcoder_finished_shards,
        }


CENTRAL1_V5P_DEPLOYMENT = Deployment(
    deployment_id="central1-v5p",
    name="pinlin_calvin_xu/data_mixture/starcoder_wsd80_lr_onset_dense_surfaces_20260825",
    version="2026.08.25.1",
    wandb_group="starcoder_wsd80_lr_onset_dense_surfaces_20260825",
    panel_tag="starcoder_wsd80_lr_onset_dense_surfaces",
    run_id_prefix="",
    marin_prefix="gs://marin-us-central1",
    tpu_type="v5p-8",
    tpu_region="us-central1",
    tpu_zone="us-central1-a",
    output_dir=Path(__file__).parent / "manifests/starcoder_wsd80_lr_onset_dense_surfaces_v1_20260825",
    cc_review_path=REPO_ROOT / ".agents/handoffs/starcoder_wsd80_lr_onset_dense_surfaces_cc_review_20260825.md",
)
CENTRAL2_V4_DEPLOYMENT = Deployment(
    deployment_id="central2-v4",
    name="pinlin_calvin_xu/data_mixture/starcoder_wsd80_lr_onset_dense_surfaces_central2_v4_20260828",
    version="2026.08.28.1",
    wandb_group="starcoder_wsd80_lr_onset_dense_surfaces_central2_v4_20260828",
    panel_tag="starcoder_wsd80_lr_onset_dense_surfaces_central2_v4",
    run_id_prefix="c2v4_",
    marin_prefix="gs://marin-us-central2",
    tpu_type="v4-8",
    tpu_region="us-central2",
    tpu_zone="us-central2-b",
    output_dir=Path(__file__).parent / "manifests/starcoder_wsd80_lr_onset_dense_surfaces_central2_v4_v1_20260828",
    cc_review_path=REPO_ROOT
    / ".agents/handoffs/starcoder_wsd80_lr_onset_dense_surfaces_central2_v4_cc_review_20260828.md",
    allow_empty_starcoder_finished_shards=True,
)
DEPLOYMENTS = {deployment.deployment_id: deployment for deployment in (CENTRAL1_V5P_DEPLOYMENT, CENTRAL2_V4_DEPLOYMENT)}


@dataclass(frozen=True)
class DenseOnsetRun:
    """One frozen optimizer-schedule, policy, support, and seed row."""

    row_id: str
    run_order: int
    run_name: str
    stage: str
    cell_id: str
    cell_slug: str
    rung: int
    hidden_size: int
    total_steps: int
    boundary_step: int
    materialized_tokens: int
    total_parameters: int
    non_embedding_parameters: int
    support_id: str
    support_role: str
    epoch_multiplier: float
    starcoder_support_batches: int
    starcoder_realized_support_tokens: int
    starcoder_support_fraction: float
    coordinate_id: str
    coordinate_sources: list[str]
    selection_class: str
    phase_0_starcoder: float
    phase_1_starcoder: float
    aggregate_starcoder: float
    phase_contrast: float
    starcoder_phase_0_sequences: int
    starcoder_phase_1_sequences: int
    starcoder_total_sequences: int
    starcoder_phase_0_epochs: float
    starcoder_phase_1_epochs: float
    starcoder_support_wraps: bool
    arm_id: str
    arm_role: str
    decay_onset_fraction: float | None
    peak_lr_multiplier: float
    optimizer: dict[str, Any]
    data_seed: int
    trainer_seed: int


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_payload() -> dict[str, Any]:
    payload = json.loads(gzip.decompress(DESIGN_PATH.read_bytes()))
    claimed_hash = payload.pop("design_sha256", None)
    observed_hash = _canonical_sha256(payload)
    if claimed_hash != EXPECTED_DESIGN_SHA256 or observed_hash != EXPECTED_DESIGN_SHA256:
        raise ValueError(f"Design hash drifted: {observed_hash} != {claimed_hash}")
    if payload.get("training_environment") != EXPECTED_TRAINING_ENVIRONMENT:
        raise ValueError("Frozen training environment drifted")
    if payload.get("placement") != SOURCE_PLACEMENT:
        raise ValueError("Frozen source-design placement drifted")
    payload["design_sha256"] = claimed_hash
    return payload


def load_design(
    *,
    selected_stage: str | None = None,
    selected_arms: frozenset[str] | None = None,
    selected_runs: frozenset[str] | None = None,
) -> tuple[dict[str, Any], tuple[DenseOnsetRun, ...]]:
    """Load and validate the frozen row inventory, optionally filtering it."""
    payload = _load_payload()
    rows = tuple(DenseOnsetRun(**row) for row in payload["runs"])
    if len(rows) != EXPECTED_RUN_COUNT or payload.get("expected_run_count") != EXPECTED_RUN_COUNT:
        raise ValueError(f"Expected {EXPECTED_RUN_COUNT} rows, got {len(rows)}")
    stage_counts = {stage: sum(row.stage == stage for row in rows) for stage in EXPECTED_STAGE_COUNTS}
    if stage_counts != EXPECTED_STAGE_COUNTS or payload.get("stage_counts") != EXPECTED_STAGE_COUNTS:
        raise ValueError(f"Stage inventory drifted: {stage_counts}")
    if {row.cell_id for row in rows} != {EXPECTED_CELL_ID}:
        raise ValueError("Cell inventory drifted")
    if {row.support_id for row in rows} != EXPECTED_SUPPORT_IDS:
        raise ValueError("Support inventory drifted")
    if {row.arm_id for row in rows} != EXPECTED_ARM_IDS:
        raise ValueError("Schedule-arm inventory drifted")
    if len({row.row_id for row in rows}) != len(rows) or len({row.run_name for row in rows}) != len(rows):
        raise ValueError("Run identities are not unique")
    if [row.run_order for row in rows] != list(range(len(rows))):
        raise ValueError("Run order is not contiguous")

    if selected_stage is not None:
        if selected_stage not in EXPECTED_STAGE_COUNTS:
            raise ValueError(f"Unknown stage: {selected_stage}")
        rows = tuple(row for row in rows if row.stage == selected_stage)
    if selected_arms is not None:
        unknown = selected_arms - EXPECTED_ARM_IDS
        if unknown:
            raise ValueError(f"Unknown schedule arms: {sorted(unknown)}")
        rows = tuple(row for row in rows if row.arm_id in selected_arms)
    if selected_runs is not None:
        available = {row.run_name for row in rows}
        unknown = selected_runs - available
        if unknown:
            raise ValueError(f"Unknown runs after stage/arm filtering: {sorted(unknown)}")
        rows = tuple(row for row in rows if row.run_name in selected_runs)
    if not rows:
        raise ValueError("Launch filters selected no rows")
    return payload, rows


def _optimizer(row: DenseOnsetRun):
    optimizer = base._optimizer(row.materialized_tokens)
    profile = row.optimizer
    configured = replace(
        optimizer,
        learning_rate=float(profile["learning_rate"]),
        adam_lr=float(profile["adam_lr"]),
        decay=int(profile["decay_steps"]),
        min_lr_ratio=float(profile["min_lr_ratio"]),
    )
    if configured.warmup != EXPECTED_WARMUP_STEPS or configured.lr_schedule != "cosine":
        raise ValueError(f"{row.run_name}: non-treatment optimizer fields drifted")
    return configured


def _keep_policy(row: DenseOnsetRun) -> list[dict[str, int | None]]:
    terminal = {"every": row.total_steps - 1, "until": None}
    if row.stage == "surface_discovery":
        return [terminal]
    return [{"every": row.boundary_step, "until": row.boundary_step}, terminal]


def _configure_training(
    training: ArtifactStep[LevanterCheckpoint],
    *,
    train_datasets: dict[ArtifactStep[TokenizedCache], float],
    validation_datasets: tuple[ArtifactStep[TokenizedCache], ...],
    phase_weights: list[tuple[int, dict[str, float]]],
    starcoder_name: str,
    row: DenseOnsetRun,
) -> ArtifactStep[LevanterCheckpoint]:
    """Install the frozen support, phase, seed, and checkpoint contract."""

    def build_config(ctx: StepContext) -> TrainLmOnPodConfig:
        pod_config = training.build_config(ctx)
        train_config = cast(TrainLmConfig, pod_config.train_config)
        data_config = mixture(ctx, train_datasets, validation=validation_datasets)
        if starcoder_name not in data_config.components:
            raise ValueError(f"{row.run_name}: StarCoder support key is absent from mixture components")
        data_config = replace(
            data_config,
            train_weights=phase_weights,
            mixture_block_size=base.MIXTURE_BLOCK_SIZE,
            experiment_budget=None,
            target_budget=None,
            simulated_epoch_subset_seed=None,
            max_train_batches={starcoder_name: row.starcoder_support_batches},
            max_train_batches_subset_seed=None,
            max_train_batches_start=None,
            train_holdout_sequences=None,
            train_holdout_seed=None,
            train_holdout_partition=None,
        )
        trainer = replace(
            train_config.trainer,
            seed=row.trainer_seed,
            checkpointer=replace(
                train_config.trainer.checkpointer,
                save_interval=TEMPORARY_CHECKPOINT_INTERVAL,
                keep=_keep_policy(row),
                keep_last_temporary_checkpoints=1,
            ),
        )
        train_config = replace(
            train_config,
            data=data_config,
            data_seed=row.data_seed,
            trainer=trainer,
        )
        return replace(pod_config, train_config=train_config)

    return replace(training, build_config=build_config)


def _runtime_environment() -> dict[str, str | bool]:
    return {
        "jax_version": jax.__version__,
        "numpy_version": np.__version__,
        "jax_default_prng_impl": jax.config.jax_default_prng_impl,
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "uv_lock_sha256": _file_sha256(REPO_ROOT / "uv.lock"),
    }


def _validate_runtime_environment() -> None:
    observed = _runtime_environment()
    if observed != EXPECTED_TRAINING_ENVIRONMENT:
        raise ValueError(f"Runtime scientific environment drifted: {observed}")


def _configure_parent_jax() -> None:
    # The launcher audits schedules on a CPU coordinator; TPU training happens in separate child tasks.
    jax.config.update("jax_platforms", "cpu")
    if jax.default_backend() != "cpu":
        raise RuntimeError("The LR-onset launcher audit requires a CPU JAX backend")


def _validate_model(payload: dict[str, Any]) -> Any:
    cell = payload["cell"]
    if (
        cell["cell_id"] != EXPECTED_CELL_ID
        or cell["total_steps"] != EXPECTED_TOTAL_STEPS
        or cell["boundary_step"] != EXPECTED_BOUNDARY_STEP
        or cell["materialized_tokens"] != EXPECTED_MATERIALIZED_TOKENS
    ):
        raise ValueError("Frozen cell geometry drifted")
    model = CompletedAdamHHeuristic()._build_model_config(cell["hidden_size"], seq_len=base.SEQ_LEN)
    if model.total_trainable_params(llama3_tokenizer_vocab_size) != EXPECTED_TOTAL_PARAMETERS:
        raise ValueError("Total model parameter count drifted")
    if model.total_trainable_params(0) != EXPECTED_NON_EMBEDDING_PARAMETERS:
        raise ValueError("Non-embedding parameter count drifted")
    return model


def build_training_steps(
    *,
    deployment: Deployment = CENTRAL1_V5P_DEPLOYMENT,
    selected_stage: str | None = None,
    selected_arms: frozenset[str] | None = None,
    selected_runs: frozenset[str] | None = None,
) -> tuple[tuple[DenseOnsetRun, ...], tuple[ArtifactStep[LevanterCheckpoint], ...]]:
    """Build independently resumable training artifacts for frozen rows."""
    payload, rows = load_design(
        selected_stage=selected_stage,
        selected_arms=selected_arms,
        selected_runs=selected_runs,
    )
    model = _validate_model(payload)
    nemotron = nemotron_datasets(tokenizer=llama3_tokenizer)
    starcoder = dolma_datasets(tokenizer=llama3_tokenizer)["dolma/starcoder"]
    training_handles = tuple([nemotron[split] for split in base.NEMOTRON_TOKEN_COUNTS] + [starcoder])
    expected_names = (*tuple(nemotron[split].name for split in base.NEMOTRON_TOKEN_COUNTS), starcoder.name)
    if tuple(handle.name for handle in training_handles) != expected_names:
        raise ValueError("Training component ordering drifted")
    validation = (
        *paloma_datasets(tokenizer=llama3_tokenizer).values(),
        *uncheatable_datasets(tokenizer=llama3_tokenizer).values(),
    )
    resources = ResourceConfig.with_tpu(
        deployment.tpu_type,
        regions=(deployment.tpu_region,),
        zone=deployment.tpu_zone,
    )

    steps: list[ArtifactStep[LevanterCheckpoint]] = []
    for row in rows:
        phase_0_weights = base._phase_leaf_weights(
            row.phase_0_starcoder,
            nemotron=nemotron,
            starcoder=starcoder,
        )
        phase_1_weights = base._phase_leaf_weights(
            row.phase_1_starcoder,
            nemotron=nemotron,
            starcoder=starcoder,
        )
        static_weights = {handle: phase_0_weights[handle.name] for handle in training_handles}
        training = train_lm(
            name=f"checkpoints/{deployment.name}/{row.run_name}",
            version=deployment.version,
            model=model,
            optimizer=_optimizer(row),
            datasets=static_weights,
            validation=validation,
            batch_size=base.BATCH_SIZE,
            seq_len=base.SEQ_LEN,
            num_train_steps=row.total_steps,
            z_loss_weight=None,
            evals=None,
            resources=resources,
            steps_per_eval=1_000,
            wandb_project="marin",
            wandb_group=deployment.wandb_group,
            run_id=f"{deployment.run_id_prefix}{row.run_name}",
            tags=(
                deployment.panel_tag,
                deployment.deployment_id,
                deployment.tpu_type,
                row.stage,
                row.support_id,
                row.arm_id,
                row.coordinate_id,
                "starcoder",
                "wsd80_20",
            ),
            env_vars={"HF_ALLOW_CODE_EVAL": "1"},
        )
        steps.append(
            _configure_training(
                training,
                train_datasets=static_weights,
                validation_datasets=validation,
                phase_weights=[(0, phase_0_weights), (row.boundary_step, phase_1_weights)],
                starcoder_name=starcoder.name,
                row=row,
            )
        )
    return rows, tuple(steps)


def _schedule_vector(row: DenseOnsetRun) -> np.ndarray:
    return np.asarray(_optimizer(row).lr_scheduler(row.total_steps)(jnp.arange(row.total_steps)))


def _data_contract(train_config: TrainLmConfig) -> dict[str, Any]:
    weights = train_config.data.train_weights
    if not isinstance(weights, list):
        raise ValueError("Expected phase-indexed train weights")
    return {
        "components": tuple(sorted(train_config.data.components)),
        "train_weights": tuple(
            (boundary, tuple(sorted((name, float(weight)) for name, weight in phase.items())))
            for boundary, phase in weights
        ),
        "mixture_block_size": train_config.data.mixture_block_size,
        "experiment_budget": train_config.data.experiment_budget,
        "target_budget": train_config.data.target_budget,
        "simulated_epoch_subset_seed": train_config.data.simulated_epoch_subset_seed,
        "max_train_batches": train_config.data.max_train_batches,
        "max_train_batches_subset_seed": train_config.data.max_train_batches_subset_seed,
        "max_train_batches_start": train_config.data.max_train_batches_start,
        "train_holdout_sequences": train_config.data.train_holdout_sequences,
        "train_holdout_seed": train_config.data.train_holdout_seed,
        "train_holdout_partition": train_config.data.train_holdout_partition,
        "data_seed": train_config.data_seed,
        "trainer_seed": train_config.trainer.seed,
    }


def audit_materialized_runtime_configs(
    rows: tuple[DenseOnsetRun, ...],
    steps: tuple[ArtifactStep[LevanterCheckpoint], ...],
    *,
    deployment: Deployment = CENTRAL1_V5P_DEPLOYMENT,
) -> int:
    """Materialize representatives and fail closed on every treatment field."""
    if len(rows) != len(steps):
        raise ValueError("Row/step cardinality mismatch")
    chosen_indices: set[int] = set()
    seen_groups: set[tuple[str, str, str]] = set()
    for index, row in enumerate(rows):
        group = (row.stage, row.support_id, row.arm_id)
        if group not in seen_groups:
            chosen_indices.add(index)
            seen_groups.add(group)
    primary_identity = [
        index
        for index, row in enumerate(rows)
        if row.stage == "primary_spine"
        and row.support_id == "m100"
        and row.coordinate_id == "c109"
        and row.data_seed == EXPECTED_FIRST_CONFIRMATION_SEED
    ]
    chosen_indices.update(primary_identity)

    data_contracts: dict[tuple[str, str, int], dict[str, Any]] = {}
    for index in sorted(chosen_indices):
        row = rows[index]
        pod_config = cast(TrainLmOnPodConfig, materialized_config(steps[index], deployment.marin_prefix))
        train_config = cast(TrainLmConfig, pod_config.train_config)
        optimizer = train_config.optimizer
        profile = row.optimizer
        if train_config.trainer.num_train_steps != row.total_steps:
            raise ValueError(f"{row.run_name}: training horizon drifted")
        if train_config.optimizer_schedule_num_train_steps is not None:
            raise ValueError(f"{row.run_name}: optimizer schedule horizon override leaked in")
        if (
            optimizer.decay != profile["decay_steps"]
            or optimizer.min_lr_ratio != profile["min_lr_ratio"]
            or optimizer.learning_rate != profile["learning_rate"]
            or optimizer.adam_lr != profile["adam_lr"]
            or optimizer.warmup != EXPECTED_WARMUP_STEPS
        ):
            raise ValueError(f"{row.run_name}: optimizer treatment drifted")
        contract = _data_contract(train_config)
        if contract["train_weights"][0][0] != 0 or contract["train_weights"][1][0] != row.boundary_step:
            raise ValueError(f"{row.run_name}: data phase boundary drifted")
        if contract["max_train_batches"] != {"dolma/starcoder": row.starcoder_support_batches}:
            raise ValueError(f"{row.run_name}: StarCoder support cap drifted")
        if contract["train_holdout_sequences"] is not None:
            raise ValueError(f"{row.run_name}: a training holdout leaked into the dense family")
        if contract["data_seed"] != row.data_seed or contract["trainer_seed"] != row.trainer_seed:
            raise ValueError(f"{row.run_name}: seed drifted")
        if train_config.trainer.checkpointer.keep != _keep_policy(row):
            raise ValueError(f"{row.run_name}: phase-boundary checkpoint retention drifted")
        if train_config.trainer.checkpointer.save_interval != TEMPORARY_CHECKPOINT_INTERVAL:
            raise ValueError(f"{row.run_name}: temporary checkpoint interval drifted")
        identity = (row.support_id, row.coordinate_id, row.data_seed)
        previous = data_contracts.setdefault(identity, contract)
        if previous != contract:
            raise ValueError(f"{row.run_name}: data stream differs across optimizer arms")

    _, all_rows = load_design()
    representative = {
        row.arm_id: row for row in all_rows if row.stage == "surface_discovery" and row.coordinate_id == "c000"
    }
    earliest_onset = min(
        int(representative[arm_id].optimizer["decay_onset_step"]) for arm_id in EXPECTED_MAIN_ARM_IDS - {"no_decay"}
    )
    reference = _schedule_vector(representative["decay_0p80"])[:earliest_onset]
    for arm_id in EXPECTED_MAIN_ARM_IDS:
        observed = _schedule_vector(representative[arm_id])[:earliest_onset]
        if not np.array_equal(observed, reference):
            raise ValueError(f"{arm_id}: main-arm LR schedule differs before the earliest treatment")
    no_decay = _schedule_vector(representative["no_decay"])
    if not np.all(no_decay[EXPECTED_WARMUP_STEPS:] == no_decay[EXPECTED_WARMUP_STEPS]):
        raise ValueError("no_decay is not flat after warmup")
    phase_1_integrals = [
        representative[arm_id].optimizer["normalized_phase_1_lr_integral"]
        for arm_id in ("decay_0p60", "decay_0p80", "decay_0p90", "no_decay")
    ]
    if phase_1_integrals != sorted(phase_1_integrals):
        raise ValueError(f"Phase-1 optimizer budgets are not ordered: {phase_1_integrals}")
    return len(chosen_indices)


def _validate_starcoder_source(deployment: Deployment, rows: tuple[DenseOnsetRun, ...]) -> str:
    if not deployment.allow_empty_starcoder_finished_shards:
        return source_surface._validate_starcoder_source(deployment.marin_prefix, cast(Any, rows))

    starcoder = dolma_datasets(tokenizer=llama3_tokenizer)["dolma/starcoder"]
    expected_cache_dir = source_surface.prefix_join(
        deployment.marin_prefix,
        DOLMA_LLAMA3_OVERRIDES["starcoder"],
    )
    observed_cache_dir = starcoder.path(deployment.marin_prefix)
    if observed_cache_dir != expected_cache_dir:
        raise ValueError(f"StarCoder cache identity drifted: {observed_cache_dir!r} != {expected_cache_dir!r}")

    ledger = source_surface.CacheLedger.load(source_surface.prefix_join(observed_cache_dir, "train"))
    expected_shards = {f"{index:02d}_json_gz" for index in range(source_surface.EXPECTED_STARCODER_CACHE_SHARDS)}
    if not ledger.is_finished or ledger.layout != source_surface.CACHE_LAYOUT_CONSOLIDATED:
        raise ValueError(f"StarCoder cache is not a finished consolidated cache: {observed_cache_dir}")
    if ledger.total_num_rows != source_surface.EXPECTED_STARCODER_CACHE_DOCUMENTS:
        raise ValueError("StarCoder cache document count drifted")
    if set(ledger.shard_rows) != expected_shards:
        raise ValueError("StarCoder cache shard-row identities drifted")
    if ledger.finished_shards and set(ledger.finished_shards) != expected_shards:
        raise ValueError("StarCoder cache finished-shard identities drifted")
    if ledger.metadata.preprocessor_metadata != source_surface.EXPECTED_STARCODER_CACHE_TOKENIZER_METADATA:
        raise ValueError("StarCoder cache tokenizer metadata drifted")

    object_contract = _cache_object_contract(observed_cache_dir)
    if object_contract != CENTRAL2_STARCODER_INPUT_IDS_CONTRACT:
        raise ValueError(
            "Central2 StarCoder final-object contract drifted: "
            f"{object_contract} != {CENTRAL2_STARCODER_INPUT_IDS_CONTRACT}"
        )

    finite_support_tokens = [row.starcoder_realized_support_tokens for row in rows]
    if finite_support_tokens and max(finite_support_tokens) >= source_surface.EXPECTED_STARCODER_SOURCE_TOKENS:
        raise ValueError("A finite StarCoder support cap no longer binds below the frozen physical source")
    logger.info(
        "Validated central2 StarCoder cache %s (%d documents, %d shard rows, %d finished-shard entries)",
        observed_cache_dir,
        ledger.total_num_rows,
        len(ledger.shard_rows),
        len(ledger.finished_shards),
    )
    return observed_cache_dir


def _cache_object_contract(cache_dir: str) -> dict[str, int | str]:
    """Fingerprint final cache objects without reading their payload bytes."""
    root = f"{cache_dir.removeprefix('gs://').rstrip('/')}/train"
    metadata = gcsfs.GCSFileSystem(token="google_default").find(root, detail=True)
    rows = []
    for path, item in metadata.items():
        relative_path = path.removeprefix(f"{root}/")
        if relative_path in NON_DATA_CACHE_OBJECTS or relative_path.startswith("___temp/"):
            continue
        rows.append(
            {
                "path": relative_path,
                "size": int(item["size"]),
                "crc32c": item.get("crc32c"),
                "md5": item.get("md5Hash"),
            }
        )
    rows.sort(key=lambda row: cast(str, row["path"]))
    return {
        "object_count": len(rows),
        "total_bytes": sum(cast(int, row["size"]) for row in rows),
        "metadata_sha256": _canonical_sha256(rows),
    }


def _freeze_release(deployment: Deployment) -> dict[str, Any]:
    if not deployment.cc_review_path.exists() or not deployment.cc_review_path.read_text(
        encoding="utf-8"
    ).rstrip().endswith("VERDICT: PASS"):
        raise ValueError("A completed CC review ending in VERDICT: PASS is required before release freeze")
    payload = _load_payload()
    release = {
        "schema_version": "2026-08-28-starcoder-wsd80-lr-onset-dense-deployment-v2",
        "design_sha256": payload["design_sha256"],
        "design_path": str(DESIGN_PATH.relative_to(REPO_ROOT)),
        "source_design_placement": SOURCE_PLACEMENT,
        "deployment": deployment.release_record(),
        "central2_starcoder_input_ids_contract": (
            CENTRAL2_STARCODER_INPUT_IDS_CONTRACT if deployment.deployment_id == "central2-v4" else None
        ),
        "design_file_sha256": _file_sha256(DESIGN_PATH),
        "launcher_path": str(Path(__file__).relative_to(REPO_ROOT)),
        "launcher_sha256": _file_sha256(Path(__file__)),
        "design_generator_path": str(DESIGN_GENERATOR_PATH.relative_to(REPO_ROOT)),
        "design_generator_sha256": _file_sha256(DESIGN_GENERATOR_PATH),
        "cc_review_path": str(deployment.cc_review_path.relative_to(REPO_ROOT)),
        "cc_review_sha256": _file_sha256(deployment.cc_review_path),
        "training_environment": EXPECTED_TRAINING_ENVIRONMENT,
        "maximum_concurrent": MAX_CONCURRENT,
        "confirmation": FULL_LAUNCH_CONFIRMATION,
        "release_sha256": "",
    }
    release["release_sha256"] = _canonical_sha256(release)
    deployment.output_dir.mkdir(parents=True, exist_ok=True)
    deployment.release_path.write_text(json.dumps(release, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return release


def _load_release(deployment: Deployment) -> dict[str, Any]:
    release = json.loads(deployment.release_path.read_text(encoding="utf-8"))
    claimed_hash = release["release_sha256"]
    observed_hash = _canonical_sha256({**release, "release_sha256": ""})
    if claimed_hash != observed_hash:
        raise ValueError("Release hash drifted")
    if release.get("deployment") != deployment.release_record():
        raise ValueError("Release deployment drifted")
    checks = {
        REPO_ROOT / release["design_path"]: release["design_file_sha256"],
        REPO_ROOT / release["launcher_path"]: release["launcher_sha256"],
        REPO_ROOT / release["design_generator_path"]: release["design_generator_sha256"],
        REPO_ROOT / release["cc_review_path"]: release["cc_review_sha256"],
        REPO_ROOT / "uv.lock": release["training_environment"]["uv_lock_sha256"],
    }
    drifted = [str(path) for path, expected in checks.items() if _file_sha256(path) != expected]
    if drifted:
        raise ValueError(f"Frozen release files drifted: {drifted}")
    return release


def _parse_csv_set(value: str | None) -> frozenset[str] | None:
    if value is None:
        return None
    values = frozenset(item.strip() for item in value.split(",") if item.strip())
    if not values:
        raise ValueError("Comma-separated filter is empty")
    return values


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment", choices=tuple(DEPLOYMENTS), default=CENTRAL1_V5P_DEPLOYMENT.deployment_id)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stage", choices=tuple(EXPECTED_STAGE_COUNTS))
    parser.add_argument("--arms")
    parser.add_argument("--runs")
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT)
    parser.add_argument("--confirmation")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    if os.getenv("CI") is not None:
        logger.info("Skipping StarCoder WSD80 LR-onset dense surfaces in CI")
        return
    _configure_parent_jax()
    args = _parse_args()
    deployment = DEPLOYMENTS[args.deployment]
    if args.freeze:
        release = _freeze_release(deployment)
        logger.info("Frozen release %s", release["release_sha256"])
        return
    _validate_runtime_environment()
    selected_arms = _parse_csv_set(args.arms)
    selected_runs = _parse_csv_set(args.runs)
    rows, steps = build_training_steps(
        deployment=deployment,
        selected_stage=args.stage,
        selected_arms=selected_arms,
        selected_runs=selected_runs,
    )
    audited = audit_materialized_runtime_configs(rows, steps, deployment=deployment)
    if args.audit:
        _validate_starcoder_source(deployment, rows)
        logger.info("Audited %d rows and %d representative runtime configs", len(rows), audited)
        return
    if args.dry_run:
        for step in steps:
            lower(step)
        logger.info("Lowered %d frozen training graphs", len(steps))
        return

    release = _load_release(deployment)
    if args.confirmation != release["confirmation"]:
        raise ValueError("Full launch confirmation is missing or incorrect")
    if args.stage is None:
        raise ValueError("External launch requires an explicit stage")
    if args.max_concurrent < 1 or args.max_concurrent > release["maximum_concurrent"]:
        raise ValueError("Requested concurrency is outside the frozen release")
    if os.getenv("MARIN_PREFIX", deployment.marin_prefix) != deployment.marin_prefix:
        raise ValueError(f"The {deployment.deployment_id} release must remain region-local")
    os.environ["MARIN_PREFIX"] = deployment.marin_prefix
    _validate_starcoder_source(deployment, rows)
    run(*steps, max_concurrent=min(args.max_concurrent, len(steps)), force_run_failed=True)


if __name__ == "__main__":
    main()
