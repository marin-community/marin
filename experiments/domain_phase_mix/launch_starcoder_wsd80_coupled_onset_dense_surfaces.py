# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch dense StarCoder surfaces with the phase and LR-decay onsets coupled."""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import jax.numpy as jnp
import numpy as np
from fray.types import ResourceConfig
from levanter.main.train_lm import TrainLmConfig
from marin.execution.lazy import ArtifactStep, lower, materialized_config, run
from marin.experiment.train import train_lm
from marin.training.training import LevanterCheckpoint, TrainLmOnPodConfig

from experiments.datasets.dolma import dolma_datasets
from experiments.datasets.nemotron import nemotron_datasets
from experiments.datasets.paloma import paloma_datasets
from experiments.datasets.uncheatable import uncheatable_datasets
from experiments.domain_phase_mix import launch_starcoder_wsd80_lr_onset_dense_surfaces as lr_only
from experiments.domain_phase_mix import launch_starcoder_wsd_80_20_surface as base
from experiments.llama import llama3_tokenizer, llama3_tokenizer_vocab_size
from experiments.scaling_law_sweeps.completed_adamh import CompletedAdamHHeuristic

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DESIGN_PATH = Path(__file__).with_name("starcoder_wsd80_coupled_onset_dense_surface_design_20260830.json.gz")
DESIGN_GENERATOR_PATH = (
    Path(__file__).parent / "exploratory/two_phase_many/design_starcoder_wsd80_coupled_onset_dense_surfaces_20260830.py"
)
EXPECTED_DESIGN_SHA256 = "423bfb51e546181f78c04863b6298d57a32a38b33b25eae6a1d1464a737010eb"
EXPECTED_RUN_COUNT = 375
EXPECTED_STAGE_COUNTS = {"surface_discovery": 375}
EXPECTED_ARM_IDS = frozenset({"coupled_0p60", "coupled_0p80", "coupled_0p90"})
EXPECTED_CELL_ID = "r3_increase_d_h0640_s28260"
EXPECTED_SUPPORT_ID = "m100"
EXPECTED_TOTAL_STEPS = 28_260
EXPECTED_WARMUP_STEPS = 282
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
FULL_LAUNCH_CONFIRMATION = "I_AUTHORIZE_THE_STARCODER_WSD80_COUPLED_ONSET_DENSE_SURFACES"

CENTRAL2_V4_DEPLOYMENT = lr_only.Deployment(
    deployment_id="central2-v4-coupled",
    name="pinlin_calvin_xu/data_mixture/starcoder_wsd80_coupled_onset_dense_surfaces_central2_v4_20260830",
    version="2026.08.30.1",
    wandb_group="starcoder_wsd80_coupled_onset_dense_surfaces_central2_v4_20260830",
    panel_tag="starcoder_wsd80_coupled_onset_dense_surfaces_central2_v4",
    run_id_prefix="c2v4c_",
    marin_prefix="gs://marin-us-central2",
    tpu_type="v4-8",
    tpu_region="us-central2",
    tpu_zone="us-central2-b",
    output_dir=Path(__file__).parent / "manifests/starcoder_wsd80_coupled_onset_dense_surfaces_central2_v4_v1_20260830",
    cc_review_path=Path(__file__),
    allow_empty_starcoder_finished_shards=True,
)


@dataclass(frozen=True)
class CoupledOnsetRun:
    """One frozen coupled phase-boundary and optimizer-schedule row."""

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
    requested_onset_fraction: float
    realized_onset_fraction: float
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
    normalized_fiber_position: float
    starcoder_phase_0_sequences: int
    starcoder_phase_1_sequences: int
    starcoder_total_sequences: int
    starcoder_phase_0_epochs: float
    starcoder_phase_1_epochs: float
    starcoder_support_wraps: bool
    arm_id: str
    arm_role: str
    decay_onset_fraction: float
    peak_lr_multiplier: float
    optimizer: dict[str, Any]
    data_seed: int
    trainer_seed: int


def _load_payload() -> dict[str, Any]:
    payload = json.loads(gzip.decompress(DESIGN_PATH.read_bytes()))
    claimed_hash = payload.pop("design_sha256", None)
    observed_hash = lr_only._canonical_sha256(payload)
    if claimed_hash != EXPECTED_DESIGN_SHA256 or observed_hash != EXPECTED_DESIGN_SHA256:
        raise ValueError(f"Design hash drifted: {observed_hash} != {claimed_hash}")
    if payload.get("training_environment") != EXPECTED_TRAINING_ENVIRONMENT:
        raise ValueError("Frozen training environment drifted")
    payload["design_sha256"] = claimed_hash
    return payload


def load_design(
    *,
    selected_arms: frozenset[str] | None = None,
    selected_runs: frozenset[str] | None = None,
) -> tuple[dict[str, Any], tuple[CoupledOnsetRun, ...]]:
    """Load and validate the frozen discovery inventory."""
    payload = _load_payload()
    rows = tuple(CoupledOnsetRun(**row) for row in payload["runs"])
    if len(rows) != EXPECTED_RUN_COUNT or payload.get("expected_run_count") != EXPECTED_RUN_COUNT:
        raise ValueError(f"Expected {EXPECTED_RUN_COUNT} rows, got {len(rows)}")
    if payload.get("stage_counts") != EXPECTED_STAGE_COUNTS or {row.stage for row in rows} != {"surface_discovery"}:
        raise ValueError("Stage inventory drifted")
    if {row.arm_id for row in rows} != EXPECTED_ARM_IDS:
        raise ValueError("Coupled-onset arm inventory drifted")
    if {row.cell_id for row in rows} != {EXPECTED_CELL_ID} or {row.support_id for row in rows} != {EXPECTED_SUPPORT_ID}:
        raise ValueError("Cell or support inventory drifted")
    if len({row.row_id for row in rows}) != len(rows) or len({row.run_name for row in rows}) != len(rows):
        raise ValueError("Run identities are not unique")
    if [row.run_order for row in rows] != list(range(len(rows))):
        raise ValueError("Run order is not contiguous")
    if any(row.boundary_step != row.optimizer["decay_onset_step"] for row in rows):
        raise ValueError("Phase and LR-decay onsets are not coupled")

    if selected_arms is not None:
        unknown = selected_arms - EXPECTED_ARM_IDS
        if unknown:
            raise ValueError(f"Unknown coupled-onset arms: {sorted(unknown)}")
        rows = tuple(row for row in rows if row.arm_id in selected_arms)
    if selected_runs is not None:
        available = {row.run_name for row in rows}
        unknown = selected_runs - available
        if unknown:
            raise ValueError(f"Unknown runs after arm filtering: {sorted(unknown)}")
        rows = tuple(row for row in rows if row.run_name in selected_runs)
    if not rows:
        raise ValueError("Launch filters selected no rows")
    return payload, rows


def _validate_model(payload: dict[str, Any]) -> Any:
    cell = payload["cell"]
    if (
        cell["cell_id"] != EXPECTED_CELL_ID
        or cell["total_steps"] != EXPECTED_TOTAL_STEPS
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
    deployment: lr_only.Deployment = CENTRAL2_V4_DEPLOYMENT,
    selected_arms: frozenset[str] | None = None,
    selected_runs: frozenset[str] | None = None,
) -> tuple[tuple[CoupledOnsetRun, ...], tuple[ArtifactStep[LevanterCheckpoint], ...]]:
    """Build independently resumable training artifacts for frozen rows."""
    payload, rows = load_design(selected_arms=selected_arms, selected_runs=selected_runs)
    model = _validate_model(payload)
    nemotron = nemotron_datasets(tokenizer=llama3_tokenizer)
    starcoder = dolma_datasets(tokenizer=llama3_tokenizer)["dolma/starcoder"]
    training_handles = tuple([nemotron[split] for split in base.NEMOTRON_TOKEN_COUNTS] + [starcoder])
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
            optimizer=lr_only._optimizer(cast(Any, row)),
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
                "coupled_phase_lr_onset",
            ),
            env_vars={"HF_ALLOW_CODE_EVAL": "1"},
        )
        steps.append(
            lr_only._configure_training(
                training,
                train_datasets=static_weights,
                validation_datasets=validation,
                phase_weights=[(0, phase_0_weights), (row.boundary_step, phase_1_weights)],
                starcoder_name=starcoder.name,
                row=cast(Any, row),
            )
        )
    return rows, tuple(steps)


def _schedule_vector(row: CoupledOnsetRun) -> np.ndarray:
    return np.asarray(lr_only._optimizer(cast(Any, row)).lr_scheduler(row.total_steps)(jnp.arange(row.total_steps)))


def audit_materialized_runtime_configs(
    rows: tuple[CoupledOnsetRun, ...],
    steps: tuple[ArtifactStep[LevanterCheckpoint], ...],
    *,
    deployment: lr_only.Deployment = CENTRAL2_V4_DEPLOYMENT,
) -> int:
    """Materialize one common coordinate per arm and fail closed on coupling."""
    if len(rows) != len(steps):
        raise ValueError("Row/step cardinality mismatch")
    chosen_indices: list[int] = []
    for arm_id in sorted({row.arm_id for row in rows}):
        chosen_indices.append(next(i for i, row in enumerate(rows) if row.arm_id == arm_id))

    starcoder_name = "dolma/starcoder"
    for index in chosen_indices:
        row = rows[index]
        pod_config = cast(TrainLmOnPodConfig, materialized_config(steps[index], deployment.marin_prefix))
        train_config = cast(TrainLmConfig, pod_config.train_config)
        optimizer = train_config.optimizer
        if row.boundary_step != row.optimizer["decay_onset_step"]:
            raise ValueError(f"{row.run_name}: manifest onsets are not coupled")
        if optimizer.decay != row.total_steps - row.boundary_step or optimizer.min_lr_ratio != 0.0:
            raise ValueError(f"{row.run_name}: optimizer treatment drifted")
        if optimizer.warmup != EXPECTED_WARMUP_STEPS or train_config.trainer.num_train_steps != row.total_steps:
            raise ValueError(f"{row.run_name}: shared training geometry drifted")
        weights = train_config.data.train_weights
        if not isinstance(weights, list) or [boundary for boundary, _ in weights] != [0, row.boundary_step]:
            raise ValueError(f"{row.run_name}: phase boundary drifted")
        if not np.isclose(weights[0][1][starcoder_name], row.phase_0_starcoder, atol=1e-12):
            raise ValueError(f"{row.run_name}: phase-0 StarCoder weight drifted")
        if not np.isclose(weights[1][1][starcoder_name], row.phase_1_starcoder, atol=1e-12):
            raise ValueError(f"{row.run_name}: phase-1 StarCoder weight drifted")
        if train_config.data.max_train_batches != {starcoder_name: row.starcoder_support_batches}:
            raise ValueError(f"{row.run_name}: finite support cap drifted")
        if train_config.data_seed != row.data_seed or train_config.trainer.seed != row.trainer_seed:
            raise ValueError(f"{row.run_name}: seed drifted")
        if train_config.trainer.checkpointer.keep != lr_only._keep_policy(cast(Any, row)):
            raise ValueError(f"{row.run_name}: checkpoint retention drifted")

    _, all_rows = load_design()
    representative = {row.arm_id: row for row in all_rows if row.coordinate_id == "c000"}
    earliest_onset = min(row.boundary_step for row in representative.values())
    reference = _schedule_vector(representative["coupled_0p80"])[:earliest_onset]
    for arm_id, row in representative.items():
        if not np.array_equal(_schedule_vector(row)[:earliest_onset], reference):
            raise ValueError(f"{arm_id}: LR schedule differs before the earliest coupled onset")
    return len(chosen_indices)


def _freeze_release(deployment: lr_only.Deployment = CENTRAL2_V4_DEPLOYMENT) -> dict[str, Any]:
    payload = _load_payload()
    release = {
        "schema_version": "2026-08-30-starcoder-wsd80-coupled-onset-v1",
        "design_sha256": payload["design_sha256"],
        "design_path": str(DESIGN_PATH.relative_to(REPO_ROOT)),
        "design_file_sha256": lr_only._file_sha256(DESIGN_PATH),
        "launcher_path": str(Path(__file__).relative_to(REPO_ROOT)),
        "launcher_sha256": lr_only._file_sha256(Path(__file__)),
        "design_generator_path": str(DESIGN_GENERATOR_PATH.relative_to(REPO_ROOT)),
        "design_generator_sha256": lr_only._file_sha256(DESIGN_GENERATOR_PATH),
        "deployment": deployment.release_record(),
        "training_environment": EXPECTED_TRAINING_ENVIRONMENT,
        "maximum_concurrent": MAX_CONCURRENT,
        "confirmation": FULL_LAUNCH_CONFIRMATION,
        "release_sha256": "",
    }
    release["release_sha256"] = lr_only._canonical_sha256(release)
    deployment.output_dir.mkdir(parents=True, exist_ok=True)
    deployment.release_path.write_text(json.dumps(release, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return release


def _load_release(deployment: lr_only.Deployment = CENTRAL2_V4_DEPLOYMENT) -> dict[str, Any]:
    release = json.loads(deployment.release_path.read_text(encoding="utf-8"))
    if release["release_sha256"] != lr_only._canonical_sha256({**release, "release_sha256": ""}):
        raise ValueError("Release hash drifted")
    if release.get("deployment") != deployment.release_record():
        raise ValueError("Release deployment drifted")
    checks = {
        REPO_ROOT / release["design_path"]: release["design_file_sha256"],
        REPO_ROOT / release["launcher_path"]: release["launcher_sha256"],
        REPO_ROOT / release["design_generator_path"]: release["design_generator_sha256"],
        REPO_ROOT / "uv.lock": release["training_environment"]["uv_lock_sha256"],
    }
    drifted = [str(path) for path, expected in checks.items() if lr_only._file_sha256(path) != expected]
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
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--arms")
    parser.add_argument("--runs")
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT)
    parser.add_argument("--confirmation")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    if os.getenv("CI") is not None:
        logger.info("Skipping StarCoder coupled-onset dense surfaces in CI")
        return
    lr_only._configure_parent_jax()
    args = _parse_args()
    if args.freeze:
        release = _freeze_release()
        logger.info("Frozen release %s", release["release_sha256"])
        return
    lr_only._validate_runtime_environment()
    rows, steps = build_training_steps(
        selected_arms=_parse_csv_set(args.arms),
        selected_runs=_parse_csv_set(args.runs),
    )
    audited = audit_materialized_runtime_configs(rows, steps)
    if args.audit:
        lr_only._validate_starcoder_source(CENTRAL2_V4_DEPLOYMENT, cast(Any, rows))
        logger.info("Audited %d rows and %d representative runtime configs", len(rows), audited)
        return
    if args.dry_run:
        for step in steps:
            lower(step)
        logger.info("Lowered %d frozen training graphs", len(steps))
        return

    release = _load_release()
    if args.confirmation != release["confirmation"]:
        raise ValueError("Full launch confirmation is missing or incorrect")
    if args.max_concurrent < 1 or args.max_concurrent > release["maximum_concurrent"]:
        raise ValueError("Requested concurrency is outside the frozen release")
    if os.getenv("MARIN_PREFIX", CENTRAL2_V4_DEPLOYMENT.marin_prefix) != CENTRAL2_V4_DEPLOYMENT.marin_prefix:
        raise ValueError("The coupled-onset release must remain region-local to Central2")
    os.environ["MARIN_PREFIX"] = CENTRAL2_V4_DEPLOYMENT.marin_prefix
    lr_only._validate_starcoder_source(CENTRAL2_V4_DEPLOYMENT, cast(Any, rows))
    run(*steps, max_concurrent=min(args.max_concurrent, len(steps)), force_run_failed=True)


if __name__ == "__main__":
    main()
