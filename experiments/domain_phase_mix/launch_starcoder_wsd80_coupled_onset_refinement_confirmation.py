# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch coupled-onset BO refinement and fresh fixed-policy confirmation."""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
from dataclasses import dataclass, replace
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
from experiments.domain_phase_mix import launch_starcoder_wsd80_coupled_onset_dense_surfaces as source
from experiments.domain_phase_mix import launch_starcoder_wsd80_lr_onset_dense_surfaces as lr_only
from experiments.domain_phase_mix import launch_starcoder_wsd_80_20_surface as base
from experiments.llama import llama3_tokenizer

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DESIGN_PATH = Path(__file__).with_name("starcoder_wsd80_coupled_onset_refinement_confirmation_design_20260901.json.gz")
DESIGN_GENERATOR_PATH = (
    Path(__file__).parent
    / "exploratory/two_phase_many/design_starcoder_wsd80_coupled_onset_refinement_confirmation_20260901.py"
)
CC_REVIEW_PATH = (
    REPO_ROOT / ".agents/handoffs/starcoder_coupled_onset_refinement_confirmation_cc_review_20260901_FINAL_RESPONSE.md"
)
EXPECTED_DESIGN_SHA256 = "79943e36932e942e9c42a5070663fae00f2c5b4e3cdb5b942fdeb1af7abac8a5"
EXPECTED_RUN_COUNT = 96
EXPECTED_STAGE_COUNTS = {"bayesian_refinement_discovery": 24, "fresh_confirmation": 72}
EXPECTED_ARM_IDS = source.EXPECTED_ARM_IDS
EXPECTED_CONFIRMATION_COUNTS = {"coupled_0p60": 32, "coupled_0p80": 16, "coupled_0p90": 24}
EXPECTED_BO_COUNTS = {"coupled_0p60": 8, "coupled_0p80": 8, "coupled_0p90": 8}
MAX_CONCURRENT = 96
MAX_WANDB_TAG_LENGTH = 64
FULL_LAUNCH_CONFIRMATION = "I_AUTHORIZE_THE_STARCODER_WSD80_COUPLED_ONSET_REFINEMENT_CONFIRMATION"

CENTRAL2_V4_DEPLOYMENT = lr_only.Deployment(
    deployment_id="central2-v4-coupled-successor",
    name="pinlin_calvin_xu/data_mixture/starcoder_wsd80_coupled_onset_refinement_confirmation_central2_v4_20260901",
    version="2026.09.01.1",
    wandb_group="starcoder_wsd80_coupled_onset_refinement_confirmation_central2_v4_20260901",
    panel_tag="wsd80_coupled_onset_successor_c2v4",
    run_id_prefix="c2v4r_",
    marin_prefix="gs://marin-us-central2",
    tpu_type="v4-8",
    tpu_region="us-central2",
    tpu_zone="us-central2-b",
    output_dir=Path(__file__).parent
    / "manifests/starcoder_wsd80_coupled_onset_refinement_confirmation_central2_v4_v1_20260901",
    cc_review_path=CC_REVIEW_PATH,
    allow_empty_starcoder_finished_shards=True,
)


@dataclass(frozen=True)
class SuccessorRun(source.CoupledOnsetRun):
    """One adaptive discovery or fixed confirmation row."""

    acquisition: dict[str, float] | None = None


def _load_payload() -> dict[str, Any]:
    payload = json.loads(gzip.decompress(DESIGN_PATH.read_bytes()))
    claimed_hash = payload.pop("design_sha256", None)
    observed_hash = lr_only._canonical_sha256(payload)
    if claimed_hash != EXPECTED_DESIGN_SHA256 or observed_hash != EXPECTED_DESIGN_SHA256:
        raise ValueError(f"Design hash drifted: {observed_hash} != {claimed_hash}")
    if payload.get("training_environment") != source.EXPECTED_TRAINING_ENVIRONMENT:
        raise ValueError("Frozen training environment drifted")
    payload["design_sha256"] = claimed_hash
    return payload


def load_design(
    *,
    selected_arms: frozenset[str] | None = None,
    selected_stages: frozenset[str] | None = None,
    selected_runs: frozenset[str] | None = None,
) -> tuple[dict[str, Any], tuple[SuccessorRun, ...]]:
    """Load and validate the disjoint adaptive and confirmatory inventories."""
    payload = _load_payload()
    rows = tuple(SuccessorRun(**row) for row in payload["rows"])
    if len(rows) != EXPECTED_RUN_COUNT or payload.get("expected_run_count") != EXPECTED_RUN_COUNT:
        raise ValueError(f"Expected {EXPECTED_RUN_COUNT} rows, got {len(rows)}")
    if payload.get("stage_counts") != EXPECTED_STAGE_COUNTS:
        raise ValueError("Successor stage inventory drifted")
    if payload.get("metrics") != {
        "broad_secondary": "eval/paloma/c4_en-llama3/bpb",
        "direction": "lower_is_better",
        "endpoint_step": source.EXPECTED_TOTAL_STEPS - 1,
        "primary": "eval/paloma/dolma_100_programing_languages-llama3/bpb",
    }:
        raise ValueError("Endpoint metric contract drifted")
    if payload.get("checkpoint_contract", {}).get("all_rows") != "terminal permanent checkpoint only":
        raise ValueError("Checkpoint contract drifted")
    if "never drop, replace, or reselect" not in payload.get("completeness_contract", {}).get("failure_rule", ""):
        raise ValueError("No-drop completeness contract drifted")
    if {row.arm_id for row in rows} != EXPECTED_ARM_IDS:
        raise ValueError("Coupled-onset arm inventory drifted")
    if len({row.row_id for row in rows}) != len(rows) or len({row.run_name for row in rows}) != len(rows):
        raise ValueError("Run identities are not unique")
    if [row.run_order for row in rows] != list(range(len(rows))):
        raise ValueError("Run order is not contiguous")
    if any(row.boundary_step != row.optimizer["decay_onset_step"] for row in rows):
        raise ValueError("Phase and LR-decay onsets are not coupled")

    bo_rows = tuple(row for row in rows if row.stage == "bayesian_refinement_discovery")
    confirmation_rows = tuple(row for row in rows if row.stage == "fresh_confirmation")
    bo_counts = {arm_id: sum(row.arm_id == arm_id for row in bo_rows) for arm_id in EXPECTED_ARM_IDS}
    confirmation_counts = {arm_id: sum(row.arm_id == arm_id for row in confirmation_rows) for arm_id in EXPECTED_ARM_IDS}
    if bo_counts != EXPECTED_BO_COUNTS or confirmation_counts != EXPECTED_CONFIRMATION_COUNTS:
        raise ValueError("Per-arm successor inventory drifted")
    if {row.data_seed for row in bo_rows} != {payload["discovery_seed"]}:
        raise ValueError("Adaptive discovery seed drifted")
    if {row.data_seed for row in confirmation_rows} != set(payload["confirmation_seeds"]):
        raise ValueError("Fresh confirmation seeds drifted")
    if payload["discovery_seed"] in payload["confirmation_seeds"]:
        raise ValueError("Adaptive discovery and confirmation seeds overlap")
    if any(row.selection_class != "eligible_untied" or row.acquisition is None for row in bo_rows):
        raise ValueError("Adaptive rows must be eligible untied acquisitions")
    if any(row.acquisition is not None for row in confirmation_rows):
        raise ValueError("Confirmation rows cannot carry adaptive acquisition metadata")
    expected_policies = {
        arm_id: set(policies) for arm_id, policies in payload["confirmation_contract"]["policies_by_arm"].items()
    }
    observed_policies = {
        arm_id: {row.coordinate_id for row in confirmation_rows if row.arm_id == arm_id} for arm_id in EXPECTED_ARM_IDS
    }
    if observed_policies != expected_policies:
        raise ValueError("Frozen confirmation policy cells drifted")

    if selected_arms is not None:
        unknown = selected_arms - EXPECTED_ARM_IDS
        if unknown:
            raise ValueError(f"Unknown coupled-onset arms: {sorted(unknown)}")
        rows = tuple(row for row in rows if row.arm_id in selected_arms)
    if selected_stages is not None:
        unknown = selected_stages - set(EXPECTED_STAGE_COUNTS)
        if unknown:
            raise ValueError(f"Unknown stages: {sorted(unknown)}")
        rows = tuple(row for row in rows if row.stage in selected_stages)
    if selected_runs is not None:
        available = {row.run_name for row in rows}
        unknown = selected_runs - available
        if unknown:
            raise ValueError(f"Unknown runs after filtering: {sorted(unknown)}")
        rows = tuple(row for row in rows if row.run_name in selected_runs)
    if not rows:
        raise ValueError("Launch filters selected no rows")
    return payload, rows


def _keep_policy(row: SuccessorRun) -> list[dict[str, int | None]]:
    return [{"every": row.total_steps - 1, "until": None}]


def _configure_training(
    training: ArtifactStep[LevanterCheckpoint],
    *,
    train_datasets: dict[ArtifactStep[lr_only.TokenizedCache], float],
    validation_datasets: tuple[ArtifactStep[lr_only.TokenizedCache], ...],
    phase_weights: list[tuple[int, dict[str, float]]],
    starcoder_name: str,
    row: SuccessorRun,
) -> ArtifactStep[LevanterCheckpoint]:
    """Install the unchanged terminal-only checkpoint and data contracts."""

    def build_config(ctx: lr_only.StepContext) -> TrainLmOnPodConfig:
        pod_config = training.build_config(ctx)
        train_config = cast(TrainLmConfig, pod_config.train_config)
        data_config = lr_only.mixture(ctx, train_datasets, validation=validation_datasets)
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
                save_interval=lr_only.TEMPORARY_CHECKPOINT_INTERVAL,
                keep=_keep_policy(row),
                keep_last_temporary_checkpoints=1,
            ),
        )
        train_config = replace(train_config, data=data_config, data_seed=row.data_seed, trainer=trainer)
        return replace(pod_config, train_config=train_config)

    return replace(training, build_config=build_config)


def build_training_steps(
    *,
    deployment: lr_only.Deployment = CENTRAL2_V4_DEPLOYMENT,
    selected_arms: frozenset[str] | None = None,
    selected_stages: frozenset[str] | None = None,
    selected_runs: frozenset[str] | None = None,
) -> tuple[tuple[SuccessorRun, ...], tuple[ArtifactStep[LevanterCheckpoint], ...]]:
    """Build independently resumable training artifacts for frozen successor rows."""
    payload, rows = load_design(
        selected_arms=selected_arms,
        selected_stages=selected_stages,
        selected_runs=selected_runs,
    )
    model = source._validate_model(payload)
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
        phase_0_weights = base._phase_leaf_weights(row.phase_0_starcoder, nemotron=nemotron, starcoder=starcoder)
        phase_1_weights = base._phase_leaf_weights(row.phase_1_starcoder, nemotron=nemotron, starcoder=starcoder)
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
            _configure_training(
                training,
                train_datasets=static_weights,
                validation_datasets=validation,
                phase_weights=[(0, phase_0_weights), (row.boundary_step, phase_1_weights)],
                starcoder_name=starcoder.name,
                row=cast(Any, row),
            )
        )
    return rows, tuple(steps)


def _schedule_vector(row: SuccessorRun) -> np.ndarray:
    scheduler = lr_only._optimizer(cast(Any, row)).lr_scheduler(row.total_steps)
    return np.asarray(scheduler(jnp.arange(row.total_steps)))


def audit_materialized_runtime_configs(
    rows: tuple[SuccessorRun, ...],
    steps: tuple[ArtifactStep[LevanterCheckpoint], ...],
    *,
    deployment: lr_only.Deployment = CENTRAL2_V4_DEPLOYMENT,
) -> int:
    """Materialize one row per represented arm and stage and fail closed on drift."""
    if len(rows) != len(steps):
        raise ValueError("Row/step cardinality mismatch")
    groups = sorted({(row.arm_id, row.stage) for row in rows})
    chosen_indices = [next(i for i, row in enumerate(rows) if (row.arm_id, row.stage) == group) for group in groups]
    starcoder_name = "dolma/starcoder"
    for index in chosen_indices:
        row = rows[index]
        pod_config = cast(TrainLmOnPodConfig, materialized_config(steps[index], deployment.marin_prefix))
        train_config = cast(TrainLmConfig, pod_config.train_config)
        if train_config.optimizer.decay != row.total_steps - row.boundary_step:
            raise ValueError(f"{row.run_name}: optimizer decay drifted")
        if train_config.optimizer.min_lr_ratio != 0.0:
            raise ValueError(f"{row.run_name}: optimizer minimum LR drifted")
        if train_config.optimizer.warmup != source.EXPECTED_WARMUP_STEPS:
            raise ValueError(f"{row.run_name}: optimizer warmup drifted")
        if train_config.trainer.num_train_steps != row.total_steps:
            raise ValueError(f"{row.run_name}: total training steps drifted")
        weights = train_config.data.train_weights
        if not isinstance(weights, list) or [boundary for boundary, _ in weights] != [0, row.boundary_step]:
            raise ValueError(f"{row.run_name}: phase boundary drifted")
        if not np.isclose(weights[0][1][starcoder_name], row.phase_0_starcoder, atol=1e-12):
            raise ValueError(f"{row.run_name}: phase-0 weight drifted")
        if not np.isclose(weights[1][1][starcoder_name], row.phase_1_starcoder, atol=1e-12):
            raise ValueError(f"{row.run_name}: phase-1 weight drifted")
        if train_config.data.max_train_batches != {starcoder_name: row.starcoder_support_batches}:
            raise ValueError(f"{row.run_name}: finite support cap drifted")
        if train_config.data_seed != row.data_seed or train_config.trainer.seed != row.trainer_seed:
            raise ValueError(f"{row.run_name}: seed drifted")
        if train_config.trainer.checkpointer.keep != _keep_policy(row):
            raise ValueError(f"{row.run_name}: terminal-only checkpoint contract drifted")

    _, all_rows = load_design()
    representative = {arm_id: next(row for row in all_rows if row.arm_id == arm_id) for arm_id in EXPECTED_ARM_IDS}
    earliest_onset = min(row.boundary_step for row in representative.values())
    reference = _schedule_vector(representative["coupled_0p80"])[:earliest_onset]
    for arm_id, row in representative.items():
        if not np.array_equal(_schedule_vector(row)[:earliest_onset], reference):
            raise ValueError(f"{arm_id}: LR schedule differs before the earliest coupled onset")
    return len(chosen_indices)


def _freeze_release(deployment: lr_only.Deployment = CENTRAL2_V4_DEPLOYMENT) -> dict[str, Any]:
    payload = _load_payload()
    review_path = deployment.cc_review_path
    review = review_path.read_text(encoding="utf-8").rstrip()
    if not review.endswith("APPROVE"):
        raise ValueError("Claude Code review must end with APPROVE before release freeze")
    serialized_review_path = (
        str(review_path.relative_to(REPO_ROOT)) if review_path.is_relative_to(REPO_ROOT) else str(review_path)
    )
    release = {
        "schema_version": "2026-09-01-starcoder-wsd80-coupled-onset-successor-v1",
        "design_sha256": payload["design_sha256"],
        "design_path": str(DESIGN_PATH.relative_to(REPO_ROOT)),
        "design_file_sha256": lr_only._file_sha256(DESIGN_PATH),
        "launcher_path": str(Path(__file__).relative_to(REPO_ROOT)),
        "launcher_sha256": lr_only._file_sha256(Path(__file__)),
        "design_generator_path": str(DESIGN_GENERATOR_PATH.relative_to(REPO_ROOT)),
        "design_generator_sha256": lr_only._file_sha256(DESIGN_GENERATOR_PATH),
        "cc_review_path": serialized_review_path,
        "cc_review_sha256": lr_only._file_sha256(review_path),
        "deployment": deployment.release_record(),
        "training_environment": source.EXPECTED_TRAINING_ENVIRONMENT,
        "stage_counts": EXPECTED_STAGE_COUNTS,
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
        REPO_ROOT / release["cc_review_path"]: release["cc_review_sha256"],
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
    parser.add_argument("--stages")
    parser.add_argument("--runs")
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT)
    parser.add_argument("--confirmation")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    if os.getenv("CI") is not None:
        logger.info("Skipping StarCoder coupled-onset successor in CI")
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
        selected_stages=_parse_csv_set(args.stages),
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
        raise ValueError("The coupled-onset successor must remain region-local to Central2")
    os.environ["MARIN_PREFIX"] = CENTRAL2_V4_DEPLOYMENT.marin_prefix
    lr_only._validate_starcoder_source(CENTRAL2_V4_DEPLOYMENT, cast(Any, rows))
    run(*steps, max_concurrent=min(args.max_concurrent, len(steps)), force_run_failed=True)


if __name__ == "__main__":
    main()
