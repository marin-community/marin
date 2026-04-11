# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Two-phase StarCoder determinism + seed-jitter sweep with WSD-aligned schedule."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

import fsspec
from fray.cluster import ResourceConfig
from levanter.optim import MuonHConfig

from marin.evaluation.eval_dataset_cache import create_cache_eval_datasets_step
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.utils import create_cache_tokenizer_step

from experiments.domain_phase_mix.analysis import create_analysis_step
from experiments.domain_phase_mix.config import PhaseSchedule, WeightConfig
from experiments.domain_phase_mix.determinism_analysis import RUN_MANIFEST_FILE, create_determinism_report_step
from experiments.domain_phase_mix.domains import NEMOTRON_FULL_DOMAIN, STARCODER_DOMAIN
from experiments.domain_phase_mix.experiment import MixtureExperiment
from experiments.domain_phase_mix.experiment import DEFAULT_MUON_CONFIG as BASE_MUON_CONFIG
from experiments.domain_phase_mix.proxy_sweep import regmix_60m_proxy
from experiments.evals.task_configs import CODE_TASKS, CORE_TASKS
from experiments.llama import llama3_tokenizer
from experiments.pretraining_datasets.dolma import DOLMA_LLAMA3_OVERRIDES
from experiments.pretraining_datasets.nemotron import NEMOTRON_LLAMA3_OVERRIDES

logger = logging.getLogger(__name__)

NAME = "pinlin_calvin_xu/data_mixture/two_phase_starcoder_determinism_wsd"
OBJECTIVE_METRIC = "eval/paloma/dolma_100_programing_languages/bpb"

EXPERIMENT_BUDGET = 1_000_000_000
TARGET_BUDGET = 5_729_908_864_777
BATCH_SIZE = 128
SEQ_LEN = 2048
MIXTURE_BLOCK_SIZE = 2048

WARMUP_FRACTION = 0.01
TOTAL_STEPS = EXPERIMENT_BUDGET // (BATCH_SIZE * SEQ_LEN)
ALIGNED_PHASE_BOUNDARY_STEP = 1904
PHASE_BOUNDARY_FRACTION = ALIGNED_PHASE_BOUNDARY_STEP / TOTAL_STEPS

FIXED_PHASE_WEIGHTS = {
    "phase_0": {"starcoder": 0.218, "nemotron_full": 0.782},
    "phase_1": {"starcoder": 0.243, "nemotron_full": 0.757},
}

N_SEED_RUNS = 20
N_CONTROL_RUNS = 2
SEED_START = 10_000
CONTROL_SEED = 424_242

WANDB_ENTITY = "marin-community"
WANDB_PROJECT = "marin"

TOKENIZER_NAME = llama3_tokenizer

ANALYSIS_METRICS = [
    "eval/loss",
    OBJECTIVE_METRIC,
]

CLUSTER_REGION_ORDER = (
    ("infra/marin-us-central1.yaml", "gs://marin-us-central1"),
    ("infra/marin-us-east5-a.yaml", "gs://marin-us-east5"),
    ("infra/marin-eu-west4-a.yaml", "gs://marin-eu-west4"),
)


@dataclass(frozen=True)
class DeterminismSweepConfig:
    """Top-level configuration for this determinism/seed-jitter sweep."""

    name: str
    objective_metric: str
    n_seed_runs: int
    n_control_runs: int
    phase_weights: dict[str, dict[str, float]]
    warmup_fraction: float
    phase_boundary_fraction: float
    batch_size: int
    seq_len: int
    experiment_budget: int
    target_budget: int


@dataclass(frozen=True)
class RunSpec:
    """Manifest entry for one run in the sweep."""

    run_id: int
    run_name: str
    cohort: Literal["seed_sweep", "determinism_control"]
    data_seed: int
    phase_weights: dict[str, dict[str, float]]


@dataclass(frozen=True)
class WsdScheduleSpec:
    """Resolved WSD schedule parameters in absolute training steps."""

    total_steps: int
    phase_boundary_step: int
    warmup_steps: int
    decay_steps: int


@dataclass(frozen=True)
class SaveRunManifestConfig:
    """Config for writing the deterministic sweep manifest."""

    output_path: str
    experiment_name: str
    objective_metric: str
    warmup_steps: int
    phase_boundary_step: int
    decay_steps: int
    run_specs_json: str


def save_run_manifest(config: SaveRunManifestConfig) -> None:
    """Persist run manifest used by downstream analysis/reporting."""
    run_specs = json.loads(config.run_specs_json)
    payload = {
        "experiment_name": config.experiment_name,
        "objective_metric": config.objective_metric,
        "n_runs": len(run_specs),
        "warmup_steps": config.warmup_steps,
        "phase_boundary_step": config.phase_boundary_step,
        "decay_steps": config.decay_steps,
        "runs": run_specs,
    }
    manifest_path = os.path.join(config.output_path, RUN_MANIFEST_FILE)
    with fsspec.open(manifest_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    logger.info("Wrote run manifest with %d rows to %s", len(run_specs), manifest_path)


def default_sweep_config(name: str = NAME) -> DeterminismSweepConfig:
    """Return the default fixed-config determinism sweep."""
    return DeterminismSweepConfig(
        name=name,
        objective_metric=OBJECTIVE_METRIC,
        n_seed_runs=N_SEED_RUNS,
        n_control_runs=N_CONTROL_RUNS,
        phase_weights=copy.deepcopy(FIXED_PHASE_WEIGHTS),
        warmup_fraction=WARMUP_FRACTION,
        phase_boundary_fraction=PHASE_BOUNDARY_FRACTION,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        experiment_budget=EXPERIMENT_BUDGET,
        target_budget=TARGET_BUDGET,
    )


def _phase_step_alignment(batch_size: int, mixture_block_size: int = MIXTURE_BLOCK_SIZE) -> int:
    return mixture_block_size // math.gcd(batch_size, mixture_block_size)


def resolve_wsd_schedule(config: DeterminismSweepConfig) -> WsdScheduleSpec:
    """Resolve and validate WSD schedule values in absolute steps."""
    total_steps = config.experiment_budget // (config.batch_size * config.seq_len)
    warmup_steps = int(total_steps * config.warmup_fraction)
    phase_boundary_step = int(total_steps * config.phase_boundary_fraction)

    alignment = _phase_step_alignment(config.batch_size, MIXTURE_BLOCK_SIZE)
    if phase_boundary_step % alignment != 0:
        raise ValueError(
            "Phase boundary step must align with mixture block constraints: "
            f"step={phase_boundary_step}, alignment={alignment}"
        )

    decay_steps = total_steps - phase_boundary_step
    if decay_steps <= 0:
        raise ValueError(f"Invalid decay length: total_steps={total_steps}, boundary={phase_boundary_step}")

    return WsdScheduleSpec(
        total_steps=total_steps,
        phase_boundary_step=phase_boundary_step,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
    )


def build_run_specs(
    sweep_config: DeterminismSweepConfig,
    *,
    seed_start: int = SEED_START,
    control_seed: int = CONTROL_SEED,
) -> list[RunSpec]:
    """Build the fixed 22-run manifest (seed sweep + duplicate controls)."""
    specs: list[RunSpec] = []
    next_run_id = 0

    for i in range(sweep_config.n_seed_runs):
        specs.append(
            RunSpec(
                run_id=next_run_id,
                run_name=f"run_{next_run_id:05d}",
                cohort="seed_sweep",
                data_seed=seed_start + i,
                phase_weights=copy.deepcopy(sweep_config.phase_weights),
            )
        )
        next_run_id += 1

    for _ in range(sweep_config.n_control_runs):
        specs.append(
            RunSpec(
                run_id=next_run_id,
                run_name=f"run_{next_run_id:05d}",
                cohort="determinism_control",
                data_seed=control_seed,
                phase_weights=copy.deepcopy(sweep_config.phase_weights),
            )
        )
        next_run_id += 1

    return specs


def _required_cache_relative_paths() -> tuple[str, ...]:
    required_nemotron_splits = (
        "hq_actual",
        "hq_synth",
        "medium_high",
        "medium",
        "medium_low",
        "low_actual",
    )

    tokenized_paths = [NEMOTRON_LLAMA3_OVERRIDES[split] for split in required_nemotron_splits]
    tokenized_paths.append(DOLMA_LLAMA3_OVERRIDES["starcoder"])

    tokenizer_cache = os.path.join("raw", "tokenizers", TOKENIZER_NAME.replace("/", "--"))
    eval_cache = os.path.join("raw", "eval-datasets", "code-tasks")

    return tuple([eval_cache, tokenizer_cache, *tokenized_paths])


def _status_is_success(path: str) -> bool:
    status_path = os.path.join(path.rstrip("/"), ".executor_status")
    try:
        with fsspec.open(status_path, "r") as f:
            status = f.read().strip()
        return status == "SUCCESS"
    except FileNotFoundError:
        return False


def _preflight_prefix(prefix: str) -> tuple[bool, list[str]]:
    missing: list[str] = []
    for rel in _required_cache_relative_paths():
        full = f"{prefix.rstrip('/')}/{rel}"
        if not _status_is_success(full):
            missing.append(full)
    return len(missing) == 0, missing


def _resolve_prefix_with_fallback(
    *,
    explicit_prefix: str | None,
    skip_preflight: bool,
) -> tuple[str, str | None, dict[str, Any]]:
    """Resolve MARIN prefix with optional region-order preflight checks."""
    if explicit_prefix:
        if skip_preflight:
            return explicit_prefix, None, {"checked": [], "selected": explicit_prefix, "missing": []}
        ok, missing = _preflight_prefix(explicit_prefix)
        if not ok:
            raise RuntimeError(
                "Explicit --marin_prefix failed cache preflight. Missing SUCCESS status for:\n" + "\n".join(missing)
            )
        return explicit_prefix, None, {"checked": [explicit_prefix], "selected": explicit_prefix, "missing": []}

    env_prefix = os.environ.get("MARIN_PREFIX")
    if env_prefix:
        if skip_preflight:
            return env_prefix, None, {"checked": [], "selected": env_prefix, "missing": []}
        ok, missing = _preflight_prefix(env_prefix)
        if not ok:
            raise RuntimeError("MARIN_PREFIX failed cache preflight. Missing SUCCESS status for:\n" + "\n".join(missing))
        return env_prefix, None, {"checked": [env_prefix], "selected": env_prefix, "missing": []}

    if skip_preflight:
        # No explicit/env prefix and no preflight requested: preserve historical default.
        return (
            "gs://marin-us-central1",
            CLUSTER_REGION_ORDER[0][0],
            {
                "checked": [],
                "selected": "gs://marin-us-central1",
                "missing": [],
            },
        )

    checks: list[dict[str, Any]] = []
    for cluster_yaml, prefix in CLUSTER_REGION_ORDER:
        ok, missing = _preflight_prefix(prefix)
        checks.append(
            {
                "cluster": cluster_yaml,
                "prefix": prefix,
                "ok": ok,
                "missing": missing,
            }
        )
        if ok:
            return prefix, cluster_yaml, {"checked": checks, "selected": prefix, "missing": []}

    detail = "\n\n".join(
        f"{item['cluster']} ({item['prefix']}) missing:\n" + "\n".join(item["missing"]) for item in checks
    )
    raise RuntimeError(f"No fallback region passed cache preflight.\n\n{detail}")


def create_run_manifest_step(
    *,
    name_prefix: str,
    objective_metric: str,
    schedule: WsdScheduleSpec,
    run_specs: list[RunSpec],
) -> ExecutorStep:
    """Create step that persists run manifest for later reporting."""
    return ExecutorStep(
        name=f"{name_prefix}/run_manifest",
        description=f"Save deterministic run manifest ({len(run_specs)} runs)",
        fn=save_run_manifest,
        config=SaveRunManifestConfig(
            output_path=this_output_path(),
            experiment_name=name_prefix,
            objective_metric=objective_metric,
            warmup_steps=schedule.warmup_steps,
            phase_boundary_step=schedule.phase_boundary_step,
            decay_steps=schedule.decay_steps,
            run_specs_json=json.dumps([asdict(spec) for spec in run_specs]),
        ),
    )


def create_two_phase_determinism_experiment(
    *,
    sweep_config: DeterminismSweepConfig,
    optimizer_config: MuonHConfig,
    eval_datasets_cache_path: str,
    tpu_type: str,
) -> MixtureExperiment:
    """Create the fixed-weight two-phase StarCoder determinism experiment."""
    phase_schedule = PhaseSchedule.from_boundaries(
        boundaries=[sweep_config.phase_boundary_fraction],
        names=["phase_0", "phase_1"],
    )

    return MixtureExperiment(
        name=sweep_config.name,
        domains=[NEMOTRON_FULL_DOMAIN, STARCODER_DOMAIN],
        phase_schedule=phase_schedule,
        model_config=regmix_60m_proxy,
        batch_size=sweep_config.batch_size,
        seq_len=sweep_config.seq_len,
        experiment_budget=sweep_config.experiment_budget,
        target_budget=sweep_config.target_budget,
        eval_harness_tasks=CORE_TASKS + CODE_TASKS,
        optimizer_config=optimizer_config,
        eval_datasets_cache_path=eval_datasets_cache_path,
        resources=ResourceConfig.with_tpu(tpu_type),
    )


def build_executor_steps(
    *,
    sweep_config: DeterminismSweepConfig,
    run_specs: list[RunSpec],
    wandb_entity: str,
    wandb_project: str,
    marin_prefix: str,
    tpu_type: str,
) -> tuple[list[ExecutorStep], WsdScheduleSpec]:
    """Build full deterministic sweep DAG."""
    schedule = resolve_wsd_schedule(sweep_config)
    optimizer_config = replace(
        BASE_MUON_CONFIG,
        warmup=schedule.warmup_steps,
        decay=schedule.decay_steps,
        lr_schedule="cosine",
    )

    tokenizer_cache_base = f"{marin_prefix.rstrip('/')}/raw/tokenizers"
    eval_datasets_cache_path = f"{marin_prefix.rstrip('/')}/raw/eval-datasets/code-tasks"

    os.environ["MARIN_TOKENIZER_CACHE_PATH"] = tokenizer_cache_base
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    cache_tokenizer_step = create_cache_tokenizer_step(
        tokenizer_name=TOKENIZER_NAME,
        gcs_path=os.path.join(tokenizer_cache_base, TOKENIZER_NAME.replace("/", "--")),
        name_prefix=sweep_config.name,
    )
    cache_eval_datasets_step = create_cache_eval_datasets_step(
        eval_tasks=CORE_TASKS + CODE_TASKS,
        gcs_path=eval_datasets_cache_path,
        name_prefix=sweep_config.name,
    )

    experiment = create_two_phase_determinism_experiment(
        sweep_config=sweep_config,
        optimizer_config=optimizer_config,
        eval_datasets_cache_path=eval_datasets_cache_path,
        tpu_type=tpu_type,
    )

    weight_configs = [
        WeightConfig(run_id=spec.run_id, phase_weights=copy.deepcopy(spec.phase_weights)) for spec in run_specs
    ]
    summary = {
        "kind": "determinism_seed_jitter",
        "n_seed_runs": sweep_config.n_seed_runs,
        "n_control_runs": sweep_config.n_control_runs,
        "fixed_phase_weights": sweep_config.phase_weights,
        "warmup_steps": schedule.warmup_steps,
        "phase_boundary_step": schedule.phase_boundary_step,
        "decay_steps": schedule.decay_steps,
        "data_seeds": [spec.data_seed for spec in run_specs],
    }

    run_manifest_step = create_run_manifest_step(
        name_prefix=sweep_config.name,
        objective_metric=sweep_config.objective_metric,
        schedule=schedule,
        run_specs=run_specs,
    )
    weight_configs_step = experiment.create_weight_configs_step(
        configs=weight_configs,
        summary=summary,
        seed=-1,
        name_prefix=sweep_config.name,
    )

    training_steps: list[ExecutorStep] = []
    for spec in run_specs:
        step = experiment.create_training_step(
            weight_config=WeightConfig(run_id=spec.run_id, phase_weights=copy.deepcopy(spec.phase_weights)),
            name_prefix=sweep_config.name,
            run_name=spec.run_name,
            data_seed=spec.data_seed,
        )
        training_steps.append(step)

    analysis_step = create_analysis_step(
        weight_configs_step=weight_configs_step,
        name_prefix=sweep_config.name,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        metrics=ANALYSIS_METRICS,
        depends_on=training_steps,
    )
    determinism_report_step = create_determinism_report_step(
        name_prefix=sweep_config.name,
        objective_metric=sweep_config.objective_metric,
        run_manifest_step=run_manifest_step,
        analysis_step=analysis_step,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
    )

    all_steps = [
        cache_tokenizer_step,
        cache_eval_datasets_step,
        run_manifest_step,
        weight_configs_step,
        *training_steps,
        analysis_step,
        determinism_report_step,
    ]
    return all_steps, schedule


def run_experiment(
    *,
    name_prefix: str = NAME,
    n_seed_runs: int = N_SEED_RUNS,
    n_control_runs: int = N_CONTROL_RUNS,
    seed_start: int = SEED_START,
    control_seed: int = CONTROL_SEED,
    objective_metric: str = OBJECTIVE_METRIC,
    wandb_entity: str = WANDB_ENTITY,
    wandb_project: str = WANDB_PROJECT,
    tpu_type: str = "v5p-8",
    marin_prefix: str | None = None,
    skip_cache_preflight: bool = False,
    preflight_only: bool = False,
) -> None:
    """Run the full 22-run determinism and seed-jitter sweep."""
    if os.getenv("CI") is not None:
        logger.info("Skipping experiment execution on CI environment.")
        return

    resolved_prefix, cluster_hint, preflight = _resolve_prefix_with_fallback(
        explicit_prefix=marin_prefix,
        skip_preflight=skip_cache_preflight,
    )

    if preflight_only:
        logger.info("Cache preflight passed. selected_prefix=%s cluster_hint=%s", resolved_prefix, cluster_hint)
        logger.info("Preflight details: %s", json.dumps(preflight, indent=2, sort_keys=True))
        return

    sweep_config = DeterminismSweepConfig(
        name=name_prefix,
        objective_metric=objective_metric,
        n_seed_runs=n_seed_runs,
        n_control_runs=n_control_runs,
        phase_weights=copy.deepcopy(FIXED_PHASE_WEIGHTS),
        warmup_fraction=WARMUP_FRACTION,
        phase_boundary_fraction=PHASE_BOUNDARY_FRACTION,
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        experiment_budget=EXPERIMENT_BUDGET,
        target_budget=TARGET_BUDGET,
    )
    run_specs = build_run_specs(
        sweep_config,
        seed_start=seed_start,
        control_seed=control_seed,
    )

    steps, schedule = build_executor_steps(
        sweep_config=sweep_config,
        run_specs=run_specs,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        marin_prefix=resolved_prefix,
        tpu_type=tpu_type,
    )

    logger.info("Prepared %d runs (%d seed sweep + %d control)", len(run_specs), n_seed_runs, n_control_runs)
    logger.info(
        "WSD schedule aligned to phase boundary: total_steps=%d warmup=%d boundary=%d decay=%d",
        schedule.total_steps,
        schedule.warmup_steps,
        schedule.phase_boundary_step,
        schedule.decay_steps,
    )
    if cluster_hint:
        logger.info("Selected prefix %s; recommended cluster fallback target: %s", resolved_prefix, cluster_hint)

    executor_main(
        steps=steps,
        description=f"Two-phase StarCoder determinism + seed jitter sweep ({len(run_specs)} runs)",
    )


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Two-phase StarCoder determinism + seed-jitter sweep (WSD-aligned).")
    parser.add_argument("--name_prefix", type=str, default=NAME, help="W&B/executor name prefix.")
    parser.add_argument("--n_seed_runs", type=int, default=N_SEED_RUNS, help="Number of seed-sweep runs.")
    parser.add_argument("--n_control_runs", type=int, default=N_CONTROL_RUNS, help="Number of duplicate-seed controls.")
    parser.add_argument("--seed_start", type=int, default=SEED_START, help="Starting data_seed for seed-sweep cohort.")
    parser.add_argument("--control_seed", type=int, default=CONTROL_SEED, help="Shared data_seed for control cohort.")
    parser.add_argument("--objective_metric", type=str, default=OBJECTIVE_METRIC, help="Primary objective metric.")
    parser.add_argument("--wandb_entity", type=str, default=WANDB_ENTITY, help="W&B entity.")
    parser.add_argument("--wandb_project", type=str, default=WANDB_PROJECT, help="W&B project.")
    parser.add_argument("--tpu_type", type=str, default="v5p-8", help="TPU type for training runs.")
    parser.add_argument(
        "--marin_prefix",
        type=str,
        default=None,
        help="Storage prefix for executor outputs/caches. If unset, region-order preflight is used.",
    )
    parser.add_argument(
        "--skip_cache_preflight",
        action="store_true",
        help="Skip cache status checks and use explicit/env/default prefix directly.",
    )
    parser.add_argument(
        "--preflight_only",
        action="store_true",
        help="Only run cache preflight and print selected prefix/cluster recommendation.",
    )
    return parser.parse_known_args()


if __name__ == "__main__":
    import sys

    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]

    run_experiment(
        name_prefix=args.name_prefix,
        n_seed_runs=args.n_seed_runs,
        n_control_runs=args.n_control_runs,
        seed_start=args.seed_start,
        control_seed=args.control_seed,
        objective_metric=args.objective_metric,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        tpu_type=args.tpu_type,
        marin_prefix=args.marin_prefix,
        skip_cache_preflight=args.skip_cache_preflight,
        preflight_only=args.preflight_only,
    )
