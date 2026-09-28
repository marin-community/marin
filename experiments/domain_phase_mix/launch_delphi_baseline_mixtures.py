# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train Delphi-scale objective-agnostic data-mixing baselines.

This launcher is the issue #6607 training half: proportional and UniMax-8 over
the Dolma3/Dolmino top-level buckets at a small Delphi scaling ladder.  It uses
the CompletedAdamH/Delphi model, optimizer, and mesh logic while only replacing
the training data mixture.  The UniMax epoch-cap sweep (caps 1, 4 and 12 beside
the ladder's 8) reuses the same graph with ``--with-table9-eval``, which chains
the Marin-native Table 9 BPB evaluator after each training step.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field, replace
from datetime import timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

import fsspec
import jmp
from fray.cluster import ResourceConfig
from haliax.partitioning import ResourceAxis
from levanter.checkpoint import CheckpointerConfig
from levanter.data.text.datasets import DatasetComponent
from levanter.main import train_lm
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig
from levanter.utils.mesh import MeshConfig
from marin.evaluation.olmo_base_eval.run import olmo_base_eval_step
from marin.execution.context import executor_context
from marin.execution.executor import ExecutorMainConfig, executor_main
from marin.execution.types import ExecutorStep, InputName, this_output_path
from marin.processing.tokenize import step_to_lm_mixture_component
from marin.scaling_laws import ScalingFit, predict_optimal_config
from marin.training.training import TrainLmOnPodConfig, run_levanter_train_lm
from rigging.filesystem import marin_prefix_for_region

from experiments.defaults import default_validation_sets
from experiments.domain_phase_mix.config import PhaseSchedule, WeightConfig
from experiments.domain_phase_mix.dolma3_dolmino_top_level_domains import (
    TARGET_BUDGET_DOLMA3_COMMON_CRAWL,
    TOP_LEVEL_DOMAIN_TOKEN_COUNTS,
    TOP_LEVEL_TOTAL_AVAILABLE_TOKENS,
)
from experiments.domain_phase_mix.experiment import MixtureExperiment
from experiments.domain_phase_mix.qsplit240_replay import SKIP_EVAL_HARNESS_ENV_VAR
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (
    DEFAULT_RUNTIME_CACHE_REGION,
    DOMAIN_NAMES,
    PHASE_NAMES,
    build_top_level_domains,
)
from experiments.domain_phase_mix.weight_sampler import compute_unimax_weights
from experiments.llama import llama3_tokenizer
from experiments.scaling_law_sweeps.completed_adamh import completed_adamh_heuristic

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_ARTIFACT_DIR = (
    SCRIPT_DIR / "exploratory" / "two_phase_many" / "reference_outputs" / "delphi_baseline_mixtures_issue6607_20260623"
)

EXPERIMENT_NAME = "pinlin_calvin_xu/data_mixture/delphi_baseline_mixtures_issue6607_20260623"
DEFAULT_ANALYSIS_OUTPUT_PATH = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_baseline_mixtures_issue6607_20260623/analysis-af9355"
)
LABEL = "adamh_scaling_v6"
SEQ_LEN_DELPHI = 4096
PHASE_SCHEDULE = PhaseSchedule.uniform(2)
TABLE9_REQUEST_SET_DIR = InputName.hardcoded("raw/eval-datasets/olmo_base_eval_table9/v2")
DEFAULT_TABLE9_TPU_ZONE = "us-east5-b"
TABLE9_WANDB_GROUP = "olmo_base_eval_table9_delphi_baseline_mixtures"
SIMULATED_EPOCH_TARGET_BUDGET = TARGET_BUDGET_DOLMA3_COMMON_CRAWL
DEFAULT_TPU_REGION = "us-east5"
DEFAULT_TPU_ZONE = "us-east5-a"
DEFAULT_MAX_CONCURRENT = 4
RUN_ID_BASE = 660_700

# FLOPs -> (east5 TPU type, global batch size). These are the four rungs selected
# for a tractable scaling-baseline panel. Keep every rung on smaller preemptible
# v5p slices so the launcher is not blocked on scarce v5p-128 allocation.
TARGET_BUDGETS: dict[float, tuple[str, int]] = {
    3e18: ("v5p-8", 128),
    2e19: ("v5p-16", 128),
    3e20: ("v5p-32", 256),
    1e21: ("v5p-64", 512),
}


class DelphiBaselineMixture(StrEnum):
    """Objective-agnostic baseline mixtures: proportional and UniMax at several epoch caps."""

    PROPORTIONAL = "proportional"
    UNIMAX_1 = "unimax1"
    UNIMAX_4 = "unimax4"
    UNIMAX_8 = "unimax8"
    UNIMAX_12 = "unimax12"


# Epoch cap of each UniMax mixture at the fixed target budget; the cap binds only on the smallest buckets.
UNIMAX_EPOCH_CAPS: dict[DelphiBaselineMixture, float] = {
    DelphiBaselineMixture.UNIMAX_1: 1.0,
    DelphiBaselineMixture.UNIMAX_4: 4.0,
    DelphiBaselineMixture.UNIMAX_8: 8.0,
    DelphiBaselineMixture.UNIMAX_12: 12.0,
}


def _run_name(mixture: DelphiBaselineMixture, target_flops: float, trainer_seed: int) -> str:
    """Run name; trainer-seed repeats of a run carry a ``_t<seed>`` suffix, the original run none."""
    suffix = f"_t{trainer_seed}" if trainer_seed != 0 else ""
    return f"{mixture.value}_{_slug(target_flops)}{suffix}"


@dataclass(frozen=True)
class DelphiBaselineRunSpec:
    """One Delphi baseline training run."""

    run_order: int
    run_id: int
    run_name: str
    mixture: str
    target_flops: float
    tpu_type: str
    tpu_region: str
    tpu_zone: str
    batch_size: int
    train_tokens: int
    train_steps: int
    realized_train_tokens: int
    expected_checkpoint_step: int
    model_hidden_dim: int
    model_layers: int
    non_embedding_params: int
    total_trainable_params: int
    tensor_parallel_size: int
    data_seed: int
    trainer_seed: int
    phase_weights: dict[str, dict[str, float]]


@dataclass(frozen=True)
class DelphiBaselineTrainingConfig:
    """Config resolved by the executor before one Delphi baseline train."""

    analysis_output_path: str
    target_flops: float
    tpu_type: str
    tpu_region: str
    tpu_zone: str
    batch_size: int
    mixture: DelphiBaselineMixture
    label: str
    output_path: str
    run_id: int
    run_name: str
    data_seed: int
    trainer_seed: int = 0
    validation_configs: dict[str, DatasetComponent] | None = None


@dataclass(frozen=True)
class SaveDelphiBaselineManifestConfig:
    """Config for persisting a resolved launcher manifest."""

    output_path: str
    analysis_output_path: str
    mixtures: tuple[DelphiBaselineMixture, ...]
    target_budgets_json: str
    tpu_region: str
    tpu_zone: str
    run_id_base: int = RUN_ID_BASE
    trainer_seeds: tuple[int, ...] = (0,)
    data_seed: int | None = None


@dataclass(frozen=True)
class LaunchArtifacts:
    """Resolved Delphi baseline launcher graph."""

    run_specs: list[DelphiBaselineRunSpec]
    manifest_step: ExecutorStep
    training_steps: list[ExecutorStep]
    eval_steps: list[ExecutorStep] = field(default_factory=list)

    @property
    def steps(self) -> list[ExecutorStep]:
        return [self.manifest_step, *self.training_steps, *self.eval_steps]


def _slug(value: float) -> str:
    return f"{value:.0e}".replace("+", "").replace("-", "m")


def _constant_phase_weights(domain_weights: dict[str, float]) -> dict[str, dict[str, float]]:
    return {phase_name: dict(domain_weights) for phase_name in PHASE_NAMES}


def _proportional_weights() -> dict[str, float]:
    total_tokens = float(TOP_LEVEL_TOTAL_AVAILABLE_TOKENS)
    return {domain_name: TOP_LEVEL_DOMAIN_TOKEN_COUNTS[domain_name] / total_tokens for domain_name in DOMAIN_NAMES}


def _unimax_weights(max_epochs: float) -> dict[str, float]:
    weights = compute_unimax_weights(
        [TOP_LEVEL_DOMAIN_TOKEN_COUNTS[domain_name] for domain_name in DOMAIN_NAMES],
        budget=float(SIMULATED_EPOCH_TARGET_BUDGET),
        max_epochs=max_epochs,
    )
    return {domain_name: float(weight) for domain_name, weight in zip(DOMAIN_NAMES, weights, strict=True)}


def _weights_for_mixture(mixture: DelphiBaselineMixture) -> dict[str, float]:
    if mixture == DelphiBaselineMixture.PROPORTIONAL:
        return _proportional_weights()
    if mixture in UNIMAX_EPOCH_CAPS:
        return _unimax_weights(UNIMAX_EPOCH_CAPS[mixture])
    raise ValueError(f"Unsupported Delphi baseline mixture: {mixture!r}")


def _validate_phase_weights(phase_weights: dict[str, dict[str, float]], *, run_name: str) -> None:
    if set(phase_weights) != set(PHASE_NAMES):
        raise ValueError(f"{run_name} phase names mismatch: {sorted(phase_weights)}")
    phase0 = phase_weights["phase_0"]
    for phase_name, weights in phase_weights.items():
        if set(weights) != set(DOMAIN_NAMES):
            raise ValueError(f"{run_name}/{phase_name} domain names mismatch")
        total = sum(float(weight) for weight in weights.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"{run_name}/{phase_name} weights sum to {total}, expected 1.0")
        negative = {domain: weight for domain, weight in weights.items() if weight < 0}
        if negative:
            raise ValueError(f"{run_name}/{phase_name} has negative weights: {negative}")
        if weights != phase0:
            raise ValueError(f"{run_name} should use identical weights in both phases")


def _read_scaling_fits(analysis_output_path: str) -> dict[str, ScalingFit]:
    result_path = os.path.join(analysis_output_path, "isoflop_analysis_result.json")
    fs, _, _ = fsspec.get_fs_token_paths(result_path)
    with fs.open(result_path, "r") as handle:
        analysis_result = json.load(handle)

    scaling_fits: dict[str, ScalingFit] = {}
    for key, value in analysis_result["scaling_fits"].items():
        if len(value) != 2:
            raise ValueError(f"Expected 2 scaling fit values for {key!r}, got {len(value)}")
        scaling_fits[key] = ScalingFit(float(value[0]), float(value[1]))
    return scaling_fits


def _tensor_parallel_size(hidden_dim: int, tpu_type: str) -> int:
    cores = int(tpu_type.split("-")[1])
    chips = cores // 2
    tp = 1
    while hidden_dim % (chips // tp) != 0:
        tp *= 2
        if tp > chips:
            raise ValueError(f"Could not find tensor parallel size for hidden_dim={hidden_dim}, {tpu_type=}")
    return tp


def _candidate_for_budget(*, scaling_fits: dict[str, ScalingFit], target_flops: float, batch_size: int):
    candidate = predict_optimal_config(
        scaling_fits=scaling_fits,
        target_flops=target_flops,
        label=LABEL,
        heuristic=completed_adamh_heuristic,
        seq_len=SEQ_LEN_DELPHI,
    )
    if candidate is None:
        raise RuntimeError(f"Could not find optimal config for target_flops={target_flops:.2e}")

    train_steps = round(candidate.tokens / (batch_size * SEQ_LEN_DELPHI))
    optimizer_config = completed_adamh_heuristic.build_optimizer_config(batch_size, candidate.tokens)
    return replace(candidate, batch_size=batch_size, train_steps=train_steps, optimizer_config=optimizer_config)


def _build_mixture_data(
    mixture: DelphiBaselineMixture, train_tokens: int, model_config, batch_size: int, train_steps: int
):
    domain_weights = _weights_for_mixture(mixture)
    phase_weights = _constant_phase_weights(domain_weights)
    _validate_phase_weights(phase_weights, run_name=mixture.value)
    experiment = MixtureExperiment(
        name=EXPERIMENT_NAME,
        domains=build_top_level_domains(runtime_cache_region=DEFAULT_RUNTIME_CACHE_REGION),
        phase_schedule=PHASE_SCHEDULE,
        model_config=model_config,
        batch_size=batch_size,
        seq_len=SEQ_LEN_DELPHI,
        num_train_steps=train_steps,
        target_budget=None,
        resources=ResourceConfig.with_tpu("v5p-8", regions=[DEFAULT_TPU_REGION], zone=DEFAULT_TPU_ZONE),
        eval_harness_tasks=(),
        optimizer_config=None,
        eval_datasets_cache_path=None,
        hierarchical_runtime_domains=True,
    )
    data = experiment.create_mixture_config(WeightConfig(run_id=0, phase_weights=phase_weights))
    if train_tokens > SIMULATED_EPOCH_TARGET_BUDGET:
        raise ValueError(
            f"Delphi baseline train_tokens={train_tokens} exceeds top-level target budget "
            f"{SIMULATED_EPOCH_TARGET_BUDGET}; simulated epoching would be ill-defined."
        )
    data = replace(
        data,
        target_budget=SIMULATED_EPOCH_TARGET_BUDGET,
        experiment_budget=train_tokens,
        simulated_epoch_subset_seed=None,
    )
    return data, phase_weights


def _add_validation_components(data, validation_configs: dict[str, DatasetComponent] | None):
    if not validation_configs:
        return data

    new_components = {
        **data.components,
        **{key: value for key, value in validation_configs.items() if key not in data.components},
    }
    if isinstance(data.train_weights, dict):
        new_weights = {
            **data.train_weights,
            **{name: 0.0 for name in validation_configs if name not in data.train_weights},
        }
    else:
        new_weights = [
            (step_idx, {**weights, **{name: 0.0 for name in validation_configs if name not in weights}})
            for step_idx, weights in data.train_weights
        ]
    return replace(data, components=new_components, train_weights=new_weights)


def run_delphi_baseline_training(config: DelphiBaselineTrainingConfig) -> None:
    """Run one Delphi baseline training job."""
    scaling_fits = _read_scaling_fits(config.analysis_output_path)
    candidate = _candidate_for_budget(
        scaling_fits=scaling_fits,
        target_flops=config.target_flops,
        batch_size=config.batch_size,
    )
    params = candidate.model_config.total_trainable_params(completed_adamh_heuristic.vocab_size)
    realized_train_tokens = candidate.train_steps * config.batch_size * SEQ_LEN_DELPHI
    tp = _tensor_parallel_size(candidate.model_config.hidden_dim, config.tpu_type)

    logger.info(
        "Delphi baseline %s/%s: hidden_dim=%d layers=%d params=%.2e tokens=%.2e "
        "realized_tokens=%d batch_size=%d steps=%d tpu=%s tp=%d",
        config.mixture.value,
        _slug(config.target_flops),
        candidate.model_config.hidden_dim,
        candidate.model_config.num_layers,
        params,
        candidate.tokens,
        realized_train_tokens,
        config.batch_size,
        candidate.train_steps,
        config.tpu_type,
        tp,
    )

    data, phase_weights = _build_mixture_data(
        config.mixture,
        realized_train_tokens,
        candidate.model_config,
        config.batch_size,
        candidate.train_steps,
    )
    _validate_phase_weights(phase_weights, run_name=config.run_name)
    data = _add_validation_components(data, config.validation_configs)

    inner_config = train_lm.TrainLmConfig(
        data=data,
        trainer=TrainerConfig(
            tracker=WandbConfig(
                entity="marin-community",
                project="marin",
                tags=[
                    "issue-6607",
                    "delphi-baseline-mixtures",
                    "completed-adamh",
                    config.mixture.value,
                    f"FLOPs={config.target_flops:.1e}",
                    f"label={config.label}",
                    f"N={params:.1e}",
                    f"data_seed={config.data_seed}",
                    f"trainer_seed={config.trainer_seed}",
                ],
            ),
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            train_batch_size=candidate.batch_size,
            per_device_parallelism=-1,
            num_train_steps=candidate.train_steps,
            steps_per_eval=1000,
            checkpointer=CheckpointerConfig(
                save_interval=timedelta(minutes=10),
                keep=[dict(every=5000)],
            ),
            mesh=MeshConfig(
                axes={"data": -1, "replica": 1, "model": tp},
                compute_mapping={
                    "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                    "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                },
            ),
            seed=config.trainer_seed,
            allow_nondivisible_batch_size=True,
        ),
        train_seq_len=SEQ_LEN_DELPHI,
        model=candidate.model_config,
        optimizer=candidate.optimizer_config,
        data_seed=config.data_seed,
    )

    resources = ResourceConfig.with_tpu(config.tpu_type, regions=[config.tpu_region], zone=config.tpu_zone)
    pod_config = TrainLmOnPodConfig(
        train_config=inner_config,
        resources=resources,
        output_path=config.output_path,
        env_vars={
            "MARIN_PREFIX": marin_prefix_for_region(config.tpu_region),
            SKIP_EVAL_HARNESS_ENV_VAR: "1",
        },
    )
    run_levanter_train_lm(pod_config)


def _predict_run_spec(
    *,
    scaling_fits: dict[str, ScalingFit],
    mixture: DelphiBaselineMixture,
    target_flops: float,
    tpu_type: str,
    tpu_region: str,
    tpu_zone: str,
    batch_size: int,
    run_order: int,
    run_id_base: int = RUN_ID_BASE,
    trainer_seed: int = 0,
    data_seed: int | None = None,
) -> DelphiBaselineRunSpec:
    candidate = _candidate_for_budget(
        scaling_fits=scaling_fits,
        target_flops=target_flops,
        batch_size=batch_size,
    )
    train_tokens = round(candidate.tokens)
    realized_train_tokens = candidate.train_steps * batch_size * SEQ_LEN_DELPHI
    phase_weights = _constant_phase_weights(_weights_for_mixture(mixture))
    run_name = _run_name(mixture, target_flops, trainer_seed)
    _validate_phase_weights(phase_weights, run_name=run_name)
    non_embedding_params = int(candidate.model_config.total_trainable_params(0))
    total_params = int(candidate.model_config.total_trainable_params(completed_adamh_heuristic.vocab_size))
    return DelphiBaselineRunSpec(
        run_order=run_order,
        run_id=run_id_base + run_order,
        run_name=run_name,
        mixture=mixture.value,
        target_flops=target_flops,
        tpu_type=tpu_type,
        tpu_region=tpu_region,
        tpu_zone=tpu_zone,
        batch_size=batch_size,
        train_tokens=train_tokens,
        train_steps=candidate.train_steps,
        realized_train_tokens=realized_train_tokens,
        expected_checkpoint_step=candidate.train_steps - 1,
        model_hidden_dim=int(candidate.model_config.hidden_dim),
        model_layers=int(candidate.model_config.num_layers),
        non_embedding_params=non_embedding_params,
        total_trainable_params=total_params,
        tensor_parallel_size=_tensor_parallel_size(candidate.model_config.hidden_dim, tpu_type),
        data_seed=run_id_base + run_order if data_seed is None else data_seed,
        trainer_seed=trainer_seed,
        phase_weights=phase_weights,
    )


def save_delphi_baseline_manifest(config: SaveDelphiBaselineManifestConfig) -> None:
    """Persist run specs as JSON and CSV artifacts."""
    target_budgets = {
        float(item["target_flops"]): (str(item["tpu_type"]), int(item["batch_size"]))
        for item in json.loads(config.target_budgets_json)
    }
    scaling_fits = _read_scaling_fits(config.analysis_output_path)
    run_specs: list[DelphiBaselineRunSpec] = []
    for target_flops, (tpu_type, batch_size) in target_budgets.items():
        for mixture in config.mixtures:
            for trainer_seed in config.trainer_seeds:
                run_specs.append(
                    _predict_run_spec(
                        scaling_fits=scaling_fits,
                        mixture=mixture,
                        target_flops=target_flops,
                        tpu_type=tpu_type,
                        tpu_region=config.tpu_region,
                        tpu_zone=config.tpu_zone,
                        batch_size=batch_size,
                        run_order=len(run_specs),
                        run_id_base=config.run_id_base,
                        trainer_seed=trainer_seed,
                        data_seed=config.data_seed,
                    )
                )
    fs, _, _ = fsspec.get_fs_token_paths(config.output_path)
    fs.makedirs(config.output_path, exist_ok=True)
    with fs.open(os.path.join(config.output_path, "run_specs.json"), "w") as handle:
        json.dump([asdict(run_spec) for run_spec in run_specs], handle, indent=2, sort_keys=True)
    csv_buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        csv_buffer,
        fieldnames=[
            "run_order",
            "run_id",
            "run_name",
            "mixture",
            "target_flops",
            "tpu_type",
            "tpu_region",
            "tpu_zone",
            "batch_size",
            "train_tokens",
            "train_steps",
            "realized_train_tokens",
            "expected_checkpoint_step",
            "model_hidden_dim",
            "model_layers",
            "non_embedding_params",
            "total_trainable_params",
            "tensor_parallel_size",
            "data_seed",
            "trainer_seed",
        ],
    )
    writer.writeheader()
    for run_spec in run_specs:
        row = asdict(run_spec)
        row.pop("phase_weights")
        writer.writerow(row)
    with fs.open(os.path.join(config.output_path, "training_manifest.csv"), "w") as handle:
        handle.write(csv_buffer.getvalue())
    summary: dict[str, Any] = {
        "n_runs": len(run_specs),
        "mixtures": sorted({run_spec.mixture for run_spec in run_specs}),
        "target_flops": sorted({run_spec.target_flops for run_spec in run_specs}),
        "source_experiment": EXPERIMENT_NAME,
        "target_budget_tokens": SIMULATED_EPOCH_TARGET_BUDGET,
        "available_top_level_tokens": TOP_LEVEL_TOTAL_AVAILABLE_TOKENS,
        "unimax_epoch_caps": {
            mixture.value: cap for mixture, cap in UNIMAX_EPOCH_CAPS.items() if mixture in config.mixtures
        },
        "run_id_base": config.run_id_base,
    }
    with fs.open(os.path.join(config.output_path, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def _selected_target_budgets(values: tuple[str, ...]) -> dict[float, tuple[str, int]]:
    if not values:
        return dict(TARGET_BUDGETS)
    selected: dict[float, tuple[str, int]] = {}
    unknown: list[str] = []
    for value in values:
        target = float(value)
        if target not in TARGET_BUDGETS:
            unknown.append(value)
            continue
        selected[target] = TARGET_BUDGETS[target]
    if unknown:
        allowed = ", ".join(f"{budget:.0e}" for budget in TARGET_BUDGETS)
        raise ValueError(f"Unknown target budget(s): {unknown}. Allowed: {allowed}")
    return selected


def build_launch_artifacts(
    *,
    analysis_output_path: str,
    validation_configs: dict[str, DatasetComponent],
    mixtures: tuple[DelphiBaselineMixture, ...],
    target_budgets: dict[float, tuple[str, int]],
    tpu_region: str,
    tpu_zone: str,
    run_id_base: int = RUN_ID_BASE,
    with_table9_eval: bool = False,
    table9_tpu_zone: str = DEFAULT_TABLE9_TPU_ZONE,
    trainer_seeds: tuple[int, ...] = (0,),
    data_seed: int | None = None,
) -> LaunchArtifacts:
    """Build the executor graph for selected mixtures and FLOP budgets, optionally with Table 9 evaluations."""
    training_steps: list[ExecutorStep] = []
    eval_steps: list[ExecutorStep] = []
    scaling_fits = _read_scaling_fits(analysis_output_path) if with_table9_eval else None
    for target_flops, (tpu_type, batch_size) in target_budgets.items():
        for mixture in mixtures:
            for trainer_seed in trainer_seeds:
                run_order = len(training_steps)
                run_name = _run_name(mixture, target_flops, trainer_seed)
                training_step = ExecutorStep(
                    name=f"{EXPERIMENT_NAME}/{run_name}",
                    fn=run_delphi_baseline_training,
                    resources=ResourceConfig.with_tpu(tpu_type, regions=[tpu_region], zone=tpu_zone),
                    config=DelphiBaselineTrainingConfig(
                        analysis_output_path=analysis_output_path,
                        target_flops=target_flops,
                        tpu_type=tpu_type,
                        tpu_region=tpu_region,
                        tpu_zone=tpu_zone,
                        batch_size=batch_size,
                        mixture=mixture,
                        label=LABEL,
                        output_path=this_output_path(),
                        run_id=run_id_base + run_order,
                        run_name=run_name,
                        data_seed=run_id_base + run_order if data_seed is None else data_seed,
                        trainer_seed=trainer_seed,
                        validation_configs=validation_configs,
                    ),
                )
                training_steps.append(training_step)
                if scaling_fits is not None:
                    candidate = _candidate_for_budget(
                        scaling_fits=scaling_fits, target_flops=target_flops, batch_size=batch_size
                    )
                    eval_steps.append(
                        olmo_base_eval_step(
                            name=f"t9_{run_name}",
                            checkpoint=training_step / f"hf/step-{candidate.train_steps - 1}",
                            request_set_dir=TABLE9_REQUEST_SET_DIR,
                            resource_config=ResourceConfig.with_tpu(
                                "v6e-8", regions=[tpu_region], zone=table9_tpu_zone, disk="80g"
                            ),
                            wandb_group=TABLE9_WANDB_GROUP,
                            provenance={
                                "evaluator": "marin-native-table9-bpb",
                                "panel": "delphi_baseline_mixtures_issue6607",
                                "scale": _slug(target_flops),
                                "mixture": mixture.value,
                                "run_name": run_name,
                                "trainer_seed": str(trainer_seed),
                            },
                        )
                    )

    manifest_step = ExecutorStep(
        name=f"{EXPERIMENT_NAME}/manifest",
        fn=save_delphi_baseline_manifest,
        config=SaveDelphiBaselineManifestConfig(
            output_path=this_output_path(),
            analysis_output_path=analysis_output_path,
            mixtures=mixtures,
            target_budgets_json=json.dumps(
                [
                    {"target_flops": target_flops, "tpu_type": tpu_type, "batch_size": batch_size}
                    for target_flops, (tpu_type, batch_size) in target_budgets.items()
                ],
                sort_keys=True,
            ),
            tpu_region=tpu_region,
            tpu_zone=tpu_zone,
            run_id_base=run_id_base,
            trainer_seeds=trainer_seeds,
            data_seed=data_seed,
        ),
    )
    return LaunchArtifacts(
        run_specs=[], manifest_step=manifest_step, training_steps=training_steps, eval_steps=eval_steps
    )


def _write_local_dry_run_manifest(
    *,
    analysis_output_path: str,
    mixtures: tuple[DelphiBaselineMixture, ...],
    target_budgets: dict[float, tuple[str, int]],
    tpu_region: str,
    tpu_zone: str,
    run_id_base: int = RUN_ID_BASE,
    trainer_seeds: tuple[int, ...] = (0,),
    data_seed: int | None = None,
) -> list[DelphiBaselineRunSpec]:
    scaling_fits = _read_scaling_fits(analysis_output_path)
    run_specs: list[DelphiBaselineRunSpec] = []
    for target_flops, (tpu_type, batch_size) in target_budgets.items():
        for mixture in mixtures:
            for trainer_seed in trainer_seeds:
                run_specs.append(
                    _predict_run_spec(
                        scaling_fits=scaling_fits,
                        mixture=mixture,
                        target_flops=target_flops,
                        tpu_type=tpu_type,
                        tpu_region=tpu_region,
                        tpu_zone=tpu_zone,
                        batch_size=batch_size,
                        run_order=len(run_specs),
                        run_id_base=run_id_base,
                        trainer_seed=trainer_seed,
                        data_seed=data_seed,
                    )
                )
    LOCAL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    save_delphi_baseline_manifest(
        SaveDelphiBaselineManifestConfig(
            output_path=str(LOCAL_ARTIFACT_DIR),
            analysis_output_path=analysis_output_path,
            mixtures=mixtures,
            target_budgets_json=json.dumps(
                [
                    {"target_flops": target_flops, "tpu_type": tpu_type, "batch_size": batch_size}
                    for target_flops, (tpu_type, batch_size) in target_budgets.items()
                ],
                sort_keys=True,
            ),
            tpu_region=tpu_region,
            tpu_zone=tpu_zone,
            run_id_base=run_id_base,
            trainer_seeds=trainer_seeds,
            data_seed=data_seed,
        )
    )
    return run_specs


def _parse_mixtures(values: tuple[str, ...]) -> tuple[DelphiBaselineMixture, ...]:
    if not values:
        return tuple(DelphiBaselineMixture)
    return tuple(DelphiBaselineMixture(value) for value in values)


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixtures", nargs="*", default=[])
    parser.add_argument("--target-budgets", nargs="*", default=[])
    parser.add_argument("--tpu-region", default=DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=DEFAULT_TPU_ZONE)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--analysis-output-path", default=DEFAULT_ANALYSIS_OUTPUT_PATH)
    parser.add_argument("--run-id-base", type=int, default=RUN_ID_BASE, help="first run id and data seed of this launch")
    parser.add_argument("--with-table9-eval", action="store_true", help="chain the Table 9 BPB evaluator after each run")
    parser.add_argument(
        "--table9-tpu-zone", default=DEFAULT_TABLE9_TPU_ZONE, help="zone of the v6e-8 Table 9 evaluations"
    )
    parser.add_argument(
        "--trainer-seeds",
        nargs="+",
        type=int,
        default=[0],
        help="one run per trainer seed; seeds other than 0 add a _t<seed> suffix to the run name",
    )
    parser.add_argument(
        "--data-seed",
        type=int,
        default=None,
        help="fixed data seed for every run (trainer-seed repeats of one run); default: the run id",
    )
    return parser.parse_known_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]

    if args.tpu_region != DEFAULT_TPU_REGION or args.tpu_zone != DEFAULT_TPU_ZONE:
        raise ValueError(f"This launcher is pinned to {DEFAULT_TPU_REGION}/{DEFAULT_TPU_ZONE}")
    expected_prefix = marin_prefix_for_region(args.tpu_region)
    current_prefix = os.environ.get("MARIN_PREFIX")
    if current_prefix is not None and current_prefix != expected_prefix:
        raise ValueError(f"MARIN_PREFIX={current_prefix!r} does not match required east5 prefix {expected_prefix!r}")
    os.environ["MARIN_PREFIX"] = expected_prefix

    mixtures = _parse_mixtures(tuple(args.mixtures))
    target_budgets = _selected_target_budgets(tuple(args.target_budgets))
    validation_steps = default_validation_sets(tokenizer=llama3_tokenizer)
    validation_configs = {
        name: step_to_lm_mixture_component(step, include_raw_paths=False) for name, step in validation_steps.items()
    }

    if not args.analysis_output_path:
        raise ValueError("--analysis-output-path must be set; do not rerun isoflop analysis in this parent")

    if args.dry_run:
        run_specs = _write_local_dry_run_manifest(
            analysis_output_path=args.analysis_output_path,
            mixtures=mixtures,
            target_budgets=target_budgets,
            tpu_region=args.tpu_region,
            tpu_zone=args.tpu_zone,
            run_id_base=args.run_id_base,
            trainer_seeds=tuple(args.trainer_seeds),
            data_seed=args.data_seed,
        )
        logger.info("Wrote %d dry-run specs under %s", len(run_specs), LOCAL_ARTIFACT_DIR)
        return

    with executor_context():
        analysis_output_path = args.analysis_output_path
        artifacts = build_launch_artifacts(
            analysis_output_path=analysis_output_path,
            validation_configs=validation_configs,
            mixtures=mixtures,
            target_budgets=target_budgets,
            tpu_region=args.tpu_region,
            tpu_zone=args.tpu_zone,
            run_id_base=args.run_id_base,
            with_table9_eval=args.with_table9_eval,
            table9_tpu_zone=args.table9_tpu_zone,
            trainer_seeds=tuple(args.trainer_seeds),
            data_seed=args.data_seed,
        )
    steps = [*artifacts.steps]
    if os.getenv("CI") is not None:
        logger.info(
            "Built Delphi baseline-mixture graph with %d training and %d evaluation steps; skipping executor launch.",
            len(artifacts.training_steps),
            len(artifacts.eval_steps),
        )
        return

    executor_main(
        ExecutorMainConfig(max_concurrent=args.max_concurrent),
        steps=steps,
        description=f"{EXPERIMENT_NAME}: issue #6607 Delphi scaling baselines",
    )


if __name__ == "__main__":
    main()
