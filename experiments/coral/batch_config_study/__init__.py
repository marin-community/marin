# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Empirical runs for comparing Coral batch memory estimates to TPU HBM use.

The study uses a pretokenized DCLM cache already present in ``europe-west4``.
Each training run computes ``per_device_parallelism`` from
``experiments.coral.batch_config`` and leaves gradient accumulation to Levanter's
standard ``train_batch_size / (per_device_parallelism * data_axis_size)`` rule.

Preview the plan without submitting TPU work:

    python -m experiments.coral.batch_config_study --plan
"""

import argparse
import os
from collections.abc import Sequence
from dataclasses import dataclass, replace

from fray.cluster import ResourceConfig
from levanter.models.llama import LlamaConfig
from levanter.optim import AdamConfig
from marin.execution.lazy import ArtifactStep, lower
from marin.execution.step_runner import StepRunner
from marin.experiment.train import train_lm
from marin.processing.tokenize.tokenize import TokenizedCache
from marin.training.training import LevanterCheckpoint

from experiments.coral.batch_config import (
    BYTES_PER_GIB,
    adam_optimizer_bytes,
    batch_memory_bytes,
    dense_transformer_bytes,
    tpu_batch_config,
    tpu_hbm_capacity_bytes,
)
from experiments.llama import llama3_tokenizer_vocab_size, llama_150m, llama_600m

REGION = "europe-west4"
REGION_PREFIX = "gs://marin-eu-west4"
DCLM_BASELINE_100M_CACHE = f"{REGION_PREFIX}/tokenized/dclm_baseline_100m_llama3-f42a23"
DEFAULT_VERSION = "2026.07.09"
DEFAULT_NUM_TRAIN_STEPS = 20
DEFAULT_OVERHEAD_FACTOR = 1.0
STUDY_ID = "reps80g-nohf"
WANDB_GROUP_PREFIX = "coral-batch-config-study"
TPU_HOST_RAM = "80g"

TPUS = ("v5litepod-4", "v5litepod-8", "v6e-4")
BATCH_SIZES = (128, 256, 512)
REPLICATES = (1, 2, 3)

llama_2_4b = LlamaConfig(
    max_seq_len=1024,
    hidden_dim=2048,
    intermediate_dim=8192,
    num_heads=16,
    num_kv_heads=16,
    num_layers=28,
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model: LlamaConfig


@dataclass(frozen=True)
class StudyCase:
    name: str
    tpu: str
    model_name: str
    model: LlamaConfig
    batch_size: int
    replicate: int


@dataclass(frozen=True)
class CaseEstimate:
    parameter_count: int
    param_bytes: int
    optimizer_bytes: int
    activation_bytes: int
    total_bytes: int
    hbm_capacity_bytes: int
    per_device_parallelism: int
    gradient_accumulation: int


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("llama150m", llama_150m),
    ModelSpec("llama600m", llama_600m),
    ModelSpec("llama2p4b", llama_2_4b),
)


def _case_name(model_name: str, tpu: str, batch_size: int, replicate: int) -> str:
    tpu_name = tpu.replace("v5litepod-", "v5litepod").replace("v6e-", "v6e")
    return f"{model_name}-{tpu_name}-bs{batch_size}-r{replicate}"


CASES: tuple[StudyCase, ...] = tuple(
    StudyCase(
        _case_name(model_spec.name, tpu, batch_size, replicate),
        tpu,
        model_spec.name,
        model_spec.model,
        batch_size,
        replicate,
    )
    for tpu in TPUS
    for batch_size in BATCH_SIZES
    for model_spec in MODEL_SPECS
    for replicate in REPLICATES
)


def dclm_baseline_100m_europe_west4() -> ArtifactStep[TokenizedCache]:
    """Adopt the existing regional DCLM cache."""
    return ArtifactStep.adopt(
        "coral/data/dclm-baseline-100m-europe-west4",
        DEFAULT_VERSION,
        source=DCLM_BASELINE_100M_CACHE,
        kind=TokenizedCache,
        config={
            "tokenizer": "meta-llama/Meta-Llama-3.1-8B",
            "format": {"text_key": "text"},
            "tags": ["dclm", "100m", REGION],
        },
    )


def estimate_case(
    case: StudyCase,
    *,
    overhead_factor: float = DEFAULT_OVERHEAD_FACTOR,
) -> CaseEstimate:
    """Estimate memory and Levanter batch settings for one study case."""
    parameter_count = case.model.total_trainable_params(llama3_tokenizer_vocab_size)
    param_bytes, activation_bytes = dense_transformer_bytes(
        parameter_count=parameter_count,
        batch_size=case.batch_size,
        seq_len=case.model.max_seq_len,
        hidden_dim=case.model.hidden_dim,
        intermediate_dim=case.model.intermediate_dim,
        num_layers=case.model.num_layers,
    )
    optimizer_bytes = adam_optimizer_bytes(parameter_count)
    total_bytes = batch_memory_bytes(
        param_bytes=param_bytes,
        optimizer_bytes=optimizer_bytes,
        activation_bytes=activation_bytes,
        overhead_factor=overhead_factor,
    )
    per_device_parallelism, gradient_accumulation = tpu_batch_config(
        case.tpu,
        case.batch_size,
        total_bytes,
    )
    return CaseEstimate(
        parameter_count=parameter_count,
        param_bytes=param_bytes,
        optimizer_bytes=optimizer_bytes,
        activation_bytes=activation_bytes,
        total_bytes=total_bytes,
        hbm_capacity_bytes=tpu_hbm_capacity_bytes(case.tpu),
        per_device_parallelism=per_device_parallelism,
        gradient_accumulation=gradient_accumulation,
    )


def build_case(
    case: StudyCase,
    *,
    version: str = DEFAULT_VERSION,
    num_train_steps: int = DEFAULT_NUM_TRAIN_STEPS,
    wandb_project: str = "marin",
    overhead_factor: float = DEFAULT_OVERHEAD_FACTOR,
) -> ArtifactStep[LevanterCheckpoint]:
    """Build one training artifact with estimated batch settings applied."""
    estimate = estimate_case(case, overhead_factor=overhead_factor)
    run_id = f"coral-batch-config-{STUDY_ID}-{case.name}-{version}"
    dataset = dclm_baseline_100m_europe_west4()
    base = train_lm(
        name=f"checkpoints/coral/batch-config-study-{STUDY_ID}/{case.name}",
        version=version,
        model=case.model,
        optimizer=AdamConfig(learning_rate=6e-4, weight_decay=0.1),
        datasets={dataset: 1.0},
        batch_size=case.batch_size,
        seq_len=case.model.max_seq_len,
        num_train_steps=num_train_steps,
        z_loss_weight=None,
        evals=None,
        resources=ResourceConfig.with_tpu(case.tpu, regions=[REGION], ram=TPU_HOST_RAM),
        wandb_project=wandb_project,
        wandb_group=f"{WANDB_GROUP_PREFIX}-{STUDY_ID}-{version}",
        run_id=run_id,
        tags=[
            "coral",
            "batch-config-study",
            STUDY_ID,
            "dclm-baseline-100m-europe-west4",
            case.tpu,
            case.model_name,
            f"batch-{case.batch_size}",
            f"replicate-{case.replicate}",
            f"per-device-parallelism-{estimate.per_device_parallelism}",
            f"gradient-accumulation-{estimate.gradient_accumulation}",
            f"estimated-hbm-gib-{_gib(estimate.total_bytes):.1f}",
        ],
        env_vars=_training_env(wandb_project),
    )
    return _with_study_overrides(base, estimate.per_device_parallelism)


def build(
    *,
    case_names: Sequence[str] = (),
    version: str = DEFAULT_VERSION,
    num_train_steps: int = DEFAULT_NUM_TRAIN_STEPS,
    wandb_project: str = "marin",
    overhead_factor: float = DEFAULT_OVERHEAD_FACTOR,
) -> list[ArtifactStep[LevanterCheckpoint]]:
    """Build the selected study cases."""
    return [
        build_case(
            case,
            version=version,
            num_train_steps=num_train_steps,
            wandb_project=wandb_project,
            overhead_factor=overhead_factor,
        )
        for case in _selected_cases(case_names)
    ]


def _with_study_overrides(
    base: ArtifactStep[LevanterCheckpoint], per_device_parallelism: int
) -> ArtifactStep[LevanterCheckpoint]:
    def build_config(ctx):
        pod_config = base.build_config(ctx)
        trainer = replace(
            pod_config.train_config.trainer,
            per_device_parallelism=per_device_parallelism,
        )
        train_config = replace(
            pod_config.train_config,
            trainer=trainer,
            hf_save_steps=None,
        )
        return replace(pod_config, train_config=train_config)

    return ArtifactStep(
        name=base.name,
        version=base.version,
        artifact_type=base.artifact_type,
        run=base.run,
        build_config=build_config,
        deps=base.deps,
        runtime_args=base.runtime_args,
        override_path=base.override_path,
        expected_fingerprint=base.expected_fingerprint,
    )


def _training_env(wandb_project: str) -> dict[str, str]:
    env = {
        "MARIN_PREFIX": REGION_PREFIX,
        "WANDB_PROJECT": wandb_project,
    }
    for key in ("WANDB_ENTITY", "WANDB_MODE"):
        if value := os.environ.get(key):
            env[key] = value
    return env


def _selected_cases(case_names: Sequence[str]) -> list[StudyCase]:
    if not case_names:
        return list(CASES)
    by_name = {case.name: case for case in CASES}
    unknown = sorted(set(case_names) - set(by_name))
    if unknown:
        raise ValueError(f"Unknown case(s): {unknown}. Known cases: {sorted(by_name)}")
    return [by_name[name] for name in case_names]


def _print_plan(
    case_names: Sequence[str],
    *,
    overhead_factor: float,
    version: str,
    num_train_steps: int,
) -> None:
    print(f"version: {version}")
    print(f"region: {REGION}")
    print(f"data: {DCLM_BASELINE_100M_CACHE}")
    print(f"steps: {num_train_steps}")
    print(f"overhead_factor: {overhead_factor}")
    print()
    print(
        "| case | tpu | model | batch | params | estimate GiB | capacity GiB | "
        "per_device_parallelism | gradient_accumulation | replicate | run id |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for case in _selected_cases(case_names):
        estimate = estimate_case(case, overhead_factor=overhead_factor)
        run_id = f"coral-batch-config-{STUDY_ID}-{case.name}-{version}"
        print(
            f"| {case.name} | {case.tpu} | {case.model_name} | {case.batch_size} | "
            f"{estimate.parameter_count:,} | {_gib(estimate.total_bytes):.2f} | "
            f"{_gib(estimate.hbm_capacity_bytes):.2f} | {estimate.per_device_parallelism} | "
            f"{estimate.gradient_accumulation} | {case.replicate} | {run_id} |"
        )


def _gib(num_bytes: int) -> float:
    return num_bytes / BYTES_PER_GIB


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", action="store_true", help="Print the study matrix and exit without running jobs.")
    parser.add_argument("--case", action="append", default=[], choices=[case.name for case in CASES])
    parser.add_argument("--version", default=DEFAULT_VERSION)
    parser.add_argument("--num-train-steps", type=int, default=DEFAULT_NUM_TRAIN_STEPS)
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "marin"))
    parser.add_argument("--overhead-factor", type=float, default=DEFAULT_OVERHEAD_FACTOR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.plan:
        _print_plan(
            args.case,
            version=args.version,
            num_train_steps=args.num_train_steps,
            overhead_factor=args.overhead_factor,
        )
        return

    StepRunner().run(
        [
            lower(step)
            for step in build(
                case_names=args.case,
                version=args.version,
                num_train_steps=args.num_train_steps,
                wandb_project=args.wandb_project,
                overhead_factor=args.overhead_factor,
            )
        ]
    )


if __name__ == "__main__":
    main()
