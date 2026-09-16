# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Constant-mixture control sweep for Kotha and Liang Figure 7.

The original "Replaying pre-training data improves fine-tuning" Figure 7 sweep
kept one 4M-token rare-data subset fixed and varied the *schedule* by which the
subset was replayed over a 4B-token run. This launcher keeps the same broad/rare
setup and training scale, but uses a constant rare-data mixture throughout one
linear/WSD training run:

    rare weight p = target rare epochs / 1024

The important invariants are:
  - C4 remains the broad dataset.
  - StarCoder, FineMath, and FLAN are the three rare targets.
  - The rare subset is capped at one global batch, matching the old
    rare_fraction=1/1024 setup.
  - Effective train batch remains 1024. Smaller slices use Levanter
    microbatching/gradient accumulation via per_device_parallelism.
  - Optimizer warmup is left at AdamConfig's default, matching the historical
    Figure 7 launcher.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Final

from fray.cluster import ResourceConfig
from levanter.models.llama import LlamaConfig
from marin.datakit.download.dolma import DOLMA_DATASETS
from marin.execution.context import executor_context
from marin.execution.executor import ExecutorMainConfig, executor_main
from marin.execution.types import ExecutorStep, this_output_path
from marin.processing.tokenize import TokenizeConfig, tokenize
from marin.processing.tokenize.data_configs import lm_varying_mixture_data_config

from experiments.defaults import default_train
from experiments.llama import llama3_tokenizer, llama_150m
from experiments.simple_train_config import SimpleTrainConfig

logger = logging.getLogger(__name__)

DEFAULT_EXPERIMENT_NAME: Final = "pinlin_calvin_xu/data_mixture/replay_constant_mixture_sweep_20260706"
DEFAULT_PREFIX: Final = "gs://marin-us-east5"
DEFAULT_TPU_REGION: Final = "us-east5"
DEFAULT_TPU_ZONE: Final = "us-east5-a"

SEQ_LEN: Final = 4096
NUM_TRAIN_STEPS: Final = 1024
TRAIN_BATCH_SIZE: Final = 1024
RARE_FRACTION_DENOMINATOR: Final = 1024
RARE_BATCHES: Final = 1
VALIDATION_SEQUENCES_PER_COMPONENT: Final = 10_240
_DOLMA_V1_7_PATH: Final = "raw/dolma/v1.7"

TARGETS: Final = ("starcoder", "finemath", "flan")
EPOCH_GRID: Final = (1, 2, 4, 8, 12, 16, 24, 32)

DOLMA_LLAMA3_OVERRIDES: Final = {
    "c4": "tokenized/dolma/c4-e0e5ec",
    "flan": "tokenized/dolma/flan-a99cb2",
    "starcoder": "tokenized/dolma/starcoder-8b6089",
}

SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_ARTIFACT_DIR = (
    SCRIPT_DIR.parent
    / "domain_phase_mix"
    / "exploratory"
    / "two_phase_many"
    / "reference_outputs"
    / "replay_constant_mixture_sweep_20260706"
)


@dataclass(frozen=True)
class ConstantMixtureRunSpec:
    """One constant broad/rare mixture row."""

    run_order: int
    experiment_name: str
    run_name: str
    target: str
    target_epochs: int
    rare_weight: float
    common_weight: float
    rare_batches: int
    num_train_steps: int
    train_batch_size: int
    train_seq_len: int
    train_tokens: int
    tpu_type: str
    tpu_region: str
    tpu_zone: str
    per_device_parallelism: int
    output_path: str
    wandb_run_id: str


def _model_150m_4k() -> LlamaConfig:
    return replace(llama_150m, max_seq_len=SEQ_LEN)


def _finemath_3plus_tokenized() -> ExecutorStep:
    """Pinned FineMath-3+ tokenized cache without importing stale dataset modules."""

    return ExecutorStep(
        name="tokenized/finemath_3_plus",
        fn=tokenize,
        config=TokenizeConfig(
            train_paths=["raw/finemath/finemath-3plus"],
            validation_paths=[],
            cache_path=this_output_path(),
            tokenizer=llama3_tokenizer,
        ),
    ).with_output_path("tokenized/finemath_3_plus-a26b0f/")


def _dolma_tokenized(dataset: str) -> ExecutorStep:
    if dataset not in DOLMA_LLAMA3_OVERRIDES:
        raise ValueError(f"No pinned Dolma/Llama3 cache configured for {dataset}")
    return ExecutorStep(
        name=f"tokenized/dolma/{dataset}",
        fn=tokenize,
        config=TokenizeConfig(
            train_paths=[f"{_DOLMA_V1_7_PATH}/{file}" for file in DOLMA_DATASETS[dataset]],
            validation_paths=[],
            cache_path=this_output_path(),
            tokenizer=llama3_tokenizer,
        ),
    ).with_output_path(DOLMA_LLAMA3_OVERRIDES[dataset])


def _all_component_steps() -> dict[str, ExecutorStep]:
    return {
        "c4": _dolma_tokenized("c4"),
        "starcoder": _dolma_tokenized("starcoder"),
        "flan": _dolma_tokenized("flan"),
        "finemath": _finemath_3plus_tokenized(),
    }


def _components_for_target(target: str, component_steps: dict[str, ExecutorStep]) -> dict[str, ExecutorStep]:
    return {
        "c4": component_steps["c4"],
        target: component_steps[target],
    }


def _weights_for_epochs(target: str, target_epochs: int) -> dict[str, float]:
    rare_weight = target_epochs / RARE_FRACTION_DENOMINATOR
    if not 0.0 < rare_weight < 1.0:
        raise ValueError(f"target_epochs={target_epochs} gives invalid rare weight {rare_weight}")
    return {"c4": 1.0 - rare_weight, target: rare_weight}


def _run_name(target: str, target_epochs: int) -> str:
    return f"replay_const_fig7_{target}_e{target_epochs:02d}"


def _output_path(experiment_name: str, target: str, target_epochs: int) -> str:
    return f"checkpoints/{experiment_name}/{target}/epochs_{target_epochs:02d}"


def _build_data_config(target: str, target_epochs: int, component_steps: dict[str, ExecutorStep]):
    components = _components_for_target(target, component_steps)
    weights = _weights_for_epochs(target, target_epochs)
    return lm_varying_mixture_data_config(
        components=components,
        weights_list=[(0, weights)],
        max_train_batches={target: RARE_BATCHES},
        num_validation_sequences={
            "c4": VALIDATION_SEQUENCES_PER_COMPONENT,
            target: VALIDATION_SEQUENCES_PER_COMPONENT,
        },
    )


def _build_train_step(
    *,
    target: str,
    target_epochs: int,
    experiment_name: str,
    tpu_type: str,
    tpu_region: str,
    tpu_zone: str,
    per_device_parallelism: int,
    component_steps: dict[str, ExecutorStep],
) -> ExecutorStep:
    run_name = _run_name(target, target_epochs)
    train_config = SimpleTrainConfig(
        resources=ResourceConfig.with_tpu(tpu_type, regions=[tpu_region], zone=tpu_zone),
        train_batch_size=TRAIN_BATCH_SIZE,
        per_device_parallelism=per_device_parallelism,
        num_train_steps=NUM_TRAIN_STEPS,
        learning_rate=3e-3,
        lr_schedule="linear",
        decay=0.1,
        weight_decay=0.1,
        min_lr_ratio=0.0,
        train_seq_len=SEQ_LEN,
        steps_per_eval=NUM_TRAIN_STEPS // 20,
        steps_per_export=None,
        steps_per_hf_export=-1,
        data_seed=42,
        trainer_seed=0,
        # Levanter otherwise falls back to the output-path basename. The leaf is
        # epochs_XX for all targets, so set a target-specific W&B run id.
        env_vars={"RUN_ID": run_name},
    )
    return default_train(
        name=run_name,
        tokenized=_build_data_config(target, target_epochs, component_steps),
        model_config=_model_150m_4k(),
        train_config=train_config,
        tags=[
            "replay-constant-mixture-sweep",
            "figure-7-control",
            "constant-mixture",
            f"target={target}",
            f"epochs={target_epochs}",
        ],
        use_default_validation=False,
        eval_harness_tasks=[],
        wandb_name=run_name,
        wandb_group=experiment_name,
        override_output_path=_output_path(experiment_name, target, target_epochs),
    )


def _build_specs(
    *,
    targets: tuple[str, ...],
    epoch_grid: tuple[int, ...],
    experiment_name: str,
    tpu_type: str,
    tpu_region: str,
    tpu_zone: str,
    per_device_parallelism: int,
) -> list[ConstantMixtureRunSpec]:
    specs: list[ConstantMixtureRunSpec] = []
    for target in targets:
        for target_epochs in epoch_grid:
            weights = _weights_for_epochs(target, target_epochs)
            specs.append(
                ConstantMixtureRunSpec(
                    run_order=len(specs),
                    experiment_name=experiment_name,
                    run_name=_run_name(target, target_epochs),
                    target=target,
                    target_epochs=target_epochs,
                    rare_weight=weights[target],
                    common_weight=weights["c4"],
                    rare_batches=RARE_BATCHES,
                    num_train_steps=NUM_TRAIN_STEPS,
                    train_batch_size=TRAIN_BATCH_SIZE,
                    train_seq_len=SEQ_LEN,
                    train_tokens=NUM_TRAIN_STEPS * TRAIN_BATCH_SIZE * SEQ_LEN,
                    tpu_type=tpu_type,
                    tpu_region=tpu_region,
                    tpu_zone=tpu_zone,
                    per_device_parallelism=per_device_parallelism,
                    output_path=_output_path(experiment_name, target, target_epochs),
                    wandb_run_id=_run_name(target, target_epochs),
                )
            )
    return specs


def _write_manifest(specs: list[ConstantMixtureRunSpec]) -> Path:
    LOCAL_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = LOCAL_ARTIFACT_DIR / "constant_mixture_manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(specs[0]).keys()))
        writer.writeheader()
        for spec in specs:
            writer.writerow(asdict(spec))

    summary_path = LOCAL_ARTIFACT_DIR / "constant_mixture_manifest_summary.json"
    summary = {
        "experiment_name": specs[0].experiment_name,
        "num_runs": len(specs),
        "targets": sorted({spec.target for spec in specs}),
        "target_epochs": sorted({spec.target_epochs for spec in specs}),
        "train_tokens_per_run": NUM_TRAIN_STEPS * TRAIN_BATCH_SIZE * SEQ_LEN,
        "rare_fraction_denominator": RARE_FRACTION_DENOMINATOR,
        "rare_batches": RARE_BATCHES,
        "notes": [
            "Constant rare-data weight is target_epochs / 1024.",
            "Rare-data subset is capped at one global batch, matching the Figure 7 rare_fraction=1/1024 setup.",
            "Effective global batch remains 1024; per_device_parallelism controls microbatching/gradient accumulation.",
            "Optimizer warmup is inherited from AdamConfig's default, matching the historical Figure 7 launcher.",
            "W&B run id is set to replay_const_fig7_<target>_eXX so rows cannot collide across targets.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return manifest_path


def _parse_targets(values: list[str]) -> tuple[str, ...]:
    targets = tuple(values or TARGETS)
    unknown = sorted(set(targets) - set(TARGETS))
    if unknown:
        raise ValueError(f"Unknown targets: {unknown}; expected subset of {TARGETS}")
    return targets


def _parse_epoch_grid(values: list[int]) -> tuple[int, ...]:
    epoch_grid = tuple(values or EPOCH_GRID)
    if any(epoch <= 0 for epoch in epoch_grid):
        raise ValueError(f"Epoch grid must be positive; got {epoch_grid}")
    if any(epoch >= RARE_FRACTION_DENOMINATOR for epoch in epoch_grid):
        raise ValueError(f"Epoch grid must be < {RARE_FRACTION_DENOMINATOR}; got {epoch_grid}")
    return epoch_grid


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", nargs="*", default=list(TARGETS), choices=TARGETS)
    parser.add_argument("--target-epochs", nargs="*", type=int, default=list(EPOCH_GRID))
    parser.add_argument("--tpu-type", default="v5p-8")
    parser.add_argument("--tpu-region", default=DEFAULT_TPU_REGION)
    parser.add_argument("--tpu-zone", default=DEFAULT_TPU_ZONE)
    parser.add_argument("--per-device-parallelism", type=int, default=16)
    parser.add_argument("--max-concurrent", type=int, default=12)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--dry-run", "--dry_run", action="store_true")
    parser.add_argument("--force-run-failed", "--force_run_failed", action="store_true", default=True)
    parser.add_argument("--no-force-run-failed", dest="force_run_failed", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.tpu_region != DEFAULT_TPU_REGION or args.tpu_zone != DEFAULT_TPU_ZONE:
        raise ValueError("This replay-paper sweep is currently constrained to us-east5/us-east5-a")
    if args.prefix != DEFAULT_PREFIX:
        raise ValueError(f"This replay-paper sweep must use prefix {DEFAULT_PREFIX}")
    current_prefix = os.environ.get("MARIN_PREFIX")
    if current_prefix is not None and current_prefix != DEFAULT_PREFIX:
        raise ValueError(f"MARIN_PREFIX={current_prefix!r} does not match required prefix {DEFAULT_PREFIX!r}")
    os.environ["MARIN_PREFIX"] = DEFAULT_PREFIX

    targets = _parse_targets(args.targets)
    epoch_grid = _parse_epoch_grid(args.target_epochs)
    specs = _build_specs(
        targets=targets,
        epoch_grid=epoch_grid,
        experiment_name=args.experiment_name,
        tpu_type=args.tpu_type,
        tpu_region=args.tpu_region,
        tpu_zone=args.tpu_zone,
        per_device_parallelism=args.per_device_parallelism,
    )
    manifest_path = _write_manifest(specs)
    logger.info("Wrote manifest for %d runs to %s", len(specs), manifest_path)

    with executor_context():
        component_steps = _all_component_steps()
        steps = [
            _build_train_step(
                target=spec.target,
                target_epochs=spec.target_epochs,
                experiment_name=spec.experiment_name,
                tpu_type=spec.tpu_type,
                tpu_region=spec.tpu_region,
                tpu_zone=spec.tpu_zone,
                per_device_parallelism=spec.per_device_parallelism,
                component_steps=component_steps,
            )
            for spec in specs
        ]
    executor_main(
        ExecutorMainConfig(
            prefix=args.prefix,
            dry_run=args.dry_run,
            force_run_failed=args.force_run_failed,
            max_concurrent=args.max_concurrent,
        ),
        steps=steps,
        description=(f"{args.experiment_name}: constant C4/rare mixture controls for " "Kotha and Liang Figure 7"),
    )


if __name__ == "__main__":
    main()
