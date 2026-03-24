# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from experiments.ising_tokenizer.base.data import (
    BklIsingConfig,
    CRITICAL_TEMPERATURE_2D,
    SyntheticSplitConfig,
    TrajectoryTokenizerConfig,
    build_synthetic_split,
    temperature_normalization_stats,
)
from experiments.ising_tokenizer.base.model import ISING_TOKENIZER_V0_MODEL
from experiments.ising_tokenizer.base.train import (
    IsingRunConfig,
    IsingRolloutConfig,
    IsingTrainerConfig,
    IsingWandbConfig,
    run_local_ising_experiment,
)


@dataclass(frozen=True)
class IsingSmokeDataConfig:
    """Synthetic local data configuration for the first Ising smoke."""

    dynamics: BklIsingConfig = dataclasses.field(default_factory=BklIsingConfig)
    tokenizer: TrajectoryTokenizerConfig = dataclasses.field(default_factory=TrajectoryTokenizerConfig)
    train_split: SyntheticSplitConfig = dataclasses.field(
        default_factory=lambda: SyntheticSplitConfig(
            name="train",
            temperatures=(1.5, 1.8, 2.8, 3.1),
            num_examples=96,
            seed=0,
        )
    )
    validation_split: SyntheticSplitConfig = dataclasses.field(
        default_factory=lambda: SyntheticSplitConfig(
            name="validation",
            temperatures=(1.6, 2.9),
            num_examples=24,
            seed=1,
        )
    )
    critical_probe_split: SyntheticSplitConfig = dataclasses.field(
        default_factory=lambda: SyntheticSplitConfig(
            name="critical_probe",
            temperatures=(CRITICAL_TEMPERATURE_2D,),
            num_examples=24,
            seed=2,
        )
    )


DEFAULT_OUTPUT_DIR = "artifacts/ising_tokenizer/small_bkl_v0"


def build_local_smoke_config(data_config: IsingSmokeDataConfig) -> IsingRunConfig:
    """Build the default grug-style local smoke config."""

    return IsingRunConfig(
        model=dataclasses.replace(
            ISING_TOKENIZER_V0_MODEL,
            vocab_size=data_config.tokenizer.vocab_size(data_config.dynamics.lattice_size),
            max_seq_len=data_config.dynamics.seq_len,
        ),
        trainer=IsingTrainerConfig(),
    )


def build_datasets(data_config: IsingSmokeDataConfig):
    """Build train, validation, and critical-probe splits."""

    temperature_mean, temperature_std = temperature_normalization_stats(data_config.train_split.temperatures)
    train_dataset = build_synthetic_split(
        split_config=data_config.train_split,
        dynamics_config=data_config.dynamics,
        tokenizer_config=data_config.tokenizer,
        temperature_mean=temperature_mean,
        temperature_std=temperature_std,
    )
    validation_dataset = build_synthetic_split(
        split_config=data_config.validation_split,
        dynamics_config=data_config.dynamics,
        tokenizer_config=data_config.tokenizer,
        temperature_mean=temperature_mean,
        temperature_std=temperature_std,
    )
    critical_probe_dataset = build_synthetic_split(
        split_config=data_config.critical_probe_split,
        dynamics_config=data_config.dynamics,
        tokenizer_config=data_config.tokenizer,
        temperature_mean=temperature_mean,
        temperature_std=temperature_std,
    )
    return train_dataset, validation_dataset, critical_probe_dataset


def run_local_smoke(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    num_train_steps: int | None = None,
    train_examples: int | None = None,
    rollout_examples: int | None = None,
    wandb_project: str | None = None,
    wandb_entity: str | None = "marin-community",
    wandb_group: str | None = None,
    wandb_run_name: str | None = None,
    wandb_tags: tuple[str, ...] = (),
) -> dict[str, object]:
    """Run the first local Ising smoke experiment."""

    data_config = IsingSmokeDataConfig()
    if train_examples is not None:
        data_config = dataclasses.replace(
            data_config,
            train_split=dataclasses.replace(data_config.train_split, num_examples=train_examples),
        )

    run_config = build_local_smoke_config(data_config)
    if num_train_steps is not None:
        run_config = dataclasses.replace(
            run_config,
            trainer=dataclasses.replace(run_config.trainer, num_train_steps=num_train_steps),
        )
    if rollout_examples is not None:
        run_config = dataclasses.replace(
            run_config,
            rollout=IsingRolloutConfig(
                num_examples_per_temperature=rollout_examples,
                sample_seed=run_config.rollout.sample_seed,
            ),
        )
    if wandb_project is not None:
        run_config = dataclasses.replace(
            run_config,
            wandb=IsingWandbConfig(
                project=wandb_project,
                entity=wandb_entity,
                group=wandb_group,
                name=wandb_run_name,
                tags=wandb_tags,
            ),
        )

    train_dataset, validation_dataset, critical_probe_dataset = build_datasets(data_config)
    summary = run_local_ising_experiment(
        run_config,
        dynamics_config=data_config.dynamics,
        tokenizer_config=data_config.tokenizer,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        critical_probe_dataset=critical_probe_dataset,
        output_dir=output_dir,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local Ising tokenizer smoke.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--train-examples", type=int, default=None)
    parser.add_argument("--rollout-examples", type=int, default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-entity", default="marin-community")
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=[])
    args = parser.parse_args()

    summary = run_local_smoke(
        output_dir=args.output_dir,
        num_train_steps=args.steps,
        train_examples=args.train_examples,
        rollout_examples=args.rollout_examples,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        wandb_run_name=args.wandb_run_name,
        wandb_tags=tuple(args.wandb_tags),
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


if __name__ == "__main__":
    main()
