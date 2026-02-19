# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Grugformer native speedrun entrypoint.

This runs directly through the grug-native loop:
- no LmConfig subclass
- no GrugWrapper in the training path
- no default_speedrun/default_train harness

Usage:
  uv run -m experiments.speedrun.grugformer_20260129.grugformer_speedrun \
    --size 130m \
    --train-cache-dir /path/to/tokenized_cache_root
"""

# nodryrun

import argparse
import logging
import os
from dataclasses import dataclass
from datetime import timedelta

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh

from levanter.checkpoint import CheckpointerConfig
from levanter.data.text import DatasetComponent, LmDataConfig, TextLmDatasetFormat
from levanter.grug.model import GrugModelConfig, init_parameters
from levanter.grug_native import (
    GrugEvalConfig,
    GrugNativeRunConfig,
    GrugTrainerConfig,
    run_grug_native,
)
from levanter.optim import AdamConfig
from levanter.tracker import NoopConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig

from experiments.marin_models import marin_tokenizer

logger = logging.getLogger(__name__)


def _get_num_train_steps(param_count: int, batch_size: int, max_seq_len: int, tpp: int = 20) -> int:
    total_tokens = param_count * tpp
    return max(1, total_tokens // (batch_size * max_seq_len))


def _size_presets() -> dict[str, GrugModelConfig]:
    base = dict(max_seq_len=2048, head_dim=None)
    return {
        "130m": GrugModelConfig(
            vocab_size=128_256,
            hidden_dim=512,
            intermediate_dim=1792,
            num_layers=6,
            num_heads=8,
            num_kv_heads=8,
            **base,
        ),
        "300m": GrugModelConfig(
            vocab_size=128_256,
            hidden_dim=768,
            intermediate_dim=2688,
            num_layers=12,
            num_heads=12,
            num_kv_heads=12,
            **base,
        ),
        "520m": GrugModelConfig(
            vocab_size=128_256,
            hidden_dim=1024,
            intermediate_dim=3584,
            num_layers=24,
            num_heads=16,
            num_kv_heads=16,
            **base,
        ),
        "1_2b": GrugModelConfig(
            vocab_size=128_256,
            hidden_dim=2048,
            intermediate_dim=7168,
            num_layers=16,
            num_heads=16,
            num_kv_heads=16,
            **base,
        ),
    }


def _batch_sizes() -> dict[str, int]:
    return {"130m": 128, "300m": 128, "520m": 128, "1_2b": 128}


def _total_trainable_params(cfg: GrugModelConfig) -> int:
    single_device = np.array(jax.devices()[:1]).reshape((1, 1))
    mesh = Mesh(single_device, ("data", "model"), axis_types=(AxisType.Explicit, AxisType.Explicit))
    with jax.set_mesh(mesh):
        params_shape = eqx.filter_eval_shape(init_parameters, cfg, key=jax.random.PRNGKey(0))

    total = 0
    for leaf in jax.tree_util.tree_leaves(params_shape):
        if hasattr(leaf, "dtype") and hasattr(leaf, "shape") and jnp.issubdtype(leaf.dtype, jnp.floating):
            total += int(np.prod(leaf.shape))
    return total


@dataclass(frozen=True)
class SpeedrunArgs:
    size: str
    train_cache_dir: str
    validation_cache_dir: str | None
    tokenizer: str
    steps: int | None
    batch_size: int | None
    learning_rate: float
    weight_decay: float
    steps_per_eval: int
    max_eval_batches: int | None
    output_dir: str
    run_id: str | None
    disable_wandb: bool
    seed: int
    ema_beta: float | None
    require_accelerator: bool


def _build_data_config(args: SpeedrunArgs) -> LmDataConfig:
    components: dict[str, DatasetComponent] = {
        "train": DatasetComponent(
            cache_dir=args.train_cache_dir,
            source=None,
            format=TextLmDatasetFormat(),
        )
    }
    train_weights: dict[str, float] = {"train": 1.0}

    if args.validation_cache_dir is not None:
        components["validation"] = DatasetComponent(
            cache_dir=args.validation_cache_dir,
            source=None,
            format=TextLmDatasetFormat(),
        )
        train_weights["validation"] = 0.0

    return LmDataConfig(
        tokenizer=args.tokenizer,
        components=components,
        train_weights=train_weights,
        cache_dir=None,
        shuffle=True,
        permutation_type="feistel",
    )


def build_run_config(args: SpeedrunArgs) -> GrugNativeRunConfig:
    sizes = _size_presets()
    if args.size not in sizes:
        raise ValueError(f"Unknown size: {args.size}")

    model_cfg = sizes[args.size]
    batch = args.batch_size if args.batch_size is not None else _batch_sizes()[args.size]
    params = _total_trainable_params(model_cfg)
    steps = args.steps if args.steps is not None else _get_num_train_steps(params, batch, model_cfg.max_seq_len, tpp=20)

    run_id = args.run_id or f"grugformer_native_{args.size}"
    tracker = (
        NoopConfig()
        if args.disable_wandb
        else WandbConfig(
            project="marin",
            name=run_id,
            tags=["grug-native", "speedrun", f"size:{args.size}"],
        )
    )

    trainer = TrainerConfig(
        id=run_id,
        seed=args.seed,
        train_batch_size=batch,
        num_train_steps=steps,
        steps_per_eval=max(1, args.steps_per_eval),
        tracker=tracker,
        use_explicit_mesh_axes=True,
        require_accelerator=args.require_accelerator,
        allow_nondivisible_batch_size=True,
        checkpointer=CheckpointerConfig(
            base_path=os.path.join(args.output_dir, args.size),
            append_run_id_to_base_path=False,
            save_interval=timedelta(minutes=10),
            keep=[{"every": 1000}],
        ),
    )

    data = _build_data_config(args)

    return GrugNativeRunConfig(
        model=model_cfg,
        data=data,
        optimizer=AdamConfig(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        ),
        trainer=GrugTrainerConfig(
            trainer=trainer,
            log_every=1,
            ema_beta=args.ema_beta,
        ),
        eval=GrugEvalConfig(
            steps_per_eval=args.steps_per_eval,
            max_eval_batches=args.max_eval_batches,
            eval_current=True,
            eval_ema=bool(args.ema_beta is not None),
        ),
    )


def _parse_args() -> SpeedrunArgs:
    parser = argparse.ArgumentParser(description="Run grugformer speedrun with grug-native training loop.")
    parser.add_argument("--size", type=str, default="130m", choices=list(_size_presets().keys()))
    parser.add_argument("--train-cache-dir", type=str, required=True)
    parser.add_argument("--validation-cache-dir", type=str, default=None)
    parser.add_argument("--tokenizer", type=str, default=marin_tokenizer)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--steps-per-eval", type=int, default=500)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="checkpoints/grug_native_speedrun")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ema-beta", type=float, default=0.999)
    parser.add_argument("--require-accelerator", action="store_true")
    parsed = parser.parse_args()

    return SpeedrunArgs(
        size=parsed.size,
        train_cache_dir=parsed.train_cache_dir,
        validation_cache_dir=parsed.validation_cache_dir,
        tokenizer=parsed.tokenizer,
        steps=parsed.steps,
        batch_size=parsed.batch_size,
        learning_rate=parsed.learning_rate,
        weight_decay=parsed.weight_decay,
        steps_per_eval=parsed.steps_per_eval,
        max_eval_batches=parsed.max_eval_batches,
        output_dir=parsed.output_dir,
        run_id=parsed.run_id,
        disable_wandb=parsed.disable_wandb,
        seed=parsed.seed,
        ema_beta=parsed.ema_beta,
        require_accelerator=parsed.require_accelerator,
    )


def main() -> None:
    args = _parse_args()
    run_config = build_run_config(args)
    logger.info(
        "Starting grug-native speedrun size=%s batch=%s steps=%s",
        args.size,
        run_config.trainer.trainer.train_batch_size,
        run_config.trainer.trainer.num_train_steps,
    )
    run_grug_native(run_config)


if __name__ == "__main__":
    main()
