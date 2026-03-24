# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from experiments.grug.hybrid_mamba3.train import _make_train_step, initial_state
from experiments.speedrun.grug_hybrid_mamba3_sweep import _build_model_config, _build_train_config
from levanter.data.text.examples import GrugLmExample
from levanter.grug.attention import AttentionMask
from levanter.trainer import TrainerConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one synthetic native Grug hybrid MIMO train step on TPU.")
    parser.add_argument("--width-label", type=str, default="d512")
    parser.add_argument("--pattern-label", type=str, default="swa-linear-swa-full")
    parser.add_argument("--sliding-window", type=int, default=1024)
    parser.add_argument("--linear-mode", type=str, default="mimo", choices=("siso", "mimo"))
    parser.add_argument("--steps", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model_cfg = _build_model_config(
        width_label=args.width_label,
        pattern_label=args.pattern_label,
        sliding_window=args.sliding_window,
        linear_mode=args.linear_mode,
    )
    train_cfg = _build_train_config(model_cfg)
    trainer = TrainerConfig(
        train_batch_size=train_cfg.train_batch_size,
        num_train_steps=args.steps,
        use_explicit_mesh_axes=True,
        tracker=(),
        log_jaxprs=False,
        log_xla_hlo=False,
    )
    optimizer = train_cfg.optimizer_config.build(args.steps)
    train_step = _make_train_step(optimizer, trainer.mp, z_loss_weight=0.0, ema_beta=None)

    print(
        {
            "devices": len(jax.devices()),
            "backend": jax.default_backend(),
            "width_label": model_cfg.width_label,
            "pattern_label": model_cfg.pattern_label,
            "linear_mode": model_cfg.linear_mode,
            "batch_size": train_cfg.train_batch_size,
            "seq_len": model_cfg.max_seq_len,
        }
    )

    with trainer.use_device_mesh():
        state = initial_state(model_cfg, optimizer=optimizer, mp=trainer.mp, key=jax.random.PRNGKey(0), ema_beta=None)
        batch_sharding = NamedSharding(trainer.device_mesh, P("data", None))
        tokens = jax.device_put(
            jnp.zeros((train_cfg.train_batch_size, model_cfg.max_seq_len), dtype=jnp.int32), batch_sharding
        )
        loss_weight = jax.device_put(
            jnp.ones((train_cfg.train_batch_size, model_cfg.max_seq_len), dtype=jnp.float32), batch_sharding
        )
        batch = GrugLmExample(tokens=tokens, loss_weight=loss_weight, attn_mask=AttentionMask.causal())
        for _ in range(args.steps):
            state, metrics, _ = train_step(state, batch, compute_watch=False)
            loss = jax.block_until_ready(metrics["train/loss"])
            print({"loss": float(loss), "step": int(state.step)})


if __name__ == "__main__":
    main()
