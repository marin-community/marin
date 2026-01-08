# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import argparse
from dataclasses import dataclass, replace
from typing import Iterator

import jax
import numpy as np
import optax
from jax import numpy as jnp
from jax.tree_util import register_dataclass
from jax.sharding import AxisType

from levanter.store.cache import TreeCache

from levanter.grug.config import GrugModelConfig, GrugTrainingConfig, validate_config
from levanter.grug.data import DEFAULT_AXIS_MAPPING, build_token_loader
from levanter.grug.model import GrugModelParameters, forward, init_parameters


def synthetic_batch_iterator(
    *,
    rng: jax.Array,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
) -> Iterator[dict[str, jax.Array]]:
    """Infinite generator of random token/label pairs."""

    def _step(key: jax.Array) -> dict[str, jax.Array]:
        tokens = jax.random.randint(key, (batch_size, seq_len), 0, vocab_size)
        return {"tokens": tokens[:, :-1], "labels": tokens[:, 1:]}

    while True:
        rng, key = jax.random.split(rng)
        yield _step(key)


def dataloader_iterator(
    loader,
    *,
    seq_len: int,
) -> Iterator[dict[str, jax.Array]]:
    while True:
        batch = next(loader)
        tokens = batch[:, :seq_len]
        yield {"tokens": tokens[:, :-1], "labels": tokens[:, 1:]}


def create_mesh() -> jax.sharding.Mesh:
    devices = jax.devices()
    if not devices:
        raise RuntimeError("No JAX devices available")
    mesh = jax.make_mesh(
        (1, 1, len(devices)),
        axis_names=("replica_dcn", "replica", "data", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )
    jax.set_mesh(mesh)
    return mesh


def make_train_step(
    model_cfg: GrugModelConfig,
    optimizer: optax.GradientTransformation,
):
    def loss_and_metrics(params: GrugModelParameters, batch: dict[str, jax.Array]):
        logits = forward(params, batch["tokens"], model_cfg, mask=None)
        loss = cross_entropy_loss(logits, batch["labels"])
        metrics = {"loss": loss, "ppl": jnp.exp(loss)}
        return loss, metrics

    def step(state, batch):
        (loss, metrics), grads = jax.value_and_grad(loss_and_metrics, has_aux=True)(state.params, batch)
        updates, new_opt_state = optimizer.update(grads, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)
        new_state = replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
        return new_state, metrics

    return jax.jit(step)


def cross_entropy_loss(logits: jax.Array, labels: jax.Array) -> jax.Array:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    gathered = jnp.take_along_axis(log_probs, labels[..., None], axis=-1)
    return -jnp.mean(gathered)


def load_cache(cache_dir: str, *, seq_len: int) -> TreeCache[dict]:
    exemplar = {"input_ids": np.zeros(seq_len, dtype=np.int32)}
    return TreeCache.load(cache_dir, exemplar)


def run_training(train_cfg: GrugTrainingConfig, *, cache_dir: str | None = None) -> None:
    validate_config(train_cfg.model)
    mesh = create_mesh()

    rng = jax.random.key(train_cfg.seed)
    rng, init_rng = jax.random.split(rng)
    params = init_parameters(train_cfg.model, key=init_rng)
    optimizer = optax.adamw(learning_rate=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)
    opt_state = optimizer.init(params)
    state = TrainingState(step=0, params=params, opt_state=opt_state)

    seq_len = train_cfg.model.max_seq_len
    train_step = make_train_step(train_cfg.model, optimizer)

    if cache_dir:
        cache = load_cache(cache_dir, seq_len=seq_len)
        loader = build_token_loader(
            cache=cache,
            seq_len=seq_len,
            batch_size=train_cfg.global_batch_size,
            mesh=mesh,
            axis_mapping=DEFAULT_AXIS_MAPPING,
        )
        batch_iter = dataloader_iterator(iter(loader), seq_len=seq_len)
    else:
        batch_iter = synthetic_batch_iterator(
            rng=rng,
            batch_size=train_cfg.global_batch_size,
            seq_len=seq_len,
            vocab_size=train_cfg.model.vocab_size,
        )

    for _ in range(train_cfg.steps):
        batch = next(batch_iter)
        state, metrics = train_step(state, batch)
        print(f"step={state.step:03d} loss={float(metrics['loss']):.4f} ppl={float(metrics['ppl']):.2f}")


@dataclass(frozen=True)
class TrainingState:
    step: int
    params: GrugModelParameters
    opt_state: optax.OptState


register_dataclass(TrainingState)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Grug trainer.")
    parser.add_argument("--cache-dir", type=str, default=None, help="Optional TreeCache directory for real data.")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=50257)
    return parser.parse_args()


def build_training_config(args: argparse.Namespace) -> GrugTrainingConfig:
    model_cfg = GrugModelConfig(
        vocab_size=args.vocab_size,
        hidden_dim=args.hidden_dim,
        intermediate_dim=4 * args.hidden_dim,
        num_layers=args.layers,
        num_heads=args.heads,
        num_kv_heads=args.heads,
        max_seq_len=args.seq_len,
    )
    train_cfg = GrugTrainingConfig(
        model=model_cfg,
        learning_rate=1e-3,
        weight_decay=0.01,
        steps=args.steps,
        global_batch_size=args.batch_size,
        seed=0,
    )
    return train_cfg


def main() -> None:
    args = parse_args()
    cfg = build_training_config(args)
    run_training(cfg, cache_dir=args.cache_dir)


if __name__ == "__main__":
    main()
