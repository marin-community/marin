from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import random
from jax.sharding import AxisType, PartitionSpec as P
from jax.tree_util import register_dataclass
import optax

from .config import AttentionRuntimeConfig, GruGPTModelConfig, TrainingConfig, validate_config
from .model import GruGPTModelParameters, forward, init_parameters


@dataclass(frozen=True)
class TrainingState:
    step: int
    params: GruGPTModelParameters
    opt_state: optax.OptState


register_dataclass(TrainingState)


def create_mesh() -> jax.sharding.Mesh:
    devices = jax.devices()
    if not devices:
        raise RuntimeError("No JAX devices available")
    mesh = jax.make_mesh(
        (1, 1, len(devices)),
        axis_names=("replica", "data", "tensor"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
        devices=devices,
    )
    jax.set_mesh(mesh)
    return mesh


def synthetic_batch(key: jax.Array, *, batch_size: int, seq_len: int, vocab_size: int) -> dict[str, jax.Array]:
    tokens = random.randint(key, (batch_size, seq_len + 1), 0, vocab_size)
    tokens = jax.device_put(tokens, P(("replica", "data")))
    return {"tokens": tokens[:, :-1], "labels": tokens[:, 1:]}


def cross_entropy_loss(logits: jax.Array, labels: jax.Array) -> jax.Array:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    gathered = jnp.take_along_axis(log_probs, labels[..., None], axis=-1)
    return -jnp.mean(gathered)


def make_train_step(
    model_cfg: GruGPTModelConfig,
    runtime_cfg: AttentionRuntimeConfig,
    optimizer: optax.GradientTransformation,
    *,
    causal: bool = True,
):
    def loss_and_metrics(params: GruGPTModelParameters, batch: dict[str, jax.Array]):
        logits = forward(params, batch["tokens"], model_cfg, runtime_cfg, causal=causal)
        loss = cross_entropy_loss(logits, batch["labels"])
        metrics = {"loss": loss, "ppl": jnp.exp(loss)}
        return loss, metrics

    def step(state: TrainingState, batch: dict[str, jax.Array]):
        (loss, metrics), grads = jax.value_and_grad(loss_and_metrics, has_aux=True)(state.params, batch)
        updates, new_opt_state = optimizer.update(grads, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)
        return TrainingState(step=state.step + 1, params=new_params, opt_state=new_opt_state), metrics

    return jax.jit(step)


def main() -> None:
    # Tiny default so the demo can run on CPU.
    train_cfg = TrainingConfig(
        model=GruGPTModelConfig(
            vocab_size=1000,
            hidden_dim=128,
            intermediate_dim=512,
            num_layers=2,
            num_heads=4,
            num_kv_heads=4,
            max_seq_len=32,
        ),
        attention=AttentionRuntimeConfig(backend="reference"),
        learning_rate=1e-3,
        weight_decay=0.01,
        steps=10,
        global_batch_size=4,
        seed=0,
    )

    validate_config(train_cfg.model)
    create_mesh()

    rng = random.key(train_cfg.seed)
    rng, init_rng = random.split(rng)
    params = init_parameters(train_cfg.model, key=init_rng)

    optimizer = optax.adamw(learning_rate=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)
    opt_state = optimizer.init(params)
    state = TrainingState(step=0, params=params, opt_state=opt_state)

    seq_len = train_cfg.model.max_seq_len - 1
    train_step = make_train_step(train_cfg.model, train_cfg.attention, optimizer, causal=True)

    for _ in range(train_cfg.steps):
        rng, batch_rng = random.split(rng)
        batch = synthetic_batch(
            batch_rng,
            batch_size=train_cfg.global_batch_size,
            seq_len=seq_len,
            vocab_size=train_cfg.model.vocab_size,
        )
        state, metrics = train_step(state, batch)
        print(f"step={state.step:03d} loss={float(metrics['loss']):.4f} ppl={float(metrics['ppl']):.2f}")


if __name__ == "__main__":
    main()
