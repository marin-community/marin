# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Training loop for the grug-MoE zero-bubble pipeline.

Drives the reshard-free grug-MoE model (``model.py``) through the pipeline
primitive (``pipeline.py``) with an optax optimizer. The whole step — pipelined
forward/backward, optimizer update, parameter apply — is a single jitted
function run under a mesh whose only non-trivial axis is ``stage`` (8-way PP on a
v6e-8 puts one stage per chip).

`run_synthetic_smoke` trains on a deterministic arithmetic-sequence task whose
next token is fully predictable, so a correct pipeline must drive the loss toward
zero. It is the end-to-end "does it actually train" gate, runnable on a forced
multi-device CPU mesh or on a real TPU.
"""

from __future__ import annotations

import enum
import logging
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from experiments.grug.moe_zb.model import GrugMoEConfig, build_pipeline_params
from experiments.grug.moe_zb.pipeline import (
    STAGE_AXIS,
    PipelineModel,
    PipelineParams,
    pipeline_value_and_grad,
    zero_bubble_value_and_grad,
)

logger = logging.getLogger(__name__)


class Schedule(enum.Enum):
    """Which pipeline backend the training step uses."""

    GPIPE = enum.auto()
    ZERO_BUBBLE = enum.auto()


def make_stage_mesh(num_stages: int) -> Mesh:
    """Build a 1-D explicit mesh with ``num_stages`` devices on the stage axis."""
    devices = np.array(jax.devices())
    if devices.size < num_stages:
        raise ValueError(f"need >= {num_stages} devices for {num_stages}-way PP, have {devices.size}")
    return Mesh(devices[:num_stages].reshape(num_stages), (STAGE_AXIS,), axis_types=(AxisType.Explicit,))


def _value_and_grad_for(schedule: Schedule) -> Callable:
    if schedule is Schedule.GPIPE:
        return pipeline_value_and_grad
    return zero_bubble_value_and_grad


def shard_params(params: PipelineParams, mesh: Mesh) -> PipelineParams:
    """Reshard params onto the mesh: embed/head replicated, stage over ``stage``."""
    repl = NamedSharding(mesh, P())
    stage_sh = NamedSharding(mesh, P(STAGE_AXIS))
    return PipelineParams(
        embed=jax.tree_util.tree_map(lambda x: jax.reshard(x, repl), params.embed),
        stage=jax.tree_util.tree_map(lambda x: jax.reshard(x, stage_sh), params.stage),
        head=jax.tree_util.tree_map(lambda x: jax.reshard(x, repl), params.head),
    )


def make_train_step(
    model: PipelineModel,
    optimizer: optax.GradientTransformation,
    mesh: Mesh,
    *,
    num_microbatches: int,
    hidden_shape: tuple[int, ...],
    schedule: Schedule,
):
    """Build a jitted training step: pipelined grads -> optimizer -> apply."""
    value_and_grad = _value_and_grad_for(schedule)

    @jax.jit
    def train_step(params, opt_state, tokens):
        loss, grads = value_and_grad(
            params,
            tokens,
            tokens,
            model=model,
            mesh=mesh,
            num_microbatches=num_microbatches,
            hidden_shape=hidden_shape,
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    return train_step


def _arithmetic_batch(key, *, num_microbatches: int, microbatch: int, seq_len: int, vocab_size: int) -> jax.Array:
    """A learnable next-token task: each row is ``(start + i) mod vocab``.

    The next token is fully determined by the current one, so loss must fall
    toward zero once the model learns the increment.
    """
    batch = num_microbatches * microbatch
    starts = jax.random.randint(key, (batch, 1), 0, vocab_size)
    offsets = jnp.arange(seq_len)[None, :]
    tokens = (starts + offsets) % vocab_size
    return tokens.reshape(num_microbatches, microbatch, seq_len).astype(jnp.int32)


def run_synthetic_smoke(
    *,
    num_stages: int = 8,
    num_layers: int = 8,
    hidden_dim: int = 64,
    num_experts: int = 4,
    num_experts_per_token: int = 2,
    num_microbatches: int = 8,
    microbatch: int = 4,
    seq_len: int = 32,
    vocab_size: int = 64,
    steps: int = 60,
    learning_rate: float = 3e-3,
    schedule: Schedule = Schedule.ZERO_BUBBLE,
    seed: int = 0,
) -> list[float]:
    """Train the pipeline on the arithmetic task and return the per-step losses."""
    cfg = GrugMoEConfig(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        intermediate_dim=2 * hidden_dim,
        num_experts=num_experts,
        num_experts_per_token=num_experts_per_token,
        num_layers=num_layers,
        num_heads=4,
        num_kv_heads=4,
        max_seq_len=seq_len,
        num_stages=num_stages,
    )
    mesh = make_stage_mesh(num_stages)
    key = jax.random.PRNGKey(seed)
    init_key, data_key = jax.random.split(key)

    params, model = build_pipeline_params(cfg, key=init_key)
    optimizer = optax.adam(learning_rate)
    hidden_shape = (microbatch, seq_len, hidden_dim)

    losses: list[float] = []
    with jax.set_mesh(mesh):
        params = shard_params(params, mesh)
        opt_state = optimizer.init(params)
        train_step = make_train_step(
            model,
            optimizer,
            mesh,
            num_microbatches=num_microbatches,
            hidden_shape=hidden_shape,
            schedule=schedule,
        )
        for step in range(steps):
            data_key, batch_key = jax.random.split(data_key)
            tokens = _arithmetic_batch(
                batch_key,
                num_microbatches=num_microbatches,
                microbatch=microbatch,
                seq_len=seq_len,
                vocab_size=vocab_size,
            )
            tokens = jax.reshard(tokens, NamedSharding(mesh, P()))
            params, opt_state, loss = train_step(params, opt_state, tokens)
            loss_f = float(loss)
            losses.append(loss_f)
            if step % 5 == 0 or step == steps - 1:
                logger.info("step %3d  loss %.4f", step, loss_f)
    return losses


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    losses = run_synthetic_smoke()
    first, last = losses[0], losses[-1]
    logger.info("first loss %.4f -> last loss %.4f", first, last)
    # A correct pipeline learns the deterministic task: loss must drop sharply.
    ok = last < 0.5 * first
    logger.info("RESULT: %s", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
