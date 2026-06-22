# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Step 1 de-risk: does the PRODUCTION grug-MoE forward lower + match the oracle
inside a stage-manual ``shard_map``?

Builds a tiny real ``Transformer`` (real ``Block``, ring-EP ``moe_mlp``, embed,
fused-CE head) and runs a single pipelined forward via
:func:`experiments.grug.moe_pp.pipeline.pipeline_forward_loss`, comparing the loss
against the UNMODIFIED non-pipelined ``Transformer.next_token_loss`` oracle (run at
``stage=1`` on the same params/tokens).

The model keeps its real sharding -- FSDP ``data``, ring-EP ``expert`` (a NESTED
``shard_map`` that manualizes ``expert``), vocab-TP ``model``; only ``stage`` is
manualized by the outer ``shard_map``. Three configs are checked:

- ``PP x EP``        : ``(stage=2, data=1, expert=2)`` -- expert parallelism under
  pipeline parallelism (the heart of the bet).
- ``PP x FSDP``      : ``(stage=2, data=4, expert=1)`` -- FSDP under PP.
- ``PP x FSDP x EP`` : ``(stage=2, data=2, expert=2)`` -- both GSPMD axes live.

All three lower and match the oracle. The ``data>1 AND expert>1`` case avoids the
XLA two-GSPMD-axes-under-a-manual-axis partitioner crash (the one the toy
``moe_zb`` documents) only because ``pipeline_forward_loss`` pre-places every
shard_map input on the mesh explicitly -- feeding params with their init-time
``(data, expert)`` shardings instead aborts XLA at ``spmd_partitioner_util.cc:497``.

Run:

    XLA_FLAGS=--xla_force_host_platform_device_count=8 \\
        uv run python -m experiments.grug.moe_pp.check_forward
"""

from __future__ import annotations

import logging
import os

if "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax
import jax.numpy as jnp
import numpy as np
from haliax.partitioning import set_mesh
from levanter.grug.sharding import compact_grug_mesh

from experiments.grug.moe.model import GrugModelConfig, Transformer
from experiments.grug.moe_pp.oracle import oracle_loss
from experiments.grug.moe_pp.pipeline import pipeline_forward_loss, stack_blocks_for_stages

logger = logging.getLogger(__name__)

STAGE = 2
BATCH = 16
SEQ_LEN = 64
LOSS_TOL = 1e-4

CONFIG = GrugModelConfig(
    vocab_size=512,
    hidden_dim=128,
    intermediate_dim=256,
    shared_expert_intermediate_dim=256,
    num_experts=4,
    num_experts_per_token=2,
    num_layers=4,
    num_heads=4,
    num_kv_heads=4,
    max_seq_len=SEQ_LEN,
    sliding_window=SEQ_LEN,
    moe_implementation="ring",
    # Reference (plain-JAX) attention: the parity checks run short sequences, and the
    # TPU splash kernel requires kv seq length to be a multiple of 128. The pipeline
    # differentiates attention inside the shard_map, where reference (einsum) attention
    # transposes cleanly; the EP ring is what these checks exercise.
    attention_implementation="reference",
)


def _run_config(label: str, expert: int, replica: int, model_key, tokens, weight) -> bool:
    pipe_mesh = compact_grug_mesh(
        expert_axis_size=expert, replica_axis_size=replica, model_axis_size=1, stage_axis_size=STAGE
    )
    oracle_mesh = compact_grug_mesh(
        expert_axis_size=expert, replica_axis_size=replica, model_axis_size=1, stage_axis_size=1
    )

    # Oracle: non-pipelined production loss on stage=1. `Transformer.init` is
    # deterministic in `model_key`, so re-initializing under each mesh yields
    # numerically identical params with that mesh's canonical shardings.
    with set_mesh(oracle_mesh):
        oracle_model = Transformer.init(CONFIG, key=model_key)
        ref_loss = float(np.asarray(jax.jit(oracle_loss)(oracle_model, tokens, weight)))

    with set_mesh(pipe_mesh):
        pipe_model = Transformer.init(CONFIG, key=model_key)
        stage_arrays, block_static = stack_blocks_for_stages(pipe_model, STAGE)

        @jax.jit
        def run(model, stage_arrays):
            return pipeline_forward_loss(
                model, stage_arrays, block_static, tokens, weight, mesh=pipe_mesh, num_stages=STAGE
            )

        pipe_loss = float(np.asarray(run(pipe_model, stage_arrays)))

    diff = abs(ref_loss - pipe_loss)
    ok = np.isfinite(pipe_loss) and diff < LOSS_TOL
    logger.info(
        "[%s] mesh=%s pipe_loss=%.6f oracle_loss=%.6f diff=%.2e -> %s",
        label,
        dict(pipe_mesh.shape),
        pipe_loss,
        ref_loss,
        diff,
        "PASS" if ok else "FAIL",
    )
    return ok


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    key = jax.random.PRNGKey(0)
    k_model, k_tokens = jax.random.split(key, 2)
    tokens = jax.random.randint(k_tokens, (BATCH, SEQ_LEN), 0, CONFIG.vocab_size, dtype=jnp.int32)
    weight = jnp.ones((BATCH, SEQ_LEN), dtype=jnp.float32)

    # data axis = 8 / (stage * replica * expert). replica=2 forces data=1 for the
    # PP x EP case; replica=1 lets data absorb the remainder for the others.
    ok_ep = _run_config(
        "PP x EP        (stage=2,data=1,expert=2)", expert=2, replica=2, model_key=k_model, tokens=tokens, weight=weight
    )
    ok_fsdp = _run_config(
        "PP x FSDP      (stage=2,data=4,expert=1)", expert=1, replica=1, model_key=k_model, tokens=tokens, weight=weight
    )
    ok_both = _run_config(
        "PP x FSDP x EP (stage=2,data=2,expert=2)", expert=2, replica=1, model_key=k_model, tokens=tokens, weight=weight
    )

    ok = ok_ep and ok_fsdp and ok_both
    logger.info("RESULT: %s", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
