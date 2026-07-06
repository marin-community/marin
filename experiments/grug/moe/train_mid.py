# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Training loop for the shared-expert + routed-MoE transformer (``model_mid``).

Thin wrapper over ``train_basic``: every layer is (no-norm attention) + (SwiGLU
shared expert at width ``hidden_dim``) + (QB-routed MoE, top-k of E), so only the
model constructor and the analytic FLOP count differ from the dense path. The
mesh/data/optimizer/ZeRO-1/callback/checkpoint machinery is reused verbatim via
:func:`experiments.grug.moe.train_basic._run_grug_local`.
"""

import jax.numpy as jnp
import jmp
import optax
from jaxtyping import PRNGKeyArray

from experiments.grug.dispatch import dispatch_grug_training_run
from experiments.grug.moe import model_basic, model_mid
from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.train import GrugRunConfig
from experiments.grug.moe.train_basic import BasicTrainState, _run_grug_local, _zero1_shard_tree


def _mid_compute_flops(model_config: GrugModelConfig) -> tuple[float, dict[str, float]]:
    """Analytic FLOPs for the shared-expert + routed-MoE transformer.

    Every layer contributes attention + a full-width shared SwiGLU expert
    (``6*H*I_shared``) + the routed top-k experts (``k*6*H*I_exp``) + the router
    projection (``2*H*E``). There are no dense or MLP-only layers.
    """
    h = model_config.hidden_dim
    seq = model_config.max_seq_len

    def attn_flops() -> float:
        head_dim = h / model_config.num_heads
        qkv_proj = 2 * h * (model_config.num_heads * head_dim + 2 * model_config.num_kv_heads * head_dim)
        dense_proj = 2 * h * h
        seq_flops = 2 * seq**2 * model_config.num_heads * head_dim
        seq_flops += 3 * seq * seq * model_config.num_heads
        seq_flops += 2 * seq * seq * head_dim * model_config.num_heads
        return qkv_proj + dense_proj + seq_flops / seq

    # Shared expert is a SwiGLU FFN (gate + up + down = 3 matmuls = 6*H*I_shared), run for
    # every token (no routing).
    shared_inter = model_mid.shared_expert_intermediate(model_config)
    shared_mlp = 2 * 3 * h * shared_inter
    # Routed experts: each active expert is a SwiGLU FFN (6*H*I_exp); only the top-k run per
    # token. Plus the dense router projection (2*H*E).
    expert_inter = model_basic.moe_expert_intermediate(model_config)
    routed_mlp = model_config.num_experts_per_token * 2 * 3 * h * expert_inter + 2 * h * model_config.num_experts

    per_layer = attn_flops() + shared_mlp + routed_mlp  # every layer is shared + routed MoE
    lm_head = 2 * h * model_config.vocab_size
    flops_per_token = model_config.num_layers * per_layer + lm_head
    flops_per_example = 3 * flops_per_token * seq
    return flops_per_example, {
        "throughput/flops_per_token_analytic": flops_per_token,
        "throughput/flops_per_example_analytic": flops_per_example,
    }


def initial_state(
    model_config: GrugModelConfig,
    *,
    optimizer: optax.GradientTransformation,
    mp: jmp.Policy,
    key: PRNGKeyArray,
    ema_beta: float | None,
    zero1: bool = False,
) -> BasicTrainState:
    """Build the initial train state for the ``model_mid`` transformer."""
    params = mp.cast_to_param(model_mid.Transformer.init(model_config, key=key))
    opt_state = optimizer.init(params)
    if zero1:
        opt_state = _zero1_shard_tree(opt_state)
    return BasicTrainState(
        step=jnp.array(0, dtype=jnp.int32),
        params=params,
        opt_state=opt_state,
        ema_params=params if ema_beta is not None else None,
    )


def _run_grug_mid_local(config: GrugRunConfig) -> None:
    """Entry point for the shared-expert + routed-MoE (``model_mid``) training loop."""
    _run_grug_local(config, init_state_fn=initial_state, compute_flops_fn=_mid_compute_flops)


def run_grug_mid(config: GrugRunConfig) -> None:
    """Dispatch shared-expert + routed-MoE training through Fray jobs."""
    trainer = config.trainer.trainer
    if trainer.id is None:
        raise ValueError("trainer.id must be set before dispatching grug training.")

    dispatch_grug_training_run(
        run_id=trainer.id,
        config=config,
        local_entrypoint=_run_grug_mid_local,
        resources=config.resources,
        processes_per_task=config.processes_per_task,
    )


__all__ = [
    "initial_state",
    "run_grug_mid",
]
