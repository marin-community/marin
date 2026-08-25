# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Emit a compact Levanter oracle for PyTorch Grug training parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import MappingProxyType
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from haliax import Axis
from jax.sharding import AxisType, Mesh
from jax.sharding import PartitionSpec as P
from levanter.grug.attention import AttentionMask
from levanter.grug.sharding import unshard
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel, snowball_to_state_dict
from safetensors.numpy import save_file

TINY_CONFIG = MappingProxyType(
    {
        "vocab_size": 16,
        "hidden_dim": 8,
        "intermediate_dim": 10,
        "shared_expert_intermediate_dim": 12,
        "num_experts": 4,
        "num_experts_per_token": 2,
        "num_layers": 5,
        "num_heads": 2,
        "num_kv_heads": 1,
        "head_dim": 8,
        "max_seq_len": 16,
        "sliding_window": 3,
        "qk_mult": 1.37,
        "layer_norm_eps": 1e-5,
        "initializer_std": 0.02,
        "attention_implementation": "reference",
    }
)
SGD_LEARNING_RATE = 1e-3
GRADIENT_NAMES = (
    "model.embed_tokens.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.2.mlp.router.weight",
    "lm_head.weight",
)
_GRUG_MESH_AXIS_NAMES = ("replica_dcn", "data", "expert", "model")


def _oracle_mesh() -> Mesh:
    """Keep the one-example oracle on one device when a host exposes many."""
    devices = np.asarray(jax.local_devices()[:1], dtype=object).reshape((1, 1, 1, 1))
    axis_types = tuple(AxisType.Explicit for _ in _GRUG_MESH_AXIS_NAMES)
    return Mesh(devices, _GRUG_MESH_AXIS_NAMES, axis_types=axis_types)


class _RoutingObservation(NamedTuple):
    selected_experts: jax.Array
    combine_weights: jax.Array
    beta: jax.Array
    route_margin: jax.Array


def _reference_routed_mlp(
    expert_mlp,
    hidden_states: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
) -> jax.Array:
    """Small, ordinary-JAX MoE used only by this correctness oracle."""
    gate = jnp.einsum("td,edi->tei", hidden_states, unshard(expert_mlp.w_gate))
    up = jnp.einsum("td,edi->tei", hidden_states, unshard(expert_mlp.w_up))
    expert_outputs = jnp.einsum("tei,eid->ted", jax.nn.silu(gate) * up, unshard(expert_mlp.w_down))
    dispatch_weights = jax.nn.one_hot(selected_experts, expert_mlp.w_gate.shape[0])
    dispatch_weights = jnp.einsum("tke,tk->te", dispatch_weights, combine_weights)
    return jnp.einsum("te,ted->td", dispatch_weights, expert_outputs)


def _reference_block_components(
    block,
    hidden: jax.Array,
    short_mask: AttentionMask,
    long_mask: AttentionMask,
    is_long,
):
    attention_input = block.attn_gated_norm(block.rms_attn(hidden))
    attention_output = jax.lax.cond(
        is_long,
        lambda _: block.attn(attention_input, long_mask, disable_rope=True),
        lambda _: block.attn(attention_input, short_mask, disable_rope=False),
        operand=None,
    )
    hidden = hidden + attention_output
    mlp_input = block.mlp_gated_norm(block.rms_mlp(hidden))
    flat = mlp_input.reshape(-1, block.mlp.cfg.hidden_dim)
    router_logits = jnp.einsum("td,de->te", flat, block.mlp.router).astype(jnp.float32)
    top_values, top_indices = jax.lax.top_k(
        router_logits + block.mlp.router_bias,
        block.mlp.cfg.num_experts_per_token + 1,
    )
    alpha = top_values[:, -1:]
    route_margin = (top_values[:, -2] - top_values[:, -1]).min()
    selected = top_indices[:, :-1]
    selected_logits = jnp.take_along_axis(router_logits, selected, axis=-1)
    weights = jax.nn.sigmoid(selected_logits)
    weights = weights * (2.5 / (weights.sum(axis=-1, keepdims=True) + 1e-9))
    routed = _reference_routed_mlp(block.mlp.expert_mlp, flat, selected, weights).reshape(hidden.shape)
    hidden = hidden + routed + block.shared(mlp_input)
    q = max(1, flat.shape[0] * block.mlp.cfg.num_experts_per_token // block.mlp.cfg.num_experts)
    beta = jax.lax.top_k((router_logits - alpha).T, q)[0][:, -1]
    return hidden, _RoutingObservation(selected, weights, beta, route_margin)


def _reference_block(block, hidden: jax.Array, short_mask: AttentionMask, long_mask: AttentionMask, is_long):
    hidden, _ = _reference_block_components(block, hidden, short_mask, long_mask, is_long)
    return hidden


def _reference_transformer(model: SnowballLMHeadModel, tokens: jax.Array) -> jax.Array:
    transformer = model.transformer
    config = transformer.config
    short_mask = AttentionMask(is_causal=True, sliding_window=config.sliding_window)
    long_mask = AttentionMask(is_causal=True, sliding_window=None)
    hidden = transformer.token_embed.at[tokens].get(out_sharding=P(("replica_dcn", "data", "expert")))
    hidden = transformer.embed_gated_norm(transformer.embed_norm(hidden))
    stacked = jax.tree_util.tree_map(lambda *layers: jnp.stack(layers), *transformer.blocks)
    layer_indices = jnp.arange(len(transformer.blocks))
    long_schedule = (layer_indices % 4 == 3) | (layer_indices == len(transformer.blocks) - 1)

    def scan_layer(carry, layer_and_flag):
        layer, is_long = layer_and_flag
        return _reference_block(layer, carry, short_mask, long_mask, is_long), None

    hidden, _ = jax.lax.scan(scan_layer, hidden, (stacked, long_schedule))
    return transformer.final_gated_norm(transformer.final_norm(hidden))


def _capture(model: SnowballLMHeadModel, tokens: jax.Array) -> dict[str, jax.Array]:
    transformer = model.transformer
    config = transformer.config
    short_mask = AttentionMask(is_causal=True, sliding_window=config.sliding_window)
    long_mask = AttentionMask(is_causal=True, sliding_window=None)
    hidden = transformer.token_embed.at[tokens].get(out_sharding=P(("replica_dcn", "data", "expert")))
    hidden = transformer.embed_gated_norm(transformer.embed_norm(hidden))
    observations: dict[str, jax.Array] = {"hidden.embed": hidden}
    stacked = jax.tree_util.tree_map(lambda *layers: jnp.stack(layers), *transformer.blocks)
    layer_indices = jnp.arange(len(transformer.blocks))
    long_schedule = (layer_indices % 4 == 3) | (layer_indices == len(transformer.blocks) - 1)

    def scan_layer(carry, layer_and_flag):
        layer, is_long = layer_and_flag
        next_hidden, layer_observations = _reference_block_components(
            layer,
            carry,
            short_mask,
            long_mask,
            is_long,
        )
        return next_hidden, (
            next_hidden,
            layer_observations.selected_experts,
            layer_observations.combine_weights,
            layer_observations.beta,
            layer_observations.route_margin,
        )

    hidden, (layer_hiddens, routes, weights, betas, route_margins) = jax.lax.scan(
        scan_layer,
        hidden,
        (stacked, long_schedule),
    )
    for layer_idx in range(config.num_layers):
        observations[f"hidden.layer.{layer_idx}"] = layer_hiddens[layer_idx]
        observations[f"route.{layer_idx}"] = routes[layer_idx]
        observations[f"weight.{layer_idx}"] = weights[layer_idx]
        observations[f"beta.{layer_idx}"] = betas[layer_idx]
        observations[f"route_margin.{layer_idx}"] = route_margins[layer_idx]

    final_hidden = transformer.final_gated_norm(transformer.final_norm(hidden))
    logits = jnp.einsum("bsh,hv->bsv", final_hidden, transformer.output_proj)
    observations["hidden.final"] = final_hidden
    observations["logits"] = logits
    log_probs = jax.nn.log_softmax(unshard(logits[:, :-1]).astype(jnp.float32), axis=-1)
    observations["loss"] = -log_probs[0, jnp.arange(tokens.shape[1] - 1), tokens[0, 1:]].mean()
    observations["next_bias"] = -betas + betas.mean(axis=-1, keepdims=True)
    return observations


def _loss(model: SnowballLMHeadModel, tokens: jax.Array) -> jax.Array:
    hidden = _reference_transformer(model, tokens)
    logits = jnp.einsum("bsh,hv->bsv", hidden, model.transformer.output_proj)
    log_probs = jax.nn.log_softmax(unshard(logits[:, :-1]).astype(jnp.float32), axis=-1)
    return -log_probs[0, jnp.arange(tokens.shape[1] - 1), tokens[0, 1:]].mean()


def _selected_parameters(model: SnowballLMHeadModel) -> tuple[jax.Array, ...]:
    return (
        model.transformer.token_embed,
        model.transformer.blocks[0].attn.w_q,
        model.transformer.blocks[2].mlp.router,
        model.transformer.output_proj,
    )


def _apply_update(
    model: SnowballLMHeadModel,
    gradients: tuple[jax.Array, ...],
    next_bias: jax.Array,
) -> SnowballLMHeadModel:
    parameters = _selected_parameters(model)
    updated_parameters = tuple(
        parameter - SGD_LEARNING_RATE * gradient for parameter, gradient in zip(parameters, gradients, strict=True)
    )
    updated = eqx.tree_at(
        _selected_parameters,
        model,
        updated_parameters,
    )
    transformer = updated.transformer
    for layer_idx in range(len(transformer.blocks)):
        transformer = eqx.tree_at(
            lambda tree, i=layer_idx: tree.blocks[i].mlp.router_bias,
            transformer,
            next_bias[layer_idx],
        )
    return eqx.tree_at(lambda tree: tree.transformer, updated, transformer)


def emit_oracle(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = SnowballConfig(**TINY_CONFIG)
    input_ids = (np.arange(6, dtype=np.int32).reshape(1, 6) * 7 + 3) % TINY_CONFIG["vocab_size"]
    tokens = jnp.asarray(input_ids)
    with jax.set_mesh(_oracle_mesh()), jax.default_matmul_precision("highest"):
        # Canonical routing relies on JAX's documented lower-index tie order;
        # keep this independent from the margin-separated numerical fixture.
        tie_indices = jax.lax.top_k(jnp.zeros(4, dtype=jnp.float32), 3)[1]
        np.testing.assert_array_equal(np.asarray(jax.device_get(tie_indices)), np.array([0, 1, 2]))
        model = SnowballLMHeadModel.init(Axis("vocab", config.vocab_size), config, key=jax.random.key(7))
        state = {
            key: np.ascontiguousarray(np.asarray(jax.device_get(value)))
            for key, value in snowball_to_state_dict(model.transformer).items()
        }
        save_file(state, output_dir / "model.safetensors")
        del state
        hf_config = config.to_hf_config(config.vocab_size).to_dict()
        (output_dir / "config.json").write_text(json.dumps(hf_config, indent=2, sort_keys=True) + "\n")
        selected_parameters = _selected_parameters(model)

        def selected_loss(parameters):
            selected_model = eqx.tree_at(
                _selected_parameters,
                model,
                parameters,
            )
            return _loss(selected_model, tokens)

        _, gradients = jax.value_and_grad(selected_loss)(selected_parameters)
        observations = _capture(model, tokens)
        for key, gradient in zip(GRADIENT_NAMES, gradients, strict=True):
            observations[f"gradient.{key}"] = gradient if key == GRADIENT_NAMES[0] else gradient.T

        updated = _apply_update(model, gradients, observations["next_bias"])
        updated_observations = _capture(updated, tokens)
        for layer_idx in range(config.num_layers):
            observations[f"next_route.{layer_idx}"] = updated_observations[f"route.{layer_idx}"]

        host_observations = {key: np.asarray(jax.device_get(value)) for key, value in observations.items()}
        del gradients, observations, selected_loss, selected_parameters, updated, updated_observations
        jax.clear_caches()
        np.savez_compressed(
            output_dir / "observations.npz",
            input_ids=input_ids,
            **host_observations,
        )
        manifest = {
            "schema_version": 1,
            "producer": "levanter.models.snowball",
            "jax_backend": jax.default_backend(),
            "sgd_learning_rate": SGD_LEARNING_RATE,
            "update_parameter_names": GRADIENT_NAMES,
            "padding": "none",
            "matmul_precision": "highest",
        }
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    emit_oracle(args.output)


if __name__ == "__main__":
    main()
