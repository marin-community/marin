# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Schema-v2 Hero weights in the native and per-expert Hugging Face layouts."""

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from haliax.state_dict import StateDict, with_prefix
from jax.sharding import reshard

from levanter.models.hero_model import HeroBlock, HeroTransformer
from levanter.sharding import partition_spec_of


class _Parameter(NamedTuple):
    name: str
    value: jax.Array
    transpose: bool = False


def _global_parameters(model: HeroTransformer) -> tuple[_Parameter, ...]:
    return (
        _Parameter("model.embed_tokens.weight", model.token_embed),
        _Parameter("model.embed_norm.weight", model.embed_norm.weight),
        _Parameter("model.embed_gated_norm.down_proj.weight", model.embed_gated_norm.w_down, True),
        _Parameter("model.embed_gated_norm.up_proj.weight", model.embed_gated_norm.w_up, True),
        _Parameter("model.norm.weight", model.final_norm.weight),
        _Parameter("model.final_gated_norm.down_proj.weight", model.final_gated_norm.w_down, True),
        _Parameter("model.final_gated_norm.up_proj.weight", model.final_gated_norm.w_up, True),
        _Parameter("lm_head.weight", model.output_proj, True),
    )


def _block_parameters(block: HeroBlock) -> tuple[_Parameter, ...]:
    parameters = [
        _Parameter("input_layernorm.weight", block.rms_attn.weight),
        _Parameter("attn_gated_norm.down_proj.weight", block.attn_gated_norm.w_down, True),
        _Parameter("attn_gated_norm.up_proj.weight", block.attn_gated_norm.w_up, True),
        _Parameter("self_attn.q_proj.weight", block.attn.w_q, True),
        _Parameter("self_attn.k_proj.weight", block.attn.w_k, True),
        _Parameter("self_attn.v_proj.weight", block.attn.w_v, True),
        _Parameter("self_attn.o_proj.weight", block.attn.w_o, True),
        _Parameter("self_attn.attn_gate.weight", block.attn.attn_gate, True),
        _Parameter("post_attention_layernorm.weight", block.rms_mlp.weight),
        _Parameter("mlp_gated_norm.down_proj.weight", block.mlp_gated_norm.w_down, True),
        _Parameter("mlp_gated_norm.up_proj.weight", block.mlp_gated_norm.w_up, True),
        _Parameter("mlp.router.weight", block.mlp.router, True),
        _Parameter("mlp.router.bias", block.mlp.router_bias),
        _Parameter("mlp.experts.gate_proj.weight", block.mlp.expert_mlp.w_gate, True),
        _Parameter("mlp.experts.up_proj.weight", block.mlp.expert_mlp.w_up, True),
        _Parameter("mlp.experts.down_proj.weight", block.mlp.expert_mlp.w_down, True),
    ]
    if block.mlp.w_latent_down is not None:
        assert block.mlp.latent_norm is not None and block.mlp.w_latent_up is not None
        parameters.extend(
            (
                _Parameter("mlp.latent_down_proj.weight", block.mlp.w_latent_down, True),
                _Parameter("mlp.latent_norm.weight", block.mlp.latent_norm.weight),
                _Parameter("mlp.latent_up_proj.weight", block.mlp.w_latent_up, True),
            )
        )
    for name, conv in (
        ("self_attn.sconv_k", block.attn.sconv_k),
        ("sconv_attn", block.sconv_attn),
        ("sconv_mlp", block.sconv_mlp),
    ):
        if conv is not None:
            parameters.append(_Parameter(f"{name}.weight", conv.weight))
    if block.shared is not None:
        for i, expert in enumerate(block.shared):
            parameters.extend(
                (
                    _Parameter(f"shared_experts.{i}.gate_proj.weight", expert.w_gate, True),
                    _Parameter(f"shared_experts.{i}.up_proj.weight", expert.w_up, True),
                    _Parameter(f"shared_experts.{i}.down_proj.weight", expert.w_down, True),
                )
            )
    return tuple(parameters)


def _transpose(value: jax.Array, transpose: bool) -> jax.Array:
    return jnp.swapaxes(value, -1, -2) if transpose else value


def hero_to_state_dict(model: HeroTransformer, prefix: str | None = None) -> StateDict:
    """Export Hero with packed expert banks and canonical schema-v2 parameter names."""
    state = {with_prefix(prefix, p.name): _transpose(p.value, p.transpose) for p in _global_parameters(model)}
    for p in _block_parameters(model.stacked_blocks.stacked):
        for layer in range(model.config.num_layers):
            state[with_prefix(prefix, f"model.layers.{layer}.{p.name}")] = _transpose(p.value[layer], p.transpose)
    return state


def _read_parameter(state: StateDict, name: str, parameter: _Parameter, *, num_experts: int) -> jax.Array:
    if ".mlp.experts." in name and name not in state:
        # Native state dicts contain [E, out, in] banks; the vLLM exporter writes one
        # [out, in] tensor per expert. Both layouts represent the same schema-v2 model.
        stem, projection = name.split(".mlp.experts.")
        value = jnp.stack([jnp.asarray(state[f"{stem}.mlp.experts.{i}.{projection}"]) for i in range(num_experts)])
    else:
        value = jnp.asarray(state[name])
    return _transpose(value, parameter.transpose)


def _match_template(value: jax.Array, parameter: _Parameter) -> jax.Array:
    if value.shape != parameter.value.shape:
        raise ValueError(f"Hero weight {parameter.name}: expected {parameter.value.shape}, received {value.shape}")
    spec = partition_spec_of(parameter.value)
    return value if spec is None else reshard(value, spec)


def hero_from_state_dict(
    template: HeroTransformer, state_dict: StateDict, prefix: str | None = None
) -> HeroTransformer:
    """Load native banks or per-expert HF weights while preserving the template mesh."""
    globals_loaded = tuple(
        _match_template(
            _read_parameter(state_dict, with_prefix(prefix, p.name), p, num_experts=template.config.num_experts), p
        )
        for p in _global_parameters(template)
    )
    model = eqx.tree_at(lambda m: tuple(p.value for p in _global_parameters(m)), template, globals_loaded)
    stacked_loaded = []
    for p in _block_parameters(model.stacked_blocks.stacked):
        layers = [
            _read_parameter(
                state_dict,
                with_prefix(prefix, f"model.layers.{layer}.{p.name}"),
                p,
                num_experts=model.config.num_experts,
            )
            for layer in range(model.config.num_layers)
        ]
        stacked_loaded.append(_match_template(jnp.stack(layers), p))
    return eqx.tree_at(
        lambda m: tuple(p.value for p in _block_parameters(m.stacked_blocks.stacked)), model, tuple(stacked_loaded)
    )
