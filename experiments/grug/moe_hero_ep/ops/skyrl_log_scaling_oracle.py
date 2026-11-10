# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Write the tiny log-scaled Hero oracle that SkyRL's Megatron Grug model is checked against.

The fixture holds a randomly initialized Hero-shaped model in the artifact-schema-3 HF layout and
the JAX next-token log-probabilities of two documents, with log scaling on and off. The floor is 16
keys, so most positions are scaled. Weights are rounded to bf16 before the JAX forward, so the only
difference left for the Megatron side is its bf16 compute.

    JAX_PLATFORMS=cpu uv run python -m experiments.grug.moe_hero_ep.ops.skyrl_log_scaling_oracle \\
        --output vendor/skyrl/skyrl-train/tests/fixtures/grug_log_scaling_oracle
"""

import argparse
import dataclasses
import json
import math
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.sharding import AxisType, Mesh
from levanter.grug.attention import AttentionMask
from safetensors.torch import save_file

from experiments.grug.moe_hero_ep.heuristic import HERO_MODEL
from experiments.grug.moe_hero_ep.model import (
    Transformer,
    grugmoe_inference_state_dict,
)
from experiments.grug.moe_hero_ep.ops.export_metadata import bf16_hf_config, split_expert_names

# Hero's layer layout (sconv, latent experts, two shared experts, fused RoPE)
# at a width that Transformer Engine runs and context parallelism of two divides.
ORACLE_MODEL = dataclasses.replace(
    HERO_MODEL,
    vocab_size=512,
    hidden_dim=64,
    intermediate_dim=32,
    shared_expert_intermediate_dim=32,
    latent_dim=32,
    num_experts=8,
    num_experts_per_token=2,
    num_layers=2,
    num_heads=4,
    num_kv_heads=2,
    local_kv_heads=2,
    global_kv_heads=2,
    head_dim=32,
    max_seq_len=128,
    sliding_window=16,
    global_every=2,
    # Every token keeps its experts, as in Megatron's dropless dispatch.
    capacity_factor=8.0,
    # Twice the Hero's 0.5 / sqrt(hidden): log scaling moves past-floor log-probabilities by about 12x
    # the bf16 noise, and that noise is the same below and past the floor.
    initializer_std=1.0 / math.sqrt(64),
    log_scaling_n_floor=16,
    log_scaling_alpha=1.0,
    log_scaling_power=2.0,
    attention_implementation="reference",
    moe_implementation="fixed_all_to_all",
)
DOCUMENT_LENGTHS = (56, 40)
LOG_SCALING_FIELDS = ("log_scaling_n_floor", "log_scaling_alpha", "log_scaling_power")


def _mesh() -> Mesh:
    return Mesh(
        np.asarray(jax.devices()[:1]).reshape(1, 1, 1, 1, 1),
        ("replica_dcn", "data", "expert", "context", "model"),
        axis_types=(AxisType.Explicit,) * 5,
    )


def _next_token_logprobs(model: Transformer, tokens: np.ndarray, segment_ids: np.ndarray) -> np.ndarray:
    """Return log p(token[t + 1] | tokens[: t + 1]) for one packed row."""
    mask = AttentionMask.causal().with_segment_ids(jnp.asarray(segment_ids[None]))
    logits = eqx.filter_jit(model.logits)(jnp.asarray(tokens[None]), mask)[0].astype(jnp.float32)
    logprobs = jax.nn.log_softmax(logits, axis=-1)
    return np.asarray(jnp.take_along_axis(logprobs[:-1], jnp.asarray(tokens[1:, None]), axis=-1)[:, 0])


def _bf16_rounded(model: Transformer) -> Transformer:
    return jax.tree.map(lambda x: x.astype(jnp.bfloat16).astype(jnp.float32) if eqx.is_inexact_array(x) else x, model)


def _hf_tensors(model: Transformer, num_experts: int) -> dict[str, torch.Tensor]:
    tensors = {}
    for name, value in grugmoe_inference_state_dict(model).items():
        array = torch.from_numpy(np.array(value, dtype=np.float32)).to(torch.bfloat16)
        names = split_expert_names(name, num_experts)
        if names == (name,):
            tensors[name] = array.contiguous()
            continue
        for expert, expert_name in enumerate(names):
            tensors[expert_name] = array[expert].contiguous()
    return tensors


def write_oracle(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(23)
    documents = [rng.integers(1, ORACLE_MODEL.vocab_size, size=length, dtype=np.int32) for length in DOCUMENT_LENGTHS]
    packed = np.concatenate(documents)
    segment_ids = np.concatenate([np.full(length, index, np.int32) for index, length in enumerate(DOCUMENT_LENGTHS)])
    unscaled_config = dataclasses.replace(ORACLE_MODEL, **dict.fromkeys(LOG_SCALING_FIELDS))
    observations = {f"tokens_{index}": document for index, document in enumerate(documents)}
    with jax.set_mesh(_mesh()):
        for label, config in (("on", ORACLE_MODEL), ("off", unscaled_config)):
            # Log scaling adds no parameters, so the same key gives both configs the same weights.
            model = _bf16_rounded(Transformer.init(config, key=jax.random.key(29)))
            row = _next_token_logprobs(model, packed, segment_ids)
            start = 0
            for index, length in enumerate(DOCUMENT_LENGTHS):
                # Drop the boundary prediction, whose target is the next document's first token.
                observations[f"logprobs_{label}_{index}"] = row[start : start + length - 1]
                start += length
        save_file(_hf_tensors(model, ORACLE_MODEL.num_experts), output / "model.safetensors")
    np.savez(output / "observations.npz", **observations)
    (output / "config.json").write_text(json.dumps(bf16_hf_config(ORACLE_MODEL), indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    write_oracle(parser.parse_args().output)


if __name__ == "__main__":
    main()
