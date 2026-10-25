# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tiny, synthetic CPU-only torch/Snowball comparison; never loads real weights."""

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import torch
from jax.sharding import AxisType, Mesh
from levanter.compute_dtype import cast_model_for_compute
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel
from safetensors.torch import load_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from experiments.interp.snowball_identity.harness import import_grug
from experiments.interp.snowball_identity.multihost import gather_global

PAD_ID = 128001
VOCAB_SIZE = 128008
REAL_LENGTH = 8


def create_fixture(path: Path) -> None:
    """Write one synthetic mixed-precision checkpoint through the torch HF writer."""
    torch.set_num_threads(1)
    torch.manual_seed(310)
    grug = import_grug()
    config = grug.GrugMoeConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=16,
        intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_local_experts=8,
        num_experts_per_tok=4,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        sliding_window=4,
        max_position_embeddings=32,
        qk_mult=1.75,
    )
    config._attn_implementation = "eager"
    model = grug.GrugMoeForCausalLM(config).eval().requires_grad_(False)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, 0.2)
    model.to(torch.bfloat16)
    for block in model.model.layers:
        block.mlp.router.bias.data = torch.linspace(-0.10317, 0.11739, 8, dtype=torch.float32)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"<unk>": 0}, unk_token="<unk>")))
    tokenizer.save_pretrained(path)


def torch_reference(path: Path, tokens: np.ndarray, *, compute_float32: bool = False) -> dict:
    """Run the stock eager HF reference and capture router outputs without replacing experts."""
    grug = import_grug()
    config = grug.GrugMoeConfig(**json.loads((path / "config.json").read_text()))
    config._attn_implementation = "eager"
    model = grug.GrugMoeForCausalLM(config).eval().requires_grad_(False).to(torch.bfloat16)
    weights = load_file(str(path / "model.safetensors"))
    model.load_state_dict(weights, strict=True, assign=True)
    if compute_float32:
        model.to(torch.float32)
    routes, router_logits = [], []

    def capture(module, inputs, output):
        routes.append(output.selected_experts.reshape(*tokens.shape, 4).detach().numpy())
        router_logits.append(output.router_logits.reshape(*tokens.shape, 8).detach().numpy())

    handles = [block.mlp.router.register_forward_hook(capture) for block in model.model.layers]
    with torch.no_grad():
        logits = model(input_ids=torch.from_numpy(tokens).long()).logits.float().numpy()
    for handle in handles:
        handle.remove()
    return {"logits": logits, "routes": np.stack(routes), "router_logits": np.stack(router_logits)}


def model_mesh(experts: int) -> Mesh:
    context = 4 if jax.device_count() == 32 else 1
    data = jax.device_count() // (experts * context)
    devices = sorted(jax.devices(), key=lambda device: (device.process_index, device.id))
    grid = np.asarray(devices).reshape(context, data, experts).transpose(1, 2, 0)
    return Mesh(
        grid.reshape(1, data, experts, context, 1, 1),
        ("replica_dcn", "data", "expert", "context", "replica", "model"),
        axis_types=(AxisType.Explicit,) * 6,
    )


def run_grid(path: Path, experts: int) -> dict:
    """Measure a fixed padding/capacity grid; numerical labels are diagnostic, not gate changes."""
    torch.set_num_threads(1)
    reference_config = import_grug().GrugMoeConfig(**json.loads((path / "config.json").read_text()))
    config = dataclasses.replace(
        SnowballConfig.from_hf_config(reference_config),
        attention_implementation="reference",
        moe_implementation="ring_gather_combine",
        router_dot_fp32=True,
        router_combine_fp32=False,
    )
    tokenizer = PreTrainedTokenizerFast.from_pretrained(path)
    converter = config.hf_checkpoint_converter().replaced(tokenizer=tokenizer)
    cells = []
    weights = load_file(str(path / "model.safetensors"))
    mesh = model_mesh(experts)
    for padded in (False, True):
        length = 16 if padded else REAL_LENGTH
        tokens = np.full((8, length), PAD_ID, dtype=np.int32)
        tokens[:, :REAL_LENGTH] = (np.arange(8 * REAL_LENGTH).reshape(8, REAL_LENGTH) % 97) + 1
        tokens[:, 0] = 128000
        reference = torch_reference(path, tokens)
        reference_fp32 = torch_reference(path, tokens, compute_float32=True)
        rounding_error = np.abs(reference["logits"][:, :REAL_LENGTH] - reference_fp32["logits"][:, :REAL_LENGTH])
        for capacity in dict.fromkeys((1.0, float(experts))):
            for masked in ((False, True) if padded else (False,)):
                with jax.set_mesh(mesh):
                    model = converter.load_pretrained(
                        SnowballLMHeadModel,
                        ref=str(path),
                        config=dataclasses.replace(config, moe_capacity_factor=capacity),
                        dtype=jnp.float32,
                        axis_mapping={"embed": ["data", "context"]},
                    )
                    if not cells:
                        loaded = model.to_state_dict()
                        if loaded.keys() != weights.keys():
                            raise ValueError("Converter keysets differ")
                        for name, value in loaded.items():
                            np.testing.assert_array_equal(
                                gather_global(value), weights[name].float().numpy(), err_msg=name
                            )
                    view = cast_model_for_compute(
                        model,
                        jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16"),
                        {"router_bias": "float32"},
                    )
                    axes = (hax.Axis("batch", 8), hax.Axis("position", length))
                    ids = hax.named(jnp.asarray(tokens), axes)
                    valid_array = np.broadcast_to(np.arange(length)[None, :] < REAL_LENGTH, tokens.shape)
                    valid = hax.named(jnp.asarray(valid_array if masked else np.ones(tokens.shape, dtype=bool)), axes)
                    hidden, stats = eqx.filter_jit(view.activations_with_stats)(ids, valid)
                    logits = jnp.einsum("bsd,dv->bsv", hidden.array, view.transformer.output_proj)
                    float_hidden, _ = eqx.filter_jit(model.activations_with_stats)(ids, valid)
                    float_logits = jnp.einsum("bsd,dv->bsv", float_hidden.array, model.transformer.output_proj)
                    actual = gather_global(logits).astype(np.float32)
                    actual_fp32 = gather_global(float_logits)
                    local_rounding = np.abs(actual[:, :REAL_LENGTH] - actual_fp32[:, :REAL_LENGTH])
                    actual_routes = gather_global(stats.routes)
                    drop_counts = stats.dropped
                    diff = np.abs(actual[:, :REAL_LENGTH] - reference["logits"][:, :REAL_LENGTH])
                    scale = float(np.max(np.abs(reference["logits"][:, :REAL_LENGTH])))
                    noise_bound = 2 * float(np.finfo(np.dtype("float32")).eps) + scale / 64
                    route_set = np.mean(
                        np.all(
                            np.sort(actual_routes[:, :, :REAL_LENGTH], axis=-1)
                            == np.sort(reference["routes"][:, :, :REAL_LENGTH], axis=-1),
                            axis=-1,
                        ),
                        axis=(1, 2),
                    )
                    cells.append(
                        {
                            "padded": padded,
                            "token_valid": masked,
                            "capacity_factor": capacity,
                            "experts_mesh": experts,
                            "router_combine_fp32": False,
                            "logits_float32_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
                            "fp32_logits_sha256": hashlib.sha256(actual_fp32.tobytes()).hexdigest(),
                            "routes_sha256": hashlib.sha256(actual_routes.tobytes()).hexdigest(),
                            "max_abs_logit_diff": float(diff.max()),
                            "mean_abs_logit_diff": float(diff.mean()),
                            "reference_max_abs": scale,
                            "diagnostic_noise_bound": noise_bound,
                            "within_noise": bool(diff.max() <= noise_bound),
                            "route_set_agreement_by_layer": route_set.tolist(),
                            "pad_route_agreement_by_layer": (
                                np.mean(
                                    np.all(
                                        np.sort(actual_routes[:, :, REAL_LENGTH:], axis=-1)
                                        == np.sort(reference["routes"][:, :, REAL_LENGTH:], axis=-1),
                                        axis=-1,
                                    ),
                                    axis=(1, 2),
                                ).tolist()
                                if padded
                                else None
                            ),
                            "drop_counts_by_layer": gather_global(drop_counts).tolist(),
                            "torch_bf16_vs_fp32_error": {
                                "p50": float(np.quantile(rounding_error, 0.5)),
                                "p99": float(np.quantile(rounding_error, 0.99)),
                                "max": float(rounding_error.max()),
                            },
                            "levanter_bf16_vs_fp32_error": {
                                "p50": float(np.quantile(local_rounding, 0.5)),
                                "p99": float(np.quantile(local_rounding, 0.99)),
                                "max": float(local_rounding.max()),
                            },
                            "fp32_cross_stack_max_abs": float(
                                np.max(np.abs(actual_fp32[:, :REAL_LENGTH] - reference_fp32["logits"][:, :REAL_LENGTH]))
                            ),
                        }
                    )
    return {
        "devices": jax.device_count(),
        "processes": jax.process_count(),
        "mesh": dict(mesh.shape),
        "fixture_sha256": hashlib.sha256((path / "model.safetensors").read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256((path / "config.json").read_bytes()).hexdigest(),
        "test_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "converted_master_equality": "all keys and arrays equal losslessly upcast source",
        "synthetic_seed": 310,
        "real_length": REAL_LENGTH,
        "pad_id": PAD_ID,
        "noise_rule": "max absolute logit error <= reference max absolute / 64 + 2*float32 epsilon; diagnostic only",
        "cells": cells,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--create-fixture", action="store_true")
    parser.add_argument("--experts", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--coordinator")
    args = parser.parse_args()
    if args.rank is not None:
        jax.config.update("jax_cpu_collectives_implementation", "gloo")
        jax.distributed.initialize(
            args.coordinator,
            num_processes=4,
            process_id=args.rank,
            heartbeat_timeout_seconds=600,
            initialization_timeout=600,
        )
    if args.create_fixture:
        create_fixture(args.fixture)
    report = run_grid(args.fixture, args.experts)
    if jax.process_index() == 0:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
