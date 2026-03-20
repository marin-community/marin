#!/usr/bin/env python3
# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the Grug Muon orthogonalization path on real Grug MoE Muon leaves."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

import equinox as eqx
import jax
import jmp
import numpy as np
from jax import random
from jax.sharding import AxisType, Mesh

from experiments.grug.moe.model import GrugModelConfig, Transformer
from levanter.optim.grugmuon import (
    GrugMuonConfig,
    ORTHOGONALIZATION_LAYOUTS,
    STACK_BATCH_SHARDED,
    _grug_scale_with_muon,
)


DEFAULT_VOCAB_SIZE = 1024
DEFAULT_HIDDEN_DIM = 4096
DEFAULT_NUM_LAYERS = 1
DEFAULT_NUM_HEADS = 32
DEFAULT_NUM_KV_HEADS = 8
DEFAULT_HEAD_DIM = 128
DEFAULT_NUM_EXPERTS = 64
DEFAULT_EXPERT_PARALLELISM = 4
DEFAULT_NUM_EXPERTS_PER_TOKEN = 4
DEFAULT_ROUTED_EXPERT_WIDTH = 1024
DEFAULT_SHARED_EXPERT_WIDTH = 1024


@dataclass(frozen=True)
class BenchmarkResult:
    orthogonalization: str
    num_devices: int
    process_count: int
    mesh_shape: dict[str, int]
    muon_leaf_count: int
    muon_parameter_count: int
    compile_s: float
    mean_step_s: float
    min_step_s: float
    max_step_s: float
    steps: int
    warmup_steps: int


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--orthogonalization",
        choices=ORTHOGONALIZATION_LAYOUTS,
        default=STACK_BATCH_SHARDED,
        help="Muon orthogonalization layout to benchmark.",
    )
    parser.add_argument("--steps", type=int, default=10, help="Measured optimizer steps.")
    parser.add_argument("--warmup-steps", type=int, default=2, help="Warmup steps before timing.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--num-layers", type=int, default=DEFAULT_NUM_LAYERS)
    parser.add_argument("--num-heads", type=int, default=DEFAULT_NUM_HEADS)
    parser.add_argument("--num-kv-heads", type=int, default=DEFAULT_NUM_KV_HEADS)
    parser.add_argument("--head-dim", type=int, default=DEFAULT_HEAD_DIM)
    parser.add_argument("--num-experts", type=int, default=DEFAULT_NUM_EXPERTS)
    parser.add_argument("--expert-axis-size", type=int, default=DEFAULT_EXPERT_PARALLELISM)
    parser.add_argument(
        "--model-axis-size",
        type=int,
        default=None,
        help="Override the model-parallel axis size. Defaults to 8 when available, else 1.",
    )
    parser.add_argument("--num-experts-per-token", type=int, default=DEFAULT_NUM_EXPERTS_PER_TOKEN)
    parser.add_argument("--routed-expert-width", type=int, default=DEFAULT_ROUTED_EXPERT_WIDTH)
    parser.add_argument("--shared-expert-width", type=int, default=DEFAULT_SHARED_EXPERT_WIDTH)
    return parser


def _mesh_from_devices(*, num_devices: int, expert_axis_size: int, model_axis_size: int | None) -> Mesh:
    if num_devices <= 0:
        raise ValueError(f"num_devices must be positive, got {num_devices}")
    if expert_axis_size <= 0:
        raise ValueError(f"expert_axis_size must be positive, got {expert_axis_size}")

    if model_axis_size is None:
        model_axis_size = 8 if num_devices % (expert_axis_size * 8) == 0 else 1
    elif model_axis_size <= 0:
        raise ValueError(f"model_axis_size must be positive, got {model_axis_size}")

    if num_devices % (expert_axis_size * model_axis_size) != 0:
        raise ValueError(
            f"Cannot form data/expert/model mesh from {num_devices} devices with expert={expert_axis_size} "
            f"and model={model_axis_size}"
        )

    data_axis_size = num_devices // (expert_axis_size * model_axis_size)
    devices = np.array(jax.devices()).reshape(data_axis_size, expert_axis_size, model_axis_size)
    return Mesh(
        devices,
        axis_names=("data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )


def _model_config_from_args(args: argparse.Namespace) -> GrugModelConfig:
    return GrugModelConfig(
        vocab_size=args.vocab_size,
        hidden_dim=args.hidden_dim,
        intermediate_dim=args.routed_expert_width,
        shared_expert_intermediate_dim=args.shared_expert_width,
        num_experts=args.num_experts,
        num_experts_per_token=args.num_experts_per_token,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        max_seq_len=4096,
    )


def _random_like_tuple(key: jax.Array, leaves: tuple[jax.Array, ...]) -> tuple[jax.Array, ...]:
    keys = random.split(key, len(leaves))
    out = []
    for leaf, leaf_key in zip(leaves, keys, strict=True):
        grad = random.normal(leaf_key, leaf.shape, dtype=leaf.dtype)
        grad = jax.sharding.reshard(grad, leaf.sharding)
        out.append(grad)
    return tuple(out)


def _muon_leaves(params) -> tuple[tuple[jax.Array, ...], int]:
    config = GrugMuonConfig()
    mask = config.create_mask(params)
    muon = []
    total_params = 0
    for param, label in zip(jax.tree.leaves(params), jax.tree.leaves(mask), strict=True):
        if label != "muon":
            continue
        assert eqx.is_inexact_array(param)
        muon.append(param)
        total_params += int(np.prod(param.shape))
    return tuple(muon), total_params


def _benchmark(args: argparse.Namespace) -> BenchmarkResult:
    mesh = _mesh_from_devices(
        num_devices=jax.device_count(),
        expert_axis_size=args.expert_axis_size,
        model_axis_size=args.model_axis_size,
    )
    model_config = _model_config_from_args(args)
    mp = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")
    tx = _grug_scale_with_muon(
        momentum=0.95,
        nesterov=True,
        steps=5,
        muon_eps=1e-8,
        use_kimi_scaling=False,
        coefficient_type="quintic",
        orthogonalization_layout=args.orthogonalization,
    )

    with jax.set_mesh(mesh):
        init_key, grad_key = random.split(random.PRNGKey(args.seed))
        params = mp.cast_to_param(Transformer.init(model_config, key=init_key))
        muon_params, muon_parameter_count = _muon_leaves(params)
        state = tx.init(muon_params)
        grads = _random_like_tuple(grad_key, muon_params)

        @eqx.filter_jit
        def muon_step(grad_leaves, muon_state, param_leaves):
            return tx.update(grad_leaves, muon_state, param_leaves)

        start = time.perf_counter()
        updates, state = muon_step(grads, state, muon_params)
        jax.block_until_ready((updates, state))
        compile_s = time.perf_counter() - start

        for _ in range(args.warmup_steps):
            updates, state = muon_step(grads, state, muon_params)
            jax.block_until_ready((updates, state))

        step_times = []
        for _ in range(args.steps):
            start = time.perf_counter()
            updates, state = muon_step(grads, state, muon_params)
            jax.block_until_ready((updates, state))
            step_times.append(time.perf_counter() - start)

    return BenchmarkResult(
        orthogonalization=args.orthogonalization,
        num_devices=jax.device_count(),
        process_count=jax.process_count(),
        mesh_shape={name: int(size) for name, size in mesh.shape.items()},
        muon_leaf_count=len(muon_params),
        muon_parameter_count=muon_parameter_count,
        compile_s=compile_s,
        mean_step_s=float(np.mean(step_times)),
        min_step_s=float(np.min(step_times)),
        max_step_s=float(np.max(step_times)),
        steps=args.steps,
        warmup_steps=args.warmup_steps,
    )


def main():
    args = _build_parser().parse_args()
    result = _benchmark(args)
    print(json.dumps(asdict(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
