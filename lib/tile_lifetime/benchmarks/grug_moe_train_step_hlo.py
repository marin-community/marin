#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Export an ordinary Grug MoE train step and audit its StableHLO boundary."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jnp
import jmp
import numpy as np
import optax
from haliax.partitioning import set_mesh
from jax.sharding import AxisType, Mesh
from levanter.data.text.examples import GrugLmExample
from levanter.grug.attention import AttentionMask

from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.train import _make_train_step, initial_state

_STABLEHLO_OPERATION = re.compile(r"\bstablehlo\.([A-Za-z0-9_]+)")
_CUSTOM_CALL_TARGET = re.compile(r'call_target_name\s*=\s*"([^"]+)"')
_FORBIDDEN_SEMANTIC_TARGET_FRAGMENTS = (
    "deepep",
    "flash_attention",
    "flashattention",
    "fa4",
    "mok",
    "sonic",
)


def _mesh() -> Mesh:
    devices = np.asarray(jax.devices(), dtype=object)
    return Mesh(
        devices.reshape((1, devices.size, 1, 1)),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )


def _operation_inventory(stablehlo_text: str) -> dict[str, int]:
    return dict(sorted(Counter(_STABLEHLO_OPERATION.findall(stablehlo_text)).items()))


def _custom_call_targets(stablehlo_text: str) -> tuple[str, ...]:
    return tuple(sorted(set(_CUSTOM_CALL_TARGET.findall(stablehlo_text))))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stablehlo-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--intermediate-size", type=int, default=32)
    parser.add_argument("--shared-intermediate-size", type=int, default=32)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--query-heads", type=int, default=2)
    parser.add_argument("--key-value-heads", type=int, default=1)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--experts-per-token", type=int, default=2)
    parser.add_argument("--optimizer", choices=("sgd", "adamw"), default="sgd")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    os.environ["RAGGED_DOT_IMPL"] = "xla"
    config = GrugModelConfig(
        vocab_size=args.vocab_size,
        hidden_dim=args.hidden_size,
        intermediate_dim=args.intermediate_size,
        shared_expert_intermediate_dim=args.shared_intermediate_size,
        num_experts=args.experts,
        num_experts_per_token=args.experts_per_token,
        num_layers=args.layers,
        num_heads=args.query_heads,
        num_kv_heads=args.key_value_heads,
        max_seq_len=args.sequence_length,
        sliding_window=args.sequence_length,
        attention_implementation="reference",
        moe_implementation="scatter",
    )
    optimizer = optax.sgd(1e-3) if args.optimizer == "sgd" else optax.adamw(learning_rate=1e-4, weight_decay=0.1)
    mixed_precision = jmp.get_policy("params=float32,compute=bfloat16,output=bfloat16")
    mesh = _mesh()
    with set_mesh(mesh):
        state = initial_state(
            config,
            optimizer=optimizer,
            mp=mixed_precision,
            key=jax.random.PRNGKey(0),
            ema_beta=None,
        )
        tokens = jnp.arange(args.batch_size * args.sequence_length, dtype=jnp.int32).reshape(
            args.batch_size,
            args.sequence_length,
        )
        batch = GrugLmExample(
            tokens=tokens % args.vocab_size,
            loss_weight=jnp.ones(tokens.shape, dtype=jnp.float32),
            attn_mask=AttentionMask.causal(),
        )
        train_step = _make_train_step(
            optimizer,
            mixed_precision,
            z_loss_weight=0.0,
            ema_beta=None,
            watch_config=None,
        )
        lowered = train_step.lower(state, batch, compute_watch=False)
        stablehlo_text = str(lowered.compiler_ir(dialect="stablehlo"))

    operations = _operation_inventory(stablehlo_text)
    custom_targets = _custom_call_targets(stablehlo_text)
    forbidden_targets = tuple(
        target
        for target in custom_targets
        if any(fragment in target.lower() for fragment in _FORBIDDEN_SEMANTIC_TARGET_FRAGMENTS)
    )
    summary = {
        "kind": "ordinary_grug_moe_train_step_stablehlo",
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "configuration": {
            "batch_size": args.batch_size,
            "sequence_length": args.sequence_length,
            "vocab_size": args.vocab_size,
            "hidden_size": args.hidden_size,
            "intermediate_size": args.intermediate_size,
            "shared_intermediate_size": args.shared_intermediate_size,
            "layers": args.layers,
            "query_heads": args.query_heads,
            "key_value_heads": args.key_value_heads,
            "experts": args.experts,
            "experts_per_token": args.experts_per_token,
            "optimizer": args.optimizer,
        },
        "frontend_policy": {
            "attention": "ordinary_reference_tensor_algebra",
            "moe": "ordinary_scatter_relation_program",
            "ragged_contract": "jax.lax.ragged_dot_general",
        },
        "stablehlo_character_count": len(stablehlo_text),
        "operation_inventory": operations,
        "custom_call_targets": custom_targets,
        "forbidden_semantic_custom_call_targets": forbidden_targets,
        "clean_semantic_boundary": not forbidden_targets and operations.get("custom_call", 0) == 0,
    }
    if not summary["clean_semantic_boundary"]:
        raise RuntimeError(f"train-step StableHLO contains opaque semantic custom calls: {custom_targets}")
    args.stablehlo_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.stablehlo_output.write_text(stablehlo_text)
    args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
