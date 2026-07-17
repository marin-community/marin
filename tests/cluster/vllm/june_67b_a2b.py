# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Identity and checkpoint loading for the vendored June 67B A2B model."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import draccus
import equinox as eqx
import jax
import jax.numpy as jnp
from levanter.checkpoint import load_checkpoint as load_levanter_checkpoint
from rigging.filesystem import StoragePath

from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig as VendoredGrugModelConfig
from experiments.june_tpu_67b_a2b.moe.model import Transformer as VendoredTransformer
from tests.cluster.vllm.june_67b_a2b_identity import JUNE_67B_A2B


@dataclass(frozen=True)
class TokenLogprob:
    logprob: float
    text: str
    token_id: int


@dataclass(frozen=True)
class InferenceGolden:
    moe_implementation: str
    mp: str
    prompt: str
    prompt_token_ids: list[int]
    tokenizer: str
    top_logprobs: list[TokenLogprob]


def read_inference_golden(path: Path) -> InferenceGolden:
    return draccus.decode(InferenceGolden, json.loads(path.read_text()))


def read_executor_info() -> dict[str, Any]:
    return json.loads(StoragePath(JUNE_67B_A2B.executor_info_path).read_text())


def decode_vendored_config(executor_info: dict[str, Any]) -> VendoredGrugModelConfig:
    return draccus.decode(VendoredGrugModelConfig, executor_info["config"]["model"])


def load_checkpoint(
    config: VendoredGrugModelConfig,
    mesh: jax.sharding.Mesh,
) -> tuple[VendoredTransformer, jax.Array]:
    template = eqx.filter_eval_shape(VendoredTransformer.init, config, key=jax.random.PRNGKey(0))
    checkpoint_state = load_levanter_checkpoint(
        {
            "params": template,
            "pending_qb_betas": jax.ShapeDtypeStruct((config.num_layers, config.num_experts), jnp.float32),
        },
        JUNE_67B_A2B.checkpoint_path,
        mesh=mesh,
    )
    jax.block_until_ready(checkpoint_state)
    return checkpoint_state["params"], checkpoint_state["pending_qb_betas"]


def apply_pending_qb_betas(model: VendoredTransformer, pending_qb_betas: jax.Array) -> VendoredTransformer:
    assert model.stacked_blocks is not None
    # Mirrors train._apply_qb_betas without importing the training entrypoint.
    router_bias = -pending_qb_betas
    router_bias -= jnp.mean(router_bias, axis=-1, keepdims=True)
    return eqx.tree_at(lambda tree: tree.stacked_blocks.stacked.mlp.router_bias, model, router_bias)
