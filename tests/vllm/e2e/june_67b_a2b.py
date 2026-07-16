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
import numpy as np
from levanter.checkpoint import load_checkpoint as load_levanter_checkpoint
from rigging.filesystem import StoragePath

from experiments.june_tpu_67b_a2b.moe.model import GrugModelConfig as VendoredGrugModelConfig
from experiments.june_tpu_67b_a2b.moe.model import Transformer as VendoredTransformer


@dataclass(frozen=True)
class ModelIdentity:
    """Checkpoint, export, and golden values that form one model lineage."""

    run_root: str
    checkpoint_step: int
    export_sha256: str
    export_uri: str
    inference_golden_path: Path

    @property
    def executor_info_path(self) -> str:
        return f"{self.run_root}/.executor_info"

    @property
    def checkpoint_path(self) -> str:
        return f"{self.run_root}/checkpoints/step-{self.checkpoint_step}"

    @property
    def vllm_model_name(self) -> str:
        return f"june-67b-a2b-step-{self.checkpoint_step}-bf16"


JUNE_67B_A2B = ModelIdentity(
    run_root=(
        "s3://marin-us-east-02a/marin/grug/"
        "moe_67b_a2b_d2560_ep1_rep8_bs1024_seq65536_sw2k_v4_2048_muon_cooldown_step39k-79ebf3"
    ),
    checkpoint_step=42150,
    export_sha256="781bc3291c81ce282be6762520280ebd5ef5b85e88ba65129c2d0162d48ee632",
    export_uri="s3://marin-us-east-02a/marin/exports/grug/june-67b-a2b/step-42150/hf-bf16-vllm/781bc3291c81ce28/",
    inference_golden_path=Path(__file__).parent / "resources" / "june_tpu_67b_a2b_step_42150_logprobs.json",
)


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


# Cross-implementation parity bound, shared by the Snowball 67B gates. The golden's top logprobs sit
# on a 1/32 grid with exact ties, so a probability-error bound (the vLLM export test's bound) is the
# meaningful check; a bit-exact / exact-rank comparison is the wrong bar for a graph reimplementation.
GOLDEN_MAX_PROBABILITY_ERROR = 0.008


def score_next_token_against_golden(logprobs: np.ndarray, golden: InferenceGolden) -> tuple[np.ndarray, float]:
    """Score per-device next-token logprobs against the golden's top tokens, rank-independently.

    Args:
        logprobs: ``[batch, vocab]`` log-softmax of the next-token logits (one row per device).
        golden: the frozen inference golden.

    Returns ``(greedy_ids, max_probability_error)``: the argmax token per device, and the largest
    probability error between Snowball's logprob for each golden token and the golden's own logprob.
    Insensitive to the bf16 reordering of the golden's exact-tied tail tokens.
    """
    golden_ids = np.asarray([entry.token_id for entry in golden.top_logprobs])
    golden_logprobs = np.asarray([entry.logprob for entry in golden.top_logprobs])
    greedy_ids = logprobs.argmax(axis=-1)
    mine_at_golden = logprobs[:, golden_ids]  # [batch, len(top_logprobs)]
    max_probability_error = float(np.max(np.abs(np.exp(mine_at_golden) - np.exp(golden_logprobs)[None, :])))
    return greedy_ids, max_probability_error


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
