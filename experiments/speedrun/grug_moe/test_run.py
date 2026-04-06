# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
~1B active / ~2.5B total MoE Transformer.
D=1536, 12 layers, 12 heads, 8 experts (K=2), I=4608.
Replicated weights with pure data parallelism on v4-32.
Muon used for training.
"""

# nodryrun
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

from jax.experimental.shard_map import shard_map
import sys
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as random
import jax.scipy as jsp

from einops import rearrange
from fray.cluster import ResourceConfig
from jax.sharding import PartitionSpec as P
from jaxtyping import Array, Float, Int, PRNGKeyArray

from experiments.llama import llama3_tokenizer_vocab_size
from experiments.simple_train_config import SimpleTrainConfig
from haliax.jax_utils import named_call
from levanter.grug.attention import AttentionMask, attention
from levanter.grug.loss import fused_linear_softmax_cross_entropy_loss
from levanter.kernels.pallas.fused_cross_entropy_loss.config import BlockSizes
from levanter.grug.sharding import Pbatch
import levanter.tracker
from haliax.partitioning import _get_mesh
from levanter.optim import GrugMuonConfig
from marin.execution.executor import executor_main
from marin.speedrun.speedrun import Author
from levanter.models.lm_model import LmConfig, LmExample, LmHeadModel
from levanter.models.grug_wrapper import GrugWrapper
from marin.speedrun.speedrun import SpeedrunConfig, SpeedrunResultsConfig, speedrun_results
from .helpers import nemotron_cc_mixture, nemotron_only_speedrun

import argparse
import logging
from dataclasses import dataclass, replace
from typing import Iterator

import jax
import numpy as np
import optax
from jax import numpy as jnp
from jax.tree_util import register_dataclass
from jax.sharding import AxisType

from levanter.store.cache import TreeCache

from levanter.grug.data import DEFAULT_AXIS_MAPPING, build_token_loader
from levanter.grug.model_moe import GrugModelConfig, Transformer

from typing import Any, Protocol, cast

class GrugConfigLike(Protocol):
    vocab_size: int
    max_seq_len: int
    hidden_dim: int

@LmConfig.register_subclass("grugformer_h2h_125m")
@dataclass(frozen=True)
class MyLmConfig(LmConfig[GrugWrapper]):
    """LmConfig wrapper around grug core hyperparameters (for head-to-head comparisons)."""

    max_seq_len: int = 2048

    hidden_dim: int = 512
    intermediate_dim: int = 1792
    num_layers: int = 6
    num_heads: int = 8
    num_kv_heads: int = 8
    head_dim: int | None = None
    _total_trainable_params: int = None
    _flops_per_token: float = None
    grug_config: GrugConfigLike = None

    @property
    def model_type(self) -> type[GrugWrapper]:
        return GrugWrapper

    @property
    def Embed(self) -> Axis:
        # Not used by GrugWrapper (it returns logits directly), but LmConfig requires it.
        return Axis("embed", self.hidden_dim)

    def build(self, Vocab: Axis, *, key: PRNGKeyArray) -> GrugWrapper:
        return GrugWrapper(
            params=Transformer.init(self.grug_config, key=key),
            grug_config=self.grug_config
        )

    def flops_per_token(self, vocab_size: int, context_length: int) -> float | None:
        return self._flops_per_token

    def total_trainable_params(self, vocab_size: int) -> int:
        return self._total_trainable_params

def build_train_config(model_cfg: ModelConfig) -> SimpleTrainConfig:
    batch_size = 512
    num_train_steps = 100

    muon = GrugMuonConfig(
        learning_rate=0.01,
        adam_lr=0.0064,
        weight_decay=0,
        min_lr_ratio=0.1,
        warmup=0,
        momentum=0.95,
        beta1=0.8,
        beta2=0.95,
        epsilon=1e-15,
        muon_epsilon=1e-5,
        max_grad_norm=1,
        lr_schedule="linear",
        decay=0.5,
    )

    train_cfg = SimpleTrainConfig(
        resources=ResourceConfig.with_tpu("v5p-16"),
        train_batch_size=batch_size,
        learning_rate=muon.learning_rate,
        explicit_mesh_axes=True,
        #profiler=False,
        train_seq_len=model_cfg.max_seq_len,
        num_train_steps=num_train_steps,
        steps_per_hf_export=-1,
        #optimizer_config=muon,
    )
    return train_cfg





# -----------------------------------------------------------------------------
# Main




def main() -> None:
    # model_cfg = GrugModelConfig(
    #     vocab_size=llama3_tokenizer_vocab_size,
    #     hidden_dim= 512,
    #     intermediate_dim=512*3,
    #     num_layers= 6,
    #     num_heads= 4,
    #     num_kv_heads= 4,
    #     head_dim= None,
    #     max_seq_len= 1024,
    # )
    model_cfg = GrugModelConfig(
        vocab_size=llama3_tokenizer_vocab_size,
        hidden_dim= 1536,
        intermediate_dim=4608,
        num_layers= 12,
        num_heads= 12,
        num_kv_heads= 12,
        head_dim= None,
        max_seq_len= 2048,
    )

    lm_model_cfg = MyLmConfig(
        _total_trainable_params=model_cfg.total_trainable_params,
        _flops_per_token=model_cfg.flops_per_token,
        grug_config=model_cfg,
        max_seq_len=model_cfg.max_seq_len,
        hidden_dim=model_cfg.hidden_dim,
        num_heads=model_cfg.num_heads,
        num_kv_heads=model_cfg.num_kv_heads,
        intermediate_dim=model_cfg.intermediate_dim,
        num_layers=model_cfg.num_layers,
    )

    train_cfg = build_train_config(model_cfg)

    speedrun_cfg = SpeedrunConfig(
        author=Author(
            name="Larry Dial",
            affiliation="OpenAthena",
            url="https://github.com/ClassicLarry",
        ),
        description="desc",
        model_config=lm_model_cfg,
        train_config=train_cfg,
        tokenized_dataset=nemotron_cc_mixture,
    )
    speedrun = nemotron_only_speedrun("test_run_feb25_5", speedrun_cfg)
    executor_main(steps=speedrun, description="Single Nano Run")


if __name__ == "__main__":
    main()

