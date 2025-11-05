# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import jax.numpy as jnp
import pytest
from jax.random import PRNGKey

import haliax as hax
from haliax import Axis

from levanter.layers.attention import AttentionBackend, AttentionMask
from levanter.models.gpt2 import Gpt2Config, Gpt2LMHeadModel
from test_utils import check_load_config, check_model_works_with_seqlen, parameterize_with_configs


@pytest.mark.parametrize("num_blocks", [1, 4, 12])
@pytest.mark.parametrize("attn_backend", [AttentionBackend.JAX_FLASH, AttentionBackend.VANILLA])
def test_gradient_checkpointing(num_blocks, attn_backend):
    # ensure that gradient checkpointing doesn't change the output
    # (this is a regression test for a bug that caused the output to change)
    config = Gpt2Config(
        seq_len=64,
        hidden_dim=64,
        num_layers=num_blocks,
        num_heads=8,
        gradient_checkpointing=False,
        # use_flash_attention=True,
        attn_backend=attn_backend,
    )
    config_checkpoint = dataclasses.replace(config, gradient_checkpointing=True)
    key = PRNGKey(0)

    Vocab = Axis("vocab", 128)

    model = Gpt2LMHeadModel.init(Vocab, config, key=key)
    model_checkpoint = Gpt2LMHeadModel.init(Vocab, config_checkpoint, key=key)

    input_ids = hax.arange(config.Pos, dtype=jnp.int32)

    causal_mask = AttentionMask.causal()

    a1 = model(input_ids, key=key, attn_mask=causal_mask)
    a2 = model_checkpoint(input_ids, key=key, attn_mask=causal_mask)

    assert hax.all(hax.isclose(a1, a2, rtol=1e-4, atol=1e-5)), f"failed with num_blocks={num_blocks}"


@parameterize_with_configs("gpt2*.yaml")
def test_gpt2_configs(config_file):
    from levanter.main.train_lm import TrainLmConfig

    check_load_config(TrainLmConfig, config_file)


def test_pass_different_length_seq_to_gpt2():
    config = Gpt2Config(
        seq_len=64,
        hidden_dim=16,
        num_layers=4,
        num_heads=2,
        gradient_checkpointing=False,
        use_flash_attention=True,
    )
    check_model_works_with_seqlen(Gpt2LMHeadModel, config, 16)
