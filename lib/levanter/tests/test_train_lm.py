# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile

import jax
import jax.numpy as jnp
import pytest

from haliax.quantization import QuantizationConfig

import levanter.main.train_lm as train_lm
import tiny_test_corpus
from levanter.data.dataset import ListAsyncDataset
from levanter.data.mixture import MixtureDataset
from levanter.data.text import DirectDatasetComponent, GrugLmExample, LmDataConfig
from levanter.data.text.datasets import NamedLmDataset
from levanter.distributed import DistributedConfig, RayConfig
from levanter.tracker import NoopConfig


@pytest.mark.entry
def test_train_lm():
    # just testing if train_lm has a pulse
    with tempfile.TemporaryDirectory() as tmpdir:
        data_config, _ = tiny_test_corpus.construct_small_data_cache(tmpdir)
        try:
            config = train_lm.TrainLmConfig(
                data=data_config,
                model=train_lm.LlamaConfig(
                    num_layers=2,
                    num_heads=2,
                    num_kv_heads=2,
                    max_seq_len=64,
                    hidden_dim=32,
                    attn_backend=None,  # use default for platform
                ),
                trainer=train_lm.TrainerConfig(
                    num_train_steps=2,
                    train_batch_size=len(jax.devices()),
                    max_eval_batches=1,
                    tracker=NoopConfig(),
                    require_accelerator=False,
                    distributed=DistributedConfig(initialize_jax_distributed=False),
                    ray=RayConfig(auto_start_cluster=False),
                ),
            )
            train_lm.main(config)
        finally:
            try:
                os.unlink("wandb")
            except Exception:
                pass


@pytest.mark.entry
def test_train_lm_fp8():
    # just testing if train_lm has a pulse
    with tempfile.TemporaryDirectory() as tmpdir:
        data_config, _ = tiny_test_corpus.construct_small_data_cache(tmpdir)
        try:
            config = train_lm.TrainLmConfig(
                data=data_config,
                model=train_lm.LlamaConfig(
                    num_layers=2,
                    num_heads=2,
                    num_kv_heads=2,
                    max_seq_len=64,
                    hidden_dim=32,
                    attn_backend=None,  # use default for platform
                ),
                trainer=train_lm.TrainerConfig(
                    quantization=QuantizationConfig(fp8=True),
                    num_train_steps=2,
                    train_batch_size=len(jax.devices()),
                    max_eval_batches=1,
                    tracker=NoopConfig(),
                    require_accelerator=False,
                    distributed=DistributedConfig(initialize_jax_distributed=False),
                    ray=RayConfig(auto_start_cluster=False),
                ),
            )
            train_lm.main(config)
        finally:
            try:
                os.unlink("wandb")
            except Exception:
                pass


@pytest.mark.entry
def test_train_lm_direct_dataset():
    with tempfile.TemporaryDirectory():
        try:
            vocab_size = 128
            seq_len = 64
            data = []
            for i in range(8):
                tokens = jnp.full((seq_len,), i % vocab_size, dtype=jnp.int32)
                data.append(GrugLmExample.causal(tokens))
            dataset = ListAsyncDataset(data)

            component = DirectDatasetComponent(datasets={"train": dataset})
            data_config = LmDataConfig(
                components={"direct": component}, vocab_size=vocab_size, tokenizer="passthrough"
            )

            config = train_lm.TrainLmConfig(
                data=data_config,
                model=train_lm.LlamaConfig(
                    num_layers=2,
                    num_heads=2,
                    num_kv_heads=2,
                    max_seq_len=seq_len,
                    hidden_dim=32,
                    attn_backend=None,
                ),
                trainer=train_lm.TrainerConfig(
                    num_train_steps=2,
                    train_batch_size=len(jax.devices()),
                    max_eval_batches=1,
                    tracker=NoopConfig(),
                    require_accelerator=False,
                    distributed=DistributedConfig(initialize_jax_distributed=False),
                    ray=RayConfig(auto_start_cluster=False),
                ),
            )
            train_lm.main(config)
        finally:
            try:
                os.unlink("wandb")
            except Exception:
                pass


def test_find_nested_mixture_dataset_recovers_named_wrapper():
    seq_len = 8
    tokens = jnp.arange(seq_len, dtype=jnp.int32)
    example = GrugLmExample.causal(tokens)
    child = ListAsyncDataset([example, example])
    mixture = MixtureDataset(
        datasets={"a": child, "b": child},
        weights={"a": 0.75, "b": 0.25},
        block_size=8,
        key=0,
    )
    wrapped = NamedLmDataset(mixture, train_lm.Axis("position", seq_len))

    recovered = train_lm._find_nested_mixture_dataset(wrapped)

    assert recovered is mixture
