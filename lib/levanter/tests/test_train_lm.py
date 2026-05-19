# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from chex import assert_trees_all_close

from haliax import Axis
from haliax.quantization import QuantizationConfig

import levanter.main.train_lm as train_lm
import tiny_test_corpus
from levanter.adaptation import LoraAdaptationConfig
from levanter.data.dataset import ListAsyncDataset
from levanter.data.text import DirectDatasetComponent, GrugLmExample, LmDataConfig
from levanter.distributed import DistributedConfig
from levanter.trainer_state import trainables_only
from levanter.tracker import NoopConfig
from test_utils import arrays_only


def _array_leaves(tree):
    return jax.tree_util.tree_leaves(arrays_only(tree))


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
                ),
            )
            train_lm.main(config)
        finally:
            try:
                os.unlink("wandb")
            except Exception:
                pass


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
                ),
            )
            train_lm.main(config)
        finally:
            try:
                os.unlink("wandb")
            except Exception:
                pass


def test_train_lm_with_lora_adapter():
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
                    attn_backend=None,
                ),
                trainer=train_lm.TrainerConfig(
                    num_train_steps=2,
                    train_batch_size=len(jax.devices()),
                    max_eval_batches=1,
                    tracker=NoopConfig(),
                    require_accelerator=False,
                    distributed=DistributedConfig(initialize_jax_distributed=False),
                ),
                adapter=LoraAdaptationConfig(r=4),
            )
            train_lm.main(config)
        finally:
            try:
                os.unlink("wandb")
            except Exception:
                pass


def test_restore_lm_model_from_partial_checkpoint_recovers_base_model():
    config = train_lm.LlamaConfig(
        num_layers=1,
        num_heads=2,
        num_kv_heads=2,
        max_seq_len=16,
        hidden_dim=16,
        attn_backend=None,
    )
    Vocab = Axis("vocab", 32)
    base_key, wrong_base_key, adapter_key, wrong_adapter_key = jrandom.split(jrandom.PRNGKey(0), 4)

    adapter = LoraAdaptationConfig(r=4)
    trained_model = adapter.apply(config.build(Vocab, key=base_key), key=adapter_key)
    wrong_resume_skeleton = adapter.apply(config.build(Vocab, key=wrong_base_key), key=wrong_adapter_key)
    correct_source_skeleton = adapter.apply(config.build(Vocab, key=base_key), key=wrong_adapter_key)
    trainable_filter = adapter.trainable_filter(trained_model)

    checkpointed_trainables = trainables_only(trained_model, trainable_filter)
    wrong_resumed_model = eqx.combine(checkpointed_trainables, wrong_resume_skeleton)
    restored_model = train_lm._restore_lm_model_from_partial_checkpoint(
        wrong_resumed_model,
        correct_source_skeleton,
        trainable_filter,
    )

    assert_trees_all_close(_array_leaves(restored_model), _array_leaves(trained_model))


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
                ),
            )
            train_lm.main(config)
        finally:
            try:
                os.unlink("wandb")
            except Exception:
                pass
