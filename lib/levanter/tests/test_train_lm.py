# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import json
import math
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
from levanter.adaptor import LoraAdaptorConfig
from levanter.data.dataset import ListAsyncDataset
from levanter.data.text.datasets import DirectDatasetComponent, LmDataConfig
from levanter.data.text.examples import GrugLmExample
from levanter.distributed import DistributedConfig
from levanter.tracker.json_file import JsonFileTrackerConfig
from levanter.trainer_state import trainables_only
from test_utils import arrays_only


def _array_leaves(tree):
    return jax.tree_util.tree_leaves(arrays_only(tree))


def _assert_training_recorded(output_path: str) -> dict:
    """Load the JsonFileTracker record and assert training produced finite metrics.

    The smoke tests run ``train_lm.main`` end to end; the only stable observable
    effect is the metrics persisted by the tracker on ``finish()``. Asserting on
    them catches silent no-ops (no step ever logged) and NaN/inf loss blowups.
    """
    with open(os.path.join(output_path, "eval_results.json")) as f:
        metrics = json.load(f)
    assert metrics["parameter_count"] > 0
    assert "train/loss" in metrics, "per-step logging hook never fired"
    assert math.isfinite(metrics["train/loss"])
    return metrics


def test_train_lm():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_config, _ = tiny_test_corpus.construct_small_data_cache(tmpdir)
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
                tracker=JsonFileTrackerConfig(output_path=tmpdir),
                require_accelerator=False,
                distributed=DistributedConfig(initialize_jax_distributed=False),
            ),
        )
        train_lm.main(config)
        _assert_training_recorded(tmpdir)


def test_train_lm_fp8():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_config, _ = tiny_test_corpus.construct_small_data_cache(tmpdir)
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
                tracker=JsonFileTrackerConfig(output_path=tmpdir),
                require_accelerator=False,
                distributed=DistributedConfig(initialize_jax_distributed=False),
            ),
        )
        train_lm.main(config)
        _assert_training_recorded(tmpdir)


def test_train_lm_with_lora_adapter():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_config, _ = tiny_test_corpus.construct_small_data_cache(tmpdir)
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
                tracker=JsonFileTrackerConfig(output_path=tmpdir),
                require_accelerator=False,
                distributed=DistributedConfig(initialize_jax_distributed=False),
            ),
            adapter=LoraAdaptorConfig(r=4),
        )
        train_lm.main(config)
        _assert_training_recorded(tmpdir)


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

    adapter = LoraAdaptorConfig(r=4, a_init_mode="random")
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
    with tempfile.TemporaryDirectory() as tmpdir:
        vocab_size = 128
        seq_len = 64
        data = []
        for i in range(8):
            tokens = jnp.full((seq_len,), i % vocab_size, dtype=jnp.int32)
            data.append(GrugLmExample.causal(tokens))
        dataset = ListAsyncDataset(data)

        component = DirectDatasetComponent(datasets={"train": dataset})
        data_config = LmDataConfig(components={"direct": component}, vocab_size=vocab_size, tokenizer="passthrough")

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
                tracker=JsonFileTrackerConfig(output_path=tmpdir),
                require_accelerator=False,
                distributed=DistributedConfig(initialize_jax_distributed=False),
            ),
        )
        train_lm.main(config)
        _assert_training_recorded(tmpdir)
