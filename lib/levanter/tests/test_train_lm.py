# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
import math
import os
import tempfile

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest
from chex import assert_trees_all_close

from haliax import Axis
from haliax.quantization import QuantizationConfig

import levanter.main.train_lm as train_lm
import tiny_test_corpus
from levanter.adaptor import LoraAdaptorConfig
from levanter.checkpoint import CheckpointerConfig
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


def _packed_supervised_lm_config(tmp_path, *, num_docs=8, seq_len=16, batch_size=2, num_train_epochs=1):
    """A TrainLmConfig over a single packed supervised component, for epoch resolution tests."""
    data, _ = tiny_test_corpus.construct_packed_supervised_config(tmp_path, num_docs=num_docs)
    model = train_lm.LlamaConfig(
        num_layers=2, num_heads=2, num_kv_heads=2, max_seq_len=seq_len, hidden_dim=32, attn_backend=None
    )
    return train_lm.TrainLmConfig(
        data=data,
        model=model,
        trainer=train_lm.TrainerConfig(
            train_batch_size=batch_size,
            num_train_steps=999999,
            require_accelerator=False,
            distributed=DistributedConfig(initialize_jax_distributed=False),
        ),
        train_seq_len=seq_len,
        num_train_epochs=num_train_epochs,
    )


def test_num_train_steps_for_epochs_uses_packed_length(tmp_path):
    # batch_size=1 makes the oracle concrete: N packed sequences per epoch -> N steps per epoch.
    config = _packed_supervised_lm_config(tmp_path, num_docs=8, seq_len=16, batch_size=1)
    Pos = config.model.max_Pos.resize(16)
    per_epoch = config.data.num_train_sequences(Pos)
    raw_doc_count = len(config.data.build_caches("train")["sup"].as_sync_dataset())

    # A raw-document count (the DPO-style mistake) would over-count here; packing collapses documents.
    assert per_epoch < raw_doc_count
    assert train_lm.num_train_steps_for_epochs(config) == per_epoch

    three_epochs = dataclasses.replace(config, num_train_epochs=3)
    assert train_lm.num_train_steps_for_epochs(three_epochs) == 3 * per_epoch


def test_num_train_steps_for_epochs_requires_epochs_set(tmp_path):
    config = _packed_supervised_lm_config(tmp_path, num_train_epochs=None)
    with pytest.raises(ValueError):
        train_lm.num_train_steps_for_epochs(config)


def test_num_train_steps_for_epochs_rejects_nonpositive_epochs(tmp_path):
    config = _packed_supervised_lm_config(tmp_path, num_train_epochs=0)
    with pytest.raises(ValueError):
        train_lm.num_train_steps_for_epochs(config)


def test_train_lm_num_train_epochs_stops_after_one_pass():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_config, _ = tiny_test_corpus.construct_small_data_cache(
            tmpdir, num_shards=1, chunk_size=8, doc_len=64, vocab_size=1024
        )
        seq_len = 64
        config = train_lm.TrainLmConfig(
            data=data_config,
            model=train_lm.LlamaConfig(
                num_layers=2, num_heads=2, num_kv_heads=2, max_seq_len=seq_len, hidden_dim=32, attn_backend=None
            ),
            trainer=train_lm.TrainerConfig(
                # Deliberately far larger than one epoch: the epoch cap must override it.
                num_train_steps=999999,
                train_batch_size=len(jax.devices()),
                max_eval_batches=1,
                tracker=JsonFileTrackerConfig(output_path=tmpdir),
                # Keep checkpoints inside the temp dir instead of a cwd-relative "checkpoints/".
                checkpointer=CheckpointerConfig(base_path=os.path.join(tmpdir, "checkpoints")),
                require_accelerator=False,
                distributed=DistributedConfig(initialize_jax_distributed=False),
            ),
            train_seq_len=seq_len,
            num_train_epochs=1,
        )
        resolved = train_lm.num_train_steps_for_epochs(config)
        train_lm.main(config)

        with open(os.path.join(tmpdir, "eval_results.json")) as f:
            metrics = json.load(f)
        # global_step is the last (0-based) completed step, so a single full pass ends at resolved - 1.
        assert resolved < 999999
        assert metrics["global_step"] == resolved - 1
