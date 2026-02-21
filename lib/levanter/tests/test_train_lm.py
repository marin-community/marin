# Copyright 2025 The Levanter Authors
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
from levanter.data.text import DirectDatasetComponent, GrugLmExample, LmDataConfig
from levanter.distributed import DistributedConfig, RayConfig
from levanter.tracker import NoopConfig


def test_sum_tensorstore_metric_filters_by_name(monkeypatch):
    def _fake_collect(pattern: str, include_zero_metrics: bool = False):
        assert include_zero_metrics
        assert pattern == "/tensorstore/cache/hit_count"
        return [
            {"name": "/tensorstore/cache/hit_count", "values": [{"value": 3}, {"value": 5}]},
            {"name": "/tensorstore/cache/miss_count", "values": [{"value": 99}]},
            {"name": "/tensorstore/cache/hit_count", "values": [{"value": 2}]},
        ]

    monkeypatch.setattr(train_lm.ts, "experimental_collect_matching_metrics", _fake_collect)
    assert train_lm._sum_tensorstore_metric("/tensorstore/cache/hit_count", "value") == 10.0


def test_collect_kvstore_drivers_from_spec_json():
    spec_json = {
        "driver": "zarr3",
        "kvstore": {"driver": "gcs", "bucket": "demo"},
        "codecs": [
            {"name": "something"},
            {"nested": {"kvstore": {"driver": "gcs_grpc", "bucket": "demo"}}},
        ],
    }
    assert train_lm._collect_kvstore_drivers_from_spec_json(spec_json) == {"gcs", "gcs_grpc"}


def test_tensorstore_metric_specs_for_drivers():
    gcs_specs = train_lm._tensorstore_metric_specs_for_drivers({"gcs"})
    assert "gcs_read_count" in gcs_specs
    assert "gcs_grpc_read_count" not in gcs_specs

    grpc_specs = train_lm._tensorstore_metric_specs_for_drivers({"gcs_grpc"})
    assert "gcs_read_count" not in grpc_specs
    assert "gcs_grpc_read_count" in grpc_specs


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
