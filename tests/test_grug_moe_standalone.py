# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from haliax.partitioning import set_mesh
from levanter.grug.sharding import compact_grug_mesh
from levanter.utils.flop_utils import lm_flops_per_token

from experiments.grug.moe.standalone import grug_moe_mfu as benchmark


def test_model_config_from_args_resolves_7201_reference_shape():
    args = benchmark._parse(
        [
            "--run-id",
            "reference",
            "--output-dir",
            "/tmp/reference",
            "--hidden-dim",
            "6144",
            "--num-layers",
            "48",
            "--num-experts",
            "128",
            "--num-experts-per-token",
            "4",
            "--num-heads",
            "48",
            "--num-kv-heads",
            "8",
            "--head-dim",
            "128",
            "--expert-intermediate-dim",
            "3072",
            "--shared-expert-intermediate-dim",
            "6144",
            "--sliding-window",
            "512",
            "--global-every",
            "6",
            "--num-gpus",
            "64",
            "--replica-axis-size",
            "2",
        ]
    )

    config = benchmark.model_config_from_args(args)

    assert (config.hidden_dim, config.num_layers) == (6144, 48)
    assert (config.num_heads, config.num_kv_heads, config.head_dim) == (48, 8, 128)
    assert (config.intermediate_dim, config.shared_expert_intermediate_dim) == (3072, 6144)
    assert (config.sliding_window, config.global_every) == (512, 6)
    assert (args.num_gpus, args.replica_axis_size) == (64, 2)


def test_long_layer_schedule_uses_configured_frequency_and_last_layer():
    actual = benchmark.long_layer_schedule(13, global_every=6)

    assert actual.tolist() == [
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
    ]


def test_latent_norm_requires_latent_dimension():
    with pytest.raises(ValueError):
        benchmark.GrugModelConfig(vocab_size=128, moe_latent_norm=True)


def test_latent_dimension_must_compress_hidden_dimension_evenly():
    with pytest.raises(ValueError):
        benchmark.GrugModelConfig(vocab_size=128, hidden_dim=32, moe_latent_dim=20)


def test_matched_work_latent_moe_returns_full_width():
    config = benchmark.GrugModelConfig(
        vocab_size=128,
        hidden_dim=32,
        intermediate_dim=32,
        shared_expert_intermediate_dim=0,
        num_experts=4,
        num_experts_per_token=2,
        moe_latent_dim=16,
        moe_latent_norm=True,
        moe_implementation="scatter",
    )
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)

    with set_mesh(mesh):
        module = benchmark.MoEMLP.init(config, key=jax.random.PRNGKey(0))
        output, _ = module(jnp.ones((1, 4, 32), dtype=jnp.float32))

    assert output.shape == (1, 4, 32)
    assert np.isfinite(np.asarray(output)).all()


def test_compute_flops_preserves_baseline_formula():
    config = benchmark.GrugModelConfig(vocab_size=128, hidden_dim=32, intermediate_dim=16)

    flops_per_example, summary = benchmark._compute_flops(model_config=config)

    expected_per_token = lm_flops_per_token(
        hidden_dim=config.hidden_dim,
        intermediate_dim=config.intermediate_dim,
        shared_intermediate_dim=config.shared_expert_intermediate_dim,
        num_layers=config.num_layers,
        num_kv_heads=config.num_kv_heads,
        num_heads=config.num_heads,
        seq_len=config.max_seq_len,
        vocab_size=config.vocab_size,
        glu=True,
        num_experts=config.num_experts,
        num_shared_experts=1,
        num_experts_per_tok=config.num_experts_per_token,
    )
    assert flops_per_example == 3 * config.max_seq_len * expected_per_token
    assert summary["throughput/flops_per_token_analytic"] == expected_per_token


def test_compute_flops_adds_only_latent_projections_to_matched_expert_work():
    baseline = benchmark.GrugModelConfig(
        vocab_size=128,
        hidden_dim=32,
        intermediate_dim=16,
        shared_expert_intermediate_dim=0,
        num_layers=2,
        num_experts=4,
        num_experts_per_token=2,
        max_seq_len=8,
    )
    latent = dataclasses.replace(
        baseline,
        intermediate_dim=32,
        moe_latent_dim=16,
        moe_latent_norm=True,
    )

    baseline_per_example, _ = benchmark._compute_flops(model_config=baseline)
    latent_per_example, _ = benchmark._compute_flops(model_config=latent)

    projection_flops_per_token = baseline.num_layers * 4 * baseline.hidden_dim * latent.moe_latent_dim
    assert latent_per_example - baseline_per_example == 3 * baseline.max_seq_len * projection_flops_per_token


def test_model_config_summary_records_latent_and_attention_settings():
    config = benchmark.GrugModelConfig(
        vocab_size=128,
        hidden_dim=32,
        intermediate_dim=32,
        shared_expert_intermediate_dim=32,
        num_heads=4,
        num_kv_heads=1,
        sliding_window=8,
        global_every=6,
        moe_latent_dim=16,
        moe_latent_norm=True,
        moe_implementation="ring_cute",
    )

    summary = benchmark.model_config_summary(config)

    assert summary["num_kv_heads"] == 1
    assert summary["sliding_window"] == 8
    assert summary["global_every"] == 6
    assert summary["moe_latent_dim"] == 16
    assert summary["moe_latent_norm"] is True
    assert summary["moe_implementation"] == "ring_cute"
    json.dumps(summary)
