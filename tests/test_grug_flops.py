# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Analytic FLOP accounting for grug variants.

The reported MFU is `flops_per_example / step_time / device_peak`, so an error in the
analytic numerator shows up as a wrong MFU rather than as a failure. These tests tie the
numerator to the model that is actually executed: the moe forward pass is run to confirm it
windows the layers the accounting prices as windowed, and the FLOPs/token for each launched
config is pinned so an architecture change cannot move a published number silently.
"""

import dataclasses
import importlib

import jax
import jax.numpy as jnp
import pytest
from levanter.grug.attention import AttentionMask
from levanter.grug.sharding import compact_grug_mesh
from levanter.utils.flop_utils import lm_flops_per_token

from experiments.grug.moe.launch import GRUG_MOE_TRIAL_MODEL
from experiments.grug.moe.model import _LONG_LAYER_EVERY, GrugModelConfig, Transformer, long_layer_flags
from experiments.grug.moe.train import _compute_flops as moe_compute_flops


def test_moe_long_layers_are_strided_with_the_last_always_long():
    """Long layers land on a regular stride, and the final layer is long wherever the stride
    falls. Both come from ``_LONG_LAYER_EVERY``, so retuning the stride does not need an edit
    here -- only the pinned FLOPs/token below, which is where such a change should show up.
    """
    for num_layers in range(1, 33):
        expected = sorted({*range(_LONG_LAYER_EVERY - 1, num_layers, _LONG_LAYER_EVERY), num_layers - 1})
        assert [i for i, is_long in enumerate(long_layer_flags(num_layers)) if is_long] == expected


def test_moe_forward_windows_only_the_short_layers():
    """The schedule has to reach the forward pass, not just the accounting.

    A one-layer model is all-long, so narrowing the window cannot reach any attention call
    and the output is bit-identical. A two-layer model has one windowed layer, so it must
    change. This fails if ``Transformer.__call__`` ignores or inverts the flags.
    """
    assert long_layer_flags(1) == (True,)
    assert long_layer_flags(2) == (False, True)

    seq_len = 32
    tokens = jnp.arange(seq_len, dtype=jnp.int32)[None, :]
    mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)

    def forward(num_layers: int, sliding_window: int):
        cfg = dataclasses.replace(
            GrugModelConfig(vocab_size=64),
            num_layers=num_layers,
            max_seq_len=seq_len,
            sliding_window=sliding_window,
            num_heads=4,
            num_kv_heads=2,
            hidden_dim=64,
            intermediate_dim=32,
            shared_expert_intermediate_dim=32,
            num_experts=4,
            num_experts_per_token=2,
        )
        with jax.set_mesh(mesh):
            hidden, _ = Transformer.init(cfg, key=jax.random.PRNGKey(0))(tokens)
        return hidden

    assert jnp.array_equal(forward(1, seq_len), forward(1, 4))
    assert not jnp.array_equal(forward(2, seq_len), forward(2, 4))


@pytest.mark.parametrize("window", [1, 7, 64, 128])
def test_sliding_window_mask_admits_exactly_the_window(window):
    """The accounting prices a short layer at ``sliding_window`` keys per query, so the mask
    the forward builds from the same value must admit exactly that many."""
    seq_len = 128
    mask = AttentionMask(is_causal=True, sliding_window=window)
    admitted = int(mask.materialize_mask(seq_len, seq_len).sum(axis=-1).max())
    assert admitted == min(window, seq_len)


_MOE_KWARGS = dict(
    hidden_dim=2048,
    intermediate_dim=1408,
    num_layers=24,
    num_kv_heads=4,
    num_heads=16,
    seq_len=4096,
    vocab_size=128256,
    glu=True,
    num_experts=64,
    num_shared_experts=1,
    num_experts_per_tok=6,
    shared_intermediate_dim=1408,
)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            dict(
                hidden_dim=4096,
                intermediate_dim=11008,
                num_layers=32,
                num_kv_heads=32,
                num_heads=32,
                seq_len=4096,
                vocab_size=32000,
                glu=True,
            ),
            15374221312.0,
        ),
        (_MOE_KWARGS, 4751622144.0),
        (dict(_MOE_KWARGS, sliding_window=1024, num_long_layers=6, local_kv_heads=8, global_kv_heads=4), 4371480576.0),
    ],
    ids=["dense", "moe", "interleaved"],
)
def test_lm_flops_per_token_is_unchanged_for_existing_callers(kwargs, expected):
    """The other callers price uniform-attention models through this signature. The values are
    the megatron-lm estimate for each config."""
    assert lm_flops_per_token(**kwargs) == pytest.approx(expected, rel=1e-12)


def test_moe_launched_config_flops_per_token():
    """Golden value for the config the launcher builds; a change here moves every reported MFU."""
    _, summary = moe_compute_flops(model_config=GRUG_MOE_TRIAL_MODEL)
    assert summary["throughput/flops_per_token_analytic"] == pytest.approx(0.702480e9, rel=1e-6)


@pytest.mark.parametrize(
    ("variant", "expected_gflops_per_token"),
    [("moe_hero_ep", 32.911196), ("moe_hero_fsdp", 44.491407)],
)
def test_hero_launched_config_flops_per_token(variant, expected_gflops_per_token):
    heuristic = importlib.import_module(f"experiments.grug.{variant}.heuristic")
    train = importlib.import_module(f"experiments.grug.{variant}.train")

    model_config, _ = heuristic.build_hero_configs(num_train_steps=1000, batch_size=1024)
    _, summary = train._compute_flops(model_config=model_config)
    assert summary["throughput/flops_per_token_analytic"] == pytest.approx(expected_gflops_per_token * 1e9, rel=1e-6)
