# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The subprocess source stream reconstructs the exact native compute policy."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from haliax import Axis
from levanter.grug.sharding import compact_grug_mesh
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel, snowball_to_state_dict
from marin.rl.levanter_publication import MAX_PUBLICATION_BYTES, SnowballPublication


def _compute_model(mesh):
    count = jax.device_count()
    config = SnowballConfig(
        vocab_size=16 * count,
        hidden_dim=8 * count,
        intermediate_dim=8,
        shared_expert_intermediate_dim=8,
        num_experts=4 * count,
        num_experts_per_token=2,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=4,
        max_seq_len=8,
        sliding_window=4,
        attention_implementation="reference",
    )
    with jax.set_mesh(mesh):
        model = SnowballLMHeadModel.init(Axis("vocab", config.vocab_size), config, key=jax.random.key(12))
        model = jax.tree.map(lambda value: value.astype(jnp.bfloat16) if eqx.is_array(value) else value, model)
        # Router bias must keep genuine FP32 values instead of rounding through BF16.
        model = eqx.tree_at(
            lambda item: tuple(block.mlp.router_bias for block in item.transformer.blocks),
            model,
            tuple(jnp.full((config.num_experts,), 0.100123, dtype=jnp.float32) for _ in model.transformer.blocks),
        )
    return model


def _read(publication, model, version):
    assert publication.begin(model, version) == {"version": version}
    payloads = []
    while (payload := publication.next()) is not None:
        payloads.append(payload)
    assert publication.end() == {"version": version, "complete": True}
    return payloads


@pytest.mark.multidevice
def test_snowball_publication_reconstructs_experts_dense_and_router_precision():
    mesh = compact_grug_mesh(expert_axis_size=1)
    model = _compute_model(mesh)
    publication = SnowballPublication(model, mesh)
    metadata = publication.inventory(3)
    # A long-lived Trainer leaves this global mesh active across child IPC
    # commands. Extracting a local expert must not inherit the global devices.
    with jax.set_mesh(mesh):
        payloads = _read(publication, model, 3)
    assert set(metadata["devices"]) == set(range(jax.device_count()))
    assert metadata["payload_order"] == tuple((payload["root"], payload["key"]) for payload in payloads)
    assert len({payload["key"] for payload in payloads}) == len(payloads)
    assert all(payload["version"] == 3 and len(payload["data"]) <= MAX_PUBLICATION_BYTES for payload in payloads)

    expected = snowball_to_state_dict(model.transformer)
    landed = {}
    experts = {}
    for payload in payloads:
        dtype = jnp.bfloat16 if payload["dtype"] == "bfloat16" else np.float32
        value = np.frombuffer(payload["data"], dtype=dtype).reshape(payload["shape"])
        if payload["key"].startswith("expert:"):
            name = payload["key"].removeprefix("expert:")
            prefix, projection, expert = name.rsplit(".", 2)
            experts[(int(prefix.split(".")[2]), projection, int(expert.removeprefix("expert")))] = value
        else:
            _, name, offset, numel, _ = payload["key"].split(":")
            target = landed.setdefault(name, np.empty(expected[name].shape, dtype=value.dtype))
            target.reshape(-1)[int(offset) : int(offset) + int(numel)] = value
    for name, actual in landed.items():
        # Receiver FP32 router storage widens BF16 compute weights, while an FP32
        # bias override must survive without an intermediate BF16 conversion.
        expected_dtype = np.float32 if name.endswith((".mlp.router.weight", ".mlp.router.bias")) else jnp.bfloat16
        assert actual.dtype == expected_dtype
        np.testing.assert_array_equal(actual, np.asarray(expected[name]).astype(expected_dtype))
    assert set(landed) == {name for name in expected if ".mlp.experts." not in name}
    for layer, block in enumerate(model.transformer.blocks):
        for expert in range(model.config.num_experts):
            gate = np.asarray(block.mlp.expert_mlp.w_gate)[expert].T
            up = np.asarray(block.mlp.expert_mlp.w_up)[expert].T
            down = np.asarray(block.mlp.expert_mlp.w_down)[expert].T
            np.testing.assert_array_equal(experts[(layer, "fc1", expert)], np.concatenate((gate, up), axis=0))
            np.testing.assert_array_equal(experts[(layer, "fc2", expert)], down)


def test_snowball_publication_replay_and_next_version_use_the_frozen_compute_view():
    mesh = compact_grug_mesh(expert_axis_size=1)
    original = _compute_model(mesh)
    publication = SnowballPublication(original, mesh)
    first = _read(publication, original, 0)
    replay = _read(publication, original, 0)
    assert first == replay

    with jax.set_mesh(mesh):
        updated = eqx.tree_at(
            lambda model: model.transformer.blocks[0].mlp.expert_mlp.w_gate,
            original,
            original.transformer.blocks[0].mlp.expert_mlp.w_gate + jnp.bfloat16(1),
        )
    second = _read(publication, updated, 1)
    before = {payload["key"]: payload["data"] for payload in first}
    changed = {payload["key"] for payload in second if payload["data"] != before[payload["key"]]}
    assert changed == {
        f"expert:model.layers.0.mlp.experts.fc1.expert{index}" for index in range(original.config.num_experts)
    }
    assert all(payload["version"] == 1 for payload in second)
