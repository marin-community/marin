# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
import os
import struct
import subprocess
import sys

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from safetensors.numpy import load, save

from levanter.grug.sharding import compact_grug_mesh
from levanter.inference.engine import InferenceEngine, InferenceEngineConfig, Request
from levanter.inference.jit_scheduler import SeqDecodingParams
from levanter.inference.snowball_engine_replay import (
    generate_snowball_replay,
    replay_snowball_logits,
    replay_snowball_loss,
)
from levanter.inference.snowball_replay_payload import SnowballReplayPayload
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel


@pytest.fixture
def replay_case():
    config = SnowballConfig(
        vocab_size=23,
        hidden_dim=16,
        intermediate_dim=16,
        shared_expert_intermediate_dim=16,
        num_layers=1,
        num_experts=4,
        num_experts_per_token=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        max_seq_len=8,
        attention_implementation="reference",
        inference_attention_implementation="reference",
        moe_implementation="ring",
    )
    with (
        jax.set_mesh(compact_grug_mesh(expert_axis_size=1)),
        jax.default_matmul_precision("highest"),
    ):
        model = SnowballLMHeadModel.init(hax.Axis("vocab", 23), config, key=jax.random.key(41))
        engine = InferenceEngine.from_model_with_config(
            model,
            None,
            InferenceEngineConfig(
                max_seq_len=8,
                max_pages=2,
                max_seqs=1,
                page_size=4,
                max_queued_tokens=1,
                max_seqs_in_prefill=1,
                max_prefill_size=4,
                max_rounds=3,
                compute_dtype=jnp.float32,
            ),
        )
        request = Request(
            [1, 2, 7],
            709,
            dataclasses.replace(
                SeqDecodingParams.default(),
                max_num_tokens=jnp.asarray(7),
                temperature=jnp.asarray(1.0),
                key=jax.random.PRNGKey(99),
            ),
            1,
        )
        payload = generate_snowball_replay(engine, [request], policy_id="tiny-checkpoint-41").payloads[0]
        yield model, engine, request, payload


@pytest.mark.timeout(240)
def test_payload_roundtrip_fresh_process_replays_routes_and_weighted_loss(replay_case, tmp_path):
    model, _, _, payload = replay_case
    path = tmp_path / "transport.snowball"
    path.write_bytes(payload.to_bytes())
    output = tmp_path / "learner.npz"
    child = """
import json, sys
from pathlib import Path
import draccus
import haliax as hax
import jax
import numpy as np
from levanter.grug.sharding import compact_grug_mesh
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel
from levanter.inference.snowball_engine_replay import replay_snowball_logits, replay_snowball_loss
from levanter.inference.snowball_replay_payload import SnowballReplayPayload
payload = SnowballReplayPayload.from_bytes(Path(sys.argv[1]).read_bytes())
config = draccus.decode(SnowballConfig, json.loads(payload.model_binding)["config"])
with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)), jax.default_matmul_precision("highest"), hax.axis_mapping(
        {"batch": ("replica_dcn", "data", "expert"), "position": None, "vocab": "model"}
    ):
    model = SnowballLMHeadModel.init(hax.Axis("vocab", 23), config, key=jax.random.key(41))
    logits = replay_snowball_logits(model, payload, policy_id="tiny-checkpoint-41")
    loss = replay_snowball_loss(model, payload, policy_id="tiny-checkpoint-41")
    np.savez(sys.argv[2], logits=np.asarray(logits.array), loss=np.asarray(loss))
"""
    completed = subprocess.run(
        [os.path.abspath(sys.executable), "-B", "-c", child, str(path), str(output)],
        cwd=tmp_path,
        env={**os.environ, "JAX_PLATFORMS": "cpu", "JAX_NUM_CPU_DEVICES": "1", "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stderr
    with np.load(output, allow_pickle=False) as actual:
        np.testing.assert_array_equal(
            actual["logits"], np.asarray(replay_snowball_logits(model, payload, policy_id="tiny-checkpoint-41").array)
        )
        with hax.axis_mapping({"batch": ("replica_dcn", "data", "expert"), "position": None, "vocab": "model"}):
            np.testing.assert_array_equal(
                actual["loss"], np.asarray(replay_snowball_loss(model, payload, policy_id="tiny-checkpoint-41"))
            )
    restored = SnowballReplayPayload.from_bytes(path.read_bytes())
    assert restored.binding == payload.binding
    assert restored.request_id == 709 and restored.sampling.key == (0, 99)
    np.testing.assert_array_equal(restored.raw_policy_logprobs, payload.sampled_logprobs)


@pytest.mark.timeout(180)
def test_payload_rejects_altered_tokens_order_prompt_policy_and_transport(replay_case):
    model, _, _, payload = replay_case
    for changed in [
        dataclasses.replace(payload, tokens=payload.tokens[::-1].copy()),
        dataclasses.replace(payload, positions=payload.positions[::-1].copy()),
        dataclasses.replace(payload, prompt_length=payload.prompt_length + 1),
        dataclasses.replace(payload, model_binding=payload.model_binding + " "),
    ]:
        with pytest.raises(ValueError):
            replay_snowball_logits(model, changed, policy_id="tiny-checkpoint-41")
    with pytest.raises(ValueError, match="identity"):
        replay_snowball_logits(model, payload, policy_id="another-checkpoint")
    different = SnowballLMHeadModel(model.transformer, dataclasses.replace(model.config, sliding_window=3))
    with pytest.raises(ValueError, match="model architecture"):
        replay_snowball_logits(different, payload, policy_id="tiny-checkpoint-41")
    wire = payload.to_bytes()
    with pytest.raises(ValueError):
        SnowballReplayPayload.from_bytes(wire[: len(wire) // 2])
    # The version is a public wire contract; an unknown version must not be
    # silently interpreted as today's route or causal mask schema.
    header_start = wire.index(b"\n") + 1
    metadata_size = struct.unpack(">I", wire[header_start : header_start + 4])[0]
    start = header_start + 4
    metadata = json.loads(wire[start : start + metadata_size])
    metadata["version"] = 100
    encoded = json.dumps(metadata).encode()
    incompatible = wire[:header_start] + struct.pack(">I", len(encoded)) + encoded + wire[start + metadata_size :]
    with pytest.raises(ValueError, match="version"):
        SnowballReplayPayload.from_bytes(incompatible)


def test_wire_packs_expert_ids_and_preserves_the_archived_binding(replay_case):
    _, _, _, payload = replay_case
    wire = payload.to_bytes()
    metadata_start = wire.index(b"\n") + 1
    metadata_size = struct.unpack_from(">I", wire, metadata_start)[0]
    arrays_start = metadata_start + 4 + metadata_size
    arrays = load(wire[arrays_start:])
    assert arrays["layer.0.selected_experts"].dtype == np.uint8
    assert SnowballReplayPayload.from_bytes(wire).binding == payload.binding

    legacy = json.loads(wire[metadata_start + 4 : arrays_start])
    legacy["version"] = 1
    metadata = json.dumps(legacy).encode()
    arrays["layer.0.selected_experts"] = arrays["layer.0.selected_experts"].astype(np.int32)
    archived = wire[:metadata_start] + struct.pack(">I", len(metadata)) + metadata + save(arrays)
    assert len(archived) > len(wire)
    restored = SnowballReplayPayload.from_bytes(archived)
    assert restored.binding == payload.binding
    np.testing.assert_array_equal(restored.layers[0].selected_experts, payload.layers[0].selected_experts)


@pytest.mark.timeout(240)
def test_transported_route_loss_matches_independent_next_token_values_and_gradients(replay_case):
    model, _, _, payload = replay_case
    payload = SnowballReplayPayload.from_bytes(payload.to_bytes())
    # A learner router update must keep the transported expert choices. A
    # consumer that recaptures or ignores evidence instead follows fresh routes.
    router = model.transformer.blocks[0].mlp.router.at[:, 0].add(10.0)
    model = eqx.tree_at(lambda m: m.transformer.blocks[0].mlp.router, model, router)
    example, capture = payload.training_inputs(model, policy_id="tiny-checkpoint-41")
    replayed = replay_snowball_logits(model, payload, policy_id="tiny-checkpoint-41")
    direct = model(example.tokens, example.attn_mask, route_captures=capture)
    fresh = model(example.tokens, example.attn_mask)
    np.testing.assert_array_equal(np.asarray(replayed.array), np.asarray(direct.array))
    assert np.max(np.abs(np.asarray(replayed.array) - np.asarray(fresh.array))) > 0.0
    weights = jnp.asarray([1.0, 0.0, 2.0, 4.0])

    def consumed(m):
        return replay_snowball_loss(
            m, payload, policy_id="tiny-checkpoint-41", target_weights=hax.named(weights, "target")
        )

    def reference(m):
        # Full-vocabulary next-token CE independently checks shifting, endpoint
        # coverage and weighting against the public logits replay boundary.
        logits = replay_snowball_logits(m, payload, policy_id="tiny-checkpoint-41").array[
            payload.prompt_length - 1 : -1
        ]
        logits = jax.sharding.reshard(logits, jax.sharding.PartitionSpec())
        target_logits = logits[jnp.arange(len(weights)), jnp.asarray(payload.tokens[payload.prompt_length :])]
        return jnp.sum((jax.nn.logsumexp(logits, axis=-1) - target_logits) * weights) / weights.sum()

    with hax.axis_mapping({"batch": ("replica_dcn", "data", "expert"), "position": None, "vocab": "model"}):
        actual_value, actual_grad = eqx.filter_jit(eqx.filter_value_and_grad(consumed))(model)
        expected_value, expected_grad = eqx.filter_jit(eqx.filter_value_and_grad(reference))(model)
    np.testing.assert_allclose(actual_value, expected_value, rtol=1e-5, atol=1e-5)
    for actual, expected in zip(
        jax.tree_util.tree_leaves(actual_grad), jax.tree_util.tree_leaves(expected_grad), strict=True
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
    assert sum(float(jnp.sum(jnp.abs(g))) for g in jax.tree_util.tree_leaves(actual_grad)) > 0


@pytest.mark.timeout(180)
def test_transformed_sampling_payload_labels_logprobs_and_replays_raw_logits(replay_case):
    model, engine, request, _ = replay_case
    request = dataclasses.replace(
        request,
        decode_params=dataclasses.replace(
            request.decode_params,
            temperature=jnp.asarray(0.7),
            top_p=jnp.asarray(0.9),
        ),
    )
    ordinary = engine.generate([request])
    replay = generate_snowball_replay(engine, [request], policy_id="tiny-checkpoint-41")
    assert replay.generation == ordinary
    payload = SnowballReplayPayload.from_bytes(replay.payloads[0].to_bytes())
    assert np.isfinite(payload.sampled_logprobs).all()
    with pytest.raises(ValueError, match="raw-policy comparison"):
        _ = payload.raw_policy_logprobs
    logits = replay_snowball_logits(model, payload, policy_id="tiny-checkpoint-41")
    predictors = logits.array[payload.prompt_length - 1 : -1]
    logprobs = np.asarray(jax.nn.log_softmax(predictors, axis=-1))
    raw = logprobs[np.arange(len(payload.sampled_logprobs)), payload.tokens[payload.prompt_length :]]
    assert np.max(np.abs(raw - payload.sampled_logprobs)) > 0.0
