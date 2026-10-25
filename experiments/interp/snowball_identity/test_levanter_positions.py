# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU checks of the Levanter positions writer on a tiny random Levanter Snowball."""

import dataclasses
import json
import threading
from types import SimpleNamespace

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import pytest
import torch
from haliax import Axis
from jax.sharding import AbstractMesh, AxisType
from levanter.grug.sharding import compact_grug_mesh
from levanter.models.snowball import SnowballConfig, SnowballLMHeadModel

from experiments.interp.snowball_identity import levanter_positions, precision_gate
from experiments.interp.snowball_identity.levanter_positions import TOP_K, positions_records, write_positions
from experiments.interp.snowball_identity.logit_check import stored_sequences
from experiments.interp.snowball_identity.sequences import Sequence

CONFIG = SnowballConfig(
    vocab_size=64,
    hidden_dim=32,
    intermediate_dim=64,
    shared_expert_intermediate_dim=48,
    num_experts=8,
    num_experts_per_token=2,
    num_layers=5,
    num_heads=4,
    num_kv_heads=2,
    head_dim=12,
    max_seq_len=32,
    sliding_window=4,
    qk_mult=1.37,
    initializer_std=0.02,
    attention_implementation="reference",
)
SEQUENCES = [
    Sequence("check|a", "check", (3, 17, 5, 60, 2, 9, 41, 8, 11, 30)),
    Sequence("long|x", "long", tuple(range(1, 21))),
    Sequence("contrastive|r|no_system|0", "contrastive", (7, 7, 12, 50, 1)),
]


@pytest.fixture(scope="module")
def mesh_and_model():
    mesh = compact_grug_mesh(expert_axis_size=1)
    with jax.set_mesh(mesh):
        model = SnowballLMHeadModel.init(Axis("vocab", CONFIG.vocab_size), CONFIG, key=jax.random.key(3))
    return mesh, model


def test_records_match_the_model_forward(mesh_and_model):
    mesh, model = mesh_and_model
    batch = jax.device_count()
    with jax.set_mesh(mesh):
        records = positions_records(model, SEQUENCES, batch=1, pad_multiple=1, pad_id=63)
        for sequence in SEQUENCES:
            tokens = jnp.broadcast_to(jnp.asarray(sequence.tokens, dtype=jnp.int32), (batch, len(sequence.tokens)))
            ids = hax.named(tokens, (Axis("batch", batch), model.Pos.resize(len(sequence.tokens))))
            expected = np.asarray(jax.nn.log_softmax(model(ids).array[0].astype(jnp.float32), axis=-1))
            record = records[sequence.id]
            assert record["tokens"] == list(sequence.tokens)
            assert torch.isnan(record["actual"][0]) and (record["top_ids"][0] == -1).all()
            assert record["top_ids"].shape == (len(sequence.tokens), TOP_K)
            # Entry i is the distribution over token i given tokens 0..i-1.
            for i in range(1, len(sequence.tokens)):
                np.testing.assert_allclose(record["actual"][i], expected[i - 1, sequence.tokens[i]], atol=1e-4)
                assert int(record["top_ids"][i, 0]) == int(expected[i - 1].argmax())


def test_write_positions_uses_the_compute_view(mesh_and_model, tmp_path):
    mesh, model = mesh_and_model
    with jax.set_mesh(mesh):
        write_positions(
            model,
            jmp.get_policy("p=f32,c=bfloat16"),
            SEQUENCES,
            str(tmp_path / "positions.pt"),
            pad_multiple=4,
            pad_id=63,
        )
        exact = positions_records(model, SEQUENCES, pad_multiple=4, pad_id=63)
    stored = torch.load(tmp_path / "positions.pt")
    assert [s.tokens for s in stored_sequences(str(tmp_path / "positions.pt"))] == [s.tokens for s in SEQUENCES]
    # The BF16 compute view scores close to, but not exactly like, the FP32 model.
    diff = (stored["check|a"]["actual"][1:] - exact["check|a"]["actual"][1:]).abs()
    assert 0 < float(diff.max()) < 0.5


def test_batching_and_right_padding_keep_real_positions(mesh_and_model):
    mesh, model = mesh_and_model
    with jax.set_mesh(mesh):
        single = positions_records(model, SEQUENCES, batch=1, pad_multiple=1, pad_id=63)
        # Two rows per step (the last step gets a filler copy) and padding to a multiple of 8.
        batched = positions_records(model, SEQUENCES, batch=2, pad_multiple=8, pad_id=63)
    assert list(batched) == [s.id for s in SEQUENCES]
    for sequence in SEQUENCES:
        a, b = single[sequence.id], batched[sequence.id]
        assert a["tokens"] == b["tokens"] and len(b["actual"]) == len(sequence.tokens)
        torch.testing.assert_close(a["actual"], b["actual"], atol=1e-5, rtol=0, equal_nan=True)
        torch.testing.assert_close(a["top_lp"], b["top_lp"], atol=1e-5, rtol=0, equal_nan=True)
        assert torch.equal(a["top_ids"][:, :5], b["top_ids"][:, :5])


def test_routes_sidecar_matches_the_observer(mesh_and_model):
    mesh, model = mesh_and_model
    with jax.set_mesh(mesh):
        single = positions_records(model, SEQUENCES, batch=1, pad_multiple=1, pad_id=63, routes=True)
        batched = positions_records(model, SEQUENCES, batch=2, pad_multiple=8, pad_id=63, routes=True)
        sequence = SEQUENCES[0]
        ids = hax.named(
            jnp.asarray([sequence.tokens], dtype=jnp.int32), (Axis("batch", 1), model.Pos.resize(len(sequence.tokens)))
        )
        _, observed = model.activations_with_routes(ids)
    observed = np.asarray(jax.device_get(observed))[:, 0].transpose(1, 0, 2)  # [T, L, K]
    np.testing.assert_array_equal(single[sequence.id]["routes"].numpy(), observed)
    for s in SEQUENCES:
        assert single[s.id]["routes"].shape == (len(s.tokens), CONFIG.num_layers, CONFIG.num_experts_per_token)
        assert torch.equal(single[s.id]["routes"], batched[s.id]["routes"])


def test_compute_dtype_override_reaches_the_scored_view(mesh_and_model, tmp_path, monkeypatch):
    mesh, model = mesh_and_model
    policy = jmp.get_policy("p=f32,c=bfloat16")
    # Give the router biases values that BF16 cannot hold, so keeping them in FP32 is observable.
    biases = [block.mlp.router_bias for block in model.transformer.blocks]
    model = eqx.tree_at(
        lambda m: [b.mlp.router_bias for b in m.transformer.blocks],
        model,
        [b + jnp.float32(2**-12) * (1 + jnp.arange(b.shape[0])) for b in biases],
    )
    seen = []
    original = levanter_positions.positions_records

    def spy(view, *args):
        seen.append({str(b.mlp.router_bias.dtype) for b in view.transformer.blocks})
        return original(view, *args)

    monkeypatch.setattr(levanter_positions, "positions_records", spy)
    with jax.set_mesh(mesh):
        write_positions(model, policy, SEQUENCES[:1], str(tmp_path / "a.pt"), pad_multiple=4, pad_id=63)
        write_positions(
            model,
            policy,
            SEQUENCES[:1],
            str(tmp_path / "b.pt"),
            pad_multiple=4,
            pad_id=63,
            compute_dtype_overrides={"router_bias": "float32"},
        )
    assert seen == [{"bfloat16"}, {"float32"}]


GATE_SEQUENCES = [SEQUENCES[0], Sequence("check|b", "check", tuple(range(1, 21)))]
"""Check-group only: ``compare`` requires the real long length and contrastive spans."""


def gate_fixture(mesh, model, policy, tmp_path):
    """Pinned positions, a "vLLM" reference and torch routes that are all this model's own compute view."""
    pinned = str(tmp_path / "pinned.pt")
    with jax.set_mesh(mesh):
        write_positions(model, policy, GATE_SEQUENCES, pinned, batch=2, pad_multiple=8, pad_id=63, routes=True)
    ours = torch.load(pinned)
    reference = {
        sid: {
            "tokens": r["tokens"],
            "actual": [None, *r["actual"][1:].tolist()],
            "top": (
                [None]
                + [
                    dict(zip(ids[:20].tolist(), values[:20].tolist(), strict=True))
                    for ids, values in zip(r["top_ids"][1:], r["top_lp"][1:], strict=True)
                ]
            ),
        }
        for sid, r in ours.items()
    }
    (tmp_path / "vllm.json").write_text(json.dumps(reference))
    routing = tmp_path / "torch"
    routing.mkdir()
    (tmp_path / "gate").mkdir()
    torch.save({sid: r["routes"] for sid, r in ours.items()}, routing / "routes.pt")
    positions = sum(len(s.tokens) for s in GATE_SEQUENCES) * CONFIG.num_layers
    unchanged = {"positions": positions, "positions_changed": 0}
    totals = {variant: unchanged for variant in ("bf16_bias", "bf16_logits", "bf16_logits_bias")}
    (routing / "routing.json").write_text(json.dumps({"totals": totals}))
    inputs = precision_gate.PrecisionGateInputs(
        positions=pinned,
        vllm=str(tmp_path / "vllm.json"),
        identity="unused",
        tokenizer_source="unused",
        torch_routing=str(routing),
        output=str(tmp_path / "gate"),
        batch=2,
        pad_multiple=8,
    )
    trainer = SimpleNamespace(
        config=SimpleNamespace(compute_dtype_overrides=None), device_mesh=mesh, mp=policy, compute_axis_mapping={}
    )
    return inputs, trainer, SimpleNamespace(model=model)


def test_pre_update_gate_raises_on_failures_and_passes_when_clean(mesh_and_model, tmp_path, monkeypatch):
    mesh, model = mesh_and_model
    policy = jmp.get_policy("p=f32,c=bfloat16")
    inputs, trainer, state = gate_fixture(mesh, model, policy, tmp_path)
    monkeypatch.setattr(precision_gate, "candidate_spans", lambda *args: {})
    monkeypatch.setattr(levanter_positions, "PAD_ID", 63)
    monkeypatch.setattr(precision_gate, "PAD_ID", 63)
    gate = precision_gate.make_pre_update_gate(inputs)
    # The tiny sequences cannot meet the pinned coverage, so the real parity gate fails; the routes agree.
    with pytest.raises(precision_gate.PrecisionGateFailed):
        gate(trainer, state)
    report = json.loads((tmp_path / "gate" / "gate.json").read_text())
    assert report["failures"] and all(f.startswith("parity: ") for f in report["failures"])
    assert report["routes"]["totals"]["set_agree"] == report["routes"]["totals"]["positions"] > 0
    assert "jax_default_matmul_precision" in report
    # With the parity gate satisfied, the clean routes pass.
    monkeypatch.setattr(precision_gate, "parity_gate", lambda report: [])
    gate(trainer, state)
    assert json.loads((tmp_path / "gate" / "gate.json").read_text())["failures"] == []
    # One changed torch route fails the route gate.
    routes = torch.load(tmp_path / "torch" / "routes.pt")
    routes["check|a"][3, 1] = routes["check|a"][3, 1].flip(0)
    torch.save(routes, tmp_path / "torch" / "routes.pt")
    with pytest.raises(precision_gate.PrecisionGateFailed):
        gate(trainer, state)
    failures = json.loads((tmp_path / "gate" / "gate.json").read_text())["failures"]
    assert failures == ["routes: 1 position-layers differ in expert order"]


def test_pre_update_gate_fails_closed_on_missing_inputs(mesh_and_model, tmp_path, monkeypatch):
    mesh, model = mesh_and_model
    inputs, trainer, state = gate_fixture(mesh, model, jmp.get_policy("p=f32,c=bfloat16"), tmp_path)
    monkeypatch.setattr(precision_gate, "candidate_spans", lambda *args: {})
    (tmp_path / "torch" / "routing.json").unlink()
    with pytest.raises(precision_gate.PrecisionGateFailed):
        precision_gate.make_pre_update_gate(inputs)(trainer, state)
    assert "gate did not complete" in json.loads((tmp_path / "gate" / "gate.json").read_text())["failures"][0]


def test_router_dot_witness_reads_the_traced_dtypes(mesh_and_model):
    mesh, model = mesh_and_model
    with jax.set_mesh(mesh):
        bf16 = levanter_positions.router_dot_dtypes(
            levanter_positions.cast_model_for_compute(model, jmp.get_policy("p=f32,c=bfloat16"), None), jnp.bfloat16
        )
        fp32 = levanter_positions.router_dot_dtypes(model, jnp.float32)
    assert bf16 == [{"lhs": "bfloat16", "rhs": "bfloat16", "out": "bfloat16"}]
    assert not levanter_positions.router_dot_is_fp32(bf16)
    assert levanter_positions.router_dot_is_fp32(fp32) and not levanter_positions.router_dot_is_fp32([])


@pytest.mark.parametrize("router_dot_fp32", [False, True])
def test_router_dot_witness_traces_on_the_32_device_mesh(router_dot_fp32):
    # The real worker mesh (data 2 x expert 4 x context 4), abstract: no devices or arrays are needed.
    config = dataclasses.replace(CONFIG, num_layers=1, router_dot_fp32=router_dot_fp32)
    mesh = compact_grug_mesh(expert_axis_size=1)
    with jax.set_mesh(mesh):
        model = SnowballLMHeadModel.init(Axis("vocab", config.vocab_size), config, key=jax.random.key(5))
        view = levanter_positions.cast_model_for_compute(model, jmp.get_policy("p=f32,c=bfloat16"), None)
    sizes = {"data": 2, "expert": 4, "context": 4}
    names = tuple(mesh.axis_names)
    abstract = AbstractMesh(tuple(sizes.get(n, 1) for n in names), names, axis_types=(AxisType.Explicit,) * len(names))
    with jax.sharding.use_abstract_mesh(abstract):
        witness = levanter_positions.router_dot_dtypes(view, jnp.bfloat16)
    dtype = "float32" if router_dot_fp32 else "bfloat16"
    assert witness == [{"lhs": dtype, "rhs": dtype, "out": dtype}]


def test_pre_update_gate_turns_process_zero_write_failures_into_a_raise(mesh_and_model, tmp_path, monkeypatch):
    mesh, model = mesh_and_model
    inputs, trainer, state = gate_fixture(mesh, model, jmp.get_policy("p=f32,c=bfloat16"), tmp_path)
    monkeypatch.setattr(precision_gate, "candidate_spans", lambda *args: {})
    monkeypatch.setattr(precision_gate, "parity_gate", lambda report: [])
    write_json = precision_gate.write_json

    def failing_gate_json(path, value):
        if path.endswith("gate.json"):
            raise OSError("bucket unavailable")
        write_json(path, value)

    monkeypatch.setattr(precision_gate, "write_json", failing_gate_json)
    with pytest.raises(precision_gate.PrecisionGateFailed, match="did not complete"):
        precision_gate.make_pre_update_gate(inputs)(trainer, state)
    receipt = json.loads((tmp_path / "gate" / "ranks" / "0.json").read_text())
    assert receipt["stage"] == "gated" and receipt["failures"] == 1

    monkeypatch.setattr(precision_gate, "write_json", write_json)
    monkeypatch.setattr(precision_gate, "save_positions", lambda *args: (_ for _ in ()).throw(OSError("full")))
    with pytest.raises(precision_gate.PrecisionGateFailed, match="did not complete"):
        precision_gate.make_pre_update_gate(inputs)(trainer, state)
    assert "OSError('full')" in json.loads((tmp_path / "gate" / "gate.json").read_text())["failures"][0]


def test_pre_update_gate_stops_every_rank_on_unreadable_inputs(mesh_and_model, tmp_path):
    mesh, model = mesh_and_model
    inputs, trainer, state = gate_fixture(mesh, model, jmp.get_policy("p=f32,c=bfloat16"), tmp_path)
    inputs = dataclasses.replace(inputs, positions=str(tmp_path / "missing.pt"))
    with pytest.raises(precision_gate.PrecisionGateFailed, match="could not read"):
        precision_gate.make_pre_update_gate(inputs)(trainer, state)
    assert json.loads((tmp_path / "gate" / "ranks" / "0.json").read_text())["stage"] == "inputs"
    assert not (tmp_path / "gate" / "gate.json").exists()


def test_deadline_fires_unless_cancelled(mesh_and_model, tmp_path, monkeypatch):
    fired = threading.Event()
    precision_gate.Deadline(0.05, on_expire=fired.set)
    assert fired.wait(5)
    cancelled = threading.Event()
    precision_gate.Deadline(0.2, on_expire=cancelled.set).cancel()
    assert not cancelled.wait(0.5)
    # The gate cancels its deadline whether it passes or raises.
    mesh, model = mesh_and_model
    inputs, trainer, state = gate_fixture(mesh, model, jmp.get_policy("p=f32,c=bfloat16"), tmp_path)
    monkeypatch.setattr(precision_gate, "candidate_spans", lambda *args: {})
    timers = []

    def deadline(seconds):
        timers.append(precision_gate.Deadline(seconds, on_expire=lambda: pytest.fail("deadline fired")))
        return timers[-1]

    with pytest.raises(precision_gate.PrecisionGateFailed):
        precision_gate.make_pre_update_gate(inputs, deadline=deadline)(trainer, state)
    assert timers and timers[0]._done.is_set()
