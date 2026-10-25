# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CPU checks of the Levanter positions writer on a tiny random Levanter Snowball."""

import dataclasses
import hashlib
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
from levanter.testing.cpu_devices import run_on_cpu_devices

from experiments.interp.snowball_identity import levanter_positions, precision_gate, route_proxy_check
from experiments.interp.snowball_identity.levanter_positions import TOP_K, positions_records, write_positions
from experiments.interp.snowball_identity.logit_check import position_stats, reference_ids, stored_sequences
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
        records, _ = positions_records(model, SEQUENCES, batch=1, pad_multiple=1, pad_id=63)
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


def test_reference_gather_measures_ids_outside_the_top_50(mesh_and_model):
    """vLLM's ids are gathered from the same log-softmax, so a token ranked below the top 50 is still measured; the
    grid offset (position p predicts token p + 1) holds under batching and padding."""
    mesh, model = mesh_and_model
    batch = jax.device_count()
    with jax.set_mesh(mesh):
        expected = {}
        for sequence in SEQUENCES:
            tokens = jnp.broadcast_to(jnp.asarray(sequence.tokens, dtype=jnp.int32), (batch, len(sequence.tokens)))
            ids = hax.named(tokens, (Axis("batch", batch), model.Pos.resize(len(sequence.tokens))))
            expected[sequence.id] = np.asarray(jax.nn.log_softmax(model(ids).array[0].astype(jnp.float32), axis=-1))
        reference = {}
        for sequence in SEQUENCES:
            ranked = np.argsort(-expected[sequence.id], axis=-1)
            ids = np.full((len(sequence.tokens), 3), -1, dtype=np.int32)
            ids[1:, 0] = ranked[:-1, -1]  # the least likely token: outside the top 50 of 64
            ids[1:, 1] = ranked[:-1, 0]
            reference[sequence.id] = ids
        records, _ = positions_records(model, SEQUENCES, batch=2, pad_multiple=8, pad_id=63, reference_ids=reference)
    for sequence in SEQUENCES:
        record = records[sequence.id]
        assert record["reference_ids"].shape == record["reference_lp"].shape == (len(sequence.tokens), 3)
        assert (record["reference_ids"][0] == -1).all() and torch.isnan(record["reference_lp"][0]).all()
        np.testing.assert_array_equal(record["reference_ids"][1:], reference[sequence.id][1:])
        assert torch.isnan(record["reference_lp"][1:, 2]).all()
        for i in range(1, len(sequence.tokens)):
            assert int(reference[sequence.id][i, 0]) not in record["top_ids"][i].tolist()
            for column in (0, 1):
                token = reference[sequence.id][i, column]
                want = expected[sequence.id][i - 1, token]
                np.testing.assert_allclose(record["reference_lp"][i, column], want, atol=1e-4)


def test_position_stats_uses_gathered_reference_values():
    theirs = {"check|a": {"top": [None, {"5": -1.0, "9": -6.0}]}}
    record = {"top_ids": torch.tensor([[-1, -1], [5, 7]]), "top_lp": torch.tensor([[np.nan, np.nan], [-1.25, -2.0]])}
    groups = {"check": [("check|a", 1)]}
    alone = position_stats({"check|a": record}, theirs, groups)["check"]
    assert alone["missing_from_harness_top50"] == 1 and alone["tokens_compared"] == 1
    record["reference_ids"] = torch.tensor([[-1, -1], [5, 9]])
    record["reference_lp"] = torch.tensor([[np.nan, np.nan], [-1.25, -8.5]])
    gathered = position_stats({"check|a": record}, theirs, groups)["check"]
    assert gathered["missing_from_harness_top50"] == 0 and gathered["tokens_compared"] == 2
    assert gathered["max_abs_logprob_diff"] == pytest.approx(2.5)


def test_reference_ids_follow_the_record_layout(tmp_path):
    sequence = Sequence("check|a", "check", (3, 17, 5))
    stored = {"check|a": {"tokens": [3, 17, 5], "top": [None, {"17": -0.1, "4": -3.0}, {"5": -0.2}]}}
    (tmp_path / "vllm.json").write_text(json.dumps(stored))
    ids = reference_ids(str(tmp_path / "vllm.json"), [sequence])["check|a"]
    assert ids.shape == (3, 20) and ids.dtype == np.int32
    assert (ids[0] == -1).all() and ids[1, :2].tolist() == [17, 4] and ids[2, :1].tolist() == [5]
    assert (ids[1, 2:] == -1).all() and (ids[2, 1:] == -1).all()
    with pytest.raises(ValueError, match="different token IDs"):
        reference_ids(str(tmp_path / "vllm.json"), [Sequence("check|a", "check", (3, 17, 6))])


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
        exact, _ = positions_records(model, SEQUENCES, pad_multiple=4, pad_id=63)
    stored = torch.load(tmp_path / "positions.pt")
    assert [s.tokens for s in stored_sequences(str(tmp_path / "positions.pt"))] == [s.tokens for s in SEQUENCES]
    # The BF16 compute view scores close to, but not exactly like, the FP32 model.
    diff = (stored["check|a"]["actual"][1:] - exact["check|a"]["actual"][1:]).abs()
    assert 0 < float(diff.max()) < 0.5


def test_batching_and_right_padding_keep_real_positions(mesh_and_model):
    mesh, model = mesh_and_model
    with jax.set_mesh(mesh):
        single, _ = positions_records(model, SEQUENCES, batch=1, pad_multiple=1, pad_id=63)
        # Two rows per step (the last step gets a filler copy) and padding to a multiple of 8.
        batched, _ = positions_records(model, SEQUENCES, batch=2, pad_multiple=8, pad_id=63)
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
        single, _ = positions_records(model, SEQUENCES, batch=1, pad_multiple=1, pad_id=63, routes=True)
        batched, _ = positions_records(model, SEQUENCES, batch=2, pad_multiple=8, pad_id=63, routes=True)
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


def write_floor(tmp_path, monkeypatch, routes: dict) -> tuple[str, str, str]:
    """A G1 torch floor for ``routes`` (this model's own): torch batch-1 picks the same experts with a 1.0 margin; the
    floor allows 5% set disagreements per layer at own margins p50 0.02 / p99 0.04. And a G1a vLLM capture whose two
    passes pick the same experts. Each file is pinned by its own sha256."""
    k = CONFIG.num_experts_per_token
    monkeypatch.setattr(route_proxy_check, "K", k)
    top6 = {}
    for sid, chosen in routes.items():
        extra = np.broadcast_to(np.arange(k, k + 2), (*chosen.shape[:2], 2))
        ids = np.concatenate([chosen.numpy(), extra], axis=-1).astype(np.int16)
        biased = np.broadcast_to(np.arange(k + 2, 0, -1, dtype=np.float32), ids.shape).copy()
        top6[sid] = {"ids": torch.from_numpy(ids), "biased": torch.from_numpy(biased)}
    positions = sum(len(s.tokens) for s in GATE_SEQUENCES)
    own = {"margin_at_set_disagreements": {"batch1": {"count": 5, "p50": 0.02, "p99": 0.04}}}
    floor = {
        "floor": {
            "batched_vs_routes_pt": {"layers": [{"positions": positions, "set_rate": 0.95}] * CONFIG.num_layers},
            "batched_vs_batch1": {"layers": [own] * CONFIG.num_layers},
        }
    }
    (tmp_path / "floor").mkdir()
    torch.save(top6, tmp_path / "floor" / "top6.pt")
    torch.save(top6, tmp_path / "floor" / "batched-top6.pt")  # G1b's floor: torch batched, here the same routes
    (tmp_path / "floor" / "receipt.json").write_text(json.dumps(floor))
    capture = tmp_path / "capture"
    for n in (1, 2):
        (capture / f"pass-{n}").mkdir(parents=True)
        torch.save(dict(routes), capture / f"pass-{n}" / "routes.pt")
    (capture / "receipt.json").write_text("{}")
    for name, path in (
        ("TORCH_TOP6_SHA256", tmp_path / "floor" / "top6.pt"),
        ("TORCH_BATCHED_TOP6_SHA256", tmp_path / "floor" / "batched-top6.pt"),
        ("FLOOR_RECEIPT_SHA256", tmp_path / "floor" / "receipt.json"),
        ("VLLM_RECEIPT_SHA256", capture / "receipt.json"),
        ("VLLM_PASS1_SHA256", capture / "pass-1" / "routes.pt"),
        ("VLLM_PASS2_SHA256", capture / "pass-2" / "routes.pt"),
    ):
        monkeypatch.setattr(precision_gate, name, hashlib.sha256(path.read_bytes()).hexdigest())
    return str(tmp_path / "floor" / "receipt.json"), str(tmp_path / "floor" / "top6.pt"), str(capture)


def gate_fixture(mesh, model, policy, tmp_path, monkeypatch):
    """Pinned positions, a "vLLM" reference, torch routes and a G1 torch floor that are all this model's own compute
    view."""
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
    floor_receipt, torch_top6, capture = write_floor(
        tmp_path, monkeypatch, {sid: r["routes"] for sid, r in ours.items()}
    )
    inputs = precision_gate.PrecisionGateInputs(
        positions=pinned,
        vllm=str(tmp_path / "vllm.json"),
        identity="unused",
        tokenizer_source="unused",
        torch_routing=str(routing),
        floor_receipt=floor_receipt,
        torch_top6=torch_top6,
        torch_batched_top6=str(tmp_path / "floor" / "batched-top6.pt"),
        route_policy="G1",
        vllm_capture=capture,
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
    inputs, trainer, state = gate_fixture(mesh, model, policy, tmp_path, monkeypatch)
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
    zeros = [0] * CONFIG.num_layers
    assert report["moe"] == {
        "capacity_factor": 1.0,
        "expert_axis": 1,
        "router_combine_fp32": False,
        "fp32_activations": False,
        "dispatch": {
            "sender_dropped": zeros,
            "receiver_dropped": zeros,
            "dropped": zeros,
            "valid_tokens": sum(len(s.tokens) for s in GATE_SEQUENCES),
        },
    }
    assert len(report["device_memory"]) == jax.device_count()
    assert set(report["device_memory"][0]) == {"process", "local_device", *precision_gate.MEMORY_FIELDS}
    # The gate gathers at the stored vLLM ids: every reference token is measured.
    stored = torch.load(tmp_path / "gate" / "positions.pt")
    assert all("reference_lp" in r and not torch.isnan(r["reference_lp"][1:]).all() for r in stored.values())
    assert all(stats["missing_from_harness_top50"] == 0 for stats in report["parity"]["positions"].values())
    # With the parity gate satisfied, the clean routes pass.
    monkeypatch.setattr(precision_gate, "parity_gate", lambda report: [])
    gate(trainer, state)
    report = json.loads((tmp_path / "gate" / "gate.json").read_text())
    assert report["failures"] == [] and report["route_policy"] == "G1"
    assert report["route_proxy"]["G1"]["layers_passing"] == report["route_proxy"]["G1a"]["layers_passing"]
    assert report["route_proxy"]["G1"]["layers_passing"] == CONFIG.num_layers
    # A torch route in a different order is reported, not gated: G1 bounds route differences.
    routes = torch.load(tmp_path / "torch" / "routes.pt")
    routes["check|a"][3, 1] = routes["check|a"][3, 1].flip(0)
    torch.save(routes, tmp_path / "torch" / "routes.pt")
    gate(trainer, state)
    report = json.loads((tmp_path / "gate" / "gate.json").read_text())
    totals = report["routes"]["totals"]
    assert report["failures"] == [] and totals["ordered_agree"] < totals["positions"]
    # One set disagreement with torch batch-1, at a margin far above torch's own, fails G1's margin class.
    top6 = torch.load(inputs.torch_top6)
    top6["check|a"]["ids"][3, 1, [0, 2]] = top6["check|a"]["ids"][3, 1, [2, 0]]
    torch.save(top6, inputs.torch_top6)
    pinned = hashlib.sha256((tmp_path / "floor" / "top6.pt").read_bytes()).hexdigest()
    monkeypatch.setattr(precision_gate, "TORCH_TOP6_SHA256", pinned)
    with pytest.raises(precision_gate.PrecisionGateFailed):
        gate(trainer, state)
    failures = json.loads((tmp_path / "gate" / "gate.json").read_text())["failures"]
    assert failures == ["routes-G1: L1: margin p50 1.0000 > 1.5 x torch 0.0200"]


def test_pre_update_gate_fails_closed_on_missing_inputs(mesh_and_model, tmp_path, monkeypatch):
    mesh, model = mesh_and_model
    inputs, trainer, state = gate_fixture(mesh, model, jmp.get_policy("p=f32,c=bfloat16"), tmp_path, monkeypatch)
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
    inputs, trainer, state = gate_fixture(mesh, model, jmp.get_policy("p=f32,c=bfloat16"), tmp_path, monkeypatch)
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


def test_pre_update_gate_stops_every_rank_on_unreadable_inputs(mesh_and_model, tmp_path, monkeypatch):
    mesh, model = mesh_and_model
    inputs, trainer, state = gate_fixture(mesh, model, jmp.get_policy("p=f32,c=bfloat16"), tmp_path, monkeypatch)
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
    inputs, trainer, state = gate_fixture(mesh, model, jmp.get_policy("p=f32,c=bfloat16"), tmp_path, monkeypatch)
    monkeypatch.setattr(precision_gate, "candidate_spans", lambda *args: {})
    timers = []

    def deadline(seconds):
        timers.append(precision_gate.Deadline(seconds, on_expire=lambda: pytest.fail("deadline fired")))
        return timers[-1]

    with pytest.raises(precision_gate.PrecisionGateFailed):
        precision_gate.make_pre_update_gate(inputs, deadline=deadline)(trainer, state)
    assert timers and timers[0]._done.is_set()


def test_dispatch_failures_name_each_layer_with_drops():
    dispatch = {"sender_dropped": [0, 2, 0], "receiver_dropped": [0, 1, 4], "dropped": [0, 3, 4], "valid_tokens": 9}
    assert precision_gate.dispatch_failures(dispatch) == [
        "layer 1: 3 valid assignments dropped (2 sender, 1 receiver)",
        "layer 2: 4 valid assignments dropped (0 sender, 4 receiver)",
    ]
    assert (
        precision_gate.dispatch_failures(
            {**dispatch, "sender_dropped": [0] * 3, "receiver_dropped": [0] * 3, "dropped": [0] * 3}
        )
        == []
    )


def test_dispatch_counts_valid_tokens_only_on_an_ep4_mesh():
    """On 8 CPU devices (data 2 x expert 4): filler rows and padding take no dispatch, a dropless capacity drops
    nothing, and a small one drops valid assignments; the real rows score the same either way when dropless."""
    run_on_cpu_devices(
        """
        import dataclasses

        import jax
        import numpy as np
        from haliax import Axis
        from levanter.grug.sharding import compact_grug_mesh
        from levanter.models.snowball import SnowballLMHeadModel

        from experiments.interp.snowball_identity.levanter_positions import positions_records
        from experiments.interp.snowball_identity.test_levanter_positions import CONFIG, SEQUENCES

        mesh = compact_grug_mesh(expert_axis_size=4)
        lengths = sum(len(s.tokens) for s in SEQUENCES)
        results = {}
        for capacity in (4.0, 0.25):
            config = dataclasses.replace(CONFIG, moe_capacity_factor=capacity)
            with jax.set_mesh(mesh):
                model = SnowballLMHeadModel.init(Axis("vocab", config.vocab_size), config, key=jax.random.key(3))
                # Eight rows per step: three real sequences, five filler rows, padded to 16 positions.
                results[capacity] = positions_records(model, SEQUENCES, batch=8, pad_multiple=16, pad_id=63)
        records, dispatch = results[4.0]
        assert dispatch == {
            "sender_dropped": [0] * CONFIG.num_layers,
            "receiver_dropped": [0] * CONFIG.num_layers,
            "dropped": [0] * CONFIG.num_layers,
            "valid_tokens": lengths,
        }, dispatch
        _, capped = results[0.25]
        assert capped["valid_tokens"] == lengths
        assert len(capped["dropped"]) == CONFIG.num_layers and sum(capped["dropped"]) > 0, capped
        assert capped["dropped"] == [s + r for s, r in zip(capped["sender_dropped"], capped["receiver_dropped"])]
        # Every layer has K assignments per valid token; it cannot drop more.
        assert max(capped["dropped"]) <= lengths * CONFIG.num_experts_per_token
        assert all(np.isfinite(records[s.id]["actual"][1:].numpy()).all() for s in SEQUENCES)
        print("OK")
        """,
        device_count=8,
    )
