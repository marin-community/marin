# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import hashlib
import json
from types import SimpleNamespace

import haliax as hax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from haliax import Axis
from levanter.grug.sharding import compact_grug_mesh
from levanter.models.snowball import SnowballLMHeadModel

from experiments.interp.snowball_identity import moe_drop_diagnostic as diag
from experiments.interp.snowball_identity.harness import import_grug
from experiments.interp.snowball_identity.levanter_positions import PAD_ID
from experiments.interp.snowball_identity.sequences import Sequence
from experiments.interp.snowball_identity.test_levanter_positions import CONFIG, SEQUENCES


def test_gate_batches_follow_positions_records():
    """The item-5 capture must see the grids ``positions_records`` scores: sorted by length, short steps filled with
    copies of their first sequence, right-padded with PAD_ID to the multiple."""
    grids = list(diag.gate_batches(SEQUENCES, 2, 4))
    assert [g.shape for g in grids] == [(2, 12), (2, 20)]
    assert grids[0][0].tolist() == list(SEQUENCES[2].tokens) + [PAD_ID] * 7
    assert grids[0][1].tolist() == list(SEQUENCES[0].tokens) + [PAD_ID] * 2
    assert grids[1][0].tolist() == list(SEQUENCES[1].tokens) == grids[1][1].tolist()


def test_top_margins():
    logits = np.array([0.0, 3.0, 1.0, 2.0, 2.5, -1.0])
    found = diag.top_margins(logits, np.zeros(6), k=2)
    assert found["ids"] == [1, 4, 3] and found["margin_k_vs_next"] == pytest.approx(0.5)


def test_compare_dumps_skips_the_bias_entry_and_reports_each_op():
    rng = np.random.default_rng(1)
    names = ["embed"] + [f"{op}/{layer}" for layer in range(diag.DUMP_LAYERS) for op in diag.OPS]

    def dumps(noise: float) -> dict:
        record = {name: rng.normal(size=(6, 8 if not name.startswith("router") else 16)) for name in names}
        record |= {f"routes/{layer}": np.tile(np.arange(4), (6, 1)) for layer in range(diag.DUMP_LAYERS)}
        return {"check|a": {k: v + noise if k.startswith("attn_out/0") else v for k, v in record.items()}}

    reference = dumps(0.0)
    ours = {"check|a": {k: v.copy() for k, v in reference["check|a"].items()}}
    ours["check|a"]["attn_out/0"] = ours["check|a"]["attn_out/0"] + 0.5
    reference["_bias"] = {layer: np.zeros(16) for layer in range(diag.DUMP_LAYERS)}
    found = diag.compare_dumps(ours, reference, reference["_bias"])
    assert found["ops"]["embed"]["max_abs"] == 0 and found["ops"]["L0/attn_norm"]["max_abs"] == 0
    assert found["ops"]["L0/attn_out"]["max_abs"] == pytest.approx(0.5)
    assert found["routes"]["L0"] == {"positions": 6, "set_agree": 6, "ordered_agree": 6}
    assert set(found["layer0_position4"]) == {"check|a"}


def test_first_layers_config_sets_every_layer_alias():
    grug = import_grug()
    cut = diag.first_layers_config({"num_hidden_layers": 26, "num_layers": 26, "hidden_size": 64})
    assert cut == {"num_hidden_layers": 3, "num_layers": 3, "hidden_size": 64}
    assert grug.GrugMoeConfig(**cut).num_hidden_layers == 3
    with pytest.raises(ValueError):
        diag.first_layers_config({"num_hidden_layers": 4})


def test_torch_dumps_runs_the_first_layers_of_a_saved_checkpoint(tmp_path):
    """The torch phase end to end on CPU: a 26-layer tiny GrugMoE saved as HF safetensors, cut to layers 0-2."""
    grug = import_grug()
    torch.manual_seed(0)
    config = grug.GrugMoeConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=16,
        shared_expert_intermediate_size=32,
        num_local_experts=8,
        num_experts_per_tok=4,
        num_hidden_layers=26,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        sliding_window=4,
        max_position_embeddings=64,
    )
    config._attn_implementation = "eager"
    model = grug.GrugMoeForCausalLM(config).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.2)
    model.save_pretrained(tmp_path, safe_serialization=True, max_shard_size="200KB")  # sharded, with an index
    sequences = [Sequence("check|a", "check", (5, 17, 3, 99, 42, 8)), Sequence("check|b", "check", (7, 1, 2))]
    dumps = diag.torch_dumps(str(tmp_path), sequences, torch.device("cpu"))

    names = {"embed"} | {f"{op}/{layer}" for layer in range(3) for op in (*diag.OPS, "routes")}
    assert set(dumps["check|a"]) == names and set(dumps["_bias"]) == {0, 1, 2}
    assert dumps["check|a"]["embed"].shape == (6, 32) and dumps["check|b"]["resid/2"].shape == (3, 32)
    assert dumps["check|a"]["router_logits/0"].shape == (6, 8) and dumps["check|a"]["routes/1"].shape == (6, 4)
    # The cut model's layer-2 output equals the full model's at the same depth.
    with torch.no_grad():
        full = model(torch.tensor([[5, 17, 3, 99, 42, 8]]), output_hidden_states=True).hidden_states[3][0]
    np.testing.assert_allclose(dumps["check|a"]["resid/2"], full.float().numpy(), atol=1e-5)


def test_levanter_dumps_agree_with_the_model_forward():
    """The op-by-op layers 0-2 reproduce the model's own routes; the dump batch is padded to 16 rows of 64."""
    config = dataclasses.replace(CONFIG, num_layers=4)
    check = [SEQUENCES[0], SEQUENCES[2]]
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = SnowballLMHeadModel.init(Axis("vocab", config.vocab_size), config, key=jax.random.key(3))
        dumps, batch = diag.levanter_dumps(model, check)
        rows = [s.tokens for s in check] + [check[0].tokens] * (diag.DUMP_ROWS - len(check))
        grid = diag.padded(rows, diag.PAD_MULTIPLE)
        _, routes = model.activations_with_routes(
            hax.named(jnp.asarray(grid), (Axis("batch", 16), model.Pos.resize(64)))
        )
        routes = np.asarray(routes)
    assert batch["grid"] == [16, 64]
    for row, sequence in enumerate(check):
        n = len(sequence.tokens)
        assert dumps[sequence.id]["embed"].shape == (n, config.hidden_dim)
        for layer in range(3):
            np.testing.assert_array_equal(dumps[sequence.id][f"routes/{layer}"], routes[layer, row, :n])
            np.testing.assert_array_equal(batch["routes"][layer][row, :n], routes[layer, row, :n])


def test_gate_batch_of_finds_the_row_and_its_validity():
    grid, valid, row = diag.gate_batch_of(SEQUENCES, "check|a", 2, 4)
    assert grid.shape == (2, 12) and row == 1
    assert grid[row].tolist()[:10] == list(SEQUENCES[0].tokens)
    assert valid[0].sum() == len(SEQUENCES[2].tokens) and valid[1].sum() == len(SEQUENCES[0].tokens)
    # The last step's filler row (a copy of its first sequence) is not valid.
    grid, valid, row = diag.gate_batch_of(SEQUENCES, "long|x", 2, 4)
    assert row == 0 and valid[0].all() and not valid[1].any()
    with pytest.raises(KeyError):
        diag.gate_batch_of(SEQUENCES, "missing", 2, 4)


def test_item5_margins_compare_sets_and_both_margins():
    ids = np.tile(np.arange(6, dtype=np.int16), (3, 2, 1))
    biased = np.tile(np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0], dtype=np.float32), (3, 2, 1))
    ours_ids, ours = ids.copy(), biased.copy()
    ours_ids[1, 1, 3], ours_ids[1, 1, 4] = 4, 3  # Levanter swaps the 4th and 5th at position 1, layer 1
    ours[1, 1, 3], ours[1, 1, 4] = 2.05, 2.0
    found = diag.item5_margins({"ids": ours_ids, "biased": ours}, {"ids": ids, "biased": biased}, position=1)
    assert found["set_disagreements"] == 1 and found["set_disagreements_by_layer"] == [0, 1]
    assert found["levanter_margin_at_disagreements"]["max"] == pytest.approx(0.05, abs=1e-6)
    assert found["torch_margin_at_disagreements"]["p50"] == pytest.approx(1.0)
    assert found["position_1"]["set_agree_by_layer"] == [True, False]
    assert diag.margin_stats(np.array([])) == {"count": 0}


def test_router_top6_matches_the_scored_forward_routes():
    """The capture's routes equal the validity-masked forward's own, and its top-4 ids are those routes."""
    config = dataclasses.replace(CONFIG, num_layers=4)
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        model = SnowballLMHeadModel.init(Axis("vocab", config.vocab_size), config, key=jax.random.key(3))
        grid, valid, _ = diag.gate_batch_of(SEQUENCES, "check|a", 2, 4)
        ids, values, routes = (
            np.asarray(x) for x in jax.jit(diag._router_top6)(model, jnp.asarray(grid), jnp.asarray(valid))
        )
        Batch, Pos = Axis("batch", 2), model.Pos.resize(grid.shape[1])
        _, stats = model.activations_with_stats(
            hax.named(jnp.asarray(grid), (Batch, Pos)), hax.named(jnp.asarray(valid), (Batch, Pos))
        )
    assert ids.shape == values.shape == (4, 2, 12, diag.TOP)
    np.testing.assert_array_equal(routes, np.asarray(stats.routes))
    np.testing.assert_array_equal(ids[..., : config.num_experts_per_token], routes)
    assert (np.diff(values, axis=-1) <= 0).all()


def test_scoring_gathers_at_the_stored_vllm_ids(monkeypatch):
    seen = {}

    def records(view, sequences, *args, **kwargs):
        seen.update(args=args, kwargs=kwargs)
        return {}, {}

    monkeypatch.setattr(diag, "positions_records", records)
    monkeypatch.setattr(diag, "reference_ids", lambda path, sequences: {"path": path, "count": len(sequences)})
    diag.score_positions("view", ["a", "b"])
    assert seen["args"] == (diag.BATCH, diag.PAD_MULTIPLE, diag.PAD_ID)
    assert seen["kwargs"] == {"routes": True, "reference_ids": {"path": diag.INPUTS.vllm, "count": 2}}


def _launch_args(tmp_path, runtime: dict, sha: str | None = None):
    receipt = tmp_path / "receipt.json"
    receipt.write_bytes(json.dumps({"runtime_model_config": runtime}).encode())
    return SimpleNamespace(
        job_name="t",
        root=f"{diag.T310}/diag-test",
        arm="dropless-fp32-combine-fp32-activations",
        runtime_receipt=str(receipt),
        runtime_receipt_sha256=sha or hashlib.sha256(receipt.read_bytes()).hexdigest(),
        timeout_seconds=2700,
        priority="production",
        expect_commit=None,
        run=False,
    )


def test_launch_plan_binds_the_receipt_sha_and_both_checkers(tmp_path, capsys):
    args = _launch_args(tmp_path, {"router_combine_fp32": True, "fp32_activations": True})
    diag.launch_job(args)
    plan = json.loads(capsys.readouterr().out)
    assert plan["runtime_receipt_sha256"] == args.runtime_receipt_sha256
    assert plan["arm"] == {args.arm: {"router_combine_fp32": True, "fp32_activations": True}}
    assert plan["g1a"]["vllm_pass1_sha256"] == diag.route_proxy_check.VLLM_PASS1_SHA256
    assert plan["g1a"]["rule"]["k"] == 1.25
    assert plan["h100_hour_ceiling"] == 6.0


def test_launch_refuses_a_receipt_without_the_arm_flags(tmp_path):
    with pytest.raises(ValueError, match="do not match arm"):
        diag.launch_job(_launch_args(tmp_path, {"router_combine_fp32": True, "fp32_activations": False}))
    with pytest.raises(ValueError, match="differs from the reviewed receipt"):
        diag.launch_job(_launch_args(tmp_path, {"router_combine_fp32": True, "fp32_activations": True}, "0" * 64))
