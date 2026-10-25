# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch

from experiments.interp.snowball_identity import moe_drop_diagnostic as diag
from experiments.interp.snowball_identity import torch_l0_ops as l0
from experiments.interp.snowball_identity.harness import import_grug
from experiments.interp.snowball_identity.sequences import Sequence


@pytest.fixture(scope="module")
def inner():
    grug = import_grug()
    torch.manual_seed(0)
    config = grug.GrugMoeConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=16,
        shared_expert_intermediate_size=32,
        num_local_experts=8,
        num_experts_per_tok=4,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        sliding_window=4,
        max_position_embeddings=256,
    )
    config._attn_implementation = "eager"
    model = grug.GrugMoeForCausalLM(config).eval().requires_grad_(False)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.2)
    return model.model.to(torch.bfloat16)


SEQUENCES = [
    Sequence(f"{group}|r{i}", group, tuple(np.random.default_rng(i).integers(1, 127, n).tolist()))
    for i, (group, n) in enumerate(
        (
            ("check", 5),
            ("contrastive", 9),
            ("check", 3),
            ("long", 12),
            ("contrastive", 7),
            ("check", 20),
            *(("contrastive", n) for n in (6, 4, 11, 8)),
        )
    )
]


@pytest.mark.parametrize("is_long", [False, True])
def test_step_by_step_block_zero_matches_the_modules(inner, is_long):
    """Each gated norm and the attention are checked bit for bit inside ``l0_ops``; the router logits come from the
    module's own router on the step-by-step input."""
    ids = torch.tensor([list(SEQUENCES[1].tokens)])
    ops = l0.l0_ops(inner, ids, is_long)
    assert ("L0/q_rope" in ops) is not is_long
    block = inner.layers[0]
    hidden = inner.embed_gated_norm(inner.embed_norm(inner.embed_tokens(ids)))
    assert torch.equal(ops["embed"], hidden)
    positions = torch.arange(ids.shape[1]).unsqueeze(0)
    attn_in = block.attn_gated_norm(block.input_layernorm(hidden))
    resid = hidden + block.self_attn(attn_in, None, positions, is_long=is_long)
    router_in = block.mlp_gated_norm(block.post_attention_layernorm(resid))
    assert torch.equal(ops["L0/mlp_norm"], router_in)
    expected = block.mlp.router(router_in.reshape(-1, router_in.shape[-1])).router_logits
    assert torch.equal(ops["L0/router_logits"].reshape(expected.shape), expected)
    assert all(value.shape[:2] == ids.shape for value in ops.values())


def test_batched_vs_batch1_covers_every_real_position(inner, monkeypatch):
    """Ten sequences: two gate steps of 8, the second filled with copies. ``check`` counts only its sequences."""
    monkeypatch.setattr(diag, "PAD_ID", 0)  # inside the tiny vocab
    stats, check_ops = l0.compare_ops(inner, False, SEQUENCES, torch.device("cpu"))
    real = sum(len(s.tokens) for s in SEQUENCES)
    checked = sum(len(s.tokens) for s in SEQUENCES if s.group == "check")
    assert stats["all"]["embed"]["elements"] == real * 32
    assert stats["check"]["L0/router_logits"]["elements"] == checked * 8
    assert set(check_ops) == {s.id for s in SEQUENCES if s.group == "check"}
    assert set(stats["all"]) == set(stats["check"]) == set(check_ops["check|r0"])
    # The embedding lookup does not depend on the batch shape.
    assert stats["all"]["embed_tokens"]["max_abs"] == 0.0


def test_accumulated_op_diff_equals_the_diagnostic_op_diff():
    rng = np.random.default_rng(0)
    pairs = [(rng.normal(size=(n, 6)), rng.normal(size=(n, 6))) for n in (3, 7, 2)]
    accumulated = l0.OpDiff()
    for ours, ref in pairs:
        accumulated.add(torch.from_numpy(ours), torch.from_numpy(ref))
    whole = diag.op_diff(np.concatenate([o for o, _ in pairs]), np.concatenate([r for _, r in pairs]))
    report = accumulated.report()
    assert report.pop("elements") == 12 * 6
    assert report == pytest.approx(whole)


def test_rerun_equality_reads_the_rerun_keys():
    ops = {"check|a": {name: torch.ones(3, 4) for name in l0.RERUN_NAMES}}
    rerun = {"check|a": {key: np.ones((3, 4), dtype=np.float32) for key in l0.RERUN_NAMES.values()}}
    assert all(l0.rerun_equality(ops, rerun).values())
    rerun["check|a"]["mlp_norm/0"][1, 2] = 2.0
    assert l0.rerun_equality(ops, rerun) == {name: name != "L0/mlp_norm" for name in l0.RERUN_NAMES}
