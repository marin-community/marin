# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import itertools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from levanter.grug.attention import align_kv_heads, apply_rotary_embedding, attention
from levanter.grug.sharding import compact_grug_mesh
from levanter.models import snowball
from levanter.models.snowball import SnowballAttention, SnowballConfig

from experiments.interp.snowball_identity import levanter_l0_attn as lev

S = 6
TAPPED = (
    *lev.PROJECTIONS,
    *lev.NORMS,
    "L0/q_rope_rotated",
    "L0/k_rope_rotated",
    "L0/q_scaled",
    "L0/aligned_v",
    "L0/attn_head_gate",
    "L0/attn_gated",
    "L0/attn_out",
)


@pytest.fixture(scope="module")
def setup():
    cfg = SnowballConfig(
        vocab_size=48,
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
        qk_mult=1.75,
        layer_norm_eps=1e-5,
        initializer_std=0.2,
        fp32_activations=True,
    )
    counter = itertools.count()
    keys = (jax.random.fold_in(jax.random.PRNGKey(0), i) for i in counter)

    def bf16(shape, scale=1.0):
        return (scale * jax.random.normal(next(keys), shape)).astype(jnp.bfloat16)

    with jax.set_mesh(compact_grug_mesh(expert_axis_size=1)):
        attn = SnowballAttention.init(cfg, key=next(keys))
        attn = jax.tree_util.tree_map(lambda w: w.astype(jnp.bfloat16), attn)
        attn = eqx.tree_at(lambda a: a.attn_gate, attn, bf16((32, 4), 0.3))
        yield attn, bf16((1, S, 32)), bf16


def test_taps_record_every_step_and_leave_the_module_unchanged(setup):
    """Op by op (no fusion), the tapped module equals the plain module bit for bit; jitted, the taps only change which
    intermediates XLA materializes (the job reports that as composite_vs_chain)."""
    attn, x, _ = setup
    mask = lev.short_mask(attn.cfg)
    with jax.disable_jit():
        taps = lev.Taps()
        with taps.installed():
            tapped_out = attn(x, mask)
        plain = attn(x, mask)
    assert set(TAPPED) - {"L0/attn_out"} <= set(taps.found)
    np.testing.assert_array_equal(np.asarray(tapped_out), np.asarray(plain))
    assert snowball.jnp is jnp and snowball.attention is attention
    chain = lev.tapped(attn, x, None)
    assert set(TAPPED) <= set(chain)
    assert chain["implementation"] is None  # on CPU the module leaves the choice to ``attention``
    np.testing.assert_allclose(
        np.asarray(chain["L0/attn_out"].astype(jnp.float32)),
        np.asarray(lev.composite(attn, x, None).astype(jnp.float32)),
        rtol=2**-7,
        atol=2**-7,
    )


def test_isolated_feeds_each_step_torch_input(setup):
    attn, x, bf16 = setup
    cfg, half = attn.cfg, attn.cfg.inferred_head_dim // 2
    given = {
        "L0/q_proj": bf16((1, S, 4, 12)),
        "L0/k_proj": bf16((1, S, 2, 12)),
        "L0/v_proj": bf16((1, S, 2, 12)),
        "L0/q_norm": bf16((1, S, 4, 12)),
        "L0/k_norm": bf16((1, S, 2, 12)),
        "L0/q_rope_rotated": bf16((1, S, 4, half)),
        "L0/k_rope_rotated": bf16((1, S, 2, half)),
        "L0/q_scaled": bf16((1, S, 4, 12)),
        "L0/k_rope": bf16((1, S, 2, 12)),
        "L0/attn_core": bf16((1, S, 4, 12)),
        "L0/attn_head_gate": 2 * jax.nn.sigmoid(bf16((1, S, 4, 1))),
        "L0/attn_gated": bf16((1, S, 4, 12)),
    }
    out = lev.tapped(attn, x, given)
    eq = np.testing.assert_array_equal
    eq(np.asarray(out["L0/q_norm"]), np.asarray(snowball.rms_norm(given["L0/q_proj"])))
    eq(np.asarray(out["L0/k_norm"]), np.asarray(snowball.rms_norm(given["L0/k_proj"])))
    q_rot, _ = apply_rotary_embedding(
        given["L0/q_norm"][..., :half], given["L0/k_norm"][..., :half], seq_len=S, head_dim=half, rope=cfg.rope
    )
    eq(np.asarray(out["L0/q_rope_rotated"]), np.asarray(q_rot))
    q_rope = jnp.concatenate([given["L0/q_rope_rotated"], given["L0/q_norm"][..., half:]], axis=-1)
    eq(np.asarray(out["L0/q_scaled"]), np.asarray(q_rope * cfg.qk_mult))
    q, k, v = given["L0/q_scaled"], given["L0/k_rope"], given["L0/v_proj"]
    mask = lev.short_mask(cfg)
    eq(np.asarray(out["core/reference"]), np.asarray(attention(q, k, v, mask, implementation="reference")))
    eq(np.asarray(out["core/torch_order"]), np.asarray(lev.torch_order_core(q, k, v, cfg.sliding_window)))
    eq(np.asarray(out["L0/aligned_v"]), np.asarray(align_kv_heads(given["L0/v_proj"], num_q_heads=4)))
    o_proj = jnp.einsum("bsh,hd->bsd", given["L0/attn_gated"].reshape(1, S, -1), attn.w_o)
    eq(np.asarray(out["L0/attn_out"]), np.asarray(o_proj))
    # The XSA correction and the gate multiply on torch's core, v and gate.
    xsa = lev.tapped(attn, x, given, gate_one=True)["L0/attn_xsa"]
    np.testing.assert_allclose(
        np.asarray((xsa * given["L0/attn_head_gate"]).astype(jnp.float32)),
        np.asarray(out["L0/attn_gated"].astype(jnp.float32)),
        rtol=1e-2,
        atol=1e-2,
    )


def test_torch_order_core_masks_outside_the_window(setup):
    _, _, bf16 = setup
    q, k, v = bf16((1, S, 4, 12)), bf16((1, S, 2, 12)), bf16((1, S, 2, 12))
    base = lev.torch_order_core(q, k, v, 4)
    moved = lev.torch_order_core(q, k.at[:, 0].set(5.0), v.at[:, 0].set(5.0), 4)
    np.testing.assert_array_equal(np.asarray(base[:, 4:]), np.asarray(moved[:, 4:]))
    assert not np.array_equal(np.asarray(base[:, :4]), np.asarray(moved[:, :4]))


def test_extended_reference_matches_the_module_views(setup):
    attn, _, bf16 = setup
    v, q_rope = np.asarray(bf16((S, 2, 12)), dtype=np.float32), np.asarray(bf16((S, 4, 12)), dtype=np.float32)
    ref = lev.extended_reference({"a": {"L0/v_proj": v, "L0/q_rope": q_rope, "L0/k_rope": v}}, attn)["a"]
    expanded = align_kv_heads(jnp.asarray(v[None]), num_q_heads=4)[0]
    np.testing.assert_array_equal(ref["L0/aligned_v"], np.asarray(expanded))
    np.testing.assert_array_equal(ref["L0/q_rope_rotated"], q_rope[..., :6])


def test_measure_end_to_end(setup, tmp_path):
    """Every mode runs on fake torch ops and arm dumps; steps fed the same input as torch read 0 differing, and the
    replay of the arm's own attn_norm against its own attn_out reads 0."""
    attn, _, bf16 = setup
    half = attn.cfg.inferred_head_dim // 2
    reference, levanter = {}, {}
    for sid, length in (("a", 5), ("b", 7)):
        x = bf16((1, length, 32))
        chain = {
            name: np.asarray(v.astype(jnp.float32))[0]
            for name, v in lev.tapped(attn, x, None).items()
            if name != "implementation"
        }
        ops = {
            name: chain[name]
            for name in (
                *lev.PROJECTIONS,
                *lev.NORMS,
                "L0/q_scaled",
                "L0/attn_head_gate",
                "L0/attn_gated",
                "L0/attn_out",
            )
        }
        ops["L0/attn_norm"] = np.asarray(x.astype(jnp.float32))[0]
        ops["L0/q_rope"] = np.concatenate([chain["L0/q_rope_rotated"], chain["L0/q_norm"][..., half:]], axis=-1)
        ops["L0/k_rope"] = np.concatenate([chain["L0/k_rope_rotated"], chain["L0/k_norm"][..., half:]], axis=-1)
        ops["L0/attn_core"] = ops["L0/attn_xsa"] = np.asarray(bf16((length, 4, 12)).astype(jnp.float32))
        reference[sid] = ops
        levanter[sid] = {
            "attn_norm/0": ops["L0/attn_norm"],
            "attn_out/0": np.asarray(lev.composite(attn, x, None).astype(jnp.float32))[0],
        }
    result = lev.measure(attn, reference, levanter, str(tmp_path), "default")
    modes = {"isolated", "isolated_xsa", "chain", "levanter_replay", *(f"core/{a}" for a in lev.CORE_ARMS)}
    assert modes | {f"composite/{n}" for n in lev.COMPOSITE_IMPLEMENTATIONS} <= set(result)
    assert result["isolated"]["L0/q_proj"]["differing"] == 0
    assert result["isolated"]["L0/q_proj"]["elements"] == 12 * 4 * 12
    assert result["levanter_replay_vs_arm"]["L0/attn_out"]["differing"] == 0
    assert result["levanter_attn_norm_vs_torch"]["L0/attn_norm"]["differing"] == 0
    records = result["records"]
    assert records["selected_implementation"] == ["None"]
    assert records["rope_cache_dtype"] == ["float32"]
    assert records["xla_flash_block_size"] == {"a": None, "b": None}
    assert set(records["source_facts"]) == {"module", "gpu_fa4_cute", "reference"}
    assert (tmp_path / "default-attn-composite-hlo.txt").read_text().startswith("HloModule")
