# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Block AttnRes in grug fast_track, combined with MLA + Inkling relative-position attention on the global
layers and either sliding-window attention or KDA on the local layers."""

import dataclasses
import functools
import math
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import pytest
from jax.sharding import AxisType, Mesh
from jax.sharding import PartitionSpec as P
from levanter.data.text.examples import GrugLmExample
from levanter.grug.attention import AttentionMask
from levanter.testing.cpu_devices import run_on_cpu_devices

from experiments.grug.fast_track import model as model_module
from experiments.grug.fast_track.heuristic import MoeHeuristic
from experiments.grug.fast_track.model import (
    AttnResLayerBackward,
    GrugModelConfig,
    KimiDeltaAttention,
    LocalMixer,
    Transformer,
)
from experiments.grug.fast_track.train import _loss_and_grads, _make_train_step, initial_state
from experiments.grug.moe.kda import chunk_kda

_HIDDEN = 64
_SEQ = 128  # the banded Inkling bias needs a multiple of REL_BIAS_BLOCK
_BATCH = 2


def _config(
    *,
    dense: bool,
    num_layers: int,
    num_blocks: int,
    local_mixer: LocalMixer = LocalMixer.KDA,
    layer_backward: AttnResLayerBackward = AttnResLayerBackward.RECOMPUTE,
    remat_attention: bool = False,
) -> GrugModelConfig:
    """Tiny MLA + Inkling + AttnRes config mirroring the KMA recipe's switches."""
    return GrugModelConfig(
        vocab_size=128,
        hidden_dim=_HIDDEN,
        intermediate_dim=3 * _HIDDEN if dense else _HIDDEN // 2,
        shared_expert_intermediate_dim=_HIDDEN // 2,
        num_shared_experts=1,
        num_experts=4,
        num_experts_per_token=2,
        latent_dim=None if dense else _HIDDEN // 2,
        dense_mlp=dense,
        num_layers=num_layers,
        num_heads=2,
        num_kv_heads=2,
        local_kv_heads=None,
        global_kv_heads=None,
        head_dim=16,
        max_seq_len=_SEQ,
        # >= seq so the CPU reference attention and the model see the same (full causal) mask.
        sliding_window=_SEQ,
        global_every=2,
        capacity_factor=8.0,
        pooled_transport_capacity_factor=8.0,
        initializer_std=0.5 / math.sqrt(_HIDDEN),
        qk_norm=False,
        qk_mult=1.0,
        sconv_sites=("attn", "mlp"),
        mla=True,
        mla_kv_latent_dim=32,
        inkling_relpos=True,
        rel_dim=4,
        rel_extent=8,
        attn_res=True,
        attn_res_num_blocks=num_blocks,
        attn_res_layer_backward=layer_backward,
        attn_res_remat_attention=remat_attention,
        local_mixer=local_mixer,
    )


@pytest.fixture
def fp32_kda_kernel(monkeypatch):
    """Run the XLA chunked KDA kernel's intra-chunk GEMMs in fp32 (its default is bf16, ~0.5% relative
    error by design), so the parity test sees the fp32 rounding differences and not bf16 re-rounding."""
    monkeypatch.setattr(model_module, "chunk_kda", functools.partial(chunk_kda, matmul_dtype=jnp.float32))


@pytest.fixture
def mesh() -> Mesh:
    return Mesh(
        np.array(jax.devices()[:1], dtype=object).reshape((1, 1, 1, 1)),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )


def _randomize_queries(model: Transformer, key: jax.Array) -> Transformer:
    """Pseudo-queries are zero-initialized (uniform depth mixing); give them non-trivial values."""
    leaves, treedef = jax.tree_util.tree_flatten_with_path(model)
    keys = iter(jax.random.split(key, len(leaves)))
    new_leaves = []
    for path, leaf in leaves:
        if "attn_res_query" in jax.tree_util.keystr(path):
            leaf = leaf + 0.3 * jax.random.normal(next(keys), leaf.shape, leaf.dtype)
        new_leaves.append(leaf)
    return jax.tree_util.tree_unflatten(treedef, new_leaves)


def _depth_attention(sources: list[jax.Array], query: jax.Array, eps: float) -> jax.Array:
    """softmax_s(q . rms_norm(x_s)) weighted sum of the sources (paper eq., parameter-free key norm)."""
    stacked = jnp.stack([s.astype(jnp.float32) for s in sources])  # [N, B, S, D]
    keys = stacked * jax.lax.rsqrt(jnp.mean(stacked**2, axis=-1, keepdims=True) + eps)
    weights = jax.nn.softmax(jnp.einsum("nbsd,d->nbs", keys, query), axis=0)
    return jnp.einsum("nbs,nbsd->bsd", weights, stacked).astype(sources[0].dtype)


def _sliced_layers(model: Transformer) -> list:
    """Every layer in layer order, sliced out of its stack."""
    by_index = {}
    for stack, indices in zip(model.layer_stacks(), model.stack_layer_indices(), strict=True):
        for j, i in enumerate(indices):
            by_index[i] = jax.tree.map(lambda x, j=j: x[j], stack.stacked)
    return [by_index[i] for i in range(model.config.num_layers)]


def _reference_hidden(model: Transformer, tokens: jax.Array, boundary_coda_blend: jax.Array | None = None) -> jax.Array:
    """Straight-line Block AttnRes: plain autodiff, full source list, no custom VJP or precomputed logits."""
    cfg = model.config
    eps = cfg.layer_norm_eps
    mask = AttentionMask.causal()
    embedded = jax.sharding.reshard(model.token_embed[tokens], P(("replica_dcn", "data", "expert"), None, None))
    hidden = model.embed_gated_norm(model.embed_norm(embedded))
    seg_size = max(1, cfg.num_layers // cfg.attn_res_num_blocks)
    bigram = None
    if model.token_embed2 is not None:
        assert model.embed2_norm is not None
        ids = model_module._bigram_hash_ids(tokens, None, cfg.embed2_rows, cfg.embed2_ngram)
        bigram = model.embed2_norm(
            jax.sharding.reshard(model.token_embed2[ids], P(("replica_dcn", "data", "expert"), None, None))
        )
        if cfg.bigram_gate:
            assert model.bigram_gate_w is not None and model.bigram_gate_b is not None
            bigram, _ = model_module._content_gate(
                hidden,
                bigram,
                model.bigram_gate_w,
                model.bigram_gate_b,
                model.bigram_gate_a_lr,
                model.bigram_gate_b_lr,
            )
    blocks: list[jax.Array] = [] if bigram is None else [bigram]
    partial: jax.Array | None = hidden
    prelude_output = None
    prelude_len = cfg.num_layers // 3
    core_end = prelude_len + cfg.num_layers // 3 + int(cfg.num_layers % 3 > 0)
    for i, layer in enumerate(_sliced_layers(model)):
        if cfg.boundary_alpha is not None and i == prelude_len:
            assert partial is not None
            prelude_output = sum(source.astype(jnp.float32) for source in [*blocks, partial]).astype(hidden.dtype)
            summary = (cfg.boundary_alpha * prelude_output).astype(hidden.dtype)
            blocks = [bigram] if cfg.boundary_preserve_bigram else []
            partial = summary - bigram if cfg.boundary_preserve_bigram else summary
        if cfg.boundary_alpha is not None and i == core_end:
            assert partial is not None and prelude_output is not None
            core_output = sum(source.astype(jnp.float32) for source in [*blocks, partial]).astype(hidden.dtype)
            summary = (
                core_output.astype(jnp.float32)
                * jax.lax.rsqrt(jnp.mean(core_output.astype(jnp.float32) ** 2, axis=-1, keepdims=True) + eps)
                + cfg.boundary_alpha * prelude_output
            ).astype(hidden.dtype)
            if boundary_coda_blend is not None:
                summary = ((1 - boundary_coda_blend) * core_output + boundary_coda_blend * summary).astype(hidden.dtype)
            blocks = [bigram] if cfg.boundary_preserve_bigram else []
            partial = summary - bigram if cfg.boundary_preserve_bigram else summary
        if i % seg_size == 0 and i // seg_size < cfg.attn_res_num_blocks:
            assert partial is not None
            blocks.append(partial)
            partial = None
        use_long = (i + 1) % cfg.global_every == 0 or i == cfg.num_layers - 1
        live = blocks if partial is None else [*blocks, partial]
        attn_out, _ = layer.attn_branch(_depth_attention(live, layer.attn_res_query_attn, eps), mask, use_long, use_long)
        partial = attn_out if partial is None else partial + attn_out
        mlp_out, _ = layer.mlp_branch(_depth_attention([*blocks, partial], layer.attn_res_query_mlp, eps), mask)
        partial = partial + mlp_out
    assert partial is not None
    hidden = _depth_attention([*blocks, partial], model.attn_res_query_final, eps)
    return model.final_gated_norm(model.final_norm(hidden))


def _assert_matches_reference(
    cfg: GrugModelConfig, mesh: Mesh, boundary_coda_blend: jax.Array | None = None
) -> Transformer:
    """Loss and every gradient of the model match the straight-line reference, and the AttnRes queries,
    Inkling weights and KDA mixer weights all receive gradient."""
    with jax.set_mesh(mesh):
        model = _randomize_queries(Transformer.init(cfg, key=jax.random.key(0)), jax.random.key(1))
        tokens = jax.random.randint(jax.random.key(2), (_BATCH, _SEQ), 0, cfg.vocab_size)
        cotangent = jax.random.normal(jax.random.key(3), (_BATCH, _SEQ, cfg.hidden_dim), jnp.float32)

        def model_loss(m):
            return jnp.sum(m(tokens, boundary_coda_blend=boundary_coda_blend)[0].astype(jnp.float32) * cotangent)

        def reference_loss(m):
            return jnp.sum(_reference_hidden(m, tokens, boundary_coda_blend).astype(jnp.float32) * cotangent)

        loss, grads = eqx.filter_jit(eqx.filter_value_and_grad(model_loss))(model)
        ref_loss, ref_grads = eqx.filter_jit(eqx.filter_value_and_grad(reference_loss))(model)

    # The loss sums B*S*D terms of either sign, so it can nearly cancel; bound its error per term.
    np.testing.assert_allclose(loss, ref_loss, rtol=1e-5, atol=1e-7 * cotangent.size)
    flat = jax.tree_util.tree_flatten_with_path(eqx.filter(grads, eqx.is_array))[0]
    ref_flat = jax.tree_util.tree_leaves(eqx.filter(ref_grads, eqx.is_array))
    assert len(flat) == len(ref_flat)
    for (path, grad), ref_grad in zip(flat, ref_flat, strict=True):
        name = jax.tree_util.keystr(path)
        scale = float(jnp.max(jnp.abs(ref_grad))) + 1e-12
        max_abs = float(jnp.max(jnp.abs(grad - ref_grad)))
        assert max_abs / scale < 1e-4, f"{name}: max|diff|={max_abs:.3e} (scale {scale:.3e})"
    # Grouped by leaf name: layer 0's attention gate sees a single source (the embedding), so its own
    # query gradient is exactly zero.
    grad_max: dict[str, float] = {}
    for path, grad in flat:
        name = jax.tree_util.keystr(path)
        if "attn_res_query" in name or ".rel_pos." in name or ("kda_blocks" in name and ".attn." in name):
            leaf = name.rsplit(".", 1)[-1]
            grad_max[leaf] = max(grad_max.get(leaf, 0.0), float(jnp.max(jnp.abs(grad))))
    assert grad_max and all(value > 0 for value in grad_max.values()), grad_max
    return grads


@pytest.mark.parametrize("local_mixer", list(LocalMixer))
@pytest.mark.parametrize("dense", [True, False], ids=["dense", "moe"])
@pytest.mark.parametrize(
    "num_layers,num_blocks,layer_backward,remat_attention",
    [
        (6, 3, AttnResLayerBackward.RECOMPUTE, False),
        (4, 8, AttnResLayerBackward.RECOMPUTE, False),
        (5, 8, AttnResLayerBackward.SAVE, False),
        (6, 3, AttnResLayerBackward.RECOMPUTE, True),
    ],
    ids=["L6N3", "L4N8", "L5N8_save", "L6N3_remat_attention"],
)
def test_attn_res_mla_inkling_matches_straight_line_reference(
    mesh, fp32_kda_kernel, dense, num_layers, num_blocks, layer_backward, remat_attention, local_mixer
):
    cfg = _config(
        dense=dense,
        num_layers=num_layers,
        num_blocks=num_blocks,
        local_mixer=local_mixer,
        layer_backward=layer_backward,
        remat_attention=remat_attention,
    )
    if local_mixer == LocalMixer.KDA:
        # global_every=2: the even layers (and the last) are MLA softmax attention, the rest KDA.
        with jax.set_mesh(mesh):
            layers = _sliced_layers(Transformer.init(cfg, key=jax.random.key(0)))
        expected_kda = [i % 2 == 0 and i != num_layers - 1 for i in range(num_layers)]
        assert [isinstance(layer.attn, KimiDeltaAttention) for layer in layers] == expected_kda
    _assert_matches_reference(cfg, mesh)


@pytest.mark.parametrize("num_layers", [6, 8])
@pytest.mark.parametrize("layer_backward", list(AttnResLayerBackward))
def test_boundary_operator_matches_straight_line_reference(mesh, fp32_kda_kernel, num_layers, layer_backward):
    cfg = _config(dense=False, num_layers=num_layers, num_blocks=8, layer_backward=layer_backward)
    cfg = dataclasses.replace(cfg, boundary_alpha=1.0)
    _assert_matches_reference(cfg, mesh)


def test_boundary_operator_keeps_bigram_source_trainable(mesh, fp32_kda_kernel):
    cfg = dataclasses.replace(
        _config(dense=False, num_layers=6, num_blocks=8),
        boundary_alpha=1.0,
        second_embed=True,
        second_embed_bigram=True,
        embed2_rows=256,
        bigram_gate=True,
        bigram_gate_rank=4,
    )
    with jax.set_mesh(mesh):
        model = Transformer.init(cfg, key=jax.random.key(0))
        tokens = jax.random.randint(jax.random.key(1), (_BATCH, _SEQ), 0, cfg.vocab_size)
        cotangent = jax.random.normal(jax.random.key(2), (_BATCH, _SEQ, cfg.hidden_dim))

        def loss(m):
            return jnp.sum(m(tokens)[0].astype(jnp.float32) * cotangent)

        value, grads = eqx.filter_jit(eqx.filter_value_and_grad(loss))(model)

    assert np.isfinite(value)
    assert grads.token_embed2 is not None
    assert float(jnp.max(jnp.abs(grads.token_embed2))) > 0


@pytest.mark.parametrize("layer_backward", list(AttnResLayerBackward))
@pytest.mark.parametrize(
    "blend,preserve_bigram,gated_bigram",
    [
        (0.0, False, True),
        (0.5, False, True),
        (1.0, False, True),
        (None, True, True),
        (None, True, False),
        (0.5, True, True),
    ],
    ids=["warm_start", "warm_middle", "warm_end", "preserve_gated", "preserve_ungated", "combined"],
)
def test_boundary_initialization_ablations_match_reference(
    mesh, fp32_kda_kernel, layer_backward, blend, preserve_bigram, gated_bigram
):
    cfg = dataclasses.replace(
        _config(dense=False, num_layers=6, num_blocks=8, layer_backward=layer_backward),
        boundary_alpha=1.0,
        boundary_preserve_bigram=preserve_bigram,
        second_embed=True,
        second_embed_bigram=True,
        embed2_rows=256,
        bigram_gate=gated_bigram,
        bigram_gate_rank=4 if gated_bigram else 0,
    )
    grads = _assert_matches_reference(cfg, mesh, None if blend is None else jnp.asarray(blend))
    layers = _sliced_layers(grads)
    for index in [2, 4]:
        magnitude = float(jnp.max(jnp.abs(layers[index].attn_res_query_attn)))
        # Preserving the source restores a routing choice at the first core/coda attention gate.
        if preserve_bigram:
            assert magnitude > 0
        else:
            assert magnitude == 0


def test_boundary_coda_warmup_uses_current_step_and_full_operator_for_evaluation(mesh, fp32_kda_kernel):
    cfg = dataclasses.replace(
        _config(dense=True, num_layers=6, num_blocks=8), boundary_alpha=1.0, boundary_coda_warmup_steps=10
    )
    mp = jmp.get_policy("params=float32,compute=float32,output=float32")
    with jax.set_mesh(mesh):
        model = _randomize_queries(Transformer.init(cfg, key=jax.random.key(0)), jax.random.key(1))
        tokens = jax.random.randint(jax.random.key(2), (_BATCH, _SEQ), 0, cfg.vocab_size)
        batch = GrugLmExample(tokens=tokens, loss_weight=jnp.ones(tokens.shape), attn_mask=AttentionMask.causal())

        def scheduled_loss(m, step):
            return _loss_and_grads(m, batch, mp, None, step=step)

        def explicit_loss(m, blend):
            return m.next_token_loss(tokens, batch.loss_weight, boundary_coda_blend=blend)

        scheduled = eqx.filter_jit(scheduled_loss)
        explicit = eqx.filter_jit(eqx.filter_value_and_grad(explicit_loss))
        observed = []
        for step, blend in [(0, 0.0), (5, 0.5), (10, 1.0), (20, 1.0), (None, None)]:
            (value, _), grads = scheduled(model, None if step is None else jnp.asarray(step))
            expected, expected_grads = explicit(model, None if blend is None else jnp.asarray(blend))
            np.testing.assert_allclose(value, expected, rtol=1e-6, atol=1e-6)
            for grad, expected_grad in zip(jax.tree.leaves(grads), jax.tree.leaves(expected_grads), strict=True):
                np.testing.assert_allclose(grad, expected_grad, rtol=1e-5, atol=1e-6)
            observed.append(float(value))
        assert abs(observed[0] - observed[2]) > 1e-5
        np.testing.assert_allclose(observed[2:], observed[2], rtol=1e-6, atol=1e-6)


def test_kma_optimizer_groups(mesh):
    cfg = _config(dense=False, num_layers=4, num_blocks=2)
    with jax.set_mesh(mesh):
        model = Transformer.init(cfg, key=jax.random.key(0))
    optimizer = MoeHeuristic().build_optimizer_config(
        num_train_steps=100, batch_size=8, hidden_dim=cfg.hidden_dim, seq_len=_SEQ
    )
    mask = optimizer.create_mask(eqx.filter(model, eqx.is_array))
    labels = {jax.tree_util.keystr(path): label for path, label in jax.tree_util.tree_flatten_with_path(mask)[0]}

    def labels_where(predicate) -> set[str]:
        found = {label for name, label in labels.items() if predicate(name)}
        assert found, "no parameter matched"
        return found

    def kda_leaf(*leaves: str):
        return lambda name: "kda_blocks" in name and name.endswith(tuple(f".attn.{leaf}" for leaf in leaves))

    def mla_leaf(*leaves: str):
        return lambda name: "stacked_blocks" in name and name.endswith(tuple(f".attn.{leaf}" for leaf in leaves))

    # KDA local layers.
    assert labels_where(kda_leaf("w_q", "w_k", "w_v", "w_o", "w_g")) == {"muonh"}
    assert labels_where(kda_leaf("w_a_down", "w_a_up", "a_log", "dt_bias")) == {"adam"}
    assert labels_where(kda_leaf("w_beta")) == {"kda_beta"}
    assert labels_where(lambda n: "kda_blocks" in n and (".attn.sconv_" in n or ".attn.o_norm" in n)) == {"adam"}
    # MLA + Inkling global layers.
    assert labels_where(mla_leaf("w_q", "w_dkv", "w_uk", "w_uv", "w_o")) == {"muonh"}
    assert labels_where(lambda n: "kv_latent_norm" in n) == {"adam"}
    assert labels_where(lambda n: n.endswith(("rel_pos.r_proj", "rel_pos.proj"))) == {"adam"}
    assert labels_where(lambda n: n.endswith(".attn.attn_gate")) == {"adam"}
    # AttnRes pseudo-queries of both layer kinds, plus the final gate.
    assert labels_where(lambda n: "attn_res_query" in n) == {"attn_res_query"}
    assert labels_where(lambda n: n.endswith("attn_res_query_final")) == {"attn_res_query"}
    # Everything else keeps the base routing.
    assert labels_where(lambda n: n.endswith((".expert_mlp.w_gate", ".expert_mlp.w_up", ".expert_mlp.w_down"))) == {
        "muonh"
    }
    assert labels_where(lambda n: n.endswith(".output_proj")) == {"adamh"}


_TRAIN_STEPS = 3


def check_kma_train_steps() -> None:
    """A few trainer steps (forward, backward, optimizer update) of the KMA model with dense MLPs on packed
    documents: losses and params stay finite and every optimizer group moves its parameters.

    Needs a mesh axis of size > 1 (the MuonH Newton-Schulz stack shards over it), so it runs on two
    CPU devices from ``test_kma_train_steps_are_finite``; the MoE experts' distributed Newton-Schulz
    needs the GPU-only QuACK kernel, hence the dense MLPs.
    """
    mesh = Mesh(
        np.asarray(jax.devices()[:2]).reshape((1, 2, 1, 1)),
        ("replica_dcn", "data", "expert", "model"),
        axis_types=(AxisType.Explicit,) * 4,
    )
    cfg = _config(dense=True, num_layers=4, num_blocks=2)
    optimizer = MoeHeuristic().build_optimizer_config(
        num_train_steps=100, batch_size=8, hidden_dim=cfg.hidden_dim, seq_len=_SEQ
    )
    tx = optimizer.build(100)
    mp = jmp.get_policy("params=float32,compute=float32,output=float32")
    step = _make_train_step(tx, mp, z_loss_weight=1e-4)
    tokens = jax.random.randint(jax.random.key(1), (_BATCH, _SEQ), 0, cfg.vocab_size)
    segment_ids = jnp.asarray(np.repeat([0, 1], [_SEQ // 2 - 5, _SEQ // 2 + 5])[None].repeat(_BATCH, 0), jnp.int32)
    with jax.set_mesh(mesh):
        batch = GrugLmExample(
            tokens=tokens,
            loss_weight=jnp.ones(tokens.shape, jnp.float32),
            attn_mask=AttentionMask(is_causal=True, segment_ids=(segment_ids, segment_ids)),
        )
        state = initial_state(cfg, optimizer=tx, mp=mp, key=jax.random.key(0))
        before = eqx.filter(state.params, eqx.is_array)
        labels = optimizer.create_mask(before)
        before = jax.tree.map(np.asarray, before)
        for _ in range(_TRAIN_STEPS):  # step 0 runs at the warmup LR of 0
            state, metrics, _ = step(state, batch)
            assert np.isfinite(float(metrics["train/loss"]))

    after = jax.tree_util.tree_flatten_with_path(eqx.filter(state.params, eqx.is_array))[0]
    moved: dict[str, bool] = {}
    for (path, new), old, label in zip(
        after, jax.tree_util.tree_leaves(before), jax.tree_util.tree_leaves(labels), strict=True
    ):
        new = np.asarray(new)
        assert np.isfinite(new).all(), jax.tree_util.keystr(path)
        moved[label] = moved.get(label, False) or bool(np.abs(new - old).max() > 0)
    assert set(moved) == {"muonh", "adamh", "adam", "attn_res_query", "kda_beta"}, sorted(moved)
    assert all(moved.values()), moved


def test_kma_train_steps_are_finite():
    run_on_cpu_devices(
        f"import sys; sys.path.insert(0, {str(Path(__file__).parent)!r}); "
        "import test_fast_track_attn_res as t; t.check_kma_train_steps()",
        device_count=2,
    )
