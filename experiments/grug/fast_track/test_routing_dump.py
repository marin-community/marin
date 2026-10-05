# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Expert-specialization tooling on a top-1-of-8 model: the routing-count dump, its analysis report, the
router-embedding tie (single-token and centroid, and its release) and the router bias seed."""

import json

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import numpy as np
import pytest
from levanter.grug.attention import AttentionMask

import experiments.grug.fast_track.test_kda_local as t
from experiments.grug.fast_track.analyze_routing import build_report, load_dump
from experiments.grug.fast_track.launch import RouterTieClass, router_tie_class_ids, router_tie_cluster_specs
from experiments.grug.fast_track.model import RouterCombine, _routers_by_layer, tie_routers
from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig
from experiments.grug.fast_track.train import (
    GrugTrainState,
    _release_router_ties,
    _router_tie_view,
    _routing_counts_step,
    write_routing_dump,
)

_TOP1_OF_8 = dict(num_experts=8, num_experts_per_token=1, intermediate_dim=64, router_combine=RouterCombine.SIGMOID_RAW)
_MP = jmp.get_policy("params=float32,compute=float32,output=float32")


def _tokens(high: int = t._VOCAB) -> jax.Array:
    return jax.random.randint(jax.random.PRNGKey(2), (2, t._SEQ), 0, high)


def _counts(model, tokens, segment_ids):
    cfg = model.config
    shape = (cfg.num_layers, cfg.vocab_size, cfg.num_experts)
    zeros = (jnp.zeros(shape, jnp.int32), jnp.zeros(shape, jnp.int32), jnp.zeros(shape, jnp.float32))
    betas = jnp.zeros((cfg.num_layers, cfg.num_experts))
    return [np.asarray(c) for c in _routing_counts_step(_MP)(model, betas, tokens, segment_ids, zeros)]


def test_routing_dump_counts_every_assignment_and_analysis_reads_it(tmp_path):
    mesh, model = t._model(**_TOP1_OF_8)
    tokens = _tokens()
    # Second sequence: two documents, then two padding positions.
    segments = np.zeros(tokens.shape, np.int32)
    segments[1, 10:] = 1
    segments[1, -2:] = -1
    with jax.set_mesh(mesh):
        current, previous, weighted = _counts(model, tokens, jnp.asarray(segments))
        mask = AttentionMask.causal().with_segment_ids(jnp.asarray(segments))
        selected, _ = eqx.filter_jit(lambda m: m.routing_assignments(tokens, mask))(model)
    cfg = model.config
    valid = int(np.sum(segments >= 0))
    assert current.sum() == cfg.num_layers * valid * cfg.num_experts_per_token
    # Previous-token pairs exist inside a document only: not at a sequence or document start, nor at padding.
    assert previous.sum() == cfg.num_layers * (valid - 3)
    assert weighted.sum() > 0
    # The counts are the per-token assignments, bincounted.
    sel, tok = np.asarray(selected)[..., 0], np.asarray(tokens)
    for layer in range(cfg.num_layers):
        ok = segments >= 0
        expected = np.bincount(tok[ok] * cfg.num_experts + sel[layer][ok], minlength=cfg.vocab_size * cfg.num_experts)
        np.testing.assert_array_equal(current[layer].reshape(-1), expected)

    path = str(tmp_path / "routing_step0.npz")
    write_routing_dump(path, (current, previous, weighted), cfg, np.asarray(model.layer_kinds()), 0, valid)
    other = str(tmp_path / "routing_step5.npz")
    write_routing_dump(other, (current, previous, weighted), cfg, np.asarray(model.layer_kinds()), 5, valid)
    dumps = [load_dump(path), load_dump(other)]
    report, data = build_report(dumps, [chr(97 + v % 26) for v in range(cfg.vocab_size)], compare=dumps[0], min_count=1)
    assert "## Per layer" in report and "## Across seeds" in report and "## Over time" in report
    assert len(data["experts"]) == cfg.num_layers * cfg.num_experts
    # A dump matched against itself gives every routed-to expert a partner with its own profile.
    for match in data["cross_seed"]:
        load = current[match["layer"]].sum(axis=0)
        for (a, _), sim in zip(match["pairs"], match["similarity"], strict=True):
            if load[a] > 0:
                assert sim > 1 - 1e-6


def test_router_embed_tie_replaces_only_the_tied_column_and_trains_the_embedding():
    tied_token = t._VOCAB - 1  # never in the batch, so its embedding row is trained only through the tie
    tokens = _tokens(high=tied_token)
    weight = jnp.ones(tokens.shape, jnp.float32)
    mesh, plain = t._model(**_TOP1_OF_8)
    _, tied = t._model(**_TOP1_OF_8, router_embed_tie=(f"2:5:{tied_token}",))
    with jax.set_mesh(mesh):
        before, after = _routers_by_layer(tied), _routers_by_layer(tie_routers(tied))
        grads = {
            name: eqx.filter_jit(eqx.filter_grad(lambda m: m.next_token_loss(tokens, weight)))(m)
            for name, m in (("plain", plain), ("tied", tied))
        }
    for layer in before:
        changed = np.any(np.asarray(before[layer]) != np.asarray(after[layer]), axis=0)
        assert changed.tolist() == [layer == 2 and e == 5 for e in range(8)]
    column = np.asarray(after[2][:, 5])
    expected = np.asarray(tied.router_tie_alpha[0] * tied.token_embed[tied_token])
    np.testing.assert_allclose(column, expected, rtol=1e-6)
    # alpha starts the tied column at the untied column's norm.
    np.testing.assert_allclose(np.linalg.norm(column), np.linalg.norm(np.asarray(before[2][:, 5])), rtol=1e-5)
    assert float(jnp.abs(grads["plain"].token_embed[tied_token]).max()) == 0.0
    assert float(jnp.abs(grads["tied"].token_embed[tied_token]).max()) > 0.0
    assert float(jnp.abs(grads["tied"].router_tie_alpha).max()) > 0.0
    # The replaced router column gets no gradient; the same column of an untied layer does.
    tied_router_grads = _routers_by_layer(grads["tied"])
    assert float(jnp.abs(tied_router_grads[2][:, 5]).max()) == 0.0
    assert float(jnp.abs(tied_router_grads[3][:, 5]).max()) > 0.0
    groups = GrugMoeMuonHConfig().create_mask(eqx.filter(tied, eqx.is_inexact_array))
    assert groups.router_tie_alpha == "adam"


def test_router_bias_seed_routes_the_seeded_token_to_its_expert():
    tokens = _tokens()
    seeded = int(np.asarray(tokens)[0, 0])
    mesh, plain = t._model(**_TOP1_OF_8)
    _, model = t._model(**_TOP1_OF_8, router_bias_seed=(f"1:6:{seeded}:50",))
    with jax.set_mesh(mesh):
        route = eqx.filter_jit(lambda m: m.routing_assignments(tokens)[0])
        sel, sel_plain = np.asarray(route(model))[..., 0], np.asarray(route(plain))[..., 0]
    tok = np.asarray(tokens)
    assert np.all(sel[1][tok == seeded] == 6)
    # Up to the seeded layer, only the seeded token's layer-1 routing moves.
    np.testing.assert_array_equal(sel[0], sel_plain[0])
    np.testing.assert_array_equal(sel[1][tok != seeded], sel_plain[1][tok != seeded])


_CENTROID = (t._VOCAB - 3, t._VOCAB - 2, t._VOCAB - 1)  # never in the batch: their rows train only through the tie


def test_router_embed_tie_centroid_is_the_mean_row_and_trains_every_row():
    tokens = _tokens(high=_CENTROID[0])
    weight = jnp.ones(tokens.shape, jnp.float32)
    mesh, tied = t._model(**_TOP1_OF_8, router_embed_tie=("*:3:" + "|".join(map(str, _CENTROID)),))
    with jax.set_mesh(mesh):
        before, after = _routers_by_layer(tied), _routers_by_layer(tie_routers(tied))
        grads = eqx.filter_jit(eqx.filter_grad(lambda m: m.next_token_loss(tokens, weight)))(tied)
    centroid = np.asarray(tied.token_embed)[list(_CENTROID)].mean(axis=0)
    assert tied.router_tie_alpha.shape == (tied.config.num_layers,)
    for layer in before:
        column = np.asarray(after[layer][:, 3])
        np.testing.assert_allclose(column, float(tied.router_tie_alpha[layer]) * centroid, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(np.linalg.norm(column), np.linalg.norm(np.asarray(before[layer][:, 3])), rtol=1e-5)
    for token in _CENTROID:
        assert float(jnp.abs(grads.token_embed[token]).max()) > 0.0


def test_router_tie_release_is_continuous_then_frees_the_column():
    tokens = _tokens(high=_CENTROID[0])
    weight = jnp.ones(tokens.shape, jnp.float32)
    ties = ("1:5:" + "|".join(map(str, _CENTROID)),)
    mesh, tied = t._model(**_TOP1_OF_8, router_embed_tie=ties, router_embed_tie_release_step=3)
    state = GrugTrainState(
        step=jnp.array(3), params=tied, master_params=None, ema_params=tied, opt_state=(), pending_qb_betas=jnp.zeros(())
    )
    with jax.set_mesh(mesh):

        def loss(model, active):
            return model.next_token_loss(tokens, weight, reduction="none", router_tie_active=active)

        released = _release_router_ties(state)
        before = eqx.filter_jit(loss)(tied, True)
        after = eqx.filter_jit(loss)(released.params, False)
        # Evals and routing dumps run the untied program; before the release they see the ties materialized.
        viewed = eqx.filter_jit(loss)(_router_tie_view(tied, 2), None)
        grads = eqx.filter_jit(eqx.filter_grad(lambda m: jnp.mean(loss(m, None))))(released.params)
    np.testing.assert_allclose(np.asarray(after), np.asarray(before), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.asarray(viewed), np.asarray(before), rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(
        np.asarray(_routers_by_layer(released.ema_params)[1]), np.asarray(_routers_by_layer(released.params)[1])
    )
    assert _router_tie_view(released.params, 3) is released.params
    # After the release the column trains on its own; the tie's embedding rows and alpha get no gradient.
    assert float(jnp.abs(_routers_by_layer(grads)[1][:, 5]).max()) > 0.0
    for token in _CENTROID:
        assert float(jnp.abs(grads.token_embed[token]).max()) == 0.0
    assert float(jnp.abs(grads.router_tie_alpha).max()) == 0.0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (dict(router_embed_tie=("1:2",)), "'L:E:V' or 'L:E:V1|V2"),
        (dict(router_embed_tie=("1:2:3|x",)), "must be integers"),
        (dict(router_embed_tie=("1:2:3||4",)), "must be integers"),
        (dict(router_embed_tie=("1:2:3|3",)), "repeat"),
        (dict(router_embed_tie=(f"1:2:3|{t._VOCAB}",)), "out of range"),
        (dict(router_embed_tie=("9:2:3",)), "layer must be"),
        (dict(router_embed_tie_release_step=10), "needs router_embed_tie"),
        (dict(router_embed_tie=("1:2:3",), router_embed_tie_release_step=0), "must be positive"),
    ],
)
def test_router_embed_tie_spec_errors(overrides, message):
    with pytest.raises(ValueError, match=message):
        t._model(**_TOP1_OF_8, **overrides)


def test_router_tie_classes():
    strings = ["<special 0>", " 12", "3", "\\", "\\frac", "mathbf", " mathbf", "\n", " \n\n", " the", "1a"]
    ids = {c: router_tie_class_ids(c, strings) for c in RouterTieClass}
    assert ids[RouterTieClass.DIGITS] == (1, 2)
    assert ids[RouterTieClass.LATEX] == (3, 4, 5, 6)
    assert ids[RouterTieClass.NEWLINE] == (7, 8)


def test_router_tie_cluster_specs_tie_expert_e_to_cluster_e(tmp_path):
    path = tmp_path / "clusters.json"
    path.write_text(json.dumps([[5, 7], [1], [2, 3, 4]]))
    specs = router_tie_cluster_specs(f"*:{path}")
    assert specs == ("*:0:5|7", "*:1:1", "*:2:2|3|4")
    assert router_tie_cluster_specs(None) == ()
