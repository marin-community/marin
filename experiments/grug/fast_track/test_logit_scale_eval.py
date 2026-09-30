# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the logit-scale sweep (experiments/grug/fast_track/logit_scale_eval.py)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.grug.fast_track.logit_scale_eval import (
    apply_logit_soft_cap,
    logit_scale_sums,
    parse_logit_scale_grid,
    summarize_logit_scale_sums,
)


def _batch(seed: int, b: int = 3, s: int = 5, v: int = 7, t: int = 2):
    rng = np.random.default_rng(seed)
    logits = jnp.asarray(rng.standard_normal((b, s, v)).astype(np.float32) * 3)
    labels = jnp.asarray(rng.integers(0, v, size=(b, s)))
    weights = jnp.asarray((rng.random((b, s)) > 0.2).astype(np.float32))
    tags = jnp.asarray(np.eye(t, dtype=np.float32)[rng.integers(0, t, size=b)])
    return logits, labels, weights, tags


def test_unit_scale_loss_is_cross_entropy_and_sharper_logits_change_it():
    logits, labels, weights, tags = _batch(0)
    scales = (0.5, 1.0, 2.0)
    sums = logit_scale_sums(logits, labels, weights, tags, scales)
    ce = -jax.nn.log_softmax(logits, axis=-1)
    ce = jnp.take_along_axis(ce, labels[..., None], axis=-1)[..., 0]
    expected = jnp.einsum("bt,bt,bk->k", ce, weights, tags)
    np.testing.assert_allclose(sums["loss"][1], expected, rtol=1e-5)
    np.testing.assert_allclose(sums["tokens"], jnp.einsum("bt,bk->k", weights, tags))
    assert not np.allclose(sums["loss"][0], sums["loss"][1])
    assert not np.allclose(sums["loss"][2], sums["loss"][1])


def test_top1_and_rank_ignore_the_scale_and_count_correctly():
    logits, labels, weights, tags = _batch(1)
    scaled = logit_scale_sums(3.0 * logits, labels, weights, tags, (1.0,))
    plain = logit_scale_sums(logits, labels, weights, tags, (1.0,))
    np.testing.assert_allclose(scaled["top1"], plain["top1"])
    np.testing.assert_allclose(scaled["log2_rank"], plain["log2_rank"])
    # A batch whose label is always the top logit scores rank 0 (log2 1 = 0) and top-1 = every weighted token.
    labels = jnp.argmax(logits, axis=-1)
    perfect = logit_scale_sums(logits, labels, weights, tags, (1.0,))
    np.testing.assert_allclose(perfect["top1"], perfect["tokens"])
    np.testing.assert_allclose(perfect["log2_rank"], 0.0, atol=1e-6)


def test_soft_cap_forms():
    z = jnp.asarray([-50.0, 0.0, 50.0])
    np.testing.assert_allclose(apply_logit_soft_cap(z, None), z)
    np.testing.assert_allclose(apply_logit_soft_cap(z, 10.0), 10.0 * jnp.tanh(z / 10.0))
    np.testing.assert_allclose(apply_logit_soft_cap(z, (15.0, 5.0, 7.5)), 15.0 * jax.nn.sigmoid((z + 5.0) / 7.5))


def test_summary_selects_on_fit_half_and_reports_on_report_half():
    scales = (0.8, 0.9, 1.0, 1.1, 1.2)
    # Two tags; parent "paloma" holds tag 0 only. Fit half: the loss curve bottoms at 0.9 on paloma.
    fit_loss = np.array([[3.0, 2.8, 2.9, 3.1, 3.4], [5.0, 5.0, 5.0, 5.0, 5.0]]).T * 100.0
    rep_loss = np.array([[3.2, 3.0, 3.05, 3.3, 3.6], [4.0, 4.0, 4.0, 4.0, 4.0]]).T * 50.0
    fit = {
        "tokens": np.array([100.0, 100.0]),
        "loss": fit_loss,
        "top1": np.array([40.0, 10.0]),
        "log2_rank": np.zeros(2),
    }
    report = {
        "tokens": np.array([50.0, 50.0]),
        "loss": rep_loss,
        "top1": np.array([25.0, 5.0]),
        "log2_rank": np.zeros(2),
    }
    out = summarize_logit_scale_sums(fit, report, scales=scales, hierarchy={"paloma": [0], "other": [1]})
    assert out["best/scale"] == 0.9
    assert out["best/temperature"] == pytest.approx(1 / 0.9)
    assert out["unit/paloma/macro_loss"] == pytest.approx(3.05)
    assert out["best/paloma/macro_loss"] == pytest.approx(3.0)
    assert out["gain/paloma/macro_loss"] == pytest.approx(0.05)
    assert out["report/paloma/top1_acc"] == pytest.approx(0.5)
    assert out["fit/all/top1_acc"] == pytest.approx((0.4 + 0.1) / 2)
    # "all" is the mean of per-tag means, so the flat tag pulls the curve up but not the selection.
    assert out["best/all/macro_loss"] == pytest.approx((3.0 + 4.0) / 2)
    assert 0.85 < out["best/scale_refined"] < 0.95


def test_summary_rejects_grids_without_unit_scale():
    sums = {"tokens": np.ones(1), "loss": np.ones((2, 1)), "top1": np.ones(1), "log2_rank": np.ones(1)}
    with pytest.raises(ValueError, match=r"1\.0"):
        summarize_logit_scale_sums(sums, sums, scales=(0.9, 1.1), hierarchy={"paloma": [0]})


def test_parse_grid_includes_unit_scale():
    assert parse_logit_scale_grid("0.8:1.2:5") == (0.8, 0.9, 1.0, 1.1, 1.2)
    grid = parse_logit_scale_grid("0.85:1.25:5")
    assert 1.0 in grid and len(grid) == 6 and list(grid) == sorted(grid)
    with pytest.raises(ValueError):
        parse_logit_scale_grid("1.2:0.8:5")
