# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from experiments.grug.fast_track.fact_probe import EMA, RAW, FactProbeRecord, FactSpan


def test_span_scores_the_predictions_of_its_tokens(tmp_path):
    tokens = np.arange(2 * 8).reshape(2, 8)
    weights = np.ones((2, 8), np.float32)
    weights[:, -1] = 0
    # Loss at position p is the prediction of token p + 1; encode p in the loss to check the offset.
    loss = np.tile(np.arange(8, dtype=np.float32), (2, 1)) + 100 * np.arange(2)[:, None]
    span = FactSpan(data_step=5, row=1, start=3, end=6)
    path = str(tmp_path / "fact_probe.npz")
    record = FactProbeRecord((span,), path)

    scalars = record.add(5, {RAW: {5: loss}}, {5: tokens}, {5: weights})

    out = np.load(path)
    np.testing.assert_array_equal(out["span_token"], tokens[1, 3:6])
    np.testing.assert_array_equal(out["raw_span_loss"][0], [102, 103, 104])
    assert np.isnan(out["ema_span_loss"]).all()
    np.testing.assert_allclose(out["raw_row_loss"][0, 0], [3.0, 103.0])
    assert scalars == {f"fact_probe/{RAW}/span0": 103.0, f"fact_probe/{RAW}/batch5": 53.0}

    record.add(6, {RAW: {5: loss + 1}, EMA: {5: loss}}, {5: tokens}, {5: weights})
    out = np.load(path)
    np.testing.assert_array_equal(out["probe_steps"], [5, 6])
    np.testing.assert_array_equal(out["ema_span_loss"][1], [102, 103, 104])


def test_span_parse_rejects_spans_without_a_predicting_position():
    assert FactSpan.parse("10:3:7:12") == FactSpan(10, 3, 7, 12)
    with pytest.raises(ValueError):
        FactSpan.parse("10:3:0:4")
