# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import io
import json

import pytest
from marin.evaluation.likelihood import ScoringClient

from experiments.evals.vep.paced import PacedClient
from experiments.evals.vep.targets import Target


def test_changed_sentinel_stops_before_more_scores(monkeypatch):
    values = iter([-0.5, -0.5, -0.4])
    monkeypatch.setattr(ScoringClient, "score", lambda *args: {"logprob": next(values)})
    evidence = io.StringIO()
    client = PacedClient(None, "unused", "fixture", io.StringIO(), 100, 900, guard_log=evidence)
    client.sentinels = [Target("fixture", "null", "+", "none", "reference", "", "ACGT", False)]
    client.check_sentinels()
    client.check_sentinels()
    with pytest.raises(RuntimeError, match="drifted"):
        client.check_sentinels()
    assert [json.loads(line)["logprob"] for line in evidence.getvalue().splitlines()] == [-0.5, -0.5, -0.4]
