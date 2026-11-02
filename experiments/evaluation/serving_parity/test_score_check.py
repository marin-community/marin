# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy

import pytest

from experiments.evaluation.serving_parity.score_check import check_scores


def test_score_check(capsys):
    h100 = dict(
        hardware="h100",
        model="step12",
        revision="fixed",
        items=[dict(id=str(i), sha256=f"document-{i}", correct=int(i < 89)) for i in range(100)],
    )
    tpu = copy.deepcopy(h100)
    tpu["hardware"] = "v4"
    assert check_scores(h100, tpu)
    for i in range(84, 89):
        tpu["items"][i]["correct"] = 0
    assert check_scores(h100, tpu)  # Exactly five percentage points is accepted.
    tpu["items"][83]["correct"] = 0
    assert not check_scores(h100, tpu)
    assert "FAIL: H100=0.890, TPU=0.830" in capsys.readouterr().out
    tpu["items"][0]["sha256"] = "different-document"
    with pytest.raises(ValueError, match="different GSM8K items"):
        check_scores(h100, tpu)
    tpu["items"].pop()
    with pytest.raises(ValueError, match="100 unique"):
        check_scores(h100, tpu)
