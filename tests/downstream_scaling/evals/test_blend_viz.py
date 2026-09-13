# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import math

import pytest
import torch

from experiments.downstream_scaling.evals.tools.blend_viz.scoring import score_decision_rows


def test_decision_row_indexing_and_exact_kl_direction():
    # prompt length 2 and three decision rows must select positions 1, 2, 3.
    logits_a = torch.log(
        torch.tensor(
            [
                [0.01, 0.99],  # unused position 0
                [0.75, 0.25],
                [0.10, 0.90],
                [0.80, 0.20],
                [0.01, 0.99],  # unused position 4
            ]
        )
    )
    logits_b = torch.log(
        torch.tensor(
            [
                [0.01, 0.99],
                [0.50, 0.50],
                [0.10, 0.90],
                [0.80, 0.20],
                [0.01, 0.99],
            ]
        )
    )

    scored = score_decision_rows(
        logits_a,
        logits_b,
        prompt_length=2,
        committed_ids=[0, 1, -1],
        eos_token_id=2,
        topk_store=2,
    )

    assert scored.a_topk_ids[:, 0].tolist() == [0, 1, 0]
    expected_a_b = 0.75 * math.log(0.75 / 0.50) + 0.25 * math.log(0.25 / 0.50)
    expected_b_a = 0.50 * math.log(0.50 / 0.75) + 0.50 * math.log(0.50 / 0.25)
    assert scored.kl_a_b[0] == pytest.approx(expected_a_b)
    assert scored.kl_a_b[0] != pytest.approx(expected_b_a)
    assert scored.kl_a_b[1:].tolist() == pytest.approx([0.0, 0.0])
