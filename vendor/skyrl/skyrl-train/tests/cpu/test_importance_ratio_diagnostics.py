"""Mismatch statistics retain response positions and generating-policy ages."""

import pytest
import torch

from skyrl_train.utils.importance_ratio_diagnostics import mismatch_ratio_metrics


def test_mismatch_keeps_unknown_ages_and_long_response_tails_separate():
    response = torch.ones((3, 4096), dtype=torch.bool)
    response[0, 2:] = False
    mask = torch.zeros_like(response)
    mask[0, :2] = True
    mask[1, [0, 1024, 4095]] = True
    mask[2, [0, 4095]] = True
    delta = torch.full(response.shape, torch.nan, dtype=torch.float64)
    delta[0, :2] = 0.1
    delta[1, [0, 1024, 4095]] = torch.tensor([0.2, -0.4, 0.6], dtype=torch.float64)
    delta[2, 4095] = -2.0
    ages = torch.full(response.shape, -1)
    ages[0] = 0
    ages[1] = 3
    metrics = mismatch_ratio_metrics(delta, torch.zeros_like(delta), mask, ages, response)
    prefix = "policy/mismatch/"
    assert metrics[prefix + "pooled/selected_tokens"] == 7
    assert metrics[prefix + "pooled/finite_tokens"] == 6
    assert metrics[prefix + "staleness0/selected_tokens"] == 2
    assert metrics[prefix + "staleness0/log_ratio_abs_mean"] == pytest.approx(0.1)
    assert metrics[prefix + "unknown/selected_tokens"] == 2
    assert metrics[prefix + "unknown/finite_tokens"] == 1
    assert metrics[prefix + "unknown/response_ge2048/log_ratio_abs_mean"] == 2.0
    assert metrics[prefix + "staleness3/response_ge2048/selected_tokens"] == 3
    assert metrics[prefix + "staleness3/response_ge2048/log_ratio_abs_mean"] == pytest.approx(0.4)
    assert metrics[prefix + "staleness3/pos_last256/selected_tokens"] == 1
    assert metrics[prefix + "staleness0/response_ge2048/selected_tokens"] == 0
