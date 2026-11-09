"""Mismatch windows use full response length; long responses have a separate summary."""

import pytest
import torch
from skyrl_train.utils.importance_ratio_diagnostics import mismatch_ratio_metrics


def test_last_window_excludes_opening_model_tokens_before_masked_tool_output():
    response = torch.ones((1, 1200), dtype=torch.bool)
    mask = torch.zeros_like(response)
    mask[:, :100] = mask[:, 1100:] = True
    delta = torch.full(response.shape, torch.nan, dtype=torch.float64)
    delta[:, :100], delta[:, 1100:] = 0.1, 1.0
    metrics = mismatch_ratio_metrics(delta, torch.zeros_like(delta), mask, torch.zeros(response.shape, dtype=torch.int32), response)
    prefix = "policy/mismatch/pooled/"
    assert metrics[prefix + "pos_first256/selected_tokens"] == 100
    assert metrics[prefix + "pos_first256/log_ratio_abs_mean"] == pytest.approx(0.1)
    assert metrics[prefix + "pos_last256/selected_tokens"] == 100
    assert metrics[prefix + "pos_last256/log_ratio_abs_mean"] == pytest.approx(1.0)


def test_long_response_measurement_keeps_selected_and_finite_counts_separate():
    response = torch.ones((2, 2048), dtype=torch.bool)
    response[1, -1] = False  # The second response is below the 2048-token threshold.
    mask = torch.zeros_like(response)
    mask[0, [0, 2047]], mask[1, [0, 2046]] = True, True
    delta = torch.full(response.shape, torch.nan, dtype=torch.float64)
    delta[0, 0] = 0.2
    delta[1, [0, 2046]] = torch.tensor([0.4, 0.6], dtype=torch.float64)
    ages = torch.zeros(response.shape, dtype=torch.int32)
    ages[0] = 3
    metrics = mismatch_ratio_metrics(delta, torch.zeros_like(delta), mask, ages, response)
    prefix = "policy/mismatch/"
    assert metrics[prefix + "pooled/response_ge2048/selected_tokens"] == 2
    assert metrics[prefix + "pooled/response_ge2048/finite_tokens"] == 1
    assert metrics[prefix + "pooled/response_ge2048/log_ratio_abs_mean"] == pytest.approx(0.2)
    assert metrics[prefix + "staleness3/response_ge2048/selected_tokens"] == 2
    assert metrics[prefix + "staleness0/response_ge2048/selected_tokens"] == 0
