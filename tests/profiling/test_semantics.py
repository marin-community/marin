# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from marin.profiling.semantics import estimate_flop_proxy


def test_estimate_flop_proxy_attention_handles_nonpositive_dims() -> None:
    assert estimate_flop_proxy("attention_splash", "0,8,2048,64|0,8,2048,64") is None


def test_estimate_flop_proxy_loss_handles_nonpositive_dims() -> None:
    assert estimate_flop_proxy("loss_xent", "0,512|0,128256") is None
