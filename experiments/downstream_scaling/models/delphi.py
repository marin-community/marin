# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Delphi scaling-ladder checkpoint roots."""

DELPHI_CHECKPOINTS: dict[str, str] = {
    "3e18": "checkpoints/isoflop/isoflop-3e+18-d1024-L11-B8-adamh_scaling_v6/hf",
    "9e18": "checkpoints/isoflop/isoflop-9e+18-d1152-L12-B16-adamh_scaling_v6/hf",
    "2e19": "checkpoints/isoflop/isoflop-2e+19-d1408-L15-B16-adamh_scaling_v6/hf",
    "3e19": "checkpoints/isoflop/isoflop-3e+19-d1536-L16-B32-adamh_scaling_v6/hf",
    "9e19": "checkpoints/isoflop/isoflop-9e+19-d1792-L18-B64-adamh_scaling_v6/hf",
    "2e20": "checkpoints/isoflop/isoflop-2e+20-d2048-L21-B64-adamh_scaling_v6/hf",
    "3e20": "checkpoints/isoflop/isoflop-3e+20-d2304-L23-B128-adamh_scaling_v6/hf",
    "1e21": "adamh-scaling-ladder-nemotron-optimal-1e+21-v5-019021/hf",
    "1e22": "adamh-scaling-ladder-nemotron-optimal-1e+22-v5-025b0e/hf",
    "1e23": "adamh-scaling-ladder-nemotron-optimal-1e+23-v5-27f2fb/hf",
}
