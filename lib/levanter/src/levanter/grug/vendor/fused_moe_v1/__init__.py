# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Vendored TPU fused MoE reference kernel used for benchmark experiments."""

from levanter.grug.vendor.fused_moe_v1.kernel import fused_ep_moe

__all__ = ["fused_ep_moe"]
