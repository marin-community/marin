# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Storage defaults for disposable CoreWeave Grug MoE experiment launchers."""

from __future__ import annotations

import os

CW_GRUG_MOE_MARIN_PREFIX = "s3://marin-na/tmp/ttl=7d"
_STABLE_R2_ROOT_PREFIXES = {"s3://marin-na/marin", "s3://marin-na/marin/"}


def set_default_cw_grug_moe_prefix() -> None:
    """Default disposable CoreWeave runs to the R2 7-day TTL prefix."""
    if "MARIN_PREFIX" not in os.environ or os.environ["MARIN_PREFIX"] in _STABLE_R2_ROOT_PREFIXES:
        os.environ["MARIN_PREFIX"] = CW_GRUG_MOE_MARIN_PREFIX
