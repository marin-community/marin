# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from haliax import Axis
from haliax.nn.scan import Stacked


def init_scan_foldable(
    Layers: Axis,
    BlockType: Any,
    *init_args,
    scan_layers: bool,
    gradient_checkpointing: bool | str,
    key,
    **init_kwargs,
):
    """
    Initialize a scan-capable block container with a single call-site shape.

    This keeps stack selection (`Stacked` vs `BlockSeq`) centralized so models
    can migrate off Haliax scan containers in one place.
    """
    stack_cls = Stacked
    if not scan_layers:
        from haliax.nn.scan import BlockSeq

        stack_cls = BlockSeq

    return stack_cls.init(
        Layers,
        BlockType,
        gradient_checkpointing=gradient_checkpointing,
    )(
        *init_args,
        key=key,
        **init_kwargs,
    )
