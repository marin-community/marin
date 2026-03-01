# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from haliax import Axis


def _scan_stack_cls(scan_layers: bool):
    from haliax.nn.scan import BlockSeq, Stacked

    if scan_layers:
        return Stacked
    return BlockSeq


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
    stack_cls = _scan_stack_cls(scan_layers)

    return stack_cls.init(
        Layers,
        BlockType,
        gradient_checkpointing=gradient_checkpointing,
    )(
        *init_args,
        key=key,
        **init_kwargs,
    )


def init_blockseq_from_blocks(
    blocks: list[Any],
    Layers: Axis,
    *,
    gradient_checkpointing: bool | str,
):
    """Construct a `BlockSeq` with centralized checkpoint-policy resolution."""
    from haliax.nn.scan import BlockSeq, ScanCheckpointPolicy

    return BlockSeq(blocks, Layers, ScanCheckpointPolicy._mk(gradient_checkpointing))
