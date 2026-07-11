# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: PLC0415

from dataclasses import dataclass
from typing import Any

from cute_nvshmem_transport.addressing import ArenaLayout, ArenaShape, Protocol


@dataclass(frozen=True)
class ArenaSlice:
    """A byte range within a symmetric arena."""

    offset: int
    size: int


@dataclass
class SymmetricArena:
    """Own one collectively allocated byte-identical NVSHMEM arena per PE."""

    layout: ArenaLayout
    tensor: Any

    @classmethod
    def allocate(cls, shape: ArenaShape) -> "SymmetricArena":
        """Collectively allocate the fixed arena for the initialized PE team."""
        import cutlass.cute as cute
        import nvshmem.core.interop.cute as cute_interop

        layout = ArenaLayout.create(shape)
        tensor = cute_interop.tensor((layout.total_bytes,), dtype=cute.Uint8)
        return cls(layout=layout, tensor=tensor)

    def payload(self, protocol: Protocol, peer: int, slot: int, size: int | None = None) -> ArenaSlice:
        """Return a deterministic payload slice for one peer and pipeline slot."""
        payload_size = self.layout.shape.max_payload_bytes if size is None else size
        if not 0 <= payload_size <= self.layout.shape.max_payload_bytes:
            raise ValueError(f"size must be in [0, {self.layout.shape.max_payload_bytes}]")
        return ArenaSlice(self.layout.payload_offset(protocol, peer, slot), payload_size)

    def ready(self, protocol: Protocol, peer: int, slot: int) -> ArenaSlice:
        """Return the ready-epoch scalar for one protocol, peer, and slot."""
        return ArenaSlice(self.layout.epoch_offset(protocol, False, peer, slot), 8)

    def consumed(self, protocol: Protocol, peer: int, slot: int) -> ArenaSlice:
        """Return the consumed-epoch scalar for one protocol, peer, and slot."""
        return ArenaSlice(self.layout.epoch_offset(protocol, True, peer, slot), 8)

    def byte_count(self, protocol: Protocol, peer: int, slot: int) -> ArenaSlice:
        """Return the dynamic byte-count scalar for one protocol, peer, and slot."""
        return ArenaSlice(self.layout.byte_count_offset(protocol, peer, slot), 8)

    def free(self) -> None:
        """Collectively release the symmetric allocation."""
        import nvshmem.core.interop.cute as cute_interop

        cute_interop.free_tensor(self.tensor)

    def __enter__(self) -> "SymmetricArena":
        return self

    def __exit__(self, *_: object) -> None:
        self.free()
