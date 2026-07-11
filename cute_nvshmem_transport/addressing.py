# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from enum import IntEnum

UINT64_BYTES = 8


class Protocol(IntEnum):
    PUSH = 0
    PULL = 1


@dataclass(frozen=True)
class ArenaShape:
    num_pes: int
    num_slots: int
    max_payload_bytes: int

    def __post_init__(self) -> None:
        if self.num_pes < 2:
            raise ValueError("num_pes must be at least two")
        if self.num_slots < 1:
            raise ValueError("num_slots must be positive")
        if self.max_payload_bytes < 1:
            raise ValueError("max_payload_bytes must be positive")


@dataclass(frozen=True)
class ArenaLayout:
    shape: ArenaShape
    push_inbox: int
    pull_source: int
    push_ready: int
    push_consumed: int
    pull_ready: int
    pull_consumed: int
    payload_bytes: int
    total_bytes: int

    @classmethod
    def create(cls, shape: ArenaShape) -> "ArenaLayout":
        payload_region_bytes = shape.num_pes * shape.num_slots * shape.max_payload_bytes
        epoch_region_bytes = shape.num_pes * shape.num_slots * UINT64_BYTES
        byte_count_region_bytes = 2 * shape.num_pes * shape.num_slots * UINT64_BYTES

        push_inbox = 0
        pull_source = push_inbox + payload_region_bytes
        push_ready = pull_source + payload_region_bytes
        push_consumed = push_ready + epoch_region_bytes
        pull_ready = push_consumed + epoch_region_bytes
        pull_consumed = pull_ready + epoch_region_bytes
        payload_bytes = pull_consumed + epoch_region_bytes
        total_bytes = payload_bytes + byte_count_region_bytes
        return cls(
            shape=shape,
            push_inbox=push_inbox,
            pull_source=pull_source,
            push_ready=push_ready,
            push_consumed=push_consumed,
            pull_ready=pull_ready,
            pull_consumed=pull_consumed,
            payload_bytes=payload_bytes,
            total_bytes=total_bytes,
        )

    def payload_offset(self, protocol: Protocol, peer: int, slot: int) -> int:
        self._validate_peer_slot(peer, slot)
        base = self.push_inbox if protocol is Protocol.PUSH else self.pull_source
        return base + (peer * self.shape.num_slots + slot) * self.shape.max_payload_bytes

    def epoch_offset(self, protocol: Protocol, consumed: bool, peer: int, slot: int) -> int:
        self._validate_peer_slot(peer, slot)
        bases = {
            (Protocol.PUSH, False): self.push_ready,
            (Protocol.PUSH, True): self.push_consumed,
            (Protocol.PULL, False): self.pull_ready,
            (Protocol.PULL, True): self.pull_consumed,
        }
        return bases[(protocol, consumed)] + (peer * self.shape.num_slots + slot) * UINT64_BYTES

    def byte_count_offset(self, protocol: Protocol, peer: int, slot: int) -> int:
        self._validate_peer_slot(peer, slot)
        index = (int(protocol) * self.shape.num_pes + peer) * self.shape.num_slots + slot
        return self.payload_bytes + index * UINT64_BYTES

    def _validate_peer_slot(self, peer: int, slot: int) -> None:
        if not 0 <= peer < self.shape.num_pes:
            raise ValueError(f"peer must be in [0, {self.shape.num_pes})")
        if not 0 <= slot < self.shape.num_slots:
            raise ValueError(f"slot must be in [0, {self.shape.num_slots})")
