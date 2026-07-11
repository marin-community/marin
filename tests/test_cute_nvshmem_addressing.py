# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from cute_nvshmem_transport.addressing import ArenaLayout, ArenaShape, Protocol
from cute_nvshmem_transport.signals import final_slot_epochs, slot_epoch
from cute_nvshmem_transport.symmetric_arena import SymmetricArena


def test_arena_layout_regions_do_not_overlap() -> None:
    layout = ArenaLayout.create(ArenaShape(num_pes=8, num_slots=4, max_payload_bytes=6144))

    offsets = [
        layout.push_inbox,
        layout.pull_source,
        layout.push_ready,
        layout.push_consumed,
        layout.pull_ready,
        layout.pull_consumed,
        layout.payload_bytes,
        layout.total_bytes,
    ]
    assert offsets == sorted(set(offsets))


def test_arena_layout_offsets_are_deterministic_and_bounded() -> None:
    layout = ArenaLayout.create(ArenaShape(num_pes=2, num_slots=2, max_payload_bytes=128))

    assert layout.payload_offset(Protocol.PUSH, 1, 1) == 384
    assert layout.payload_offset(Protocol.PULL, 1, 1) == 896
    assert layout.epoch_offset(Protocol.PULL, True, 1, 1) == 1144
    assert layout.byte_count_offset(Protocol.PULL, 1, 1) == 1208
    assert layout.byte_count_offset(Protocol.PULL, 1, 1) + 8 == layout.total_bytes


@pytest.mark.parametrize(("peer", "slot"), [(-1, 0), (2, 0), (0, -1), (0, 2)])
def test_arena_layout_rejects_out_of_range_indices(peer: int, slot: int) -> None:
    layout = ArenaLayout.create(ArenaShape(num_pes=2, num_slots=2, max_payload_bytes=128))

    with pytest.raises(ValueError):
        layout.payload_offset(Protocol.PUSH, peer, slot)


def test_symmetric_arena_slices_match_fixed_layout() -> None:
    layout = ArenaLayout.create(ArenaShape(num_pes=2, num_slots=2, max_payload_bytes=128))
    arena = SymmetricArena(layout=layout, tensor=object())

    assert arena.payload(Protocol.PULL, 1, 1, 64).offset == 896
    assert arena.payload(Protocol.PULL, 1, 1, 64).size == 64
    assert arena.ready(Protocol.PUSH, 1, 1).offset == 1048
    assert arena.consumed(Protocol.PULL, 1, 1).offset == 1144
    assert arena.byte_count(Protocol.PULL, 1, 1).offset == 1208


def test_slot_epochs_require_prior_consumption_only_after_initial_fill() -> None:
    assert slot_epoch(1, 4).previous_epoch is None
    assert slot_epoch(4, 4).slot == 3
    assert slot_epoch(5, 4).slot == 0
    assert slot_epoch(5, 4).previous_epoch == 1
    assert final_slot_epochs(10, 4) == (9, 10, 7, 8)
