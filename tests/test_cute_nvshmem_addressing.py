# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from cute_nvshmem_transport.addressing import ArenaLayout, ArenaShape, Protocol


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
