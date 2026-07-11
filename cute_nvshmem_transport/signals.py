# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

MAX_EPOCH = (1 << 64) - 1


@dataclass(frozen=True)
class SlotEpoch:
    """Map one monotonic transfer epoch to its reusable pipeline slot."""

    epoch: int
    slot: int
    previous_epoch: int | None


def slot_epoch(epoch: int, num_slots: int) -> SlotEpoch:
    """Return the slot and required consumed epoch for a transfer."""
    if not 1 <= epoch <= MAX_EPOCH:
        raise ValueError(f"epoch must be in [1, {MAX_EPOCH}]")
    if num_slots < 1:
        raise ValueError("num_slots must be positive")
    slot = (epoch - 1) % num_slots
    previous_epoch = epoch - num_slots if epoch > num_slots else None
    return SlotEpoch(epoch=epoch, slot=slot, previous_epoch=previous_epoch)


def final_slot_epochs(num_epochs: int, num_slots: int) -> tuple[int, ...]:
    """Return the final expected epoch in every slot after a run."""
    if not 1 <= num_epochs <= MAX_EPOCH:
        raise ValueError(f"num_epochs must be in [1, {MAX_EPOCH}]")
    if num_slots < 1:
        raise ValueError("num_slots must be positive")
    return tuple(
        num_epochs - ((num_epochs - slot - 1) % num_slots) if num_epochs > slot else 0 for slot in range(num_slots)
    )
