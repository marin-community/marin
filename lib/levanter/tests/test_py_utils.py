# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import gc
import weakref

from levanter.utils.py_utils import per_instance_lru_cache


class _Boxes:
    """Returns a fresh object per evaluation, so cache hits are observable by identity."""

    @per_instance_lru_cache(maxsize=1)
    def box(self, x: int) -> list[int]:
        return [x]


def test_per_instance_cache_capacity_is_not_shared_between_instances():
    first = _Boxes()
    second = _Boxes()

    boxed = first.box(0)
    # With a single class-level cache this call would evict `first`'s only entry.
    second.box(0)

    assert first.box(0) is boxed


def test_per_instance_cache_evicts_within_an_instance():
    boxes = _Boxes()

    boxed = boxes.box(0)
    boxes.box(1)

    assert boxes.box(0) is not boxed


def test_cached_instances_are_collectable():
    boxes = _Boxes()
    boxes.box(0)
    ref = weakref.ref(boxes)

    del boxes
    gc.collect()

    assert ref() is None
