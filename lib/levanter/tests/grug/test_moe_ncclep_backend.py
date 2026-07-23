# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from levanter.grug._moe.ep_ncclep import ncclep_receive_capacity


def test_ncclep_receive_capacity_matches_ep8_training_shape() -> None:
    capacity = ncclep_receive_capacity(
        global_tokens=131_072,
        top_k=4,
        ep_size=8,
        capacity_factor=1.25,
    )

    assert capacity == 81_920


@pytest.mark.parametrize(
    ("global_tokens", "top_k", "ep_size", "capacity_factor"),
    [
        (0, 4, 8, 1.25),
        (131_072, 0, 8, 1.25),
        (131_072, 4, 0, 1.25),
        (131_071, 4, 8, 1.25),
        (131_072, 4, 8, 0.0),
    ],
)
def test_ncclep_receive_capacity_rejects_invalid_layouts(
    global_tokens: int,
    top_k: int,
    ep_size: int,
    capacity_factor: float,
) -> None:
    with pytest.raises(ValueError):
        ncclep_receive_capacity(global_tokens, top_k, ep_size, capacity_factor)
