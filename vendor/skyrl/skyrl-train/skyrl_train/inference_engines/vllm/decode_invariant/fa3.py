"""FA3 constants, and the split of a step's requests that gives each query row the same output as a one-row decode
step of that row."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

FLASH_ATTN_VERSION = 3
HOPPER_COMPUTE_CAPABILITY_MAJOR = 9
# The one head dimension a decode-invariant engine runs. At 128, FA3's bf16 causal and local kernels on Hopper
# (``tile_size_fwd_sm90``) read keys in tiles of 128 in every step, so at one split the order in which a row adds up
# its keys depends only on the row's position.
FA3_HEAD_DIM = 128


@dataclass(frozen=True)
class Fa3Requests:
    """A step's query rows regrouped into ``E`` FA3 varlen requests, in the same row order. ``query_start`` (int32,
    ``[E + 1]``) is each request's first row, ``key_lengths`` (int32, ``[E]``) its number of keys, and ``owner``
    (``[E]``) the original request it came from, whose block-table row it reads."""

    query_start: torch.Tensor
    key_lengths: torch.Tensor
    owner: torch.Tensor


def fa3_window_requests(query_start: Sequence[int], key_lengths: Sequence[int], window: int) -> Fa3Requests:
    """Split each varlen request so that each row at a position of at least ``window`` is a request of its own.

    A request's query rows are the last ``q`` of its ``k`` keys. FA3's local kernel reads a query tile's keys from the
    window start of the tile's first row, so every later row of the tile would add up other key tiles than its own
    decode step does.
    """
    query_start = np.asarray(query_start, dtype=np.int64)
    rows = np.arange(query_start[-1])
    request = np.repeat(np.arange(query_start.size - 1), np.diff(query_start))
    position = np.asarray(key_lengths, dtype=np.int64)[request] - query_start[request + 1] + rows
    starts = np.flatnonzero((rows == query_start[request]) | (position >= window))
    return Fa3Requests(
        query_start=torch.from_numpy(np.append(starts, query_start[-1])).to(torch.int32),
        key_lengths=torch.from_numpy(position[np.append(starts[1:], rows.size) - 1] + 1).to(torch.int32),
        owner=torch.from_numpy(request[starts]),
    )
