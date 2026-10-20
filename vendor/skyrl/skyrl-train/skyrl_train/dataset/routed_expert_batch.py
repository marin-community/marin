"""Compact and dense representations of response-aligned router replay targets."""

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import torch
from loguru import logger


def _routed_experts_dtype_for_num_experts(num_experts: int) -> torch.dtype:
    """Pick the narrowest integer dtype that can hold ANY valid expert id for a
    model with ``num_experts`` experts, DETERMINISTICALLY (max possible id =
    ``num_experts - 1``), independent of the per-batch observed max.

    This is load-bearing: the dtype must NOT depend on the data, or two batches
    / ranks whose observed max straddles a dtype boundary (e.g. Qwen3-Next with
    512 experts: one batch max=200 -> uint8, another max=300 -> int16) would
    pick DIFFERENT dtypes for the same field, size-mismatching a later cross-rank
    collective on this tensor -> NCCL hang. Keying on ``num_experts`` makes every
    rank/batch agree.

      * num_experts <= 256      -> uint8  (max id <= 255; Qwen3-Coder 128 -> uint8)
      * num_experts <= 32768    -> int16  (max id <= 32767; Qwen3-Next 512 -> int16, deterministic)
      * otherwise               -> int64  (defensive; no shipped MoE model exceeds int16 range)

    """
    if num_experts <= 0:
        raise ValueError(f"num_experts must be positive, got {num_experts}")
    if num_experts <= (torch.iinfo(torch.uint8).max + 1):
        return torch.uint8
    if num_experts <= (torch.iinfo(torch.int16).max + 1):
        return torch.int16
    return torch.int64


def _collate_routed_experts_from_arrays(
    routed_experts: List[np.ndarray],
    max_output_len: int,
    num_experts: int,
) -> "torch.Tensor":
    """Build a dense ``[B, response, layer, top_k]`` routed-expert tensor.

    Response rows are right-padded with zeroes in the final tensor dtype.
    """
    if any(rows.ndim != 3 for rows in routed_experts):
        raise ValueError("routed_experts must contain [token, layer, top_k] arrays")
    layers = max(rows.shape[1] for rows in routed_experts)
    top_k = max(rows.shape[2] for rows in routed_experts)
    _re_dtype = _routed_experts_dtype_for_num_experts(num_experts)
    numpy_dtype = {torch.uint8: np.uint8, torch.int16: np.int16, torch.int64: np.int64}[_re_dtype]
    out = np.zeros((len(routed_experts), max_output_len, layers, top_k), dtype=numpy_dtype)
    for index, rows in enumerate(routed_experts):
        count = min(len(rows), max_output_len)
        out[index, :count, : rows.shape[1], : rows.shape[2]] = rows[:count]
    return torch.from_numpy(out)


@dataclass(frozen=True)
class RoutedExpertRows:
    """Compact response-aligned routes carried until a worker forms a microbatch."""

    rows: tuple[np.ndarray, ...]
    response_len: int
    num_experts: int | None

    def __post_init__(self) -> None:
        if any(row.ndim != 3 or len(row) > self.response_len for row in self.rows):
            raise ValueError("routed experts must contain response-aligned [token, layer, top_k] arrays")
        if self.num_experts is None:
            # Resolve the fallback once for the full batch so every worker slice uses the same dtype.
            observed_max = max((int(row.max()) for row in self.rows if row.size), default=0)
            object.__setattr__(self, "num_experts", observed_max + 1)
            logger.warning("num_experts is unknown; routed experts use a full-batch observed-max dtype")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: slice) -> "RoutedExpertRows":
        return RoutedExpertRows(self.rows[index], self.response_len, self.num_experts)

    @property
    def dtype(self) -> torch.dtype:
        assert self.num_experts is not None
        return _routed_experts_dtype_for_num_experts(self.num_experts)

    @property
    def shape(self) -> tuple[int, int, int, int]:
        layers = max(row.shape[1] for row in self.rows)
        top_k = max(row.shape[2] for row in self.rows)
        return len(self.rows), self.response_len, layers, top_k

    @property
    def nbytes(self) -> int:
        return sum(row.nbytes for row in self.rows)

    def materialize(self, device: torch.device | int | str = "cpu") -> torch.Tensor:
        local_response_len = max((len(row) for row in self.rows), default=0)
        assert self.num_experts is not None
        return _collate_routed_experts_from_arrays(list(self.rows), local_response_len, self.num_experts).to(device)

    @classmethod
    def cat(cls, chunks: Sequence["RoutedExpertRows"]) -> "RoutedExpertRows":
        first = chunks[0]
        if any(chunk.response_len != first.response_len or chunk.num_experts != first.num_experts for chunk in chunks):
            raise ValueError("routed expert chunks must share response length and expert count")
        return cls(tuple(row for chunk in chunks for row in chunk.rows), first.response_len, first.num_experts)
