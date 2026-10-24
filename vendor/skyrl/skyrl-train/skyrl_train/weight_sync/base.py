"""Base data structures for weight synchronization."""

from dataclasses import dataclass
from functools import cached_property
from typing import List

import torch


@dataclass
class WeightChunk:
    """Represents one or more model parameters to be transferred.

    A WeightChunk can contain multiple parameters grouped together for efficient
    transfer (e.g., Q/K/V projections for FlashRL fusion).

    Attributes:
        names: List of parameter names (e.g., ["model.layer.0.weight"])
        dtypes: List of dtype strings (e.g., ["torch.bfloat16"])
        shapes: List of tensor shapes (e.g., [[4096, 4096]])
        tensors: List of actual tensor data (populated during extraction)
        total_numel: Total number of elements (cached property, auto-calculated)
        total_size_bytes: Total memory footprint (cached property, auto-calculated)
    """

    names: List[str]
    dtypes: List[str]
    shapes: List[List[int]]
    tensors: List[torch.Tensor]

    def __post_init__(self):
        """Validate that all input lists have the same length."""
        lengths = [len(self.names), len(self.dtypes), len(self.shapes), len(self.tensors)]
        if len(set(lengths)) != 1:
            raise ValueError(
                f"All lists must have the same length. Got names={len(self.names)}, "
                f"dtypes={len(self.dtypes)}, shapes={len(self.shapes)}, tensors={len(self.tensors)}"
            )

    def __len__(self) -> int:
        """Return the number of parameters in this chunk."""
        return len(self.names)

    @cached_property
    def total_numel(self) -> int:
        """Calculate total number of elements across all tensors."""
        return sum(t.numel() for t in self.tensors)

    @cached_property
    def total_size_bytes(self) -> int:
        """Calculate total memory footprint in bytes."""
        return sum(t.numel() * t.element_size() for t in self.tensors)
