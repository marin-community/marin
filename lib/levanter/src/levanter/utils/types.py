# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeAlias,
    TypeVar,
    Union,
    cast,
)

from jaxtyping import PyTree

if TYPE_CHECKING:
    from haliax.types import IntScalar, Scalar
else:
    IntScalar = Any
    Scalar = Any


M = TypeVar("M")  # Model
M_con = TypeVar("M_con", contravariant=True)  # Model
X = TypeVar("X", contravariant=True)  # Input

if TYPE_CHECKING:
    from haliax.nn.scan import BlockFoldable, ScanCheckpointPolicy
else:

    class BlockFoldable(Protocol[M]):  # type: ignore
        def fold(self, *args, **kwargs): ...

        def scan(self, *args, **kwargs): ...

    class ScanCheckpointPolicy(Protocol):  # type: ignore
        pass


PhysicalAxisSpec: TypeAlias = str | Sequence[str]
ResourceMapping: TypeAlias = Mapping[str, PhysicalAxisSpec]


class ValAndGradFn(Protocol[M, X]):
    def __call__(self, model: M, *inputs: X, **input_kwargs) -> Tuple[Scalar, M]: ...


class ValFn(Protocol[M_con, X]):
    def __call__(self, model: M_con, *inputs: X, **input_kwargs) -> Scalar: ...


FilterSpec = Union[bool, Callable[[Any], bool]]
"""
A filter specification. Typically used on a pytree to filter out certain subtrees. Boolean values are
treated as-is, while callables are called on each element of the pytree. If the callable returns True, the element
is kept, otherwise it is filtered out.
"""

FilterTree = FilterSpec | PyTree[FilterSpec]


class ComputeLossFunction(Protocol[M_con, X]):
    """
    Function signature for "compute_loss" functions in Levanter: these
    couple the computation of the logits and the evaluation of the loss
    """

    def __call__(
        self,
        model: M_con,
        *inputs: X,
        reduction: Optional[Callable[..., Any]] = cast(Optional[Callable[..., Any]], None),
        reduction_axis: Optional[Any] = None,
        **kwargs,
    ) -> Scalar | Any: ...
