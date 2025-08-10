from typing import Any, Callable, Optional, Protocol, Tuple, TypeVar, Union

from jaxtyping import PyTree

import haliax as hax
from haliax.types import Scalar


M = TypeVar("M")  # Model
M_con = TypeVar("M_con", contravariant=True)  # Model
X = TypeVar("X", contravariant=True)  # Input

try:
    from haliax.nn.scan import BlockFoldable
except ImportError:

    class BlockFoldable(Protocol[M]):  # type: ignore
        def fold(self, *args, **kwargs):
            ...

        def scan(self, *args, **kwargs):
            ...


class ValAndGradFn(Protocol[M, X]):
    def __call__(self, model: M, *inputs: X, **input_kwargs) -> Tuple[Scalar, M]:
        ...


class ValFn(Protocol[M_con, X]):
    def __call__(self, model: M_con, *inputs: X, **input_kwargs) -> Scalar:
        ...


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
        reduction: Optional[hax.ReductionFunction] = hax.mean,
        reduction_axis: Optional[hax.AxisSelection] = None,
        **kwargs,
    ) -> Scalar | hax.NamedArray:
        ...
