import enum
import typing
from functools import partial

import jax

import haliax as hax
import haliax.nn as hnn


_A = typing.TypeVar("_A", hax.Scalar, hax.NamedArray, jax.Array)
ActivationFunction = typing.Callable[[_A], _A]


class ActivationFunctionEnum(str, enum.Enum):
    relu = "relu"
    silu = "silu"
    swish = "swish"
    gelu = "gelu"
    gelu_new = "gelu_new"
    quick_gelu = "quick_gelu"
    tanh = "tanh"

    def to_fn(self) -> ActivationFunction:
        return TO_FN[self]


# type: ignore
TO_FN: dict[ActivationFunctionEnum, ActivationFunction] = {
    ActivationFunctionEnum.relu: hnn.relu,
    ActivationFunctionEnum.silu: hnn.silu,
    ActivationFunctionEnum.swish: hnn.swish,
    ActivationFunctionEnum.gelu: partial(hnn.gelu, approximate=False),
    ActivationFunctionEnum.gelu_new: partial(hnn.gelu, approximate=True),
    ActivationFunctionEnum.quick_gelu: hnn.quick_gelu,
    ActivationFunctionEnum.tanh: hax.tanh,
}
