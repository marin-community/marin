from dataclasses import dataclass
from math import prod
from typing import Optional, Tuple, Type, TypeAlias, Union

import jax
import numpy as np
from jax import ShapeDtypeStruct
from jaxtyping import PyTree

from haliax import Axis
from haliax.util import is_named_array


DType = Union[np.dtype, Type[int], Type[float], Type[bool]]

ShapeSpec: TypeAlias = ShapeDtypeStruct


@dataclass(frozen=True)
class NamedShapeSpec:
    """A shape specification with named axes."""

    shape: Optional[Tuple[Axis, ...]]
    dtype: Optional[DType]

    size = property(lambda self: prod(ax.size for ax in self.shape))
    ndim = property(lambda self: len(self.shape))


def to_raw_shape(shape: Union[ShapeSpec, NamedShapeSpec]) -> Optional[Tuple[int, ...]]:
    if isinstance(shape, ShapeDtypeStruct):
        return shape.shape
    else:
        raw = shape.shape
        if raw is None:
            return None
        return tuple(ax.size for ax in raw)


def shape_spec_of(tree: PyTree) -> PyTree[Union[ShapeSpec, NamedShapeSpec]]:
    """Get the shape specification of a tree."""

    def _leaf_spec(leaf):
        if is_named_array(leaf):
            return NamedShapeSpec(leaf.axes, leaf.dtype)
        else:
            return ShapeDtypeStruct(leaf.shape, leaf.dtype)

    return jax.tree_util.tree_map(_leaf_spec, tree, is_leaf=is_named_array)


def conforms(shape: PyTree[Union[ShapeSpec, NamedShapeSpec]], tree: PyTree) -> bool:
    """Check if a tree conforms to a shape specification."""

    def _leaf_conforms(shape_spec: Union[ShapeSpec, NamedShapeSpec], leaf):
        if isinstance(shape_spec, ShapeSpec):  # type: ignore
            return shape_spec.shape == leaf.shape and shape_spec.dtype == leaf.dtype
        else:
            return (shape_spec.shape is None or shape_spec.shape == leaf.axes) and (
                shape_spec.dtype is None or shape_spec.dtype == leaf.dtype
            )

    return jax.tree_util.tree_all(jax.tree_util.tree_map(_leaf_conforms, shape, tree, is_leaf=is_named_array))
