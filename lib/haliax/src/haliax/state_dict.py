# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0


from typing import TypeVar

import equinox

from haliax.jax_utils import is_jax_array_like
from haliax.types import FilterSpec

from ._src.state_dict import (
    ModuleWithStateDictSerialization,
    StateDict,
    flatten_modules_for_export,
    from_state_dict,
    from_torch_compatible_state_dict,
    load_state_dict,
    save_state_dict,
    to_numpy_state_dict,
    to_state_dict,
    unflatten_modules_from_export,
    with_prefix,
)

T = TypeVar("T")


def to_torch_compatible_state_dict(
    t: T, *, flatten: bool = True, prefix: str | None = None, filter: FilterSpec = is_jax_array_like
) -> StateDict:
    """
    Convert a tree to a state dict that is compatible with torch-style state dicts.

    This applies the same logic as [to_state_dict][] but also uses [haliax.state_dict.ModuleWithStateDictSerialization.flatten_for_export][] to flatten

    Args:
        t: The tree to convert
        flatten: Whether to flatten axes using flatten_for_export
        prefix: The prefix to use for the state dict keys
        filter: The filter to use for selecting which nodes to include in the state dict. By default, this includes only
            array-like objects (e.g. JAX and NumPy arrays).
    """
    t = equinox.filter(t, filter)
    if flatten:
        t = flatten_modules_for_export(t)
    return to_numpy_state_dict(t, prefix=prefix)


__all__ = [
    "ModuleWithStateDictSerialization",
    "from_torch_compatible_state_dict",
    "load_state_dict",
    "save_state_dict",
    "from_state_dict",
    "with_prefix",
    "to_state_dict",
    "to_numpy_state_dict",
    "StateDict",
    "to_torch_compatible_state_dict",
    "flatten_modules_for_export",
    "unflatten_modules_from_export",
]
