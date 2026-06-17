# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import jax
import optax
from levanter.utils.jax_utils import leaf_key_paths

EXPERT_AXIS = "expert"


def target_named_sharding(array) -> jax.sharding.NamedSharding | None:
    if array is None or not hasattr(array, "shape"):
        return None
    sharding = getattr(array, "sharding", None)
    if sharding is None:
        aval = jax.typeof(array)
        sharding = getattr(aval, "sharding", None)
    if isinstance(sharding, jax.sharding.NamedSharding):
        return sharding
    return None


def _axis_names(axis) -> tuple[str, ...]:
    if axis is None:
        return ()
    if isinstance(axis, tuple):
        return axis
    return (axis,)


def _path_str(path) -> str:
    if isinstance(path, (list, tuple)):
        return ".".join(str(part) for part in path)
    return str(path)


def assert_update_sharding_matches_params(updates, params, label: str) -> None:
    paths = leaf_key_paths(params)

    def check(update, param, path):
        if update is None or isinstance(update, optax.MaskedNode) or param is None or not hasattr(param, "shape"):
            return None

        param_sharding = target_named_sharding(param)
        if param_sharding is None:
            return None

        update_sharding = target_named_sharding(update)
        path_text = _path_str(path)
        if update_sharding is None:
            raise AssertionError(
                f"{label} lost NamedSharding at {path_text}: expected {param_sharding.spec}, got no NamedSharding"
            )

        expected_spec = param_sharding.spec
        actual_spec = update_sharding.spec
        if getattr(param, "ndim", None) == 3 and len(expected_spec) > 0 and EXPERT_AXIS in _axis_names(expected_spec[0]):
            actual_leading_axes = _axis_names(actual_spec[0]) if len(actual_spec) > 0 else ()
            if EXPERT_AXIS not in actual_leading_axes:
                raise AssertionError(
                    f"{label} lost expert-axis sharding at {path_text}: expected {expected_spec}, got {actual_spec}"
                )

        if actual_spec != expected_spec:
            raise AssertionError(f"{label} sharding drift at {path_text}: expected {expected_spec}, got {actual_spec}")

        return None

    jax.tree.map(check, updates, params, paths, is_leaf=lambda x: x is None)
