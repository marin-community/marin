# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from typing import Any, Protocol

import fsspec
import jax
from haliax.jax_utils import is_jax_array_like
from levanter.tracker import current_tracker


class GrugStateWithSharding(Protocol):
    params: object
    opt_state: object


def grug_state_sharding_dict(state: GrugStateWithSharding) -> dict[str, dict[str, str]]:
    """Return sharding specs for Grug parameters and optimizer state."""
    return {
        "params": tree_sharding_dict(state.params),
        "opt_state": tree_sharding_dict(state.opt_state),
    }


def dump_grug_state_sharding(state: GrugStateWithSharding, path: str | Path) -> None:
    """Write Grug parameter and optimizer-state sharding specs as JSON."""
    output = grug_state_sharding_dict(state)
    serialized = json.dumps(output, indent=2, sort_keys=True)
    with fsspec.open(str(path), "w") as out:
        out.write(serialized)
        out.write("\n")


def dump_grug_state_sharding_artifact(state: GrugStateWithSharding, path: str | Path) -> None:
    """Write Grug sharding specs and log them to the active tracker."""
    dump_grug_state_sharding(state, path)
    current_tracker().log_artifact(path, name="grug_sharding_spec", type="sharding")


def tree_sharding_dict(tree: Any) -> dict[str, str]:
    """Return a full PyTree path dict with each array value replaced by its sharding spec."""
    shardings: dict[str, str] = {}
    for path, value in jax.tree_util.tree_leaves_with_path(tree):
        if is_jax_array_like(value):
            shardings[jax.tree_util.keystr(path)] = _sharding_spec(value)
    return shardings


def _sharding_spec(value: Any) -> str:
    sharding = getattr(value, "sharding", None)
    if sharding is None:
        return "None"

    spec = getattr(sharding, "spec", None)
    if spec is not None:
        return repr(spec)

    return repr(sharding)
