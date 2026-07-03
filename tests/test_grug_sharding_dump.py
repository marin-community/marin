# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from experiments.grug.sharding_dump import (
    dump_grug_state_sharding,
    grug_state_sharding_dict,
)


@dataclass(frozen=True)
class _State:
    params: dict[str, jax.Array]
    opt_state: dict[str, jax.Array]


def test_grug_state_sharding_dict_reports_params_and_opt_state() -> None:
    mesh = Mesh(np.array(jax.devices()), ("data",))
    sharding = NamedSharding(mesh, P("data"))
    state = _State(
        params={"weight": jax.device_put(jnp.ones((jax.device_count(),), dtype=jnp.float32), sharding)},
        opt_state={"moment": jax.device_put(jnp.zeros((jax.device_count(),), dtype=jnp.float32), sharding)},
    )

    shardings = grug_state_sharding_dict(state)

    assert shardings == {
        "params": {"['weight']": "P('data',)"},
        "opt_state": {"['moment']": "P('data',)"},
    }


def test_dump_grug_state_sharding_writes_json(tmp_path) -> None:
    mesh = Mesh(np.array(jax.devices()), ("data",))
    sharding = NamedSharding(mesh, P("data"))
    state = _State(
        params={"weight": jax.device_put(jnp.ones((jax.device_count(),), dtype=jnp.float32), sharding)},
        opt_state={"moment": jax.device_put(jnp.zeros((jax.device_count(),), dtype=jnp.float32), sharding)},
    )
    path = tmp_path / "run" / "artifacts" / "sharding.json"

    dump_grug_state_sharding(state, path)

    assert json.loads(path.read_text()) == {
        "params": {"['weight']": "P('data',)"},
        "opt_state": {"['moment']": "P('data',)"},
    }
