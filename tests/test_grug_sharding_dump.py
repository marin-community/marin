# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.tracker import NoopTracker, current_tracker
from rigging.filesystem.storage_path import StoragePath

from experiments.grug.sharding_dump import (
    dump_grug_state_sharding,
    dump_grug_state_sharding_run_artifact,
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


def test_remote_sharding_override_preserves_uri_and_observes_dtype(tmp_path):
    uri = f"memory://tpu-sharding/{tmp_path.name}/state-sharding.json"
    state = _State(
        params={"weight": jnp.ones((4,), dtype=jnp.float32)},
        opt_state={"moment": jnp.zeros((4,), dtype=jnp.bfloat16)},
    )
    with current_tracker(NoopTracker()):
        dump_grug_state_sharding_run_artifact(state, log_dir=tmp_path, run_id="remote", path_override=uri)
    destination = StoragePath(uri).parent / "state-sharding-rank-0.json"
    report = json.loads(destination.read_text())
    assert destination.scheme == "memory"
    assert report["rank"] == 0
    assert report["arrays"]["params"]["['weight']"]["dtype"] == "float32"
    assert report["arrays"]["params"]["['weight']"]["shape"] == [4]
    assert report["arrays"]["opt_state"]["['moment']"]["dtype"] == "bfloat16"
    assert report["arrays"]["params"]["['weight']"]["local_shards"]
    assert not (tmp_path / "memory:").exists()
