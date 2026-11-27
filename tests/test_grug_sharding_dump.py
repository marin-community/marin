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
    GRUG_SHARDING_DUMP_FILENAME,
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


def test_remote_sharding_dump_override_keeps_url_and_logs_local_copy(tmp_path, monkeypatch) -> None:
    class RecordingTracker(NoopTracker):
        def __init__(self):
            self.artifacts: list[str] = []

        def log_artifact(self, artifact_path, **kwargs):
            self.artifacts.append(str(artifact_path))

    monkeypatch.chdir(tmp_path)
    state = _State(
        params={"weight": jnp.ones((4,), dtype=jnp.float32)},
        opt_state={"moment": jnp.zeros((4,), dtype=jnp.float32)},
    )
    override = f"memory://grug-sharding/{tmp_path.name}/state-sharding.json"
    tracker = RecordingTracker()

    with current_tracker(tracker):
        dump_grug_state_sharding_run_artifact(state, log_dir=tmp_path / "logs", run_id="run", path_override=override)

    expected = grug_state_sharding_dict(state)
    assert json.loads(StoragePath(override).read_text()) == expected
    local_copy = tmp_path / "logs" / "run" / "artifacts" / GRUG_SHARDING_DUMP_FILENAME
    assert tracker.artifacts == [str(local_copy)]
    assert json.loads(local_copy.read_text()) == expected
    assert not (tmp_path / "memory:").exists()
