# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from levanter.callbacks import StepInfo
from levanter.elastic import (
    ElasticTrainingConfig,
    FileBackedPeerSyncController,
    _flatten_transfer_payload,
    _restore_transfer_payload,
    read_completion,
)
from levanter.trainer import _InjectedFaults, MARIN_FAULT_INJECTION_STEPS_ENV


@dataclasses.dataclass
class DummyState:
    step: int
    model: dict[str, jax.Array]
    opt_state: None = None
    model_averaging: None = None


def _single_device_mesh() -> Mesh:
    return Mesh(np.array(jax.devices()).reshape(1), axis_names=("replica",))


def test_peer_sync_bootstrap_and_merge(tmp_path):
    elastic_root = str(tmp_path / "elastic")
    mesh = _single_device_mesh()

    peer_controller = FileBackedPeerSyncController(
        config=ElasticTrainingConfig(
            enabled=True,
            group_id="run",
            worker_id="w001",
            state_path=elastic_root,
            sync_interval_steps=1,
            publish_interval_steps=1,
        ),
        checkpoint_base_path=str(tmp_path / "checkpoints" / "run-w001"),
        run_id="run-w001",
        axis_mapping={},
        mesh=mesh,
    )
    local_controller = FileBackedPeerSyncController(
        config=ElasticTrainingConfig(
            enabled=True,
            group_id="run",
            worker_id="w000",
            state_path=elastic_root,
            sync_interval_steps=1,
            publish_interval_steps=1,
            mixing_rate=0.5,
        ),
        checkpoint_base_path=str(tmp_path / "checkpoints" / "run-w000"),
        run_id="run-w000",
        axis_mapping={},
        mesh=mesh,
    )
    assert peer_controller.transport_kind == "checkpoint"
    assert local_controller.transport_kind == "checkpoint"

    peer_state = DummyState(step=1, model={"weight": jnp.array([2.0], dtype=jnp.float32)})
    peer_controller._publish_state(peer_state, step=0)
    peer_controller._manager.wait_until_finished()

    bootstrapped = local_controller.bootstrap_state(
        DummyState(step=0, model={"weight": jnp.array([0.0], dtype=jnp.float32)})
    )
    assert float(bootstrapped.model["weight"][0]) == 2.0

    peer_controller._publish_state(
        DummyState(step=2, model={"weight": jnp.array([2.0], dtype=jnp.float32)}),
        step=1,
    )
    peer_controller._manager.wait_until_finished()

    synced_info = local_controller.maybe_update_state(
        StepInfo(
            state=DummyState(step=1, model={"weight": jnp.array([0.0], dtype=jnp.float32)}),
            loss=0.0,
            step_duration=0.0,
        )
    )
    assert float(synced_info.state.model["weight"][0]) == 1.0

    local_controller.mark_completed(synced_info)
    completion = read_completion(local_controller.paths.completion_path)
    assert completion is not None
    assert completion.worker_id == "w000"
    assert completion.completed_step == 0


def test_injected_faults_only_fire_once_across_retries(tmp_path, monkeypatch):
    checkpoint_path = str(tmp_path / "checkpoints" / "run")
    marker_root = str(tmp_path / "markers")
    monkeypatch.setenv(MARIN_FAULT_INJECTION_STEPS_ENV, "[10]")
    monkeypatch.setenv("MARIN_FAULT_INJECTION_STATE_PATH", marker_root)

    injector = _InjectedFaults.from_env(checkpoint_path=checkpoint_path, run_id="run")
    assert injector is not None

    with pytest.raises(RuntimeError, match="Injected fault"):
        injector.maybe_raise(10)

    retry_injector = _InjectedFaults.from_env(checkpoint_path=checkpoint_path, run_id="run")
    assert retry_injector is not None
    retry_injector.maybe_raise(10)


def test_transfer_payload_filters_non_arrays_and_restores_template():
    template = {
        "model": {"weight": jnp.array([1.0], dtype=jnp.float32)},
        "model_averaging": {"weight": jnp.array([2.0], dtype=jnp.float32), "tot_weight": 3.0},
        "opt_state": {"count": jnp.array(4, dtype=jnp.int32), "should_skip": True},
    }

    flat_payload = _flatten_transfer_payload(template)

    assert set(flat_payload.keys()) == {
        "model.weight",
        "model_averaging.weight",
        "opt_state.count",
    }

    restored = _restore_transfer_payload(
        template,
        {
            "model.weight": jnp.array([10.0], dtype=jnp.float32),
            "model_averaging.weight": jnp.array([20.0], dtype=jnp.float32),
            "opt_state.count": jnp.array(30, dtype=jnp.int32),
        },
    )

    assert float(restored["model"]["weight"][0]) == 10.0
    assert float(restored["model_averaging"]["weight"][0]) == 20.0
    assert restored["model_averaging"]["tot_weight"] == 3.0
    assert int(restored["opt_state"]["count"]) == 30
    assert restored["opt_state"]["should_skip"] is True
