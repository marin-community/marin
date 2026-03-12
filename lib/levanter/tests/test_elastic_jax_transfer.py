# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from datetime import timedelta

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh

from levanter.callbacks import StepInfo
from levanter.elastic import ElasticTrainingConfig, FileBackedPeerSyncController, PeerAveragingSyncConfig


@dataclasses.dataclass
class DummyState:
    step: int
    model: dict[str, jax.Array]
    opt_state: None = None
    model_averaging: None = None


def _single_device_mesh() -> Mesh:
    return Mesh(np.array(jax.devices()).reshape(1), axis_names=("replica",))


def test_jax_transfer_peer_sync_smoke(tmp_path):
    try:
        import jax.experimental.transfer  # noqa: F401
    except (ImportError, AttributeError):
        pytest.skip("jax.experimental.transfer is unavailable in this environment")

    if jax.default_backend() != "tpu":
        pytest.skip("jax transfer smoke test only runs on TPU")
    if jax.process_count() != 1:
        pytest.skip("jax transfer smoke test currently expects a single JAX process")

    elastic_root = str(tmp_path / "elastic")
    mesh = _single_device_mesh()
    config = ElasticTrainingConfig(
        enabled=True,
        group_id="run",
        state_path=elastic_root,
        sync_interval_steps=1,
        publish_interval_steps=1,
        sync=PeerAveragingSyncConfig(),
        transport="jax_transfer",
        transfer_timeout=timedelta(seconds=30),
        request_poll_interval_seconds=0.05,
    )

    peer_controller = FileBackedPeerSyncController(
        config=dataclasses.replace(config, worker_id="w001"),
        checkpoint_base_path=str(tmp_path / "checkpoints" / "run-w001"),
        run_id="run-w001",
        axis_mapping={},
        mesh=mesh,
    )
    local_controller = FileBackedPeerSyncController(
        config=dataclasses.replace(config, worker_id="w000"),
        checkpoint_base_path=str(tmp_path / "checkpoints" / "run-w000"),
        run_id="run-w000",
        axis_mapping={},
        mesh=mesh,
    )

    try:
        assert peer_controller.transport_kind == "jax_transfer"
        assert local_controller.transport_kind == "jax_transfer"

        peer_controller._publish_state(  # noqa: SLF001
            DummyState(step=1, model={"weight": jnp.array([2.0], dtype=jnp.float32)}),
            step=0,
        )
        bootstrapped = local_controller.bootstrap_state(
            DummyState(step=0, model={"weight": jnp.array([0.0], dtype=jnp.float32)})
        )
        assert float(bootstrapped.model["weight"][0]) == 2.0

        peer_controller._publish_state(  # noqa: SLF001
            DummyState(step=2, model={"weight": jnp.array([2.0], dtype=jnp.float32)}),
            step=1,
        )
        synced_info = local_controller.maybe_update_state(
            StepInfo(
                state=DummyState(step=1, model={"weight": jnp.array([0.0], dtype=jnp.float32)}),
                loss=0.0,
                step_duration=0.0,
            )
        )
        assert float(synced_info.state.model["weight"][0]) == 1.0
    finally:
        peer_controller.close()
        local_controller.close()
