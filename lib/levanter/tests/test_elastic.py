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
    DiLoCoSyncConfig,
    ElasticTrainingConfig,
    ElasticWorkerStatus,
    FileBackedPeerSyncController,
    PeerAveragingSyncConfig,
    _clip_tree_to_global_norm,
    _remove_if_exists,
    _aggregate_progress_metrics,
    _flatten_transfer_payload,
    _restore_transfer_payload,
    _tree_global_norm,
    read_completion,
    read_worker_status,
)
from levanter.schedule import BatchSchedule
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
            sync=PeerAveragingSyncConfig(),
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
            sync=PeerAveragingSyncConfig(mixing_rate=0.5),
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


def test_diloco_sync_updates_anchor_with_outer_optimizer(tmp_path):
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
            sync=DiLoCoSyncConfig(outer_optimizer="sgd", outer_learning_rate=0.25),
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
            sync=DiLoCoSyncConfig(outer_optimizer="sgd", outer_learning_rate=0.25),
        ),
        checkpoint_base_path=str(tmp_path / "checkpoints" / "run-w000"),
        run_id="run-w000",
        axis_mapping={},
        mesh=mesh,
    )

    initial_state = DummyState(step=0, model={"weight": jnp.array([0.0], dtype=jnp.float32)})
    peer_controller.bootstrap_state(initial_state)
    local_controller.bootstrap_state(initial_state)
    assert local_controller._diloco_state is not None  # noqa: SLF001
    assert local_controller._diloco_state.anchor_model["weight"] is not initial_state.model["weight"]  # noqa: SLF001

    peer_controller._publish_state(  # noqa: SLF001
        DummyState(step=1, model={"weight": jnp.array([3.0], dtype=jnp.float32)}),
        step=0,
    )
    peer_controller._manager.wait_until_finished()  # noqa: SLF001

    synced_info = local_controller.maybe_update_state(
        StepInfo(
            state=DummyState(step=1, model={"weight": jnp.array([1.0], dtype=jnp.float32)}),
            loss=0.0,
            step_duration=0.0,
        )
    )

    assert float(synced_info.state.model["weight"][0]) == pytest.approx(0.5)
    assert local_controller._diloco_state is not None  # noqa: SLF001
    assert (
        local_controller._diloco_state.anchor_model["weight"] is not synced_info.state.model["weight"]
    )  # noqa: SLF001


def test_aggregate_progress_metrics_reports_logical_and_delivered_tokens():
    metrics = _aggregate_progress_metrics(
        [
            ElasticWorkerStatus(worker_id="w000", run_id="run", step=9),
            ElasticWorkerStatus(worker_id="w001", run_id="run", step=4),
        ],
        configured_workers=4,
        batch_schedule=BatchSchedule(8),
        tokens_per_example=16,
        prefix="elastic",
    )

    assert metrics["elastic/configured_workers"] == 4
    assert metrics["elastic/reporting_workers"] == 2
    assert metrics["elastic/reporting_worker_fraction"] == 0.5
    assert metrics["elastic/logical_step"] == 9
    assert metrics["elastic/min_worker_step"] == 4
    assert metrics["elastic/step_spread"] == 5
    assert metrics["elastic/logical_total_examples"] == 80
    assert metrics["elastic/delivered_total_examples"] == 120
    assert metrics["elastic/logical_total_tokens"] == 1280
    assert metrics["elastic/delivered_total_tokens"] == 1920


def test_remove_if_exists_ignores_gcs_not_found_races(monkeypatch):
    monkeypatch.setattr(
        "levanter.elastic.fsspec_utils.remove",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("The specified key does not exist.")),
    )

    _remove_if_exists("gs://bucket/path")


def test_jax_transfer_publish_uses_staged_payload(tmp_path, monkeypatch):
    elastic_root = str(tmp_path / "elastic")
    mesh = _single_device_mesh()
    controller = FileBackedPeerSyncController(
        config=ElasticTrainingConfig(
            enabled=True,
            group_id="run",
            worker_id="w000",
            state_path=elastic_root,
            sync_interval_steps=1,
            publish_interval_steps=1,
            sync=PeerAveragingSyncConfig(),
        ),
        checkpoint_base_path=str(tmp_path / "checkpoints" / "run-w000"),
        run_id="run-w000",
        axis_mapping={},
        mesh=mesh,
    )

    class _FakeTransferRuntime:
        def __init__(self):
            self.prepare_calls = 0
            self.publish_calls: list[tuple[int, dict[str, object]]] = []

        def prepare_publish_payload(self, payload: dict[str, object]) -> dict[str, object]:
            self.prepare_calls += 1
            return {"prepared": self.prepare_calls}

        def publish_prepared(self, *, step: int, prepared_payload: dict[str, object]) -> dict[str, object]:
            self.publish_calls.append((step, prepared_payload))
            return {"address": "127.0.0.1:1234"}

    transfer_runtime = _FakeTransferRuntime()

    controller.transport_kind = "jax_transfer"
    controller._transfer_runtime = transfer_runtime  # type: ignore[assignment]
    state = DummyState(step=1, model={"weight": jnp.array([1.0], dtype=jnp.float32)})

    controller.stage_publish_state(state)
    monkeypatch.setattr(
        controller,
        "_shareable_state",
        lambda _state: (_ for _ in ()).throw(AssertionError("staged publish should not rebuild payload")),
    )

    controller._publish_state(state, step=0)
    assert transfer_runtime.prepare_calls == 1
    assert transfer_runtime.publish_calls == [(0, {"prepared": 1})]
    status = read_worker_status(controller.paths.worker_status_path(controller.worker_id))
    assert status is not None
    assert status.transport_kind == "jax_transfer"
    assert (status.transport_metadata or {})["address"] == "127.0.0.1:1234"


def test_jax_transfer_publish_prepares_payload_when_not_staged(tmp_path):
    elastic_root = str(tmp_path / "elastic")
    mesh = _single_device_mesh()
    controller = FileBackedPeerSyncController(
        config=ElasticTrainingConfig(
            enabled=True,
            group_id="run",
            worker_id="w000",
            state_path=elastic_root,
            sync_interval_steps=1,
            publish_interval_steps=1,
            sync=PeerAveragingSyncConfig(),
        ),
        checkpoint_base_path=str(tmp_path / "checkpoints" / "run-w000"),
        run_id="run-w000",
        axis_mapping={},
        mesh=mesh,
    )

    class _FakeTransferRuntime:
        def __init__(self):
            self.prepare_calls = 0
            self.publish_calls: list[tuple[int, dict[str, object]]] = []

        def prepare_publish_payload(self, payload: dict[str, object]) -> dict[str, object]:
            self.prepare_calls += 1
            return {"prepared": self.prepare_calls}

        def publish_prepared(self, *, step: int, prepared_payload: dict[str, object]) -> dict[str, object]:
            self.publish_calls.append((step, prepared_payload))
            return {"address": "127.0.0.1:1234"}

    transfer_runtime = _FakeTransferRuntime()
    controller.transport_kind = "jax_transfer"
    controller._transfer_runtime = transfer_runtime  # type: ignore[assignment]

    controller._publish_state(
        DummyState(step=1, model={"weight": jnp.array([1.0], dtype=jnp.float32)}),
        step=0,
    )
    assert transfer_runtime.prepare_calls == 1
    assert transfer_runtime.publish_calls == [(0, {"prepared": 1})]


def test_clip_tree_to_global_norm_scales_arrays():
    tree = {"x": jnp.array([3.0, 4.0], dtype=jnp.float32)}
    clipped = _clip_tree_to_global_norm(tree, max_norm=2.0)
    clipped_norm = float(_tree_global_norm(clipped))
    assert clipped_norm == pytest.approx(2.0, abs=1e-5)


def test_diloco_default_peer_staleness_filter_uses_sync_interval(tmp_path):
    elastic_root = str(tmp_path / "elastic")
    mesh = _single_device_mesh()

    peer_controller = FileBackedPeerSyncController(
        config=ElasticTrainingConfig(
            enabled=True,
            group_id="run",
            worker_id="w001",
            state_path=elastic_root,
            sync_interval_steps=3,
            publish_interval_steps=3,
            sync=DiLoCoSyncConfig(outer_optimizer="sgd", outer_learning_rate=0.25),
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
            sync_interval_steps=3,
            publish_interval_steps=3,
            sync=DiLoCoSyncConfig(outer_optimizer="sgd", outer_learning_rate=0.25),
        ),
        checkpoint_base_path=str(tmp_path / "checkpoints" / "run-w000"),
        run_id="run-w000",
        axis_mapping={},
        mesh=mesh,
    )

    peer_controller._publish_state(  # noqa: SLF001
        DummyState(step=1, model={"weight": jnp.array([2.0], dtype=jnp.float32)}),
        step=0,
    )
    peer_controller._manager.wait_until_finished()  # noqa: SLF001

    assert local_controller._candidate_peer_statuses(current_step=5)  # noqa: SLF001
    assert not local_controller._candidate_peer_statuses(current_step=7)  # noqa: SLF001
