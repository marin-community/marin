# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import abc
import dataclasses
import json
import logging
import os
import socket
import threading
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

import fsspec
import jax
import jax.tree_util as jtu
import numpy as np
import optax
import draccus
import equinox as eqx
import levanter.tracker
from draccus import field
from haliax.jax_utils import is_jax_array_like
from jax.experimental.array_serialization.serialization import GlobalAsyncCheckpointManager
from jax.sharding import Mesh

import haliax as hax
from levanter.checkpoint import load_checkpoint, save_checkpoint
from levanter.schedule import BatchSchedule
from levanter.trainer_state import init_optimizer_for_trainables, trainables_only
from levanter.utils import fsspec_utils
from levanter.utils.jax_utils import shape_dtype_struct_tree
from levanter.utils.tree_utils import key_path_to_str

if TYPE_CHECKING:
    from haliax.partitioning import ResourceMapping

    from levanter.callbacks import StepInfo
    from levanter.trainer_state import TrainerState


logger = logging.getLogger(__name__)

S = TypeVar("S", bound="TrainerState")
ElasticTransportMode = Literal["auto", "checkpoint", "jax_transfer"]
DiLoCoOuterOptimizer = Literal["adam", "sgd"]

MARIN_ELASTIC_GROUP_ID_ENV = "MARIN_ELASTIC_GROUP_ID"
MARIN_ELASTIC_WORKER_ID_ENV = "MARIN_ELASTIC_WORKER_ID"
MARIN_ELASTIC_WORKER_COUNT_ENV = "MARIN_ELASTIC_WORKER_COUNT"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _utcnow_iso() -> str:
    return _utcnow().isoformat()


def _parent_path(path: str) -> str:
    protocol, rest = fsspec.core.split_protocol(path)
    rest = rest.rstrip("/")
    parent = os.path.dirname(rest)
    if protocol is None:
        return parent
    if not parent:
        return f"{protocol}://"
    return f"{protocol}://{parent}"


def _remove_if_exists(path: str, *, recursive: bool = False) -> None:
    try:
        fsspec_utils.remove(path, recursive=recursive)
    except FileNotFoundError:
        return
    except OSError as exc:
        if "not exist" in str(exc).lower() or "notfound" in str(exc).lower():
            return
        raise


class ElasticSyncConfig(draccus.ChoiceRegistry, abc.ABC):
    @classmethod
    def default_choice_name(cls) -> str | None:
        return "peer_average"


@ElasticSyncConfig.register_subclass("peer_average")
@dataclass(frozen=True)
class PeerAveragingSyncConfig(ElasticSyncConfig):
    mixing_rate: float = 0.5
    share_optimizer_state: bool = False

    def __post_init__(self):
        if not 0.0 < self.mixing_rate <= 1.0:
            raise ValueError(f"mixing_rate must be in (0, 1], got {self.mixing_rate}")


@ElasticSyncConfig.register_subclass("diloco")
@dataclass(frozen=True)
class DiLoCoSyncConfig(ElasticSyncConfig):
    outer_learning_rate: float = 1.0
    outer_optimizer: DiLoCoOuterOptimizer = "adam"
    outer_beta1: float = 0.9
    outer_beta2: float = 0.95
    outer_epsilon: float = 1e-8
    outer_weight_decay: float = 0.0
    reset_inner_optimizer_on_sync: bool = False

    def __post_init__(self):
        if self.outer_learning_rate <= 0:
            raise ValueError(f"outer_learning_rate must be positive, got {self.outer_learning_rate}")
        if self.outer_epsilon <= 0:
            raise ValueError(f"outer_epsilon must be positive, got {self.outer_epsilon}")
        if not 0.0 <= self.outer_beta1 < 1.0:
            raise ValueError(f"outer_beta1 must be in [0, 1), got {self.outer_beta1}")
        if not 0.0 <= self.outer_beta2 < 1.0:
            raise ValueError(f"outer_beta2 must be in [0, 1), got {self.outer_beta2}")
        if self.outer_weight_decay < 0:
            raise ValueError(f"outer_weight_decay must be non-negative, got {self.outer_weight_decay}")


@dataclass(frozen=True)
class ElasticTrainingConfig:
    """Configuration for peer-synchronized elastic training across single-slice cohorts."""

    enabled: bool = False
    group_id: str | None = None
    worker_id: str | None = None
    worker_count: int | None = None
    state_path: str | None = None
    sync_interval_steps: int = 100
    publish_interval_steps: int | None = None
    peer_timeout: timedelta = timedelta(minutes=30)
    max_peer_staleness_steps: int | None = None
    max_peers: int = 1
    sync: ElasticSyncConfig = field(default_factory=PeerAveragingSyncConfig)
    bootstrap_from_peers: bool = True
    keep_published_checkpoints: int = 2
    transport: ElasticTransportMode = "auto"
    transfer_timeout: timedelta = timedelta(seconds=60)
    request_poll_interval_seconds: float = 0.2
    transfer_via_cpu: bool = False

    def __post_init__(self):
        if self.worker_count is not None and self.worker_count <= 0:
            raise ValueError(f"worker_count must be positive, got {self.worker_count}")
        if self.sync_interval_steps <= 0:
            raise ValueError(f"sync_interval_steps must be positive, got {self.sync_interval_steps}")
        if self.publish_interval_steps is not None and self.publish_interval_steps <= 0:
            raise ValueError(f"publish_interval_steps must be positive, got {self.publish_interval_steps}")
        if self.max_peer_staleness_steps is not None and self.max_peer_staleness_steps < 0:
            raise ValueError(f"max_peer_staleness_steps must be non-negative, got {self.max_peer_staleness_steps}")
        if self.max_peers <= 0:
            raise ValueError(f"max_peers must be positive, got {self.max_peers}")
        if self.keep_published_checkpoints <= 0:
            raise ValueError(f"keep_published_checkpoints must be positive, got {self.keep_published_checkpoints}")
        if self.request_poll_interval_seconds <= 0:
            raise ValueError(
                f"request_poll_interval_seconds must be positive, got {self.request_poll_interval_seconds}"
            )
        if self.transfer_timeout <= timedelta(0):
            raise ValueError(f"transfer_timeout must be positive, got {self.transfer_timeout}")


@dataclass
class _DiLoCoState:
    anchor_model: Any
    outer_opt_state: Any


@dataclass(frozen=True)
class _ElasticProgressConfig:
    tokens_per_example: int
    batch_schedule: BatchSchedule
    prefix: str = "elastic"


@dataclass(frozen=True)
class ElasticPaths:
    root_path: str
    workers_path: str
    completion_path: str

    def worker_dir(self, worker_id: str) -> str:
        return fsspec_utils.join_path(self.workers_path, worker_id)

    def worker_status_path(self, worker_id: str) -> str:
        return fsspec_utils.join_path(self.worker_dir(worker_id), "status.json")

    def worker_checkpoint_dir(self, worker_id: str) -> str:
        return fsspec_utils.join_path(self.worker_dir(worker_id), "state")

    def worker_checkpoint_path(self, worker_id: str, step: int) -> str:
        return fsspec_utils.join_path(self.worker_checkpoint_dir(worker_id), f"step-{step}")

    def worker_requests_dir(self, worker_id: str) -> str:
        return fsspec_utils.join_path(self.worker_dir(worker_id), "requests")

    def worker_request_path(self, worker_id: str, request_id: str) -> str:
        return fsspec_utils.join_path(self.worker_requests_dir(worker_id), f"{request_id}.json")


def resolve_elastic_ids(config: ElasticTrainingConfig, *, run_id: str) -> tuple[str, str]:
    group_id = config.group_id or os.environ.get(MARIN_ELASTIC_GROUP_ID_ENV) or run_id
    worker_id = config.worker_id or os.environ.get(MARIN_ELASTIC_WORKER_ID_ENV) or run_id
    return group_id, worker_id


def resolve_elastic_paths(
    checkpoint_base_path: str,
    config: ElasticTrainingConfig,
    *,
    run_id: str,
) -> ElasticPaths:
    group_id, _ = resolve_elastic_ids(config, run_id=run_id)
    if config.state_path is not None:
        root_path = config.state_path
    else:
        root_path = fsspec_utils.join_path(_parent_path(checkpoint_base_path), f"_elastic/{group_id}")
    workers_path = fsspec_utils.join_path(root_path, "workers")
    completion_path = fsspec_utils.join_path(root_path, "completed.json")
    return ElasticPaths(root_path=root_path, workers_path=workers_path, completion_path=completion_path)


@dataclass(frozen=True)
class ElasticWorkerStatus:
    worker_id: str
    run_id: str
    step: int
    checkpoint_path: str | None = None
    transport_kind: str = "checkpoint"
    transport_metadata: dict[str, Any] | None = None
    updated_at: str = field(default_factory=_utcnow_iso)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ElasticWorkerStatus":
        return cls(
            worker_id=str(payload["worker_id"]),
            run_id=str(payload["run_id"]),
            step=int(payload["step"]),
            checkpoint_path=payload.get("checkpoint_path"),
            transport_kind=str(payload.get("transport_kind", "checkpoint")),
            transport_metadata=payload.get("transport_metadata"),
            updated_at=str(payload.get("updated_at", _utcnow_iso())),
        )

    def updated_at_dt(self) -> datetime:
        return datetime.fromisoformat(self.updated_at)


@dataclass(frozen=True)
class ElasticCompletion:
    worker_id: str
    run_id: str
    completed_step: int
    updated_at: str = field(default_factory=_utcnow_iso)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ElasticCompletion":
        return cls(
            worker_id=str(payload["worker_id"]),
            run_id=str(payload["run_id"]),
            completed_step=int(payload["completed_step"]),
            updated_at=str(payload.get("updated_at", _utcnow_iso())),
        )


@dataclass(frozen=True)
class ElasticTransferRequest:
    request_id: str
    requester_id: str
    transfer_uuid: int
    created_at: str = field(default_factory=_utcnow_iso)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ElasticTransferRequest":
        return cls(
            request_id=str(payload["request_id"]),
            requester_id=str(payload["requester_id"]),
            transfer_uuid=int(payload["transfer_uuid"]),
            created_at=str(payload.get("created_at", _utcnow_iso())),
        )


def _write_json(path: str, payload: dict[str, Any]) -> None:
    fs, _, (plain_path,) = fsspec.get_fs_token_paths(path)
    parent = os.path.dirname(plain_path)
    if parent:
        fs.makedirs(parent, exist_ok=True)
    with fs.open(plain_path, "w") as f:
        json.dump(payload, f, sort_keys=True)


def _read_json(path: str) -> dict[str, Any] | None:
    fs, _, (plain_path,) = fsspec.get_fs_token_paths(path)
    if not fs.exists(plain_path):
        return None
    try:
        with fs.open(plain_path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def write_worker_status(path: str, status: ElasticWorkerStatus) -> None:
    _write_json(path, dataclasses.asdict(status))


def read_worker_status(path: str) -> ElasticWorkerStatus | None:
    payload = _read_json(path)
    if payload is None:
        return None
    return ElasticWorkerStatus.from_dict(payload)


def write_completion(path: str, completion: ElasticCompletion) -> None:
    _write_json(path, dataclasses.asdict(completion))


def read_completion(path: str) -> ElasticCompletion | None:
    payload = _read_json(path)
    if payload is None:
        return None
    return ElasticCompletion.from_dict(payload)


def write_transfer_request(path: str, request: ElasticTransferRequest) -> None:
    _write_json(path, dataclasses.asdict(request))


def read_transfer_request(path: str) -> ElasticTransferRequest | None:
    payload = _read_json(path)
    if payload is None:
        return None
    return ElasticTransferRequest.from_dict(payload)


def list_worker_statuses(paths: ElasticPaths) -> list[ElasticWorkerStatus]:
    fs, _, (workers_plain_path,) = fsspec.get_fs_token_paths(paths.workers_path)
    if not fs.exists(workers_plain_path):
        return []

    statuses: list[ElasticWorkerStatus] = []
    for candidate in fs.glob(os.path.join(workers_plain_path, "*/status.json")):
        protocol = fs.protocol if isinstance(fs.protocol, str) else fs.protocol[0]
        candidate_path = f"{protocol}://{candidate}" if protocol else candidate
        status = read_worker_status(candidate_path)
        if status is not None:
            statuses.append(status)
    return statuses


class ElasticTrainingFinished(RuntimeError):
    """Raised when another cohort has already completed the logical training run."""

    def __init__(self, info: "StepInfo[Any]", completion: ElasticCompletion):
        self.info: "StepInfo[Any]" = info
        self.completion = completion
        super().__init__(
            f"Elastic training already completed by {completion.worker_id} at step {completion.completed_step}"
        )


def _get_local_ip_from_hostname() -> str:
    return socket.gethostbyname(socket.gethostname())


def _try_import_jax_transfer():
    try:
        import jax.experimental.transfer as jax_transfer

        return jax_transfer, None
    except (ImportError, AttributeError) as exc:
        return None, exc


class _JaxTransferRuntime:
    """Peer-to-peer state transfer using JAX TransferServer with file-backed request discovery."""

    def __init__(self, *, config: ElasticTrainingConfig, paths: ElasticPaths, worker_id: str):
        jax_transfer, error = _try_import_jax_transfer()
        if jax_transfer is None:
            raise RuntimeError("jax.experimental.transfer is not available") from error

        self.config = config
        self.paths = paths
        self.worker_id = worker_id
        self._jax_transfer = jax_transfer
        backend_client = jax.devices()[0].client
        ip = _get_local_ip_from_hostname()
        self._server = jax_transfer.start_transfer_server(
            backend_client,
            f"{ip}:0",
            [f"{ip}:0"] * jax.device_count(),
        )
        self.address = self._server.address()
        self._payload_lock = threading.Lock()
        self._payload: dict[str, Any] | None = None
        self._payload_step = -1
        self._closed = threading.Event()
        self._poll_thread = threading.Thread(
            target=self._poll_requests_loop,
            name=f"elastic-transfer-{worker_id}",
            daemon=True,
        )
        self._poll_thread.start()

        self._cpu_device = None
        if config.transfer_via_cpu:
            cpu_devices = jax.devices("cpu")
            if not cpu_devices:
                raise RuntimeError("No CPU devices available for transfer_via_cpu")
            self._cpu_device = cpu_devices[0]

    def close(self) -> None:
        self._closed.set()
        self._poll_thread.join(timeout=2.0)
        self._server = None

    def publish(self, *, step: int, payload: dict[str, Any]) -> dict[str, Any]:
        flat_payload = _flatten_transfer_payload(payload)
        materialized = self._materialize_payload(flat_payload)
        with self._payload_lock:
            self._payload = materialized
            self._payload_step = step
        return {"address": self.address}

    def fetch(
        self, *, exemplar_payload: dict[str, Any], peer_status: ElasticWorkerStatus
    ) -> tuple[dict[str, Any], int]:
        metadata = peer_status.transport_metadata or {}
        address = metadata.get("address")
        if not address:
            raise RuntimeError(f"Peer {peer_status.worker_id} did not publish a transfer address")

        request = ElasticTransferRequest(
            request_id=uuid.uuid4().hex,
            requester_id=self.worker_id,
            transfer_uuid=int(np.random.randint(0, 2**31 - 1)),
        )
        request_path = self.paths.worker_request_path(peer_status.worker_id, request.request_id)
        write_transfer_request(request_path, request)

        placeholder = shape_dtype_struct_tree(_flatten_transfer_payload(exemplar_payload))
        connection = self._server.connect(address)

        def _pull():
            out = connection.pull(request.transfer_uuid, placeholder)
            if out is None:
                raise RuntimeError(f"Elastic transfer from {peer_status.worker_id} returned no payload")
            return _restore_transfer_payload(exemplar_payload, jax.block_until_ready(out))

        timeout_s = self.config.transfer_timeout.total_seconds()
        try:
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="elastic-pull") as executor:
                return executor.submit(_pull).result(timeout=timeout_s), peer_status.step
        except FuturesTimeoutError as exc:
            raise TimeoutError(
                f"Timed out after {timeout_s:.1f}s pulling elastic state from {peer_status.worker_id}"
            ) from exc
        finally:
            _remove_if_exists(request_path)

    def _poll_requests_loop(self) -> None:
        while not self._closed.is_set():
            try:
                self._drain_requests()
            except Exception:
                logger.exception("Elastic transfer request poller failed for %s", self.worker_id)
            self._closed.wait(self.config.request_poll_interval_seconds)

    def _drain_requests(self) -> None:
        request_dir = self.paths.worker_requests_dir(self.worker_id)
        fs, _, (plain_request_dir,) = fsspec.get_fs_token_paths(request_dir)
        if not fs.exists(plain_request_dir):
            return

        protocol = fs.protocol if isinstance(fs.protocol, str) else fs.protocol[0]
        for candidate in sorted(fs.glob(os.path.join(plain_request_dir, "*.json"))):
            candidate_path = f"{protocol}://{candidate}" if protocol else candidate
            request = read_transfer_request(candidate_path)
            if request is None:
                continue

            payload = self._current_payload()
            if payload is None:
                continue

            self._server.await_pull(request.transfer_uuid, payload)
            _remove_if_exists(candidate_path)

    def _current_payload(self) -> dict[str, Any] | None:
        with self._payload_lock:
            return self._payload

    def _materialize_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        def _copy_leaf(x):
            if not is_jax_array_like(x):
                return x

            host_value = np.asarray(jax.device_get(x))
            if self._cpu_device is not None:
                return jax.device_put(host_value, self._cpu_device)
            sharding = getattr(x, "sharding", None)
            if sharding is not None:
                return jax.device_put(host_value, sharding)
            return jax.device_put(host_value)

        if self._cpu_device is None:
            return cast(dict[str, Any], jtu.tree_map(_copy_leaf, payload))

        with hax.set_mesh(Mesh(np.array([self._cpu_device]), axis_names=("cpu",))):
            return cast(dict[str, Any], jtu.tree_map(_copy_leaf, payload))


class FileBackedPeerSyncController:
    """File-backed control plane with pluggable checkpoint or JAX transfer payload exchange."""

    def __init__(
        self,
        *,
        config: ElasticTrainingConfig,
        checkpoint_base_path: str,
        run_id: str,
        axis_mapping: "ResourceMapping",
        mesh: Mesh,
    ):
        self.config = config
        self.run_id = run_id
        self.axis_mapping = axis_mapping
        self.mesh = mesh
        self.group_id, self.worker_id = resolve_elastic_ids(config, run_id=run_id)
        self.paths = resolve_elastic_paths(checkpoint_base_path, config, run_id=run_id)
        self.publish_interval_steps = config.publish_interval_steps or config.sync_interval_steps
        self.control_interval_steps = min(self.publish_interval_steps, config.sync_interval_steps)
        self._manager = GlobalAsyncCheckpointManager(timeout_secs=60 * 30)
        self._published_checkpoints: deque[str] = deque(maxlen=config.keep_published_checkpoints)
        self._last_applied_peer: tuple[str, int] | None = None
        self.transport_kind = self._resolve_transport_kind()
        self._transfer_runtime = (
            _JaxTransferRuntime(config=config, paths=self.paths, worker_id=self.worker_id)
            if self.transport_kind == "jax_transfer"
            else None
        )
        self._diloco_state: _DiLoCoState | None = None
        self._progress_config: _ElasticProgressConfig | None = None

    def close(self) -> None:
        self._manager.wait_until_finished()
        if self._transfer_runtime is not None:
            self._transfer_runtime.close()

    def configure_progress_reporting(
        self,
        *,
        tokens_per_example: int,
        batch_schedule: BatchSchedule,
        prefix: str = "elastic",
    ) -> None:
        self._progress_config = _ElasticProgressConfig(
            tokens_per_example=tokens_per_example,
            batch_schedule=batch_schedule,
            prefix=prefix,
        )
        if jax.process_index() == 0:
            levanter.tracker.log_summary(
                {
                    f"{prefix}/configured_workers": self.config.worker_count or 1,
                    f"{prefix}/sync_interval_steps": self.config.sync_interval_steps,
                    f"{prefix}/publish_interval_steps": self.publish_interval_steps,
                    f"{prefix}/tokens_per_example": tokens_per_example,
                }
            )

    def bootstrap_state(self, state: S) -> S:
        if isinstance(self.config.sync, DiLoCoSyncConfig):
            self._ensure_diloco_state(state)

        if not self.config.bootstrap_from_peers or int(state.step) != 0:
            return state

        peer_statuses = self._candidate_peer_statuses(current_step=-1)
        if not peer_statuses:
            return state

        peer = peer_statuses[0]
        logger.info(
            "Bootstrapping %s from peer %s at step %s using %s",
            self.worker_id,
            peer.worker_id,
            peer.step,
            peer.transport_kind,
        )
        shareable_state, source_step = self._load_shareable_state(state, peer)
        self._last_applied_peer = (peer.worker_id, source_step)
        return self._apply_shareable_state(state, shareable_state, force_copy=True)

    def maybe_update_state(self, info: "StepInfo[S]") -> "StepInfo[S]":
        if info.next_step % self.control_interval_steps == 0:
            completion = read_completion(self.paths.completion_path)
            if completion is not None and completion.worker_id != self.worker_id:
                raise ElasticTrainingFinished(info, completion)

        if info.next_step % self.publish_interval_steps == 0:
            self._publish_state(info.state, info.step)

        if info.next_step % self.config.sync_interval_steps != 0:
            if info.next_step % self.control_interval_steps == 0:
                self._log_aggregate_progress(current_step=info.step)
            return info

        peers = self._candidate_peer_statuses(current_step=info.step)
        if not peers:
            self._log_aggregate_progress(current_step=info.step)
            return info

        shareable_states: list[dict[str, Any]] = []
        newest_peer: tuple[str, int] | None = None
        for peer in peers:
            try:
                shareable_state, source_step = self._load_shareable_state(info.state, peer)
            except Exception:
                logger.warning(
                    "Elastic sync pull from %s via %s failed",
                    peer.worker_id,
                    peer.transport_kind,
                    exc_info=True,
                )
                continue

            shareable_states.append(shareable_state)
            if newest_peer is None or source_step > newest_peer[1]:
                newest_peer = (peer.worker_id, source_step)

        if not shareable_states:
            self._log_aggregate_progress(current_step=info.step)
            return info

        if isinstance(self.config.sync, DiLoCoSyncConfig):
            new_state = self._apply_diloco_sync(info.state, shareable_states)
        else:
            averaged_peer_state = _average_shareable_states(shareable_states)
            new_state = self._apply_shareable_state(info.state, averaged_peer_state)

        if newest_peer is not None:
            self._last_applied_peer = newest_peer
            logger.info(
                "Elastic sync for %s using %d peer(s), newest peer=%s@%s via %s",
                self.worker_id,
                len(shareable_states),
                newest_peer[0],
                newest_peer[1],
                self.transport_kind,
            )

        self._log_aggregate_progress(current_step=info.step)
        return dataclasses.replace(info, state=new_state)

    def mark_completed(self, info: "StepInfo[S]") -> None:
        self._manager.wait_until_finished()
        if jax.process_index() != 0:
            return
        write_completion(
            self.paths.completion_path,
            ElasticCompletion(worker_id=self.worker_id, run_id=self.run_id, completed_step=info.step),
        )

    def _resolve_transport_kind(self) -> str:
        if self.config.transport == "checkpoint":
            return "checkpoint"

        if self.config.transport == "jax_transfer":
            self._validate_jax_transfer_or_raise()
            return "jax_transfer"

        if jax.default_backend() != "tpu":
            return "checkpoint"
        if jax.process_count() != 1:
            logger.info(
                "Falling back to checkpoint elastic transport because jax_transfer currently expects process_count == 1"
            )
            return "checkpoint"

        jax_transfer, error = _try_import_jax_transfer()
        if jax_transfer is None:
            logger.warning(
                "Falling back to checkpoint elastic transport because JAX transfer is unavailable: %s", error
            )
            return "checkpoint"

        return "jax_transfer"

    def _validate_jax_transfer_or_raise(self) -> None:
        if jax.process_count() != 1:
            raise RuntimeError("Elastic JAX transfer transport currently requires jax.process_count() == 1")
        jax_transfer, error = _try_import_jax_transfer()
        if jax_transfer is None:
            raise RuntimeError("jax.experimental.transfer is not available in this environment") from error

    def _candidate_peer_statuses(self, *, current_step: int) -> list[ElasticWorkerStatus]:
        deadline = _utcnow() - self.config.peer_timeout
        statuses = list_worker_statuses(self.paths)
        candidates: list[ElasticWorkerStatus] = []
        for status in statuses:
            if status.worker_id == self.worker_id:
                continue
            if status.updated_at_dt() < deadline:
                continue
            if self.config.max_peer_staleness_steps is not None and current_step >= 0:
                if current_step - status.step > self.config.max_peer_staleness_steps:
                    continue
            if self._last_applied_peer == (status.worker_id, status.step):
                continue
            if status.transport_kind == "checkpoint":
                if status.checkpoint_path is None or not fsspec_utils.exists(status.checkpoint_path):
                    continue
            elif status.transport_kind == "jax_transfer":
                if self._transfer_runtime is None:
                    continue
                if not (status.transport_metadata or {}).get("address"):
                    continue
            else:
                continue

            candidates.append(status)

        candidates.sort(key=lambda status: status.step, reverse=True)
        return candidates[: self.config.max_peers]

    def _shareable_state(self, state: S) -> dict[str, Any]:
        sync_state = self._sync_state_payload(state)
        opt_state = state.opt_state if self._share_optimizer_state else None
        return {
            "model": state.model,
            "model_averaging": state.model_averaging,
            "opt_state": opt_state,
            "sync_state": sync_state,
        }

    def _publish_state(self, state: S, step: int) -> None:
        if self.transport_kind == "jax_transfer":
            assert self._transfer_runtime is not None
            metadata = self._transfer_runtime.publish(step=step, payload=self._shareable_state(state))
            if jax.process_index() == 0:
                write_worker_status(
                    self.paths.worker_status_path(self.worker_id),
                    ElasticWorkerStatus(
                        worker_id=self.worker_id,
                        run_id=self.run_id,
                        step=step,
                        checkpoint_path=None,
                        transport_kind="jax_transfer",
                        transport_metadata=metadata,
                    ),
                )
            return

        checkpoint_path = self.paths.worker_checkpoint_path(self.worker_id, step)
        previous_checkpoints = list(self._published_checkpoints)
        shareable_state = self._shareable_state(state)

        def _on_commit():
            if jax.process_index() == 0:
                write_worker_status(
                    self.paths.worker_status_path(self.worker_id),
                    ElasticWorkerStatus(
                        worker_id=self.worker_id,
                        run_id=self.run_id,
                        step=step,
                        checkpoint_path=checkpoint_path,
                        transport_kind="checkpoint",
                    ),
                )
                overflow = len(previous_checkpoints) - (self.config.keep_published_checkpoints - 1)
                stale = previous_checkpoints[:overflow] if overflow > 0 else []
                for old_path in stale:
                    if old_path != checkpoint_path and fsspec_utils.exists(old_path):
                        _remove_if_exists(old_path, recursive=True)

        save_checkpoint(
            tree=shareable_state,
            step=step,
            checkpoint_path=checkpoint_path,
            manager=self._manager,
            commit_callback=_on_commit,
            is_temporary=True,
        )
        self._published_checkpoints.append(checkpoint_path)

    def _load_shareable_state(self, state: S, peer_status: ElasticWorkerStatus) -> tuple[dict[str, Any], int]:
        exemplar = self._shareable_state(state)
        if peer_status.transport_kind == "jax_transfer":
            if self._transfer_runtime is None:
                raise RuntimeError("Local elastic controller does not have a JAX transfer runtime")
            return self._transfer_runtime.fetch(exemplar_payload=exemplar, peer_status=peer_status)

        if peer_status.checkpoint_path is None:
            raise RuntimeError(f"Peer {peer_status.worker_id} did not publish a checkpoint path")

        return (
            load_checkpoint(
                exemplar,
                peer_status.checkpoint_path,
                discover_latest=False,
                axis_mapping=self.axis_mapping,
                mesh=self.mesh,
            ),
            peer_status.step,
        )

    def _apply_shareable_state(
        self,
        state: S,
        shareable_state: dict[str, Any],
        *,
        force_copy: bool = False,
    ) -> S:
        if isinstance(self.config.sync, DiLoCoSyncConfig):
            self._update_diloco_state_from_shareable(shareable_state, state=state)
            model = shareable_state["model"] if force_copy else self._diloco_anchor_model(state)
            return dataclasses.replace(state, model=model)

        sync_config = cast(PeerAveragingSyncConfig, self.config.sync)
        mixing_rate = 1.0 if force_copy else sync_config.mixing_rate
        model = _merge_tree(state.model, shareable_state["model"], mixing_rate)
        model_averaging = state.model_averaging
        peer_model_averaging = shareable_state["model_averaging"]
        if model_averaging is not None and peer_model_averaging is not None:
            model_averaging = _merge_tree(model_averaging, peer_model_averaging, mixing_rate)
        elif mixing_rate == 1.0 and peer_model_averaging is not None:
            model_averaging = peer_model_averaging

        opt_state = state.opt_state
        peer_opt_state = shareable_state["opt_state"]
        if self._share_optimizer_state and peer_opt_state is not None:
            opt_state = _merge_tree(state.opt_state, peer_opt_state, mixing_rate)

        return dataclasses.replace(state, model=model, model_averaging=model_averaging, opt_state=opt_state)

    @property
    def _share_optimizer_state(self) -> bool:
        sync_config = self.config.sync
        if isinstance(sync_config, PeerAveragingSyncConfig):
            return sync_config.share_optimizer_state
        return False

    def _sync_state_payload(self, state: S) -> dict[str, Any] | None:
        if not isinstance(self.config.sync, DiLoCoSyncConfig):
            return None
        diloco_state = self._ensure_diloco_state(state)
        return {
            "anchor_model": diloco_state.anchor_model,
            "outer_opt_state": diloco_state.outer_opt_state,
        }

    def _ensure_diloco_state(self, state: S) -> _DiLoCoState:
        if self._diloco_state is None:
            trainable_anchor = self._trainable_model(state.model, state)
            self._diloco_state = _DiLoCoState(
                anchor_model=state.model,
                outer_opt_state=self._diloco_outer_optimizer.init(trainable_anchor),
            )
        return self._diloco_state

    def _diloco_anchor_model(self, state: S) -> Any:
        return self._ensure_diloco_state(state).anchor_model

    @property
    def _diloco_outer_optimizer(self) -> optax.GradientTransformation:
        sync_config = cast(DiLoCoSyncConfig, self.config.sync)
        if sync_config.outer_optimizer == "sgd":
            components: list[optax.GradientTransformation] = []
        else:
            components = [
                optax.scale_by_adam(
                    sync_config.outer_beta1,
                    sync_config.outer_beta2,
                    sync_config.outer_epsilon,
                )
            ]
        if sync_config.outer_weight_decay > 0:
            components.append(optax.add_decayed_weights(sync_config.outer_weight_decay))
        components.append(optax.scale(-sync_config.outer_learning_rate))
        return optax.chain(*components)

    def _apply_diloco_sync(self, state: S, peer_shareable_states: list[dict[str, Any]]) -> S:
        sync_config = cast(DiLoCoSyncConfig, self.config.sync)
        self._update_diloco_state_from_shareable(peer_shareable_states[0], state=state)
        diloco_state = self._ensure_diloco_state(state)

        models = [state.model, *(peer_state["model"] for peer_state in peer_shareable_states)]
        mean_model = _average_trees(models)
        anchor_model = self._diloco_anchor_model(state)
        anchor_trainable = self._trainable_model(anchor_model, state)
        mean_trainable = self._trainable_model(mean_model, state)
        pseudo_grads = _tree_difference(anchor_trainable, mean_trainable)
        updates, new_outer_opt_state = self._diloco_outer_optimizer.update(
            pseudo_grads,
            diloco_state.outer_opt_state,
            params=anchor_trainable,
        )
        updated_trainable = optax.apply_updates(anchor_trainable, updates)
        updated_model = self._merge_trainable_model(updated_trainable, anchor_model)
        self._diloco_state = _DiLoCoState(anchor_model=updated_model, outer_opt_state=new_outer_opt_state)

        opt_state = state.opt_state
        if sync_config.reset_inner_optimizer_on_sync and hasattr(state, "optimizer"):
            opt_state = init_optimizer_for_trainables(state.optimizer, self._trainable_model(updated_model, state))

        return dataclasses.replace(state, model=updated_model, opt_state=opt_state)

    def _update_diloco_state_from_shareable(self, shareable_state: dict[str, Any], *, state: S) -> None:
        sync_state = shareable_state.get("sync_state")
        if sync_state is None:
            self._ensure_diloco_state(state)
            return
        self._diloco_state = _DiLoCoState(
            anchor_model=sync_state["anchor_model"],
            outer_opt_state=sync_state["outer_opt_state"],
        )

    def _trainable_model(self, model: Any, state: S) -> Any:
        if hasattr(state, "is_trainable"):
            return trainables_only(model, state.is_trainable)
        return model

    def _merge_trainable_model(self, trainable_model: Any, base_model: Any) -> Any:
        try:
            return eqx.combine(trainable_model, base_model)
        except Exception:
            return trainable_model

    def _log_aggregate_progress(self, *, current_step: int) -> None:
        if self._progress_config is None or jax.process_index() != 0:
            return

        statuses = {status.worker_id: status for status in list_worker_statuses(self.paths)}
        statuses[self.worker_id] = ElasticWorkerStatus(
            worker_id=self.worker_id,
            run_id=self.run_id,
            step=current_step,
            checkpoint_path=None,
            transport_kind=self.transport_kind,
        )
        metrics = _aggregate_progress_metrics(
            list(statuses.values()),
            configured_workers=self.config.worker_count or 1,
            batch_schedule=self._progress_config.batch_schedule,
            tokens_per_example=self._progress_config.tokens_per_example,
            prefix=self._progress_config.prefix,
        )
        levanter.tracker.log(metrics, step=current_step)


def _merge_tree(current: Any, incoming: Any, mixing_rate: float) -> Any:
    if mixing_rate == 1.0:
        return incoming
    return optax.incremental_update(incoming, current, mixing_rate)


def _average_trees(trees: list[Any]) -> Any:
    if not trees:
        raise ValueError("trees must be non-empty")
    if len(trees) == 1:
        return trees[0]

    average = trees[0]
    for i, tree in enumerate(trees[1:], start=2):
        average = _merge_tree(average, tree, 1.0 / i)
    return average


def _tree_difference(lhs: Any, rhs: Any) -> Any:
    def subtract(a: Any, b: Any) -> Any:
        if a is None or b is None:
            return a
        return a - b

    return jtu.tree_map(subtract, lhs, rhs)


def _average_shareable_states(shareable_states: list[dict[str, Any]]) -> dict[str, Any]:
    if not shareable_states:
        raise ValueError("shareable_states must be non-empty")
    if len(shareable_states) == 1:
        return shareable_states[0]

    average = shareable_states[0]
    for i, peer_state in enumerate(shareable_states[1:], start=2):
        average = {key: _merge_optional_tree(average[key], peer_state[key], 1.0 / i) for key in average.keys()}
    return average


def _merge_optional_tree(current: Any, incoming: Any, mixing_rate: float) -> Any:
    if current is None:
        return incoming
    if incoming is None:
        return current
    return _merge_tree(current, incoming, mixing_rate)


def _aggregate_progress_metrics(
    statuses: list[ElasticWorkerStatus],
    *,
    configured_workers: int,
    batch_schedule: BatchSchedule,
    tokens_per_example: int,
    prefix: str,
) -> dict[str, int | float]:
    if not statuses:
        return {
            f"{prefix}/configured_workers": configured_workers,
            f"{prefix}/reporting_workers": 0,
            f"{prefix}/reporting_worker_fraction": 0.0,
        }

    completed_steps = [status.step + 1 for status in statuses]
    total_examples = [batch_schedule.global_data_offset_by_step(step_count) for step_count in completed_steps]
    logical_examples = max(total_examples)
    delivered_examples = sum(total_examples)
    max_step = max(status.step for status in statuses)
    min_step = min(status.step for status in statuses)

    return {
        f"{prefix}/configured_workers": configured_workers,
        f"{prefix}/reporting_workers": len(statuses),
        f"{prefix}/reporting_worker_fraction": len(statuses) / configured_workers,
        f"{prefix}/logical_step": max_step,
        f"{prefix}/min_worker_step": min_step,
        f"{prefix}/step_spread": max_step - min_step,
        f"{prefix}/logical_total_examples": logical_examples,
        f"{prefix}/delivered_total_examples": delivered_examples,
        f"{prefix}/logical_total_tokens": logical_examples * tokens_per_example,
        f"{prefix}/delivered_total_tokens": delivered_examples * tokens_per_example,
    }


def _flatten_transfer_payload(tree: Any) -> dict[str, Any]:
    flat_payload: dict[str, Any] = {}
    for path, leaf in jtu.tree_flatten_with_path(tree)[0]:
        if not is_jax_array_like(leaf):
            continue
        key = _key_path_to_str(path)
        if key in flat_payload:
            raise ValueError(f"Duplicate elastic transfer key {key!r}")
        flat_payload[key] = leaf
    return flat_payload


def _restore_transfer_payload(template: Any, flat_payload: dict[str, Any]) -> Any:
    leaves_with_paths, treedef = jtu.tree_flatten_with_path(template)
    restored_leaves: list[Any] = []
    for path, leaf in leaves_with_paths:
        if not is_jax_array_like(leaf):
            restored_leaves.append(leaf)
            continue
        key = _key_path_to_str(path)
        try:
            restored_leaves.append(flat_payload[key])
        except KeyError as exc:
            raise KeyError(f"Elastic transfer payload missing key {key!r}") from exc
    return jtu.tree_unflatten(treedef, restored_leaves)


def _key_path_to_str(path: tuple[Any, ...]) -> str:
    return ".".join(key_path_to_str([entry]) for entry in path)
