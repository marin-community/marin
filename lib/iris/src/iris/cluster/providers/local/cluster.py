# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Local in-process cluster for testing.

Runs Controller + Autoscaler(GcpWorkerProvider + InMemoryGcpService(LOCAL)) in the current process.
Workers are threads, not VMs. No Docker, no GCS, no SSH.

This module lives inside providers/local/ to co-locate it with the provider
implementations it depends on (GcpWorkerProvider, InMemoryGcpService).

Provides:
- create_local_autoscaler: Factory for creating autoscaler with GcpWorkerProvider(LOCAL)
- LocalCluster: In-process cluster implementation for testing
- make_local_cluster_config: Build a fully-configured IrisClusterConfig for local execution
"""

import secrets
import tempfile
import threading
from pathlib import Path

from rigging.timing import Duration, Timestamp

from iris.cluster.config import make_local_config
from iris.cluster.constraints import worker_attributes_from_resources
from iris.cluster.controller.auth import create_api_key, create_controller_auth
from iris.cluster.controller.autoscaler import Autoscaler
from iris.cluster.controller.autoscaler.scaling_group import (
    DEFAULT_SCALE_DOWN_RATE_LIMIT,
    DEFAULT_SCALE_UP_RATE_LIMIT,
    ScalingGroup,
)
from iris.cluster.controller.controller import (
    Controller,
    ControllerConfig,
)
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.vm_lifecycle import ControllerStatus
from iris.cluster.controller.worker_provider import RpcWorkerStubFactory, WorkerProvider
from iris.cluster.providers.gcp.fake import InMemoryGcpService
from iris.cluster.providers.gcp.workers import GcpWorkerProvider
from iris.cluster.providers.types import find_free_port
from iris.cluster.service_mode import ServiceMode
from iris.cluster.worker.port_allocator import PortAllocator
from iris.managed_thread import ThreadContainer
from iris.rpc import config_pb2
from iris.rpc.auth import hash_token
from iris.time_proto import duration_from_proto


def create_local_autoscaler(
    config: config_pb2.IrisClusterConfig,
    controller_address: str,
    threads: ThreadContainer | None = None,
    db: ControllerDB | None = None,
) -> tuple[Autoscaler, tempfile.TemporaryDirectory]:
    """Create Autoscaler with GcpWorkerProvider(LOCAL) for all scale groups.

    Creates temp directories and a PortAllocator so that InMemoryGcpService(LOCAL)
    can spawn real Worker threads that register with the controller.

    Args:
        config: Cluster configuration (with defaults already applied)
        controller_address: Address for workers to connect to
        threads: Optional thread container for testing

    Returns:
        Tuple of (autoscaler, temp_dir). The caller owns the temp_dir and
        must call cleanup() when done.
    """
    label_prefix = config.platform.label_prefix or "iris"

    temp_dir = tempfile.TemporaryDirectory(prefix="iris_local_autoscaler_")
    temp_path = Path(temp_dir.name)
    cache_path = temp_path / "cache"
    cache_path.mkdir()
    fake_bundle = temp_path / "fake_bundle"
    fake_bundle.mkdir()
    (fake_bundle / "pyproject.toml").write_text("[project]\nname = 'test'\n")

    port_allocator = PortAllocator()

    # Extract worker attributes and GPU counts from each scale group config so
    # that InMemoryGcpService(LOCAL) can propagate them to local workers.
    worker_attributes_by_group: dict[str, dict[str, str | int | float]] = {}
    gpu_count_by_group: dict[str, int] = {}
    for name, sg_config in config.scale_groups.items():
        attrs: dict[str, str | int | float] = {}
        if sg_config.HasField("resources"):
            attrs.update(worker_attributes_from_resources(sg_config.resources))
        if sg_config.HasField("worker") and sg_config.worker.attributes:
            attrs.update(sg_config.worker.attributes)
        worker_attributes_by_group[name] = attrs
        if sg_config.resources.device_type == config_pb2.ACCELERATOR_TYPE_GPU and sg_config.resources.device_count > 0:
            gpu_count_by_group[name] = sg_config.resources.device_count

    storage_prefix = config.storage.remote_state_dir or ""

    gcp_service = InMemoryGcpService(
        mode=ServiceMode.LOCAL,
        project_id="local",
        controller_address=controller_address,
        cache_path=cache_path,
        fake_bundle=fake_bundle,
        port_allocator=port_allocator,
        threads=threads,
        worker_attributes_by_group=worker_attributes_by_group,
        gpu_count_by_group=gpu_count_by_group,
        storage_prefix=storage_prefix,
        label_prefix=label_prefix,
    )
    local_gcp_config = config_pb2.GcpPlatformConfig(project_id="local")
    platform = GcpWorkerProvider(
        gcp_config=local_gcp_config,
        label_prefix=label_prefix,
        gcp_service=gcp_service,
    )

    scale_up_delay = duration_from_proto(config.defaults.autoscaler.scale_up_delay)

    scale_groups: dict[str, ScalingGroup] = {}
    for name, sg_config in config.scale_groups.items():
        scale_groups[name] = ScalingGroup(
            config=sg_config,
            platform=platform,
            label_prefix=label_prefix,
            scale_up_cooldown=scale_up_delay,
            scale_up_rate_limit=sg_config.scale_up_rate_limit or DEFAULT_SCALE_UP_RATE_LIMIT,
            scale_down_rate_limit=sg_config.scale_down_rate_limit or DEFAULT_SCALE_DOWN_RATE_LIMIT,
            db=db,
        )

    # Build base_worker_config from defaults so auth_token (and other fields)
    # flow through the autoscaler to locally-spawned workers.
    base_worker_config: config_pb2.WorkerConfig | None = None
    if config.defaults.worker.auth_token:
        base_worker_config = config_pb2.WorkerConfig()
        base_worker_config.CopyFrom(config.defaults.worker)

    autoscaler = Autoscaler.from_config(
        scale_groups=scale_groups,
        config=config.defaults.autoscaler,
        platform=platform,
        threads=threads,
        base_worker_config=base_worker_config,
        db=db,
    )
    return autoscaler, temp_dir


class LocalCluster:
    """In-process cluster for local testing.

    Runs Controller + Autoscaler(GcpWorkerProvider + InMemoryGcpService(LOCAL)) in the
    current process. Workers are threads, not VMs. No Docker, no GCS, no SSH.

    A single instance can be stopped and restarted via restart(). The controller
    DB lives in a persistent _db_dir created at construction time, so checkpoints
    written before stop() are found and restored on the next start().
    """

    def __init__(
        self,
        config: config_pb2.IrisClusterConfig,
        threads: ThreadContainer | None = None,
    ):
        self._config = config
        self._threads = threads or ThreadContainer("local-cluster")
        self._controller: Controller | None = None
        self._temp_dir: tempfile.TemporaryDirectory | None = None
        self._autoscaler: Autoscaler | None = None
        self._autoscaler_temp_dir: tempfile.TemporaryDirectory | None = None
        self._stopped = threading.Event()
        self._auto_login_token: str | None = None
        # Persistent across stop()/start() so checkpoints survive restart().
        self._db_dir = tempfile.TemporaryDirectory(prefix="iris_local_controller_db_")

    def start(self) -> str:
        self._stopped = threading.Event()
        # Create temp dir for controller's bundle storage
        self._temp_dir = tempfile.TemporaryDirectory(prefix="iris_local_controller_")
        state_dir = Path(self._temp_dir.name) / "state"
        state_dir.mkdir()

        port = self._config.controller.local.port or find_free_port()
        address = f"http://127.0.0.1:{port}"

        db_dir = Path(self._db_dir.name)
        db = ControllerDB(db_dir=db_dir)

        # Derive auth from config proto so callers never need to wire it manually.
        auth = create_controller_auth(self._config.auth, db=db)
        if auth.worker_token:
            self._config.defaults.worker.auth_token = auth.worker_token

        controller_threads = self._threads.create_child("controller") if self._threads else None
        autoscaler_threads = controller_threads.create_child("autoscaler") if controller_threads else None

        # Autoscaler creates its own temp dirs for worker resources
        self._autoscaler, self._autoscaler_temp_dir = create_local_autoscaler(
            self._config,
            address,
            threads=autoscaler_threads,
            db=db,
        )

        self._controller = Controller(
            config=ControllerConfig(
                host="127.0.0.1",
                port=port,
                remote_state_dir=self._config.storage.remote_state_dir or f"file://{state_dir}",
                heartbeat_interval=Duration.from_seconds(0.5),
                local_state_dir=Path(self._db_dir.name),
                auth_verifier=auth.verifier,
                auth_provider=auth.provider,
                auth=auth,
            ),
            provider=WorkerProvider(stub_factory=RpcWorkerStubFactory()),
            autoscaler=self._autoscaler,
            threads=controller_threads,
            db=db,
        )
        self._controller.start()

        # Auto-login: mint a JWT via the controller's auth system.
        # Raw tokens won't work since the verifier only accepts JWTs.
        url = self._controller.url
        now = Timestamp.now()
        key_id = f"iris_k_local_{secrets.token_hex(8)}"
        db.ensure_user("local-admin", now, role="admin")
        db.set_user_role("local-admin", "admin")

        if auth.jwt_manager:
            create_api_key(
                db,
                key_id=key_id,
                key_hash=f"jwt:{key_id}",
                key_prefix="jwt",
                user_id="local-admin",
                name="local-auto-login",
                now=now,
            )
            jwt_token = auth.jwt_manager.create_token("local-admin", "admin", key_id)
        else:
            # Fallback for no-DB / no-JWT mode (shouldn't happen in practice)
            jwt_token = secrets.token_urlsafe(32)
            create_api_key(
                db,
                key_id=key_id,
                key_hash=hash_token(jwt_token),
                key_prefix=jwt_token[:8],
                user_id="local-admin",
                name="local-auto-login",
                now=now,
            )

        # Local import to break circular dependency:
        # local_cluster → cli.token_store → cli.__init__ → cli.main → client → local_cluster
        from iris.cli.token_store import store_token

        cluster_name = self._config.name or "local"
        store_token(cluster_name, url, jwt_token)
        self._auto_login_token = jwt_token

        return url

    @property
    def auto_login_token(self) -> str | None:
        return self._auto_login_token

    def stop(self) -> None:
        self._stopped.set()
        if self._controller:
            self._controller.stop()
            self._controller = None
        if self._autoscaler is not None:
            self._autoscaler = None
        if self._autoscaler_temp_dir is not None:
            self._autoscaler_temp_dir.cleanup()
            self._autoscaler_temp_dir = None
        if self._temp_dir:
            self._temp_dir.cleanup()
            self._temp_dir = None

    def close(self) -> None:
        """Stop the cluster and release all resources including the DB dir."""
        self.stop()
        self._db_dir.cleanup()

    def wait(self) -> None:
        """Block until stop() is called."""
        self._stopped.wait()

    def restart(self) -> str:
        self.stop()
        return self.start()

    def discover(self) -> str | None:
        return self._controller.url if self._controller else None

    def status(self) -> ControllerStatus:
        if self._controller:
            return ControllerStatus(
                running=True,
                address=self._controller.url,
                healthy=True,
            )
        return ControllerStatus(running=False, address="", healthy=False)

    def fetch_startup_logs(self, tail_lines: int = 100) -> str | None:
        return "(local controller — no startup logs)"


def make_local_cluster_config(max_workers: int) -> config_pb2.IrisClusterConfig:
    """Build a fully-configured IrisClusterConfig for local execution.

    Creates a minimal base config and transforms it via make_local_config()
    to ensure local defaults (fast autoscaler timings, etc.) are applied
    consistently from config.py.
    """
    base_config = config_pb2.IrisClusterConfig()

    sg = config_pb2.ScaleGroupConfig(
        name="local-cpu",
        buffer_slices=1,
        max_slices=max_workers,
        num_vms=1,
        resources=config_pb2.ScaleGroupResources(
            cpu_millicores=8000,
            memory_bytes=16 * 1024**3,
            disk_bytes=50 * 1024**3,
            device_type=config_pb2.ACCELERATOR_TYPE_CPU,
        ),
    )
    base_config.scale_groups["local-cpu"].CopyFrom(sg)

    return make_local_config(base_config)
