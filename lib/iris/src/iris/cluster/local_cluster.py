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

from finelog.client.log_client import Table
from rigging.credential_store import CredentialRecord, save_credentials
from rigging.timing import Duration

from iris.cluster.backends.rpc.backend import RpcTaskBackend, RpcWorkerStubFactory
from iris.cluster.config import (
    GcpPlatformConfig,
    IrisClusterConfig,
    ScaleGroupConfig,
    ScaleGroupResources,
    WorkerConfig,
    make_local_config,
)
from iris.cluster.constraints import worker_attributes_from_resources
from iris.cluster.controller.auth import SESSION_TOKEN_TTL_SECONDS, create_controller_auth
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
from iris.cluster.controller.log_stack import build_log_stack
from iris.cluster.platforms.gcp.fake import InMemoryGcpService
from iris.cluster.platforms.gcp.workers import GcpWorkerProvider
from iris.cluster.platforms.types import find_free_port
from iris.cluster.platforms.vm_lifecycle import ControllerStatus
from iris.cluster.service_mode import ServiceMode
from iris.cluster.types import DEFAULT_BACKEND_ID, AcceleratorType
from iris.cluster.worker.port_allocator import PortAllocator
from iris.managed_thread import ThreadContainer


def create_local_autoscaler(
    config: IrisClusterConfig,
    controller_address: str,
    threads: ThreadContainer | None = None,
    provisioning_table: Table | None = None,
) -> tuple[Autoscaler, tempfile.TemporaryDirectory]:
    """Create Autoscaler with GcpWorkerProvider(LOCAL) for all scale groups.

    Creates temp directories and a PortAllocator so that InMemoryGcpService(LOCAL)
    can spawn real Worker threads that register with the controller.

    Args:
        config: Cluster configuration (with defaults already applied)
        controller_address: Address for workers to connect to
        threads: Optional thread container for testing
        provisioning_table: finelog ``iris.provisioning`` table for slice-provisioning outcomes.

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
    cpu_millicores_by_group: dict[str, int] = {}
    for name, sg_config in config.scale_groups.items():
        attrs: dict[str, str | int | float] = {}
        resources = sg_config.resources
        if resources is not None:
            attrs.update(worker_attributes_from_resources(resources))
        if sg_config.worker.attributes:
            attrs.update(sg_config.worker.attributes)
        worker_attributes_by_group[name] = attrs
        if resources is None:
            continue
        if resources.device_type == AcceleratorType.GPU and resources.device_count > 0:
            gpu_count_by_group[name] = resources.device_count
        if resources.cpu_millicores > 0:
            cpu_millicores_by_group[name] = resources.cpu_millicores

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
        cpu_millicores_by_group=cpu_millicores_by_group,
        storage_prefix=storage_prefix,
        label_prefix=label_prefix,
    )
    local_gcp_config = GcpPlatformConfig(project_id="local")
    platform = GcpWorkerProvider(
        gcp_config=local_gcp_config,
        label_prefix=label_prefix,
        worker_port=config.defaults.worker.port,
        gcp_service=gcp_service,
    )

    scale_groups: dict[str, ScalingGroup] = {}
    for name, sg_config in config.scale_groups.items():
        scale_groups[name] = ScalingGroup(
            config=sg_config,
            platform=platform,
            label_prefix=label_prefix,
            scale_up_rate_limit=sg_config.scale_up_rate_limit or DEFAULT_SCALE_UP_RATE_LIMIT,
            scale_down_rate_limit=sg_config.scale_down_rate_limit or DEFAULT_SCALE_DOWN_RATE_LIMIT,
        )

    # Build base_worker_config from defaults so auth_token (and other fields)
    # flow through the autoscaler to locally-spawned workers.
    base_worker_config: WorkerConfig | None = None
    if config.defaults.worker.auth_token:
        base_worker_config = config.defaults.worker.model_copy(deep=True)

    autoscaler = Autoscaler.from_config(
        scale_groups=scale_groups,
        config=config.defaults.autoscaler,
        platform=platform,
        base_worker_config=base_worker_config,
        provisioning_table=provisioning_table,
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
        config: IrisClusterConfig,
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
        # No signing_key_pem: a local cluster mints with an ephemeral in-process
        # keypair (tokens live only for this process).
        auth = create_controller_auth(
            self._config.auth,
            cluster_name=self._config.name or "iris-local",
        )
        if auth.worker_token:
            self._config.defaults.worker.auth_token = auth.worker_token

        controller_threads = self._threads.create_child("controller") if self._threads else None
        autoscaler_threads = controller_threads.create_child("autoscaler") if controller_threads else None

        # The log stack is built before the autoscaler so its provisioning table
        # is a constructor arg; the Controller owns it and tears it down at stop().
        log_stack = build_log_stack(
            log_service_address="",
            local_log_dir=Path(self._db_dir.name) / "log-server",
            host="127.0.0.1",
            worker_token=auth.worker_token if auth.worker_token else None,
        )

        # Autoscaler creates its own temp dirs for worker resources
        self._autoscaler, self._autoscaler_temp_dir = create_local_autoscaler(
            self._config,
            address,
            threads=autoscaler_threads,
            provisioning_table=log_stack.provisioning_table,
        )

        controller_config = ControllerConfig(
            host="127.0.0.1",
            port=port,
            remote_state_dir=self._config.storage.remote_state_dir or f"file://{state_dir}",
            local_state_dir=Path(self._db_dir.name),
            auth_verifier=auth.verifier,
            auth_provider=auth.provider,
            auth=auth,
            autoscaler_evaluation_interval=self._config.defaults.autoscaler.evaluation_interval,
            # Fast worker-failure detection for local/e2e runs: reaped ~10s
            # after the last successful reconcile, vs the ~50s production grace.
            worker_unreachable_grace=Duration.from_seconds(10.0),
        )

        # The backend owns the autoscaler and constructs its own liveness tracker,
        # sized by the controller config's worker-unreachable grace. The controller
        # drives the autoscaler via backend.autoscale and persists the returned
        # state each tick.
        provider = RpcTaskBackend(
            stub_factory=RpcWorkerStubFactory(),
            unreachable_grace=controller_config.worker_unreachable_grace,
            autoscaler=self._autoscaler,
        )

        self._controller = Controller(
            config=controller_config,
            backends={DEFAULT_BACKEND_ID: provider},
            log_stack=log_stack,
            threads=controller_threads,
            db=db,
        )
        self._controller.start()

        # Auto-login: mint an in-process admin JWT so the local dashboard can open a
        # browser session (the `?session_token=` link). Raw tokens won't work since
        # the verifier only accepts JWTs; the CLI itself authenticates by loopback
        # trust (the controller binds 127.0.0.1) and needs no cached token.
        url = self._controller.url
        # jti is for log correlation only — the local session token is stateless
        # (nothing persisted, never revocable), like every other iris token. The
        # admin role is config-derived (RolePolicy), so no user row is created.
        key_id = f"iris_s_local_{secrets.token_hex(8)}"
        assert auth.jwt_manager is not None  # create_controller_auth always builds one
        jwt_token = auth.jwt_manager.create_token("local-admin", "admin", key_id, ttl_seconds=SESSION_TOKEN_TTL_SECONDS)

        cluster_name = self._config.name or "local"
        save_credentials(CredentialRecord(cluster=cluster_name, endpoint=url))
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


def make_local_cluster_config(max_workers: int) -> IrisClusterConfig:
    """Build a fully-configured IrisClusterConfig for local execution.

    Creates a minimal base config and transforms it via make_local_config()
    to ensure local defaults (fast autoscaler timings, etc.) are applied
    consistently from config.py.
    """
    sg = ScaleGroupConfig(
        name="local-cpu",
        buffer_slices=1,
        max_slices=max_workers,
        num_vms=1,
        resources=ScaleGroupResources(
            cpu_millicores=8000,
            memory_bytes=16 * 1024**3,
            disk_bytes=50 * 1024**3,
            device_type=AcceleratorType.CPU,
        ),
    )
    base_config = IrisClusterConfig(scale_groups={"local-cpu": sg})

    return make_local_config(base_config)
