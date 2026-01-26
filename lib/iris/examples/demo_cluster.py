# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Demo cluster with Jupyter notebook integration.

This script boots an iris cluster (in-process by default, no Docker required),
seeds it with quick demo jobs, and optionally launches a Jupyter notebook for
interactive exploration.

Workers are created on-demand by the autoscaler when jobs are submitted. The
autoscaler manages two scale groups:
- cpu: For jobs without device requirements
- tpu_v5e_4: For TPU jobs (v5litepod-4 topology)

Usage:
    # Validate that the cluster works (for CI)
    cd lib/iris
    uv run python examples/demo_cluster.py --validate-only

    # Launch interactive demo with Jupyter
    uv run python examples/demo_cluster.py
"""

import logging
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import click
from iris.client import IrisClient
from iris.cluster.client.local_client import (
    LocalEnvironmentProvider,
    _LocalBundleProvider,
    _LocalContainerRuntime,
    _LocalImageProvider,
)
from iris.cluster.controller.controller import Controller, ControllerConfig, RpcWorkerStubFactory
from iris.cluster.types import (
    CoschedulingConfig,
    Entrypoint,
    EnvironmentSpec,
    ResourceSpec,
    get_tpu_topology,
    tpu_device,
)
from iris.cluster.vm.autoscaler import Autoscaler, AutoscalerConfig
from iris.cluster.vm.managed_vm import ManagedVm, VmRegistry
from iris.cluster.vm.scaling_group import ScalingGroup
from iris.cluster.vm.vm_platform import VmGroupProtocol, VmGroupStatus, VmSnapshot
from iris.cluster.worker.worker import Worker, WorkerConfig
from iris.rpc import cluster_pb2, config_pb2, vm_pb2
from iris.rpc.cluster_connect import ControllerServiceClientSync
from iris.time_utils import now_ms

# The iris project root (lib/iris/) - used as workspace for the example
IRIS_ROOT = Path(__file__).parent.parent

logger = logging.getLogger(__name__)


def find_free_port() -> int:
    """Find an available port."""
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# =============================================================================
# Local VM Manager for Demo (implements VmGroupProtocol and VmManagerProtocol)
# =============================================================================


class LocalVmGroup(VmGroupProtocol):
    """In-process VM group that wraps Worker instances.

    For the demo, each VM group represents a "slice" that contains one or more
    workers. Workers become ready immediately (no bootstrap delay).
    """

    def __init__(
        self,
        group_id: str,
        scale_group: str,
        workers: list[Worker],
        worker_ids: list[str],
        vm_registry: VmRegistry,
    ):
        self._group_id = group_id
        self._scale_group = scale_group
        self._workers = workers
        self._worker_ids = worker_ids
        self._created_at_ms = now_ms()
        self._vm_registry = vm_registry
        self._terminated = False

        # Create mock ManagedVm instances for each worker (for autoscaler compatibility)
        self._managed_vms: list[ManagedVm] = []
        for i, worker_id in enumerate(worker_ids):
            # Create a minimal ManagedVm that's immediately ready
            # We don't use ManagedVm's lifecycle thread since workers are in-process
            vm_info = vm_pb2.VmInfo(
                vm_id=f"{group_id}-vm-{i}",
                slice_id=group_id,
                scale_group=scale_group,
                state=vm_pb2.VM_STATE_READY,
                address="127.0.0.1",
                zone="local",
                worker_id=worker_id,
                created_at_ms=self._created_at_ms,
                state_changed_at_ms=self._created_at_ms,
            )
            # Create a stub ManagedVm that just holds the info
            managed_vm = _StubManagedVm(vm_info)
            self._managed_vms.append(managed_vm)
            vm_registry.register(managed_vm)

    @property
    def group_id(self) -> str:
        return self._group_id

    @property
    def slice_id(self) -> str:
        return self._group_id

    @property
    def scale_group(self) -> str:
        return self._scale_group

    @property
    def created_at_ms(self) -> int:
        return self._created_at_ms

    def status(self) -> VmGroupStatus:
        if self._terminated:
            return VmGroupStatus(
                vms=[
                    VmSnapshot(
                        vm_id=vm.info.vm_id,
                        state=vm_pb2.VM_STATE_TERMINATED,
                        address="",
                        init_phase="",
                        init_error="",
                    )
                    for vm in self._managed_vms
                ]
            )
        return VmGroupStatus(
            vms=[
                VmSnapshot(
                    vm_id=vm.info.vm_id,
                    state=vm_pb2.VM_STATE_READY,
                    address="127.0.0.1",
                    init_phase="ready",
                    init_error="",
                )
                for vm in self._managed_vms
            ]
        )

    def vms(self) -> list[ManagedVm]:
        return self._managed_vms

    def terminate(self) -> None:
        if self._terminated:
            return
        self._terminated = True
        for worker in self._workers:
            worker.stop()
        for vm in self._managed_vms:
            self._vm_registry.unregister(vm.info.vm_id)

    def to_proto(self) -> vm_pb2.SliceInfo:
        return vm_pb2.SliceInfo(
            slice_id=self._group_id,
            scale_group=self._scale_group,
            created_at_ms=self._created_at_ms,
            vms=[vm.info for vm in self._managed_vms],
        )


class _StubManagedVm(ManagedVm):
    """Minimal ManagedVm stub that holds VmInfo without lifecycle management.

    Used by LocalVmGroup to satisfy the VmGroupProtocol interface without
    running actual bootstrap threads.
    """

    def __init__(self, info: vm_pb2.VmInfo):
        # Don't call super().__init__ - just set the minimal attributes
        self.info = info
        self._log_lines: list[str] = []

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def init_log(self, tail: int | None = None) -> str:
        return ""

    def check_health(self) -> bool:
        return True


class LocalVmManager:
    """VmManager for in-process demo workers.

    Creates LocalVmGroup instances containing in-process Worker instances.
    Workers are created with appropriate attributes based on the scale group
    configuration (TPU topology, etc.).
    """

    def __init__(
        self,
        scale_group_config: config_pb2.ScaleGroupConfig,
        controller_address: str,
        cache_path: Path,
        fake_bundle: Path,
        vm_registry: VmRegistry,
    ):
        self._config = scale_group_config
        self._controller_address = controller_address
        self._cache_path = cache_path
        self._fake_bundle = fake_bundle
        self._vm_registry = vm_registry
        self._slice_counter = 0

    def create_vm_group(self, tags: dict[str, str] | None = None) -> VmGroupProtocol:
        """Create a new VM group with workers."""
        slice_id = f"{self._config.name}-slice-{self._slice_counter}"
        self._slice_counter += 1

        # Determine worker count based on accelerator type
        if self._config.accelerator_type:
            try:
                topo = get_tpu_topology(self._config.accelerator_type)
                worker_count = topo.vm_count
            except ValueError:
                worker_count = 1
        else:
            worker_count = 1

        # Create workers
        workers: list[Worker] = []
        worker_ids: list[str] = []

        bundle_provider = _LocalBundleProvider(self._fake_bundle)
        image_provider = _LocalImageProvider()
        container_runtime = _LocalContainerRuntime()

        for tpu_worker_id in range(worker_count):
            worker_id = f"worker-{slice_id}-{tpu_worker_id}-{uuid.uuid4().hex[:8]}"
            worker_port = find_free_port()

            # Set up worker attributes
            attributes: dict[str, str | int | float] = {}
            if self._config.accelerator_type:
                attributes["tpu-name"] = slice_id
                attributes["tpu-worker-id"] = tpu_worker_id
                attributes["tpu-topology"] = self._config.accelerator_type

            # Create device config if accelerator is specified
            device = None
            if self._config.accelerator_type:
                topo = get_tpu_topology(self._config.accelerator_type)
                device = tpu_device(self._config.accelerator_type, count=topo.chips_per_vm)

            environment_provider = LocalEnvironmentProvider(
                cpu=1000,
                memory_gb=1000,
                attributes=attributes,
                device=device,
            )

            worker_config = WorkerConfig(
                host="127.0.0.1",
                port=worker_port,
                cache_dir=self._cache_path,
                controller_address=self._controller_address,
                worker_id=worker_id,
                poll_interval_seconds=0.1,  # Fast polling for demos
            )
            worker = Worker(
                worker_config,
                cache_dir=self._cache_path,
                bundle_provider=bundle_provider,
                image_provider=image_provider,
                container_runtime=container_runtime,
                environment_provider=environment_provider,
            )
            worker.start()
            workers.append(worker)
            worker_ids.append(worker_id)

        logger.info(
            "LocalVmManager created VM group %s with %d workers for scale group %s",
            slice_id,
            len(workers),
            self._config.name,
        )

        return LocalVmGroup(
            group_id=slice_id,
            scale_group=self._config.name,
            workers=workers,
            worker_ids=worker_ids,
            vm_registry=self._vm_registry,
        )

    def discover_vm_groups(self) -> list[VmGroupProtocol]:
        """Return empty list - no recovery for local demo."""
        return []


# =============================================================================
# GCP Cluster Demo (--cluster mode)
# =============================================================================


DEFAULT_CONTROLLER_PORT = 10000


def _wait_for_port(port: int, host: str = "localhost", timeout: float = 30.0) -> bool:
    """Wait for a port to become available.

    Returns True if port is ready, False if timeout.
    """
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return True
        except (ConnectionRefusedError, OSError, TimeoutError):
            time.sleep(0.5)
    return False


def _discover_controller_vm(zone: str, project: str) -> str | None:
    """Find the controller VM in the given zone using name-based filter."""
    result = subprocess.run(
        [
            "gcloud",
            "compute",
            "instances",
            "list",
            f"--project={project}",
            f"--zones={zone}",
            "--filter=name~^iris-controller",
            "--format=value(name)",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        logger.error("Error listing instances: %s", result.stderr)
        return None

    vm_names = result.stdout.strip().split("\n")
    vm_names = [v for v in vm_names if v]

    if not vm_names:
        return None
    if len(vm_names) > 1:
        logger.warning("Multiple controller VMs found: %s", vm_names)
    return vm_names[0]


class ClusterDemoCluster:
    """Demo cluster backed by a real GCP cluster.

    This class manages a real GCP cluster using GcpController for the controller
    VM lifecycle. An SSH tunnel is established for local access to the controller.

    Example:
        with ClusterDemoCluster(config_path) as demo:
            results = demo.seed_cluster()
            if demo.validate_seed_results(results):
                demo.launch_jupyter()
                demo.wait_for_interrupt()
    """

    def __init__(self, config_path: Path):
        from iris.cluster.vm.config import load_config
        from iris.cluster.vm.controller import GcpController

        self._config_path = config_path
        self._config = load_config(config_path)
        self._controller = GcpController(self._config)
        self._controller_url: str | None = None
        self._tunnel_proc: subprocess.Popen | None = None
        self._local_port = find_free_port()
        self._rpc_client: IrisClient | None = None

    def __enter__(self):
        """Start controller VM and establish tunnel."""
        zone = self._config.zone
        project = self._config.project_id

        # GcpController.start() is idempotent - reuses existing healthy controller
        # or creates a new one if none exists / existing is unhealthy
        logger.info("Ensuring controller VM is running and healthy...")
        controller_address = self._controller.start()
        logger.info("Controller ready at %s", controller_address)
        self._controller_url = f"http://localhost:{self._local_port}"

        # Find the VM name for SSH tunnel
        vm_name = _discover_controller_vm(zone, project)
        if not vm_name:
            raise RuntimeError("Controller VM not found after start")

        logger.info("Establishing SSH tunnel to %s:%d...", vm_name, DEFAULT_CONTROLLER_PORT)
        self._tunnel_proc = subprocess.Popen(
            [
                "gcloud",
                "compute",
                "ssh",
                vm_name,
                f"--project={project}",
                f"--zone={zone}",
                "--",
                "-L",
                f"{self._local_port}:localhost:{DEFAULT_CONTROLLER_PORT}",
                "-N",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "UserKnownHostsFile=/dev/null",
                "-o",
                "LogLevel=ERROR",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        if not _wait_for_port(self._local_port, timeout=30):
            stderr = self._tunnel_proc.stderr.read().decode() if self._tunnel_proc.stderr else ""
            self._tunnel_proc.terminate()
            self._tunnel_proc.wait()
            raise RuntimeError(f"SSH tunnel failed to establish: {stderr}")

        logger.info("SSH tunnel established on port %d", self._local_port)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup tunnel and optionally stop controller."""
        del exc_type, exc_val, exc_tb  # unused

        if self._rpc_client:
            self._rpc_client = None

        if self._tunnel_proc:
            self._tunnel_proc.terminate()
            try:
                self._tunnel_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._tunnel_proc.kill()
                self._tunnel_proc.wait()
            self._tunnel_proc = None

        # Note: We don't stop the controller by default - user can use cleanup-cluster.py
        logger.info("Demo cluster session ended. Controller VM is still running.")
        logger.info("To stop: uv run python scripts/cleanup-cluster.py --zone %s", self._config.zone)

    @property
    def controller_url(self) -> str:
        if self._controller_url is None:
            raise RuntimeError("Cluster not started")
        return self._controller_url

    @property
    def client(self) -> IrisClient:
        """IrisClient for this cluster."""
        if self._rpc_client is None:
            self._rpc_client = IrisClient.remote(
                self.controller_url,
                workspace=IRIS_ROOT,
            )
        return self._rpc_client

    def seed_cluster(self) -> list[tuple[str, str]]:
        """Submit demo jobs to the cluster (TPU jobs for real cluster)."""
        results = []

        def hello():
            print("Hello from iris!")
            return 42

        def compute(a, b):
            result = a + b
            print(f"{a} + {b} = {result}")
            return result

        # Get TPU type from config
        tpu_type = None
        for group in self._config.scale_groups.values():
            if group.accelerator_type:
                tpu_type = group.accelerator_type
                break

        if tpu_type is None:
            raise RuntimeError("No TPU scale group found in config")

        from iris.cluster.types import CoschedulingConfig, get_tpu_topology

        topo = get_tpu_topology(tpu_type)
        replicas = topo.vm_count

        # Simple TPU job
        print(f"  Submitting demo-hello (TPU: {tpu_type})...")
        job = self.client.submit(
            entrypoint=Entrypoint.from_callable(hello),
            name="demo-hello",
            resources=ResourceSpec(cpu=1, memory="512m", device=tpu_device(tpu_type)),
            environment=EnvironmentSpec(workspace="/app"),
        )
        status = job.wait(timeout=600)  # 10 min for TPU provisioning
        results.append((str(job.job_id), cluster_pb2.JobState.Name(status.state)))
        print(f"  demo-hello: {cluster_pb2.JobState.Name(status.state)}")

        # Compute job
        print(f"  Submitting demo-compute (TPU: {tpu_type})...")
        job = self.client.submit(
            entrypoint=Entrypoint.from_callable(compute, 10, 32),
            name="demo-compute",
            resources=ResourceSpec(cpu=1, memory="512m", device=tpu_device(tpu_type)),
            environment=EnvironmentSpec(workspace="/app"),
        )
        status = job.wait(timeout=600)
        results.append((str(job.job_id), cluster_pb2.JobState.Name(status.state)))
        print(f"  demo-compute: {cluster_pb2.JobState.Name(status.state)}")

        # Coscheduled TPU job
        def distributed_work():
            from iris.cluster.client import get_job_info

            info = get_job_info()
            if info is None:
                raise RuntimeError("Not running in an Iris job context")
            print(f"Task {info.task_index} of {info.num_tasks} on worker {info.worker_id}")
            return f"Task {info.task_index} done"

        print(f"  Submitting demo-coscheduled ({replicas} replicas, TPU: {tpu_type})...")
        job = self.client.submit(
            entrypoint=Entrypoint.from_callable(distributed_work),
            name="demo-coscheduled",
            resources=ResourceSpec(
                cpu=1,
                memory="512m",
                replicas=replicas,
                device=tpu_device(tpu_type),
            ),
            environment=EnvironmentSpec(workspace="/app"),
            coscheduling=CoschedulingConfig(group_by="tpu-name"),
        )
        status = job.wait(timeout=600)
        results.append((str(job.job_id), cluster_pb2.JobState.Name(status.state)))
        print(f"  demo-coscheduled: {cluster_pb2.JobState.Name(status.state)}")

        return results

    def validate_seed_results(self, results: list[tuple[str, str]]) -> bool:
        """Validate that seed jobs completed as expected."""
        expected = ["JOB_STATE_SUCCEEDED"] * len(results)
        actual = [r[1] for r in results]
        return actual == expected

    def wait_for_interrupt(self):
        """Wait for Ctrl+C, keeping the cluster running."""
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


# =============================================================================
# Demo Cluster
# =============================================================================


class DemoCluster:
    """Demo cluster with Jupyter integration.

    Supports two execution modes:
    - In-process (default): Fast, no Docker required, jobs run in threads
    - Docker: Real containers, matches production behavior

    Workers are created on-demand by the autoscaler. The autoscaler manages:
    - cpu: For jobs without device requirements (min_slices=1)
    - tpu_v5e_4: For TPU jobs (v5litepod-4 topology)

    Example:
        with DemoCluster() as demo:
            results = demo.seed_cluster()
            if demo.validate_seed_results(results):
                demo.launch_jupyter()
                demo.wait_for_interrupt()
    """

    def __init__(
        self,
        use_docker: bool = False,
    ):
        self._use_docker = use_docker
        self._controller_port = find_free_port()

        # Will be initialized in __enter__
        self._temp_dir: tempfile.TemporaryDirectory | None = None
        self._controller: Controller | None = None
        self._autoscaler: Autoscaler | None = None
        self._vm_registry: VmRegistry | None = None
        self._controller_client: ControllerServiceClientSync | None = None
        self._rpc_client: IrisClient | None = None

        # Paths initialized in __enter__
        self._cache_path: Path | None = None
        self._fake_bundle: Path | None = None

        # Jupyter integration
        self._notebook_proc: subprocess.Popen | None = None
        self._notebook_url: str | None = None

    def _create_autoscaler(self) -> Autoscaler:
        """Create the autoscaler with scale groups for CPU and TPU workers."""
        assert self._cache_path is not None
        assert self._fake_bundle is not None

        vm_registry = VmRegistry()
        self._vm_registry = vm_registry

        # Scale group configs
        cpu_config = config_pb2.ScaleGroupConfig(
            name="cpu",
            accelerator_type="",  # Empty = matches jobs without device requirements
            min_slices=1,  # Always have at least one CPU worker for simple jobs
            max_slices=4,
        )
        tpu_config = config_pb2.ScaleGroupConfig(
            name="tpu_v5e_16",
            accelerator_type="v5litepod-16",  # 4 VMs per slice to match original 4 workers/slice
            min_slices=0,
            max_slices=4,
        )

        controller_address = f"http://127.0.0.1:{self._controller_port}"

        # Create VM managers for each scale group
        cpu_vm_manager = LocalVmManager(
            scale_group_config=cpu_config,
            controller_address=controller_address,
            cache_path=self._cache_path,
            fake_bundle=self._fake_bundle,
            vm_registry=vm_registry,
        )
        tpu_vm_manager = LocalVmManager(
            scale_group_config=tpu_config,
            controller_address=controller_address,
            cache_path=self._cache_path,
            fake_bundle=self._fake_bundle,
            vm_registry=vm_registry,
        )

        # Create scale groups
        # Use fast scale-up but slow scale-down to keep workers alive during demo/notebook
        cpu_scale_group = ScalingGroup(
            config=cpu_config,
            vm_manager=cpu_vm_manager,
            scale_up_cooldown_ms=1000,  # 1 second
            scale_down_cooldown_ms=300_000,  # 5 minutes - workers stay alive for demo
        )
        tpu_scale_group = ScalingGroup(
            config=tpu_config,
            vm_manager=tpu_vm_manager,
            scale_up_cooldown_ms=1000,  # 1 second
            scale_down_cooldown_ms=300_000,  # 5 minutes - workers stay alive for demo
        )

        # Create autoscaler with fast evaluation interval
        autoscaler_config = AutoscalerConfig(
            evaluation_interval_seconds=2.0,  # Fast for demo
        )
        autoscaler = Autoscaler(
            scale_groups={
                "cpu": cpu_scale_group,
                "tpu_v5e_16": tpu_scale_group,
            },
            vm_registry=vm_registry,
            config=autoscaler_config,
        )

        return autoscaler

    def __enter__(self):
        """Start controller with autoscaler."""
        self._temp_dir = tempfile.TemporaryDirectory(prefix="demo_cluster_")
        temp_path = Path(self._temp_dir.name)
        bundle_dir = temp_path / "bundles"
        bundle_dir.mkdir()
        self._cache_path = temp_path / "cache"
        self._cache_path.mkdir()

        # Create fake bundle with minimal structure
        self._fake_bundle = temp_path / "fake_bundle"
        self._fake_bundle.mkdir()
        (self._fake_bundle / "pyproject.toml").write_text("[project]\nname = 'demo'\n")

        # Create autoscaler (needs paths set up first)
        self._autoscaler = self._create_autoscaler()

        # Start Controller with autoscaler
        controller_config = ControllerConfig(
            host="127.0.0.1",
            port=self._controller_port,
            bundle_dir=bundle_dir,
        )
        self._controller = Controller(
            config=controller_config,
            worker_stub_factory=RpcWorkerStubFactory(),
            autoscaler=self._autoscaler,
        )
        self._controller.start()

        # Create RPC client
        self._controller_client = ControllerServiceClientSync(
            address=f"http://127.0.0.1:{self._controller_port}",
            timeout_ms=30000,
        )

        # Wait for autoscaler to create initial workers (min_slices=1 for cpu)
        # Workers send heartbeats every 0.1s, and registration happens on first heartbeat
        logger.info("Waiting for autoscaler to create initial workers...")
        time.sleep(3.0)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop cluster and cleanup."""
        del exc_type, exc_val, exc_tb  # unused
        # Stop Jupyter notebook
        if self._notebook_proc:
            self._notebook_proc.terminate()
            try:
                self._notebook_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._notebook_proc.kill()
            self._notebook_proc = None

        if self._rpc_client:
            self._rpc_client = None
        if self._controller_client:
            self._controller_client.close()
        if self._controller:
            self._controller.stop()
        if self._temp_dir:
            self._temp_dir.cleanup()

    @property
    def controller_url(self) -> str:
        return f"http://127.0.0.1:{self._controller_port}"

    @property
    def client(self) -> IrisClient:
        """IrisClient for this cluster."""
        if self._rpc_client is None:
            self._rpc_client = IrisClient.remote(
                self.controller_url,
                workspace=IRIS_ROOT,
            )
        return self._rpc_client

    def submit(
        self,
        fn,
        *args,
        name: str | None = None,
        cpu: int = 1,
        memory: str = "1g",
        **kwargs,
    ):
        """Submit a job to the cluster and return Job handle."""
        entrypoint = Entrypoint.from_callable(fn, *args, **kwargs)
        environment = EnvironmentSpec(workspace="/app")
        resources = ResourceSpec(cpu=cpu, memory=memory)
        return self.client.submit(
            entrypoint=entrypoint,
            name=name or fn.__name__,
            resources=resources,
            environment=environment,
        )

    def seed_cluster(self) -> list[tuple[str, str]]:
        """Submit demo jobs to the cluster.

        Returns:
            List of (job_id, state) tuples for validation.
        """
        results = []

        def hello():
            print("Hello from iris!")
            return 42

        def compute(a, b):
            result = a + b
            print(f"{a} + {b} = {result}")
            return result

        def countdown(n):
            for i in range(n, 0, -1):
                print(f"Countdown: {i}")
                time.sleep(0.3)
            print("Liftoff!")
            return "Done!"

        def distributed_work():
            from iris.cluster.client import get_job_info

            info = get_job_info()
            if info is None:
                raise RuntimeError("Not running in an Iris job context")
            print(f"Task {info.task_index} of {info.num_tasks} on worker {info.worker_id}")
            return f"Task {info.task_index} done"

        jobs = [
            (hello, [], {}, "demo-hello"),
            (compute, [10, 32], {}, "demo-compute"),
            (countdown, [3], {}, "demo-countdown"),
        ]

        for fn, args, kwargs, name in jobs:
            job = self.submit(fn, *args, name=name, **kwargs)
            status = job.wait()
            results.append((str(job.job_id), cluster_pb2.JobState.Name(status.state)))
            print(f"  {name}: {cluster_pb2.JobState.Name(status.state)}")

        # Coscheduled TPU job - triggers autoscaler to create TPU workers
        if not self._use_docker:
            job = self.client.submit(
                entrypoint=Entrypoint.from_callable(distributed_work),
                name="demo-coscheduled",
                resources=ResourceSpec(
                    cpu=1,
                    memory="512m",
                    replicas=4,
                    device=tpu_device("v5litepod-16"),  # 4 VMs per slice for coscheduling
                ),
                environment=EnvironmentSpec(workspace="/app"),
                coscheduling=CoschedulingConfig(group_by="tpu-name"),
            )
            status = job.wait()
            results.append((str(job.job_id), cluster_pb2.JobState.Name(status.state)))
            print(f"  demo-coscheduled: {cluster_pb2.JobState.Name(status.state)}")
        else:
            print("  demo-coscheduled: SKIPPED (not available in docker mode)")

        return results

    def validate_seed_results(self, results: list[tuple[str, str]]) -> bool:
        """Validate that seed jobs completed as expected."""
        expected = ["JOB_STATE_SUCCEEDED"] * len(results)
        actual = [r[1] for r in results]
        return actual == expected

    def launch_jupyter(self, open_browser: bool = True) -> str:
        """Start Jupyter notebook server and return URL.

        Args:
            open_browser: Whether to open the browser automatically

        Returns:
            Jupyter notebook URL
        """
        env = os.environ.copy()
        env["IRIS_CONTROLLER_ADDRESS"] = self.controller_url
        env["IRIS_WORKSPACE"] = str(IRIS_ROOT)

        # Find the demo notebook
        notebook_dir = IRIS_ROOT / "examples"
        notebook_path = notebook_dir / "demo.ipynb"

        browser_flag = [] if open_browser else ["--no-browser"]

        cmd = [
            sys.executable,
            "-m",
            "jupyter",
            "notebook",
            *browser_flag,
            f"--notebook-dir={notebook_dir}",
            str(notebook_path),
        ]

        self._notebook_proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Extract Jupyter URL from output
        self._notebook_url = self._extract_jupyter_url()
        return self._notebook_url

    def _extract_jupyter_url(self, timeout: float = 30.0) -> str:
        """Extract the Jupyter URL from the notebook server output."""
        start = time.time()
        url_pattern = re.compile(r"(http://127\.0\.0\.1:\d+/\S*)")

        while time.time() - start < timeout:
            if self._notebook_proc is None or self._notebook_proc.poll() is not None:
                raise RuntimeError("Jupyter notebook process died unexpectedly")

            stdout = self._notebook_proc.stdout
            if stdout is None:
                raise RuntimeError("Jupyter notebook process has no stdout")

            line = stdout.readline()
            if line:
                match = url_pattern.search(line)
                if match:
                    return match.group(1)

        raise TimeoutError("Could not extract Jupyter URL within timeout")

    def wait_for_interrupt(self):
        """Wait for Ctrl+C, keeping the cluster running."""
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass

    def run_notebook(self, notebook_path: Path | None = None) -> bool:
        """Execute the demo notebook and validate all cells succeed.

        Args:
            notebook_path: Path to notebook (default: demo.ipynb in same dir)

        Returns:
            True if all cells executed successfully, False otherwise.
        """
        import nbformat
        from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor

        if notebook_path is None:
            notebook_path = Path(__file__).parent / "demo.ipynb"

        print(f"Running notebook: {notebook_path}")

        # Read the notebook
        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=4)

        # Set environment for the kernel (inherited by subprocess)
        os.environ["IRIS_CONTROLLER_ADDRESS"] = self.controller_url
        os.environ["IRIS_WORKSPACE"] = str(IRIS_ROOT)

        # Create executor
        ep = ExecutePreprocessor(
            timeout=120,
            kernel_name="python3",
        )

        try:
            # Execute the notebook
            ep.preprocess(nb, {"metadata": {"path": str(notebook_path.parent)}})
            print("All notebook cells executed successfully!")
            return True
        except CellExecutionError as e:
            print(f"Notebook execution failed: {e}")
            return False


@click.command()
@click.option("--no-browser", is_flag=True, help="Don't auto-open browser for Jupyter")
@click.option("--validate-only", is_flag=True, help="Run seed jobs and exit (for CI)")
@click.option("--test-notebook", is_flag=True, help="Run notebook programmatically and validate (for CI)")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--cluster", is_flag=True, help="Use real GCP cluster instead of in-process")
@click.option(
    "--config",
    "config_file",
    type=click.Path(exists=True, path_type=Path),
    help="Cluster config for --cluster mode (e.g., examples/eu-west4.yaml)",
)
def main(
    no_browser: bool,
    validate_only: bool,
    test_notebook: bool,
    verbose: bool,
    cluster: bool,
    config_file: Path | None,
):
    """Launch demo cluster with Jupyter notebook.

    By default runs in-process (no Docker required). Use --cluster to boot a
    real GCP cluster with TPU workers.

    Examples:
        # In-process mode (fast, for local testing)
        uv run python examples/demo_cluster.py --validate-only

        # Real GCP cluster mode
        uv run python examples/demo_cluster.py --cluster --config examples/eu-west4.yaml
    """
    if verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    # Validate cluster mode requirements
    if cluster and config_file is None:
        print("ERROR: --cluster requires --config (e.g., --config examples/eu-west4.yaml)")
        sys.exit(1)

    if cluster:
        assert config_file is not None  # Validated above
        print(f"Starting GCP cluster from config: {config_file}")
        print("Note: TPU provisioning can take 5-10 minutes")
        demo_class = ClusterDemoCluster(config_file)
    else:
        print("Starting demo cluster with autoscaler...")
        demo_class = DemoCluster()

    with demo_class as demo:
        print(f"Controller: {demo.controller_url}")
        print()
        print("Seeding cluster with demo jobs...")
        results = demo.seed_cluster()

        if not demo.validate_seed_results(results):
            print()
            print("ERROR: Seed jobs did not complete as expected!")
            for job_id, state in results:
                if state != "JOB_STATE_SUCCEEDED":
                    print(f"  {job_id}: {state}")
            sys.exit(1)

        print()
        print("All seed jobs succeeded!")

        if validate_only:
            print("Validation passed!")
            return

        if cluster:
            # In cluster mode, just wait for interrupt (no Jupyter)
            print()
            print("Cluster is running. Press Ctrl+C to disconnect.")
            print("Note: Controller VM will remain running. Use cleanup-cluster.py to stop.")
            demo.wait_for_interrupt()
            return

        if test_notebook:
            print()
            print("Testing notebook execution...")
            if demo.run_notebook():
                print("Notebook test passed!")
            else:
                print("Notebook test FAILED!")
                sys.exit(1)
            return

        print()
        print("Launching Jupyter notebook...")
        url = demo.launch_jupyter(open_browser=not no_browser)
        print(f"Notebook: {url}")
        print()
        print("Press Ctrl+C to stop.")
        demo.wait_for_interrupt()

    print("Shutting down...")


if __name__ == "__main__":
    main()
