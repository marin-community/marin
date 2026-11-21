# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Own a regional x86 CPU service for LOCAL task machines through an RL run."""

import platform
import signal
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace

from fray.current_client import current_client
from fray.types import ANY_REGION, CpuConfig, Entrypoint, EnvironmentConfig, JobRequest, ResourceConfig
from iris.actor.client import ActorClient
from iris.actor.resolver import FixedResolver
from iris.actor.server import ActorServer
from iris.client.client import iris_ctx
from iris.cluster.client.job_info import get_job_info
from iris.cluster.types import parse_memory_string
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.secrets import COREWEAVE_SECRET_NAMES, SecretBehavior, require_secret
from rigging.timing import Duration
from shellbox.backends.iris.local_images import LocalMachineSettings, local_machine_factory
from shellbox.backends.iris.machine_service import MachineService, MachineServiceContract
from shellbox.machine import LINUX_PROCESS_CAPABILITIES, Backend, CommandSemantics

from marin.inference.iris import _terminate_job, _wait_for_endpoint

SERVICE_PORT = "machines"
CONTROL_CPUS = 1
CONTROL_MEMORY_MB = 1024
MEGABYTE = 1024 * 1024


@dataclass(frozen=True)
class LocalMachineServiceConfig:
    local: LocalMachineSettings
    endpoint_name: str
    slots: int
    cpus_per_machine: int
    memory_mb_per_machine: int


@dataclass(frozen=True)
class LocalMachineServiceLaunch:
    config: LocalMachineServiceConfig
    resources: ResourceConfig
    environment: EnvironmentConfig
    startup_timeout: float
    rpc_timeout: float
    lifetime: Duration


@dataclass(frozen=True)
class LocalMachineServiceEndpoint:
    address: str
    actor_name: str
    contract: MachineServiceContract
    rpc_timeout: float

    def task_session_config(self) -> dict:
        """Return the public launch-config binding after verified readiness."""
        return {
            "address": self.address,
            "actor_name": self.actor_name,
            "region": self.contract.region,
            "images": list(self.contract.images),
            "rpc_timeout": self.rpc_timeout,
        }


def run_local_machine_service(config: LocalMachineServiceConfig) -> None:
    """Prepare exact regional images, register readiness, and close machines on termination."""
    info = get_job_info()
    if info is None or info.worker_region is None:
        raise ValueError("LOCAL service requires an attested Iris worker region")
    if platform.machine() != "x86_64":
        raise ValueError("LOCAL service requires an x86 CPU worker for the original image")
    if config.local.region is None or info.worker_region.casefold() != config.local.region.casefold():
        raise ValueError("LOCAL service worker differs from its run region")
    if not config.local.image_mirrors:
        raise ValueError("LOCAL service requires explicit prepared regional images")
    configure_coreweave_s3()
    factory = local_machine_factory(config.local)
    capabilities = LINUX_PROCESS_CAPABILITIES
    if factory.cgroup_parent is not None:
        capabilities = capabilities | {"retained_command_cancellation", "process_scope"}
    contract = MachineServiceContract(
        backend=Backend.LOCAL,
        semantics=CommandSemantics.LINUX_PROCESS,
        capabilities=capabilities,
        images=tuple(mirror.image for mirror in config.local.image_mirrors),
        cpus=config.cpus_per_machine,
        memory_mb=config.memory_mb_per_machine,
        region=info.worker_region,
        architecture=platform.machine(),
    )
    context = iris_ctx()
    actor_name = f"{context.namespace}/{config.endpoint_name}"
    host = MachineService(factory, contract, slots=config.slots)
    server = ActorServer(host="0.0.0.0", port=context.get_port(SERVICE_PORT))
    server.register(actor_name, host)
    stopped = threading.Event()
    previous = {signum: signal.signal(signum, lambda *_: stopped.set()) for signum in (signal.SIGTERM, signal.SIGINT)}
    try:
        port = server.serve_background()
        address = f"http://{info.advertise_host}:{port}"
        with context.registry.registered(actor_name, address):
            stopped.wait()
    finally:
        try:
            server.stop()
        finally:
            try:
                host.close()
            finally:
                for signum, handler in previous.items():
                    signal.signal(signum, handler)


def validate_local_service_launch(launch: LocalMachineServiceLaunch) -> None:
    """Reject cross-cluster placement and allocations that overcommit machine slots."""
    config = launch.config
    if config.local.region is None or not config.local.image_mirrors:
        raise ValueError("LOCAL service requires a run region and explicit image mirrors")
    if config.slots <= 0 or config.cpus_per_machine <= 0 or config.memory_mb_per_machine <= 0:
        raise ValueError("LOCAL service capacity must be positive")
    if launch.resources.target_cluster is not None:
        raise ValueError("LOCAL service must inherit the RL run cluster")
    if launch.resources.regions is not None and tuple(launch.resources.regions) != (config.local.region,):
        raise ValueError("LOCAL service placement must match its run region")
    if not isinstance(launch.resources.device, CpuConfig):
        raise ValueError("LOCAL service requires CPU resources")
    if launch.resources.cpu < config.slots * config.cpus_per_machine + CONTROL_CPUS:
        raise ValueError("LOCAL service CPU allocation cannot cover its slots and control process")
    memory_mb = parse_memory_string(launch.resources.ram) // MEGABYTE
    if memory_mb < config.slots * config.memory_mb_per_machine + CONTROL_MEMORY_MB:
        raise ValueError("LOCAL service memory allocation cannot cover its slots and control process")


@contextmanager
def local_machine_service(launch: LocalMachineServiceLaunch) -> Iterator[LocalMachineServiceEndpoint]:
    """Yield a verified private endpoint, terminating the owned CPU child on every exit."""
    validate_local_service_launch(launch)
    storage = {
        name: launch.environment.secrets[name] if name in launch.environment.secrets else require_secret(name)
        for name in COREWEAVE_SECRET_NAMES
    }
    environment = replace(
        launch.environment,
        secrets={**launch.environment.secrets, **storage},
        secret_behavior=SecretBehavior.ISOLATE,
    )
    # CPU pools lack region node labels; the inherited single-region cluster
    # and attested worker-region check establish the service's placement.
    resources = replace(launch.resources, regions=(ANY_REGION,))
    job = current_client().submit(
        JobRequest(
            name=launch.config.endpoint_name,
            entrypoint=Entrypoint.from_callable(run_local_machine_service, args=(launch.config,)),
            resources=resources,
            environment=environment,
            timeout=launch.lifetime,
            max_retries_failure=0,
            max_retries_preemption=0,
            ports=(SERVICE_PORT,),
        ),
        adopt_existing=False,
    )
    try:
        actor_name = f"{iris_ctx().namespace}/{launch.config.endpoint_name}"
        address, _ = _wait_for_endpoint(job, actor_name, launch.startup_timeout)
        client = ActorClient(
            FixedResolver({actor_name: address}), actor_name, call_timeout=launch.rpc_timeout, max_call_attempts=1
        )
        contract = client.describe()
        if (
            contract.backend != Backend.LOCAL
            or contract.architecture != "x86_64"
            or contract.region.casefold() != launch.config.local.region.casefold()
            or contract.images != tuple(mirror.image for mirror in launch.config.local.image_mirrors)
            or contract.cpus != launch.config.cpus_per_machine
            or contract.memory_mb != launch.config.memory_mb_per_machine
        ):
            raise ValueError("Ready LOCAL service differs from the run binding")
        yield LocalMachineServiceEndpoint(address, actor_name, contract, launch.rpc_timeout)
    finally:
        _terminate_job(job)
