# Fray-Zero Implementation Plan

This document outlines the staged implementation plan for the Fray-Zero system
as described in [fray-zero.md](./fray-zero.md). The implementation will be done
in a new library `lib/fluster/` to keep code separate during development.

## Architecture Overview

The system cleanly separates **cluster management** from the **actor system**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Actor Layer                                    │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────────┐  │
│  │ ActorServer │  │  ActorPool  │  │  Resolver   │  │ ActorContext  │  │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └───────────────┘  │
│         │                │                │                             │
│         └────────────────┴────────────────┘                             │
│                          │ uses                                         │
└──────────────────────────┼──────────────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Cluster Layer                                    │
│  ┌─────────────────┐       ┌──────────────────┐       ┌──────────────┐ │
│  │     Client      │       │    Controller    │       │    Worker    │ │
│  │   (Cluster)     │──RPC─→│  (Job Mgmt +     │──RPC─→│  (Per-VM)    │ │
│  └─────────────────┘       │   Registry)      │       └──────────────┘ │
│                            └────────┬─────────┘                         │
│                                     │ manages                           │
│                                     ▼                                   │
│                            ┌──────────────────┐                         │
│                            │   VM Backend     │                         │
│                            │ - LocalBackend   │                         │
│                            │ - DockerBackend  │                         │
│                            │ - GCPBackend     │                         │
│                            └──────────────────┘                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Cluster Layer (`lib/fluster/cluster/`)

The cluster layer knows nothing about actors. It provides:

**Controller (Cluster Service)**: Central RPC service that:
- Manages job lifecycle (launch, monitor, terminate)
- Owns an **Endpoint Registry** (generic name→address mapping)
- Allocates VMs via pluggable backend
- Dispatches jobs to Workers

**Endpoint Registry**: Generic service discovery (NOT actor-specific):
- Maps `(namespace, name) → list[Endpoint]` where Endpoint is `{address, job_id, metadata}`
- Tracks which job registered each endpoint
- Automatically cleans up when jobs terminate
- Multiple endpoints can register under the same name

**Worker**: Per-VM service that:
- Runs on each allocated VM
- Receives job dispatch from Controller
- Manages containers/processes locally
- Streams logs and status back

**Client (Cluster)**: Python client that:
- Connects to Controller via RPC
- Provides job lifecycle methods (launch, poll, wait, terminate)
- Exposes registry for endpoint registration/lookup

### Actor Layer (`lib/fluster/actor/`)

The actor layer is built on top of the cluster layer:

**ActorServer**: Hosts actor instances, registers with cluster's registry
**ActorPool**: Wraps registry lookup with call()/broadcast() semantics
**Resolver**: Maps actor names to ActorPools via cluster's registry
**ActorContext**: Passed to actor methods, provides cluster/resolver access

---

## Reference Code

Developers should reference these existing files for patterns:

| Pattern | Reference File |
|---------|---------------|
| Resource/Device config | `lib/fray/src/fray/cluster/base.py` (ResourceConfig, TpuConfig, etc.) |
| Environment config | `lib/fray/src/fray/cluster/base.py` (EnvironmentConfig, create_environment) |
| Job types | `lib/fray/src/fray/cluster/base.py` (JobRequest, JobInfo, JobStatus) |
| Connect RPC pattern | `lib/fray/src/fray/job/rpc/proto/fray.proto` |
| RPC server setup | `lib/fray/src/fray/job/rpc/controller.py` (FrayControllerServer) |
| RPC client pattern | `lib/fray/src/fray/job/rpc/context.py` (FrayContext) |
| Worker architecture | `worker/` (manager.py, runtime.py, builder.py, server.py) |
| Proto generation | `lib/fray/buf.yaml`, `lib/fray/buf.gen.yaml` |

---

## Stage 0: Project Scaffolding ✅ COMPLETE

**Status**: Completed 2026-01-10

**Summary**: Set up the complete `lib/fluster` project structure with Python packaging, proto definitions, and buf configuration. Clean separation between cluster management and actor system established.

**Completed Tasks**:
- ✅ Created complete directory structure with cluster/ and actor/ separation
- ✅ Configured pyproject.toml with all dependencies (connectrpc, cloudpickle, uvicorn, httpx, docker)
- ✅ Set up buf.yaml and buf.gen.yaml for proto generation
- ✅ Created empty proto files (controller.proto, worker.proto, actor.proto) as placeholders for Stage 2
- ✅ Created stub files (client.py, registry.py, backend/base.py) for future stages
- ✅ All packages importable, buf generate runs successfully

---

## Stage 1: Core Types ✅ COMPLETE

**Status**: Completed 2026-01-10

**Summary**: Implemented all core dataclasses and type definitions for both cluster and actor layers. Ported and simplified types from existing fray code with clean separation maintained between layers.

**Completed Tasks**:

**Cluster Layer (`cluster/types.py` - 480 lines)**:
- ✅ Type aliases: JobId, Namespace, WorkerId, VMId, EndpointId
- ✅ JobStatus enum with `finished()` helper method
- ✅ TPU topology information (TpuTopologyInfo) with all 34 configurations (v4, v5litepod, v5p, v6e)
- ✅ Device configs: CpuConfig, GpuConfig, TpuConfig with `chip_count()` and `vm_count()` methods
- ✅ ResourceConfig with factory methods (`with_tpu()`, `with_gpu()`, `with_cpu()`)
- ✅ EnvironmentConfig with validation and `create_environment()` helper function
- ✅ Entrypoint (simplified single-class design matching fray-zero)
- ✅ JobRequest with name validation, JobInfo, TaskStatus
- ✅ Generic Endpoint type (NOT actor-specific) for service discovery
- ✅ VMInfo for backend abstraction

**Actor Layer (`actor/types.py` - 95 lines)**:
- ✅ ActorId type alias
- ✅ ActorEndpoint dataclass wrapping cluster Endpoint with actor semantics
- ✅ ActorContext with `from_environment()` classmethod for FLUSTER_* env vars

**Testing**:
- ✅ 48 comprehensive unit tests covering all types
- ✅ Tests verify: construction, factory methods, validation, chip_count calculations, env var handling
- ✅ All tests passing, imports working correctly

**Key Design Decisions**:
- Used `str` for device variants instead of strict `Literal` types for flexibility
- Included TPU topologies for accurate chip_count calculations
- Excluded FLOPS calculations (deferred to later stages)
- Maintained zero imports from `actor/` in `cluster/` package

---

## Stage 2: Proto Definitions

**Goal**: Define proto files for Controller, Worker, and Actor services.
The cluster protos (controller, worker) have no actor knowledge. Actor RPC
is defined separately.

**Reference**: `lib/fray/src/fray/job/rpc/proto/fray.proto`

**Implementation**:

```protobuf
// src/fluster/cluster/controller/proto/controller.proto
syntax = "proto3";
package fluster.cluster.controller;

// ============ Job Management ============

message JobSpec {
  string name = 1;
  bytes serialized_entrypoint = 2;  // cloudpickle
  bytes serialized_resources = 3;
  bytes serialized_environment = 4;
}

message JobHandle {
  string job_id = 1;
  JobStatus status = 2;
  string worker_id = 3;
  string error = 4;
}

enum JobStatus {
  JOB_STATUS_UNSPECIFIED = 0;
  JOB_STATUS_PENDING = 1;
  JOB_STATUS_RUNNING = 2;
  JOB_STATUS_SUCCEEDED = 3;
  JOB_STATUS_FAILED = 4;
  JOB_STATUS_STOPPED = 5;
}

// ============ Endpoint Registry (Generic - NOT actor-specific) ============

message Endpoint {
  string endpoint_id = 1;
  string name = 2;
  string address = 3;
  string job_id = 4;
  string namespace = 5;
  map<string, string> metadata = 6;
}

message RegisterEndpointRequest {
  string name = 1;
  string address = 2;
  string job_id = 3;
  string namespace = 4;
  map<string, string> metadata = 5;
}

message LookupRequest {
  string name = 1;
  string namespace = 2;
}

message LookupResponse {
  repeated Endpoint endpoints = 1;
}

// ============ Controller Service ============

service ControllerService {
  // Job lifecycle
  rpc LaunchJob(JobSpec) returns (JobHandle);
  rpc GetJobStatus(JobHandle) returns (JobHandle);
  rpc TerminateJob(JobHandle) returns (Empty);
  rpc ListJobs(ListJobsRequest) returns (ListJobsResponse);

  // Endpoint registry (generic service discovery)
  rpc RegisterEndpoint(RegisterEndpointRequest) returns (Endpoint);
  rpc UnregisterEndpoint(Endpoint) returns (Empty);
  rpc LookupEndpoints(LookupRequest) returns (LookupResponse);
  rpc ListEndpoints(ListEndpointsRequest) returns (LookupResponse);
}

message Empty {}
message ListJobsRequest { string namespace = 1; }
message ListJobsResponse { repeated JobHandle jobs = 1; }
message ListEndpointsRequest { string prefix = 1; string namespace = 2; }
```

```protobuf
// src/fluster/cluster/worker/proto/worker.proto
syntax = "proto3";
package fluster.cluster.worker;

// ============ Job Execution ============

message RunJobRequest {
  string job_id = 1;
  bytes serialized_entrypoint = 2;
  bytes serialized_environment = 3;
  ResourceLimits limits = 4;
  map<string, string> env_vars = 5;
}

message ResourceLimits {
  int32 cpu_millicores = 1;
  int64 memory_mb = 2;
  int32 timeout_seconds = 3;
}

message RunJobResponse {
  string job_id = 1;
  JobStatus status = 2;
}

enum JobStatus {
  JOB_STATUS_UNSPECIFIED = 0;
  JOB_STATUS_PENDING = 1;
  JOB_STATUS_BUILDING = 2;
  JOB_STATUS_RUNNING = 3;
  JOB_STATUS_SUCCEEDED = 4;
  JOB_STATUS_FAILED = 5;
  JOB_STATUS_KILLED = 6;
}

message GetStatusRequest { string job_id = 1; }
message GetStatusResponse {
  string job_id = 1;
  JobStatus status = 2;
  int32 exit_code = 3;
  string error = 4;
  int64 started_at_ms = 5;
  int64 finished_at_ms = 6;
}

message LogEntry {
  int64 timestamp_ms = 1;
  string stream = 2;  // "stdout" or "stderr"
  string data = 3;
}

message StreamLogsRequest { string job_id = 1; }
message KillJobRequest { string job_id = 1; bool force = 2; }

// ============ Worker Service ============

service WorkerService {
  rpc RunJob(RunJobRequest) returns (RunJobResponse);
  rpc GetJobStatus(GetStatusRequest) returns (GetStatusResponse);
  rpc StreamLogs(StreamLogsRequest) returns (stream LogEntry);
  rpc KillJob(KillJobRequest) returns (Empty);
  rpc HealthCheck(Empty) returns (HealthResponse);
}

message Empty {}
message HealthResponse {
  bool healthy = 1;
  int64 uptime_ms = 2;
  int32 running_jobs = 3;
}
```

```protobuf
// src/fluster/actor/proto/actor.proto
syntax = "proto3";
package fluster.actor;

// ============ Actor RPC ============
// This is the protocol for calling actor methods.
// Actors register via the cluster's endpoint registry.

message ActorCall {
  string method_name = 1;
  bytes serialized_args = 2;    // cloudpickle(args)
  bytes serialized_kwargs = 3;  // cloudpickle(kwargs)
}

message ActorResponse {
  oneof result {
    bytes serialized_value = 1;  // cloudpickle(return_value)
    ActorError error = 2;
  }
}

message ActorError {
  string error_type = 1;
  string message = 2;
  bytes serialized_exception = 3;  // cloudpickle(exception) for re-raise
}

// ============ Actor Service ============
// Each ActorServer exposes this service

service ActorService {
  rpc Call(ActorCall) returns (ActorResponse);
  rpc HealthCheck(Empty) returns (HealthResponse);
}

message Empty {}
message HealthResponse { bool healthy = 1; }
```

**Exit Conditions**:

- [ ] `cluster/controller/proto/controller.proto` defines job management RPCs
- [ ] `cluster/controller/proto/controller.proto` defines generic endpoint registry RPCs
- [ ] `cluster/worker/proto/worker.proto` defines job execution RPCs
- [ ] `actor/proto/actor.proto` defines actor method call RPC
- [ ] No actor-specific terminology in cluster protos
- [ ] `buf lint` passes for all proto files
- [ ] `buf generate` produces Python Connect RPC bindings
- [ ] Generated code importable

---

## Stage 3: VM Backend Protocol

**Goal**: Define the pluggable VM backend abstraction. Implement LocalBackend
that provides a single "VM" representing the local machine.

**Implementation**:

```python
# src/fluster/backend/base.py
from typing import Protocol, AsyncIterator
from fluster.types import VMId, VMInfo, ResourceConfig

class VMBackend(Protocol):
    """Pluggable backend for VM allocation and management.

    Implementations:
    - LocalBackend: Single "VM" on local machine
    - DockerBackend: VMs as Docker containers (for testing)
    - GCPBackend: GCP Compute Engine VMs
    """

    async def allocate(self, resources: ResourceConfig) -> VMInfo:
        """Allocate a VM matching resource requirements.

        May provision a new VM or return an existing warm VM.
        """
        ...

    async def release(self, vm_id: VMId) -> None:
        """Release a VM back to the pool or terminate it."""
        ...

    async def get_status(self, vm_id: VMId) -> VMInfo:
        """Get current status of a VM."""
        ...

    async def list_vms(self) -> list[VMInfo]:
        """List all VMs managed by this backend."""
        ...

    async def wait_ready(self, vm_id: VMId, timeout: float = 300.0) -> None:
        """Wait until VM is ready to accept connections."""
        ...


# src/fluster/backend/local.py
class LocalBackend:
    """Backend that provides a single 'VM' - the local machine.

    Used for development and testing. The 'VM' runs a Worker
    in-process or as a subprocess.
    """

    def __init__(self, worker_port: int = 0):
        self._vm_id = VMId("local")
        self._worker_port = worker_port
        self._worker_process: subprocess.Popen | None = None

    async def allocate(self, resources: ResourceConfig) -> VMInfo:
        # Start local worker if not running
        if self._worker_process is None:
            self._start_local_worker()
        return VMInfo(
            vm_id=self._vm_id,
            address=f"localhost:{self._worker_port}",
            status="ready",
            resources=resources,
        )

    def _start_local_worker(self):
        """Start worker service in subprocess."""
        # python -m fluster.worker.main serve --port ...
        ...

    async def release(self, vm_id: VMId) -> None:
        # Local backend keeps worker running
        pass

    async def list_vms(self) -> list[VMInfo]:
        if self._worker_process:
            return [VMInfo(self._vm_id, f"localhost:{self._worker_port}", "ready", ...)]
        return []
```

**Exit Conditions**:

- [ ] `VMBackend` protocol defined with allocate/release/get_status/list_vms
- [ ] `LocalBackend` implements `VMBackend`
- [ ] LocalBackend starts worker as subprocess
- [ ] LocalBackend provides single "VM" at localhost
- [ ] Test: allocate VM, verify address returned
- [ ] Test: worker subprocess starts and is reachable

---

## Stage 4: Worker Service

**Goal**: Implement the Worker service that runs on each VM. Handles job
execution in containers.

**Reference**: `worker/` directory (manager.py, runtime.py, server.py)

**Implementation**:

```python
# src/fluster/worker/runtime.py
# Port from worker/runtime.py

from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class ContainerConfig:
    image: str
    command: list[str]
    env: dict[str, str]
    mounts: list[tuple[str, str]]  # (host_path, container_path)
    cpu_limit: float | None = None
    memory_limit: str | None = None
    timeout: float | None = None

@dataclass
class ContainerResult:
    exit_code: int
    started_at: float
    finished_at: float

class Runtime(ABC):
    """Container runtime abstraction."""

    @abstractmethod
    async def run(self, config: ContainerConfig) -> ContainerResult: ...

    @abstractmethod
    async def stream_logs(self, container_id: str) -> AsyncIterator[str]: ...

    @abstractmethod
    async def kill(self, container_id: str, force: bool = False) -> None: ...

class DockerRuntime(Runtime):
    """Docker-based container runtime."""
    ...

class ProcessRuntime(Runtime):
    """Direct process execution (no container)."""
    ...

def get_runtime(name: str = "auto") -> Runtime:
    """Factory for runtime selection."""
    ...


# src/fluster/worker/manager.py
# Port from worker/manager.py

class JobManager:
    """Manages job lifecycle on a single worker."""

    def __init__(
        self,
        runtime: Runtime,
        max_concurrent_jobs: int = 10,
    ):
        self._runtime = runtime
        self._semaphore = asyncio.Semaphore(max_concurrent_jobs)
        self._jobs: dict[str, Job] = {}

    async def submit_job(self, job_id: str, entrypoint: bytes, env: dict) -> None:
        """Submit job for execution (returns immediately)."""
        ...

    async def get_status(self, job_id: str) -> JobStatus:
        ...

    async def stream_logs(self, job_id: str) -> AsyncIterator[LogEntry]:
        ...

    async def kill_job(self, job_id: str, force: bool = False) -> None:
        ...


# src/fluster/worker/service.py
# RPC server implementation

class WorkerServicer:
    """Implements WorkerService RPC interface."""

    def __init__(self, manager: JobManager):
        self._manager = manager

    async def run_job(self, request: RunJobRequest, ctx) -> RunJobResponse:
        await self._manager.submit_job(
            request.job_id,
            request.serialized_entrypoint,
            dict(request.env_vars),
        )
        return RunJobResponse(job_id=request.job_id, status=JobStatus.PENDING)

    async def get_job_status(self, request: GetStatusRequest, ctx) -> GetStatusResponse:
        status = await self._manager.get_status(request.job_id)
        return GetStatusResponse(job_id=request.job_id, status=status, ...)

    # ... other methods

class WorkerServer:
    """HTTP/Connect RPC server for Worker service."""

    def __init__(self, servicer: WorkerServicer, host: str = "0.0.0.0", port: int = 0):
        ...

    def start(self) -> int:
        """Start server, return bound port."""
        ...

    def shutdown(self):
        ...
```

**Exit Conditions**:

- [ ] `Runtime` abstraction with `DockerRuntime` and `ProcessRuntime`
- [ ] `JobManager` handles concurrent job execution
- [ ] `WorkerServicer` implements proto RPC interface
- [ ] `WorkerServer` runs uvicorn with Connect RPC
- [ ] Test: submit job, check status transitions
- [ ] Test: stream logs from running job
- [ ] Test: kill running job
- [ ] Test: concurrent job limits enforced

---

## Stage 5: Controller Service

**Goal**: Implement the Controller service that manages the cluster. Handles
job dispatch to workers and maintains the endpoint registry. The controller
knows nothing about actors - it just manages jobs and a generic name→address
registry.

**Reference**: `lib/fray/src/fray/job/rpc/controller.py`

**Implementation**:

```python
# src/fluster/cluster/registry.py

class EndpointRegistry:
    """Generic endpoint registry, owned by Controller.

    Maps names to addresses, scoped by namespace. Tracks which job
    registered each endpoint for automatic cleanup. This is NOT
    actor-specific - it's a generic service discovery primitive.
    """

    def __init__(self):
        self._endpoints: dict[EndpointId, Endpoint] = {}
        self._by_name: dict[tuple[Namespace, str], list[EndpointId]] = {}
        self._by_job: dict[JobId, list[EndpointId]] = {}
        self._lock = asyncio.Lock()

    async def register(
        self,
        name: str,
        address: str,
        job_id: JobId,
        namespace: Namespace,
        metadata: dict[str, str] | None = None,
    ) -> Endpoint:
        """Register an endpoint under a name.

        Multiple endpoints can register under the same name.
        """
        ...

    async def unregister(self, endpoint_id: EndpointId) -> None:
        """Remove an endpoint registration."""
        ...

    async def lookup(self, name: str, namespace: Namespace) -> Endpoint | None:
        """Look up one endpoint by name (returns first if multiple)."""
        ...

    async def lookup_all(self, name: str, namespace: Namespace) -> list[Endpoint]:
        """Look up all endpoints registered under a name."""
        ...

    async def cleanup_job(self, job_id: JobId) -> None:
        """Remove all endpoints registered by a job.

        Called automatically when a job terminates.
        """
        ...


# src/fluster/cluster/controller/service.py

class ControllerServicer:
    """Implements ControllerService RPC interface.

    Manages job lifecycle and endpoint registry. Has no knowledge of
    actors - that's built on top by the actor layer.
    """

    def __init__(
        self,
        backend: VMBackend,
        registry: EndpointRegistry,
        namespace: Namespace,
    ):
        self._backend = backend
        self._registry = registry
        self._namespace = namespace
        self._jobs: dict[JobId, JobState] = {}
        self._lock = asyncio.Lock()

    async def launch_job(self, request: JobSpec, ctx) -> JobHandle:
        """Launch a job.

        1. Allocate VM from backend
        2. Wait for VM ready
        3. Get worker client for VM
        4. Dispatch job to worker
        5. Track job state
        """
        job_id = JobId(str(uuid.uuid4()))

        # Deserialize to get resource requirements
        resources = cloudpickle.loads(request.serialized_resources)

        # Allocate VM
        vm = await self._backend.allocate(resources)
        await self._backend.wait_ready(vm.vm_id)

        # Dispatch to worker
        worker_client = WorkerClient(vm.address)
        await worker_client.run_job(
            job_id=job_id,
            serialized_entrypoint=request.serialized_entrypoint,
            serialized_environment=request.serialized_environment,
        )

        # Track state
        self._jobs[job_id] = JobState(
            job_id=job_id,
            name=request.name,
            status=JobStatus.PENDING,
            vm_id=vm.vm_id,
            worker_address=vm.address,
        )

        return JobHandle(job_id=job_id, status=JobStatus.JOB_STATUS_PENDING)

    async def get_job_status(self, request: JobHandle, ctx) -> JobHandle:
        """Get job status by querying worker."""
        job = self._jobs.get(JobId(request.job_id))
        if not job:
            raise KeyError(f"Job {request.job_id} not found")

        # Query worker for current status
        worker_client = WorkerClient(job.worker_address)
        status = await worker_client.get_job_status(job.job_id)

        # Update local state
        job.status = status.status

        # Cleanup endpoints if job finished
        if status.status in (JobStatus.SUCCEEDED, JobStatus.FAILED):
            await self._registry.cleanup_job(job.job_id)

        return JobHandle(job_id=job.job_id, status=status.status, ...)

    async def terminate_job(self, request: JobHandle, ctx) -> Empty:
        """Terminate a running job."""
        job = self._jobs.get(JobId(request.job_id))
        if not job:
            raise KeyError(f"Job {request.job_id} not found")

        worker_client = WorkerClient(job.worker_address)
        await worker_client.kill_job(job.job_id)
        await self._registry.cleanup_job(job.job_id)

        return Empty()

    # Endpoint registry methods (generic, not actor-specific)
    async def register_endpoint(self, request: RegisterEndpointRequest, ctx) -> Endpoint:
        return await self._registry.register(
            request.name,
            request.address,
            JobId(request.job_id),
            Namespace(request.namespace),
            dict(request.metadata),
        )

    async def lookup_endpoints(self, request: LookupRequest, ctx) -> LookupResponse:
        endpoints = await self._registry.lookup_all(request.name, Namespace(request.namespace))
        return LookupResponse(endpoints=[...])


class ControllerServer:
    """HTTP/Connect RPC server for Controller service."""

    def __init__(
        self,
        backend: VMBackend,
        namespace: Namespace = Namespace("default"),
        host: str = "0.0.0.0",
        port: int = 0,
    ):
        self._registry = EndpointRegistry()
        self._servicer = ControllerServicer(backend, self._registry, namespace)
        ...

    def start(self) -> int:
        """Start server, return bound port."""
        ...

    @property
    def address(self) -> str:
        """Return address for clients to connect."""
        ...

    def shutdown(self):
        ...
```

**Exit Conditions**:

- [ ] `EndpointRegistry` tracks endpoints by name/namespace/job
- [ ] `EndpointRegistry.cleanup_job()` removes all endpoints for a job
- [ ] `ControllerServicer` implements all RPC methods
- [ ] Job launch allocates VM and dispatches to worker
- [ ] Job status queries worker and updates local state
- [ ] Job termination kills job and cleans up endpoints
- [ ] `ControllerServer` runs uvicorn with Connect RPC
- [ ] No actor-specific code in cluster/controller/
- [ ] Test: launch job via RPC, verify dispatch to worker
- [ ] Test: register endpoint, lookup endpoint
- [ ] Test: job termination cleans up endpoints

---

## Stage 6: Client Library

**Goal**: Implement the Python client library, split between cluster and actor
packages. The cluster client handles job lifecycle and endpoint registry. The
actor package provides ActorServer, ActorPool, and Resolver built on top.

**Reference**: `lib/fray/src/fray/job/rpc/context.py`

**Implementation**:

```python
# src/fluster/cluster/client.py
# Cluster client - NO actor knowledge

class Cluster:
    """Client for interacting with the Controller service.

    Handles job lifecycle and endpoint registry access. Does NOT
    know about actors - that's the actor layer's responsibility.
    """

    def __init__(self, address: str, namespace: Namespace | None = None):
        self._address = address
        self._namespace = namespace or Namespace(os.environ.get("FLUSTER_NAMESPACE", "default"))
        self._client = ControllerClient(address)

    def launch(self, request: JobRequest) -> JobId:
        """Launch a job on the cluster."""
        spec = JobSpec(
            name=request.name,
            serialized_entrypoint=cloudpickle.dumps(request.entrypoint),
            serialized_resources=cloudpickle.dumps(request.resources),
            serialized_environment=cloudpickle.dumps(request.environment),
        )
        result = asyncio.run(self._client.launch_job(spec))
        return JobId(result.job_id)

    def poll(self, job_id: JobId) -> JobInfo:
        """Get current status of a job."""
        handle = JobHandle(job_id=job_id)
        result = asyncio.run(self._client.get_job_status(handle))
        return JobInfo(
            job_id=JobId(result.job_id),
            status=_convert_status(result.status),
            ...
        )

    def wait(self, job_id: JobId | Sequence[JobId], raise_on_failure: bool = False) -> JobInfo | list[JobInfo]:
        """Block until job(s) complete."""
        ...

    def terminate(self, job_id: JobId) -> None:
        ...

    def list_jobs(self) -> list[JobInfo]:
        ...

    # Endpoint registry access (generic, not actor-specific)
    def register_endpoint(
        self,
        name: str,
        address: str,
        job_id: JobId | None = None,
        metadata: dict[str, str] | None = None,
    ) -> Endpoint:
        """Register an endpoint with the cluster's registry."""
        if job_id is None:
            job_id = JobId(os.environ.get("FLUSTER_JOB_ID", "local"))
        result = asyncio.run(self._client.register_endpoint(
            RegisterEndpointRequest(
                name=name,
                address=address,
                job_id=job_id,
                namespace=self._namespace,
                metadata=metadata or {},
            )
        ))
        return _convert_endpoint(result)

    def lookup_endpoints(self, name: str) -> list[Endpoint]:
        """Look up endpoints by name."""
        result = asyncio.run(self._client.lookup_endpoints(
            LookupRequest(name=name, namespace=self._namespace)
        ))
        return [_convert_endpoint(e) for e in result.endpoints]

    @property
    def namespace(self) -> Namespace:
        return self._namespace


def current_cluster() -> Cluster:
    """Get cluster from FLUSTER_CLUSTER_ADDRESS env var.

    Returns LocalCluster if env var not set.
    """
    address = os.environ.get("FLUSTER_CLUSTER_ADDRESS")
    if address is None:
        return LocalCluster()
    return Cluster(address)


class LocalCluster(Cluster):
    """Local cluster that starts controller in-process.

    For development and testing.
    """

    def __init__(self, namespace: Namespace | None = None):
        # Start controller with LocalBackend
        self._backend = LocalBackend()
        self._server = ControllerServer(self._backend, namespace or Namespace("local"))
        port = self._server.start()
        super().__init__(f"http://localhost:{port}", namespace)

    def shutdown(self):
        self._server.shutdown()
```

```python
# src/fluster/actor/pool.py
# Actor pool - depends on cluster

from fluster.cluster.client import Cluster
from fluster.cluster.types import Endpoint
from fluster.actor.types import ActorEndpoint
from fluster.actor.proxy import ActorProxy

class ActorPool(Generic[T]):
    """Pool of actors registered under a common name.

    Built on top of the cluster's endpoint registry.
    """

    def __init__(self, cluster: Cluster, name: str):
        self._cluster = cluster
        self._name = name
        self._endpoints: list[ActorEndpoint] = []
        self._idx = 0

    @property
    def size(self) -> int:
        self._refresh()
        return len(self._endpoints)

    def _refresh(self):
        """Refresh endpoint list from cluster's registry."""
        endpoints = self._cluster.lookup_endpoints(self._name)
        self._endpoints = [_to_actor_endpoint(e) for e in endpoints]

    def wait_for_size(self, min_size: int, timeout: float = 60.0) -> None:
        """Block until pool has at least min_size actors."""
        ...

    def call(self) -> T:
        """Get proxy for round-robin calls to one actor."""
        self._refresh()
        if not self._endpoints:
            raise RuntimeError(f"No actors registered for '{self._name}'")
        endpoint = self._endpoints[self._idx % len(self._endpoints)]
        self._idx += 1
        return ActorProxy(endpoint.address)

    def broadcast(self) -> BroadcastHandle[T]:
        """Get handle for broadcasting to all actors."""
        self._refresh()
        return BroadcastHandle(self._endpoints)


def _to_actor_endpoint(endpoint: Endpoint) -> ActorEndpoint:
    """Convert cluster Endpoint to ActorEndpoint."""
    return ActorEndpoint(
        actor_id=ActorId(endpoint.endpoint_id),
        name=endpoint.name,
        address=endpoint.address,
        job_id=endpoint.job_id,
        namespace=endpoint.namespace,
        metadata=endpoint.metadata,
    )
```

```python
# src/fluster/actor/resolver.py

from typing import Protocol
from fluster.cluster.client import Cluster
from fluster.actor.pool import ActorPool

class Resolver(Protocol):
    """Protocol for actor discovery."""
    def lookup(self, name: str) -> ActorPool: ...


class ClusterResolver:
    """Resolver backed by a Cluster's endpoint registry."""

    def __init__(self, cluster: Cluster):
        self._cluster = cluster

    def lookup(self, name: str) -> ActorPool:
        return ActorPool(self._cluster, name)


class FixedResolver:
    """Resolver with fixed actor addresses (for testing)."""

    def __init__(self, addresses: dict[str, str | list[str]]):
        self._addresses = addresses

    def lookup(self, name: str) -> ActorPool:
        ...
```

```python
# src/fluster/actor/server.py

from fluster.cluster.client import Cluster
from fluster.actor.types import ActorId, ActorContext

class ActorServer:
    """Server for hosting actors.

    Registers actors with the cluster's endpoint registry.
    Uses actor.proto for method invocation RPC.
    """

    def __init__(
        self,
        cluster: Cluster,
        host: str = "0.0.0.0",
        port: int = 0,
    ):
        self._cluster = cluster
        self._host = host
        self._port = port
        self._actors: dict[str, Any] = {}
        self._server: uvicorn.Server | None = None
        self._endpoint_ids: dict[str, EndpointId] = {}

    @property
    def address(self) -> str:
        return f"{self._host}:{self._port}"

    def register(self, name: str, actor: Any, metadata: dict[str, str] | None = None) -> ActorId:
        """Register an actor and notify cluster's registry."""
        self._actors[name] = actor

        # Register with cluster's endpoint registry
        endpoint = self._cluster.register_endpoint(
            name=name,
            address=self.address,
            metadata=metadata,
        )
        self._endpoint_ids[name] = endpoint.endpoint_id
        return ActorId(endpoint.endpoint_id)

    def serve(self) -> None:
        """Start serving requests (blocks).

        Exposes ActorService RPC for method calls.
        """
        ...

    def serve_background(self) -> None:
        """Start serving in background thread."""
        ...

    def shutdown(self):
        ...
```

**Exit Conditions**:

- [ ] `Cluster` client implements launch/poll/wait/terminate (cluster layer)
- [ ] `Cluster` client exposes register_endpoint/lookup_endpoints (cluster layer)
- [ ] `LocalCluster` starts controller with LocalBackend in-process
- [ ] `current_cluster()` returns appropriate cluster from env
- [ ] `ActorPool` supports call() and broadcast() (actor layer)
- [ ] `ClusterResolver` looks up actors via cluster's registry (actor layer)
- [ ] `ActorServer` registers with cluster's registry (actor layer)
- [ ] `ActorServer` handles RPC calls to actor methods via actor.proto
- [ ] No imports from `actor/` in `cluster/` package
- [ ] Test: LocalCluster job launch end-to-end
- [ ] Test: endpoint registration and lookup (cluster layer)
- [ ] Test: actor registration and method call (actor layer)

---

## Stage 7: Integration Tests (Local)

**Goal**: End-to-end tests using `LocalCluster` that validate the complete
system.

**Implementation**:

```python
# tests/test_integration.py

def test_job_lifecycle():
    """Test basic job launch, status, completion."""
    cluster = LocalCluster()

    def my_task():
        return 42

    job_id = cluster.launch(JobRequest(
        name="test-job",
        resources=ResourceConfig.with_cpu(),
        entrypoint=Entrypoint(callable=my_task),
        environment=EnvironmentConfig(),
    ))

    result = cluster.wait(job_id)
    assert result.status == JobStatus.SUCCEEDED

    cluster.shutdown()


def test_actor_registration_and_call():
    """Test actor server and client interaction."""
    cluster = LocalCluster()
    resolver = ClusterResolver(cluster)

    class Calculator:
        def add(self, ctx: ActorContext, a: int, b: int) -> int:
            return a + b

    # Start actor server
    server = ActorServer(cluster)
    server.register("calculator", Calculator())
    server.serve_background()

    # Client looks up and calls actor
    pool = resolver.lookup("calculator")
    pool.wait_for_size(1)

    result = pool.call().add(2, 3)
    assert result == 5

    server.shutdown()
    cluster.shutdown()


def test_worker_pool():
    """Test WorkerPool for task dispatch."""
    cluster = LocalCluster()

    def square(x: int) -> int:
        return x * x

    pool = WorkerPool(cluster, num_workers=2, resources=ResourceConfig.with_cpu())
    pool.wait_for_workers()

    futures = pool.map(square, [1, 2, 3, 4])
    results = [f.result() for f in futures]

    assert sorted(results) == [1, 4, 9, 16]

    pool.shutdown()
    cluster.shutdown()


def test_job_failure_propagation():
    """Test that job failures are properly reported."""
    cluster = LocalCluster()

    def failing_task():
        raise ValueError("intentional failure")

    job_id = cluster.launch(JobRequest(
        name="failing-job",
        resources=ResourceConfig.with_cpu(),
        entrypoint=Entrypoint(callable=failing_task),
        environment=EnvironmentConfig(),
    ))

    result = cluster.wait(job_id)
    assert result.status == JobStatus.FAILED
    assert "intentional failure" in result.error_message

    cluster.shutdown()


def test_actor_cleanup_on_job_termination():
    """Test that actors are cleaned up when job terminates."""
    cluster = LocalCluster()
    resolver = ClusterResolver(cluster)

    # This would be tested with a job that registers actors
    # then terminates, verifying actors are removed from registry
    ...
```

**Exit Conditions**:

- [ ] Job lifecycle test passes (launch → running → succeeded)
- [ ] Actor registration and call test passes
- [ ] WorkerPool test passes
- [ ] Job failure propagation test passes
- [ ] Actor cleanup on job termination test passes
- [ ] All tests run in <30 seconds

---

## Stage 8: WorkerPool

**Goal**: Implement `WorkerPool` for task dispatch patterns (like Zephyr).

**Implementation**:

```python
# src/fluster/pool.py

class WorkerPool:
    """Pool of stateless workers for task dispatch.

    Launches worker jobs that expose a task execution actor.
    Tasks are distributed round-robin.
    """

    def __init__(
        self,
        cluster: Cluster,
        num_workers: int,
        resources: ResourceConfig,
        environment: EnvironmentConfig | None = None,
        name_prefix: str = "worker",
    ):
        self._cluster = cluster
        self._num_workers = num_workers
        self._resources = resources
        self._environment = environment or EnvironmentConfig()
        self._name_prefix = name_prefix
        self._job_ids: list[JobId] = []
        self._resolver = ClusterResolver(cluster)
        self._pool: ActorPool | None = None

    def _launch_workers(self):
        """Launch worker jobs."""
        for i in range(self._num_workers):
            job_id = self._cluster.launch(JobRequest(
                name=f"{self._name_prefix}-{i}",
                resources=self._resources,
                entrypoint=Entrypoint(callable=_worker_main, args=(self._name_prefix,)),
                environment=self._environment,
            ))
            self._job_ids.append(job_id)

    def wait_for_workers(self, min_workers: int | None = None, timeout: float = 60.0) -> None:
        """Wait for workers to be ready."""
        if not self._job_ids:
            self._launch_workers()
        self._pool = self._resolver.lookup(self._name_prefix)
        self._pool.wait_for_size(min_workers or self._num_workers, timeout)

    @property
    def size(self) -> int:
        if self._pool is None:
            return 0
        return self._pool.size

    def submit(self, fn: Callable[..., T], *args, **kwargs) -> ActorFuture[T]:
        """Submit task for execution."""
        if self._pool is None:
            raise RuntimeError("Call wait_for_workers() first")
        return self._pool.call().execute(fn, args, kwargs)

    def map(self, fn: Callable[[Any], T], items: Sequence[Any]) -> list[ActorFuture[T]]:
        """Map function over items."""
        return [self.submit(fn, item) for item in items]

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown worker pool."""
        for job_id in self._job_ids:
            self._cluster.terminate(job_id)
        if wait:
            for job_id in self._job_ids:
                self._cluster.wait(job_id)


def _worker_main(pool_name: str):
    """Entrypoint for worker jobs."""
    cluster = current_cluster()
    server = ActorServer(cluster)
    server.register(pool_name, _TaskExecutor())
    server.serve()


class _TaskExecutor:
    """Actor that executes arbitrary callables."""

    def execute(self, ctx: ActorContext, fn: Callable, args: tuple, kwargs: dict) -> Any:
        return fn(*args, **kwargs)
```

**Exit Conditions**:

- [ ] `WorkerPool` launches worker jobs
- [ ] Workers register as actors under pool name
- [ ] `submit()` dispatches tasks round-robin
- [ ] `map()` submits all items
- [ ] `shutdown()` terminates worker jobs
- [ ] Test: submit tasks, collect results
- [ ] Test: map over items

---

## Stage 9: Docker Backend

**Goal**: Implement Docker-based VM backend for testing the real cluster
architecture locally. Each "VM" is a Docker container running a Worker.

**Implementation**:

```python
# src/fluster/backend/docker.py

class DockerBackend(VMBackend):
    """VM backend using Docker containers.

    Each 'VM' is a container running a Worker service.
    Used for integration testing without real VMs.
    """

    def __init__(
        self,
        image: str = "python:3.11-slim",
        network: str | None = None,
    ):
        self._image = image
        self._network = network
        self._containers: dict[VMId, ContainerInfo] = {}
        self._docker = docker.from_env()

    async def allocate(self, resources: ResourceConfig) -> VMInfo:
        """Start a container running Worker service."""
        vm_id = VMId(str(uuid.uuid4())[:8])

        # Build or use cached worker image
        worker_image = self._ensure_worker_image()

        # Start container
        container = self._docker.containers.run(
            worker_image,
            command=["python", "-m", "fluster.worker.main", "serve"],
            detach=True,
            network=self._network,
            environment={
                "FLUSTER_WORKER_ID": vm_id,
            },
            # Resource limits based on resources param
            mem_limit=resources.memory,
            cpu_period=100000,
            cpu_quota=int(resources.cpu * 100000),
        )

        # Get container IP
        container.reload()
        ip = container.attrs['NetworkSettings']['IPAddress']
        address = f"{ip}:8080"

        self._containers[vm_id] = ContainerInfo(
            vm_id=vm_id,
            container_id=container.id,
            address=address,
        )

        return VMInfo(vm_id=vm_id, address=address, status="starting", resources=resources)

    async def release(self, vm_id: VMId) -> None:
        """Stop and remove container."""
        info = self._containers.get(vm_id)
        if info:
            container = self._docker.containers.get(info.container_id)
            container.stop()
            container.remove()
            del self._containers[vm_id]

    async def wait_ready(self, vm_id: VMId, timeout: float = 60.0) -> None:
        """Wait for worker health check to pass."""
        info = self._containers[vm_id]
        client = WorkerClient(info.address)
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                health = await client.health_check()
                if health.healthy:
                    return
            except Exception:
                pass
            await asyncio.sleep(0.5)
        raise TimeoutError(f"VM {vm_id} not ready after {timeout}s")
```

**Exit Conditions**:

- [ ] `DockerBackend` starts containers as "VMs"
- [ ] Containers run Worker service
- [ ] Network communication between containers works
- [ ] `wait_ready()` polls health check
- [ ] `release()` stops and removes container
- [ ] All Stage 7 integration tests pass with `DockerBackend`
- [ ] Test: job in one container calls actor in another

---

## Stage 10: GCP Compute Engine Backend

**Goal**: Production backend using GCP Compute Engine VMs with SSH for setup.

**Implementation**:

```python
# src/fluster/backend/gcp.py

@dataclass
class GCPConfig:
    project: str
    zone: str
    service_account: str | None = None
    network: str = "default"
    subnetwork: str | None = None
    machine_type_cpu: str = "n2-standard-4"
    # TPU handled separately

class GCPBackend(VMBackend):
    """GCP Compute Engine backend.

    Provisions VMs on demand, runs Worker via SSH.
    """

    def __init__(self, config: GCPConfig):
        self._config = config
        self._compute = compute_v1.InstancesClient()
        self._vms: dict[VMId, GCPVMInfo] = {}

    async def allocate(self, resources: ResourceConfig) -> VMInfo:
        """Provision GCE VM and start Worker."""
        vm_id = VMId(f"fluster-{uuid.uuid4().hex[:8]}")

        # Determine machine type from resources
        machine_type = self._get_machine_type(resources)

        # Create instance
        instance = compute_v1.Instance(
            name=vm_id,
            machine_type=f"zones/{self._config.zone}/machineTypes/{machine_type}",
            disks=[...],
            network_interfaces=[...],
            metadata={
                "items": [
                    {"key": "startup-script", "value": self._startup_script()},
                ]
            },
        )

        operation = self._compute.insert(
            project=self._config.project,
            zone=self._config.zone,
            instance_resource=instance,
        )
        operation.result()  # Wait for creation

        # Get external IP
        instance = self._compute.get(
            project=self._config.project,
            zone=self._config.zone,
            instance=vm_id,
        )
        ip = instance.network_interfaces[0].access_configs[0].nat_i_p

        self._vms[vm_id] = GCPVMInfo(vm_id=vm_id, ip=ip, instance_name=vm_id)

        return VMInfo(vm_id=vm_id, address=f"{ip}:8080", status="starting", resources=resources)

    def _startup_script(self) -> str:
        """Script that installs and starts Worker on VM."""
        return """#!/bin/bash
        apt-get update && apt-get install -y python3-pip
        pip install fluster
        python -m fluster.worker.main serve --host 0.0.0.0 --port 8080
        """

    async def release(self, vm_id: VMId) -> None:
        """Delete VM."""
        self._compute.delete(
            project=self._config.project,
            zone=self._config.zone,
            instance=vm_id,
        )

    async def wait_ready(self, vm_id: VMId, timeout: float = 300.0) -> None:
        """Wait for VM to be ready and Worker responding."""
        ...
```

**Exit Conditions**:

- [ ] `GCPBackend` provisions VMs via Compute Engine API
- [ ] Startup script installs and starts Worker
- [ ] `wait_ready()` polls until Worker responds
- [ ] `release()` deletes VM
- [ ] Test: launch job on GCP VM (manual/integration test)
- [ ] Test: actor communication across VMs

---

## Stage 11: TPU Slice Support

**Goal**: Extend GCP backend to support TPU slices.

**Implementation**:

```python
# src/fluster/backend/gcp.py (extended)

class GCPBackend:
    def __init__(self, config: GCPConfig):
        ...
        self._tpu = tpu_v2.TpuClient()

    async def allocate(self, resources: ResourceConfig) -> VMInfo:
        if isinstance(resources.device, TpuConfig):
            return await self._allocate_tpu(resources)
        return await self._allocate_vm(resources)

    async def _allocate_tpu(self, resources: ResourceConfig) -> VMInfo:
        """Provision TPU slice."""
        tpu_config = resources.device
        vm_id = VMId(f"fluster-tpu-{uuid.uuid4().hex[:8]}")

        # Create TPU node
        node = tpu_v2.Node(
            accelerator_type=tpu_config.variant,
            runtime_version="tpu-ubuntu2204-base",
            network_config=tpu_v2.NetworkConfig(...),
            metadata={
                "startup-script": self._tpu_startup_script(),
            },
        )

        operation = self._tpu.create_node(
            parent=f"projects/{self._config.project}/locations/{self._config.zone}",
            node=node,
            node_id=vm_id,
        )
        operation.result()

        # Get network endpoints (TPU slices have multiple VMs)
        node = self._tpu.get_node(name=f".../{vm_id}")
        endpoints = node.network_endpoints

        # Return first endpoint as primary address
        # Worker runs on all VMs in slice
        primary_ip = endpoints[0].ip_address

        return VMInfo(
            vm_id=vm_id,
            address=f"{primary_ip}:8080",
            status="starting",
            resources=resources,
        )

    def _tpu_startup_script(self) -> str:
        """Startup script for TPU VMs."""
        return """#!/bin/bash
        # Install fluster and JAX TPU
        pip install fluster jax[tpu]
        # Start worker with TPU runtime
        python -m fluster.worker.main serve --host 0.0.0.0 --port 8080 --runtime process
        """
```

**Exit Conditions**:

- [ ] TPU slice provisioning works
- [ ] Worker starts on TPU VMs
- [ ] JAX environment configured correctly
- [ ] TPU slice cleanup works
- [ ] Test: launch training job on TPU (manual verification)

---

## Stage 12: Zephyr Integration

**Goal**: Create `FlusterBackend` for Zephyr.

**Implementation**:

```python
# In lib/zephyr or lib/fluster/integrations/zephyr.py

from fluster import current_cluster, WorkerPool, ResourceConfig, ClusterResolver
from zephyr.backends import Backend

class FlusterBackend(Backend):
    """Zephyr backend using Fluster for distributed execution."""

    def __init__(
        self,
        cluster: Cluster | None = None,
        max_parallelism: int = 100,
        memory_per_worker: str = "2GB",
    ):
        self._cluster = cluster or current_cluster()
        self._max_parallelism = max_parallelism
        self._memory_per_worker = memory_per_worker

    def execute(self, dataset: Dataset) -> list[Any]:
        pool = WorkerPool(
            cluster=self._cluster,
            num_workers=self._max_parallelism,
            resources=ResourceConfig.with_cpu(memory=self._memory_per_worker),
        )
        pool.wait_for_workers()

        try:
            # Execute pipeline using pool
            ...
        finally:
            pool.shutdown()

        return results
```

**Exit Conditions**:

- [ ] `FlusterBackend` implements Zephyr `Backend` protocol
- [ ] Zephyr pipelines execute on Fluster workers
- [ ] Test: run Zephyr pipeline with FlusterBackend + LocalCluster
- [ ] Test: run Zephyr pipeline with FlusterBackend + DockerBackend

---

## Stage 13: Migration and Cleanup

**Goal**: Replace fray's RPC code with fluster, delete old code.

**Implementation**:

1. Update `lib/zephyr` to use `fluster` instead of `fray.job.rpc`
2. Update any other fray consumers
3. Delete `lib/fray/src/fray/job/rpc/`
4. Update documentation

**Exit Conditions**:

- [ ] No code imports from `fray.job.rpc`
- [ ] All tests pass
- [ ] Old RPC code deleted
- [ ] Documentation updated

---

## Dependency Graph

```
Stage 0: Project Scaffolding
    │
    ▼
Stage 1: Core Types
    │
    ▼
Stage 2: Proto Definitions
    │
    ├────────────────────┐
    ▼                    ▼
Stage 3: VM Backend    Stage 4: Worker Service
    │                    │
    └────────┬───────────┘
             │
             ▼
       Stage 5: Controller Service
             │
             ▼
       Stage 6: Client Library
             │
             ▼
       Stage 7: Integration Tests (Local)
             │
             ▼
       Stage 8: WorkerPool
             │
             ├─────────────────┐
             ▼                 ▼
   Stage 9: Docker Backend   Stage 10: GCP Backend
             │                 │
             │                 ▼
             │           Stage 11: TPU Support
             │                 │
             └────────┬────────┘
                      │
                      ▼
              Stage 12: Zephyr Integration
                      │
                      ▼
              Stage 13: Migration
```

---

## Testing Strategy

**Unit Tests** (per-stage):
- Test individual classes in isolation
- Mock external dependencies (Docker, GCP APIs)
- Fast execution (<1s per test)

**Integration Tests** (Stage 7+):
- End-to-end tests with real components
- Use `LocalCluster` for speed
- Test complete workflows

**System Tests** (Stage 9+):
- Docker-based tests for realistic isolation
- GCP tests for production validation
- Manual verification for TPU

---

## Reference: Design Document Mapping

| Design Section | Implementation Stage |
|----------------|---------------------|
| Core Types | Stage 1 |
| Resource Configuration | Stage 1 |
| Cluster Interface | Stage 5 (Controller), Stage 6 (Client) |
| Metadata Service | Stage 5 (in Controller) |
| Resolver and ActorPool | Stage 6 |
| ActorServer | Stage 6 |
| WorkerPool | Stage 8 |
| Local Development | Stages 3, 6, 7 |
| Job Management | Stage 5 |
| Example Usage | Stage 7 (tests) |
