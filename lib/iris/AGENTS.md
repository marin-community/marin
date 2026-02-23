# Agent Tips

* Use the connect/RPC abstractions to implement and perform RPC calls. DO NOT use httpx or raw HTTP.
* Use scripts/generate_protos.py to regenerate files after changing the `.proto` files.
* Prefer _shallow_, _functional_ code which returns control quickly to the user, vs callbacks or inheritance.

```
class Scheduler:
  def add_job():
  def add_worker():
  def compute_schedule() -> ScheduledJobs:

class Runner:
  def run_jobs(ScheduledJobs)
```

is preferable to:

```
class Scheduler:
  def __init__(self, job_creator: JobCreator):
    self.job_creator = job_creator
  def run(self):
    ... self.job_creator.create_job()
```

* Tests should test stable behavior, not implementation details.

ABSOLUTELY DO NOT test things that are trivially caught by the type checker.
Explicitly that means:

- No tests for "constant = constant"
- No tests for "method exists"
- No tests for "create an object(x, y, z) and attributes are x, y, z"

These tests have negative value - they make our code more brittle.

Test _stable behavior_ instead. You can use mocks as needed to isolate
environments (e.g.  mock around a remote API), but prefer "fakes" -- e.g. create
a real database but with fake data -- when reasonable.

## Documentation

ALWAYS read the docs for the appropriate area.
IF they disagree with the code, ALWAYS add a task to update them.

Documentation should be kept up-to-date as code changes. When implementing new features or making significant changes, update the relevant documentation files:

@README.md - Main overview, CLI reference, and quick start

## Protocols and Testing

Non-trivial public classes should define a protocol which represents their
_important_ interface characteristics. Use this protocol in type hints for
when the class is used instead of the concrete class.

Test to this protocol, not the concrete class: the protocol should describe the
interesting behavior of the class, but not betray the implementation details.

(You may of course _instantiate_ the concrete class for testing.)

## Imports

Don't use TYPE_CHECKING. Use the real import. If there is a circular dependency:

* Prefer to resolve it with refactoring when sensible
* Otherwise use a protocol if you simply need the type information

## RPC/API Accessibility

Any functionality exposed by the worker or controller dashboards must also be
available via RPC. The dashboards should be a friendly interface on top of the
machine accessible RPC API, and should not use internal APIs (except for
efficiency). For example, if we wanted to show the scheduling status for a task,
we should define a new RPC endpoint `/TestSchedule(task_id)` and use that from
the dashboard, rather than creating a scheduler and running it manually.

## Architecture Notes

### Concurrency Model

Platform operations (`terminate`, `create_slice`, etc.) shell out to `gcloud`
via `subprocess.run` and are thread-safe. When multiple independent platform
operations need to run (e.g. tearing down N slices), use
`concurrent.futures.ThreadPoolExecutor` — not asyncio. Always apply a hard
timeout so the CLI doesn't hang on a stuck gcloud call.

## Planning

Prefer _spiral_ plans over _linear_ plans. e.g. when implementing a new feature, make a plan which has step 1 as:

Step 1:
* Add the minimal changes to the `.proto` file
* Add the server stub code
* Add a client wiring
* Add an end-to-end test

Step 2:
* Extend proto with additional field
* Update server code
* Update client code
* Update tests

...

That is _each stage of the plan_ should be a independently testable,
self-contained unit of work. THis is preferable to plans which attempt to make
all of the changes for one area (e.g. all proto changes, then all server
changes, etc.)

When adding new modules or significant features:
1. Update the README with a brief overview and usage examples
2. Add detailed documentation to the appropriate docs/ file
3. Reference the documentation from this AGENTS.md file

**Key documentation areas:**

| Area | File | Description |
|------|------|-------------|
| Architecture | README.md | High-level architecture, CLI reference, quick start |
| Autoscaler Design | docs/autoscaler-v0-design.md | Technical specification, threading model |
| Thread Safety | docs/thread-safety.md | Thread management, test synchronization best practices |
| Original Design | docs/fray-zero.md | Rationale and design decisions |
| CoreWeave Integration | (below) | Platform, runtime, and networking for CoreWeave bare metal |

### CoreWeave Integration

Iris runs on CoreWeave bare-metal GPU nodes. The integration spans three layers:

**Platform** (`cluster/platform/coreweave.py`): Uses shared NodePools with CoreWeave autoscaling.
`iris cluster start` creates one NodePool per scale group (plus a controller pool) via
`ensure_nodepools()`. NodePool names are derived from config: `{label_prefix}-{scale_group_name}`.
Each NodePool has `autoscaling: true` and `targetNodes: 0`; CoreWeave provisions nodes on demand
when Pods are scheduled. Iris manages only Pods; cleanup/stop leaves NodePools alone (they scale
to zero when idle). The controller runs as a Deployment with a `ClusterIP` Service
(`iris-controller-svc`). Workers discover the controller via in-cluster DNS
(`iris-controller-svc.iris.svc.cluster.local:10000`).

**Runtime**: `kubernetes` (`cluster/runtime/kubernetes.py`). Each task is a separate K8s Pod that
can be scheduled independently (not co-located with the worker). Task Pods claim GPU/RDMA resources
directly from the device plugin. The worker Pod must **not** request GPU/RDMA resources (the
platform skips them when `runtime: kubernetes`). Task Pods get `hostNetwork`, tolerations,
service account, S3 secret refs, and an emptyDir-backed UV cache automatically.
Docker is not available on CoreWeave bare metal.

**Networking**: All traffic stays inside the CoreWeave VPC. Worker Pods use `hostNetwork: True`
(bypassing the Kubernetes overlay for RDMA/GPU performance). Task Pods set
`hostNetwork: True` + `dnsPolicy: ClusterFirstWithHostNet` in their Pod spec.

### Light Worker Mode (CoreWeave + runtime=kubernetes)

When `runtime: kubernetes` is configured, worker Pods are intentionally "light":
- Worker Pod must not request `nvidia.com/gpu` or `rdma/ib`.
- Task Pods created by `cluster/runtime/kubernetes.py` request accelerators per task.
- Worker Pod still uses the scale-group `nodeSelector` and `hostNetwork: true`.
- Worker Pod passes control-plane env needed for task-pod creation (for example
  `IRIS_SERVICE_ACCOUNT_NAME`, and `IRIS_S3_SECRET_NAME` when S3 is enabled).

Quick verification:
- Worker create log should show `resource_limits=none`.
- `kubectl get pod <worker> -o jsonpath='{.spec.containers[0].resources}'` should be empty.
- Task pod specs should include GPU limits when task resources request GPUs.

**Disk layout**: CoreWeave bare-metal nodes have a 15 GB RAM disk (`/dev/ram0`) as the root
filesystem and a multi-TB NVMe RAID (`/dev/md127`) mounted at `/mnt/local`. Bind mounts expose
it as `/var/lib/containerd`, `/var/lib/kubelet`, `/opt`, etc. The `cache_dir` must point to the
NVMe (e.g. `/mnt/local/iris-cache`) — the default `/var/cache/iris` lands on the tiny RAM disk
and will fill up immediately when installing CUDA packages.

**K8s manifests** (`infra/coreweave/k8s/`):
- `namespace.yaml`, `service-account.yaml`, `cluster-role.yaml`, `cluster-role-binding.yaml` — RBAC/namespace prerequisites (one-time operator setup)

Controller lifecycle resources (ConfigMap, shared NodePools, Deployment, Service) are created
automatically by `iris cluster start` via `CoreweavePlatform.start_controller()`.

## Key Modules

### Time Utilities

Use `iris.time_utils` for all time-related operations instead of raw `datetime` or `time`:

| Class | Purpose |
|-------|---------|
| `Timestamp` | Point in time (epoch-based). Use for created_at, timestamps in logs, etc. |
| `Duration` | Time interval. Use for timeouts, intervals, configuration values. |
| `Deadline` | Monotonic deadline for timeout checks. Use in polling loops. |
| `Timer` | Elapsed time measurement. Use for performance tracking. |
| `ExponentialBackoff` | Retry/polling with backoff. Use `wait_until()` for condition polling. |

Example:
```python
from iris.time_utils import Timestamp, Duration, Deadline

created_at = Timestamp.now()
timeout = Duration.from_seconds(30.0)
deadline = Deadline.from_now(timeout)
deadline.wait_for(condition)

while not deadline.expired():
    if condition():
        break
    time.sleep(0.1)
```

### Deployment Topology

The controller is a plain GCE VM with no zone affinity to workers — it can run
in any zone and serve workers across all regions.

**When changing the controller zone**, update in `examples/marin.yaml`:
- `controller.gcp.zone` — the GCE zone
- `controller.image` and `defaults.bootstrap.docker_image` — use a registry in
  the same region (see below). `cluster start` auto-builds and pushes images to
  the region parsed from the image tag, so no manual push is needed.

**Docker registries** are configured in `platform/bootstrap.py` (both worker and
controller bootstrap scripts). If you add a new region's Artifact Registry, add
it to both `gcloud auth configure-docker` lines. List existing repos with:
`gcloud artifacts repositories list --project=hai-gcp-models`

**Bundle storage** (`controller.bundle_prefix`) is a GCS URI with no zone
affinity — globally accessible.

**Zone validation**: `cluster/config.py` validates that every scale group zone
appears in `platform.gcp.zones`. Multi-zone scale groups are auto-expanded by
`_expand_multi_zone_groups()`.

### Architecture Layers

Iris follows a clean layering architecture:

**Controller layer** (`cluster/controller/`): Task scheduling, autoscaling, and demand routing
- Depends on Platform layer for VM abstractions (Platform, SliceHandle, VmHandle)
- Owns autoscaling logic and scaling group state

**Platform layer** (`cluster/platform/`): Platform abstractions for managing VMs
- Provides VM lifecycle management (GCP, manual, local, CoreWeave)
- Does NOT depend on controller layer

**Cluster layer** (`cluster/`): High-level orchestration
- `connect_cluster()` and `stop_all()` free functions for cluster lifecycle
- `stop_all()` terminates controller + all slices in parallel via ThreadPoolExecutor
  with a 60s hard timeout. Timed-out operations are logged at WARNING and abandoned.
- Configuration and platform abstractions

Key files:

```
src/iris/
├── cli/                         # CLI package (cluster, build, run, debug commands)
│   ├── main.py                  # Top-level iris group
│   ├── cluster.py               # Cluster lifecycle, controller, VM ops, dashboard
│   ├── build.py                 # Image build commands
│   ├── run.py                   # Command passthrough job submission
│   └── rpc.py                   # Dynamic RPC CLI
├── cluster/
│   ├── config.py                # General Iris configuration (load_config, IrisConfig)
│   ├── manager.py               # connect_cluster() + stop_all(dry_run) free functions
│   ├── controller/
│   │   ├── controller.py        # Controller with integrated autoscaler
│   │   ├── main.py              # Controller daemon CLI (serve command)
│   │   ├── autoscaler.py        # Core autoscaling logic and demand routing
│   │   ├── scaling_group.py     # Per-group state tracking and lifecycle
│   │   ├── config.py            # Autoscaler factory functions
│   │   ├── local.py             # LocalController for in-process testing
│   │   └── vm_lifecycle.py      # Controller lifecycle (start/stop/reload via Platform)
│   └── platform/
│       ├── base.py              # Platform protocol and SliceHandle/VmHandle
│       ├── gcp.py               # GCP TPU platform
│       ├── manual.py            # Pre-existing host platform
│       ├── local.py             # Local development platform
│       ├── coreweave.py         # CoreWeave CKS platform (shared NodePools)
│       ├── bootstrap.py         # Worker bootstrap script generation
│       ├── ssh.py               # SSH connection management
│       ├── factory.py           # Platform factory from config
│       └── debug.py             # Platform debugging utilities
```

See [README.md](README.md) for CLI usage and configuration examples.

### Dashboard Frontend

The controller and worker dashboards are client-side SPAs using Preact + HTM.

**Directory structure:**
```
src/iris/cluster/static/
├── controller/          # Controller dashboard
│   ├── app.js           # Main app (tabs, state, data fetching)
│   ├── jobs-tab.js      # Jobs table with pagination/sorting/tree view
│   ├── job-detail.js    # Job detail page with task list
│   ├── workers-tab.js   # Workers table
│   └── vms-tab.js       # VM management table
├── shared/              # Shared utilities
│   ├── rpc.js           # Connect RPC client wrapper
│   ├── utils.js         # Formatting (dates, durations)
│   └── styles.css       # Consolidated CSS
├── vendor/              # Third-party ES modules
│   ├── preact.mjs       # UI framework
│   └── htm.mjs          # HTML template literals
└── worker/              # Worker dashboard components
```

**Key patterns:**
- All data fetched via Connect RPC (e.g., `ListJobs`, `GetJobStatus`)
- No REST endpoints - RPC only
- State management with Preact hooks (`useState`, `useEffect`)
- HTML templates via `htm.bind(h)` tagged template literals
- Jobs displayed as a hierarchical tree based on name structure

**When modifying the dashboard:**
1. Run dashboard tests: `uv run pytest lib/iris/tests/e2e/test_dashboard.py -x -o "addopts="`
2. Ensure any new UI features have corresponding RPC endpoints
3. Follow existing component patterns (functional components, hooks)

## Testing

All Iris E2E tests live in `tests/e2e/`. Every test is marked `e2e`.
Tests use three core fixtures:

- `cluster`: Booted local cluster with `IrisClient` and RPC access
- `page`: Playwright page pointed at the dashboard (request only when needed)
- `screenshot`: Capture labeled screenshots to `IRIS_SCREENSHOT_DIR`

Chaos injection is auto-reset between tests. Call `enable_chaos()` directly.
Docker tests use a separate `docker_cluster` fixture and are marked `docker`.

Run all E2E tests:
    uv run pytest lib/iris/tests/e2e/ -m e2e -o "addopts="

Run E2E tests without Docker (fast):
    uv run pytest lib/iris/tests/e2e/ -m "e2e and not docker" -o "addopts="

Run Docker-only tests:
    uv run pytest lib/iris/tests/e2e/ -m docker -o "addopts="

Run dashboard tests with saved screenshots:
    IRIS_SCREENSHOT_DIR=/tmp/shots uv run pytest lib/iris/tests/e2e/test_dashboard.py -o "addopts="

When modifying the dashboard:
    uv run pytest lib/iris/tests/e2e/test_dashboard.py -x -o "addopts="

## Debugging Container Failures

**Exit code 137** = 128 + 9 = SIGKILL, typically OOM. Check:
- `ContainerStatus.oom_killed` field (from `docker inspect .State.OOMKilled`)
- Job's `resources.memory_bytes` vs what was requested
- Resource flow: `JobRequest.resources` → Iris protobuf → `ContainerConfig` → `docker --memory`

**Resource propagation path:**
```
fray.v2.ResourceConfig → iris.cluster.types.ResourceSpec.to_proto()
  → cluster_pb2.ResourceSpecProto → docker.py _docker_create() --memory/--cpus
```

**Key files for container debugging:**
- `cluster/runtime/docker.py`: Docker CLI wrapper, resource limits at lines 396-403
- `cluster/runtime/types.py`: ContainerStatus with oom_killed field
- `cluster/worker/task_attempt.py`: _format_exit_error() interprets signals
