# Cluster Boot Cleanup Plan

## Overview

This plan addresses issues found in the cluster boot changes from commits 9b5774c75 and 915d014ef.

---

## Step 1: Merge Probe Scripts

**Goal**: Eliminate code duplication by merging `probe-controller.py` and `validate-cluster.py` into a single unified script.

**Files**:
- `scripts/probe-controller.py` (delete after merge)
- `scripts/validate-cluster.py` (delete after merge)
- `scripts/cluster-tools.py` (new, merged)

**Changes**:

1. Create `scripts/cluster-tools.py` with all commands from both scripts:
   - From probe-controller.py: discover, ssh-status, tunnel, health, autoscaler-status, list-workers, logs, bootstrap-logs, list-jobs
   - From validate-cluster.py: validate (renamed from main command)

2. Remove hard-coded defaults - require zone/project:
   ```python
   @click.option("--zone", required=True, help="GCP zone (e.g., europe-west4-b)")
   @click.option("--project", required=True, help="GCP project ID")
   ```

3. Single shared implementation of:
   - `discover_controller_vm()`
   - `wait_for_port()`
   - `controller_tunnel()`

4. Delete the old scripts.

**Verify**: `uv run python scripts/cluster-tools.py --zone europe-west4-b --project hai-gcp-models discover`

---

## Step 2: Remove /readiness Endpoint

**Goal**: Remove the `/readiness` endpoint from dashboard entirely. The `cluster-tools.py` script can call RPC APIs directly for debugging. Dashboard is for users, not debugging.

**Files**:
- `src/iris/cluster/controller/dashboard.py`
- `tests/cluster/controller/test_dashboard.py`

**Changes**:

1. Remove `_readiness` method from `ControllerDashboard`
2. Remove `/readiness` route from `_create_app()`
3. Remove `self._start_time` tracking (only used by readiness)
4. Delete related tests:
   - `test_readiness_endpoint_returns_ready`
   - `test_readiness_endpoint_includes_counts`

**Code to remove** (`dashboard.py:1456-1517`):
```python
# DELETE this entire method
def _readiness(self, _request: Request) -> JSONResponse:
    """Detailed readiness check for debugging startup issues..."""
    ...
```

**Route to remove** (`dashboard.py:1424`):
```python
# DELETE this line
Route("/readiness", self._readiness),
```

**Verify**: `uv run pytest tests/cluster/controller/test_dashboard.py -v`

---

## Step 3: Switch Config to Protobuf

**Goal**: Replace manual `to_dict()` with protobuf for automatic serialization. The current `to_dict()` is unmaintainable - every field change requires updating both the dataclass and the serialization method. Using protobuf provides a single source of truth since `ScaleGroupConfig` is already a proto.

**Files**:
- `src/iris/rpc/vm.proto` (add new messages)
- `src/iris/cluster/vm/config.py` (simplify to use proto)
- `tests/cluster/vm/test_controller.py` (update tests)
- `examples/*.yaml` (flatten structure to match proto)

**Changes**:

1. Add config messages to `vm.proto`:
```proto
// GCP-managed controller configuration
message GcpControllerConfig {
  string image = 1;
  string machine_type = 2;      // Default: "n2-standard-4"
  int32 boot_disk_size_gb = 3;  // Default: 50
  int32 port = 4;               // Default: 10000
}

// Manually-managed controller configuration (SSH bootstrap)
message ManualControllerConfig {
  string host = 1;
  string image = 2;
  int32 port = 3;               // Default: 10000
}

// Controller configuration using oneof for type-safe dispatch
message ControllerVmConfig {
  oneof controller {
    GcpControllerConfig gcp = 1;
    ManualControllerConfig manual = 2;
  }
}

// Full cluster configuration - single source of truth
message IrisClusterConfig {
  // Provider settings
  string provider_type = 1;     // "tpu" or "manual"
  string project_id = 2;
  string region = 3;
  string zone = 4;

  // Auth settings
  string ssh_user = 10;         // Default: "root"
  string ssh_private_key = 11;

  // Docker/worker settings
  string docker_image = 20;
  int32 worker_port = 21;       // Default: 10001

  // Controller settings
  string controller_address = 30;
  ControllerVmConfig controller_vm = 31;

  // For manual provider
  repeated string manual_hosts = 40;

  // Scale groups (name -> config)
  map<string, ScaleGroupConfig> scale_groups = 50;

  // Timeouts (inline, not nested)
  int32 boot_timeout_seconds = 60;        // Default: 300
  int32 init_timeout_seconds = 61;        // Default: 600
  int32 ssh_connect_timeout_seconds = 62; // Default: 30
  int32 ssh_poll_interval_seconds = 63;   // Default: 5

  // GCP label prefix
  string label_prefix = 70;     // Default: "iris"
}
```

2. Simplify `load_config()` to use protobuf parsing:
```python
from google.protobuf.json_format import ParseDict, MessageToDict

def load_config(config_path: Path | str) -> vm_pb2.IrisClusterConfig:
    """Load cluster config from YAML file."""
    config_path = Path(config_path)
    with open(config_path) as f:
        data = yaml.safe_load(f)

    # Expand env vars in controller_address
    if "controller_address" in data:
        data["controller_address"] = os.path.expandvars(data["controller_address"])

    return ParseDict(data, vm_pb2.IrisClusterConfig())


def config_to_dict(config: vm_pb2.IrisClusterConfig) -> dict:
    """Convert config to dict for YAML serialization."""
    return MessageToDict(config, preserving_proto_field_name=True)
```

3. Remove the dataclasses and `to_dict()` method entirely from config.py.

4. Update controller.py serialization:
```python
from iris.cluster.vm.config import config_to_dict

def _serialize_config(self) -> str:
    return yaml.dump(config_to_dict(self.config), default_flow_style=False)
```

5. Update helper methods to work with proto:
```python
def to_bootstrap_config(config: vm_pb2.IrisClusterConfig) -> vm_pb2.BootstrapConfig:
    return vm_pb2.BootstrapConfig(
        controller_address=config.controller_address,
        docker_image=config.docker_image,
        worker_port=config.worker_port,
    )

def to_timeout_config(config: vm_pb2.IrisClusterConfig) -> vm_pb2.TimeoutConfig:
    return vm_pb2.TimeoutConfig(
        boot_timeout_seconds=config.boot_timeout_seconds or 300,
        init_timeout_seconds=config.init_timeout_seconds or 600,
        ssh_connect_timeout_seconds=config.ssh_connect_timeout_seconds or 30,
        ssh_poll_interval_seconds=config.ssh_poll_interval_seconds or 5,
    )
```

6. Flatten YAML config files to match proto field names (see YAML changes below).

7. Update tests to use new API.

**YAML Config Changes**:

The YAML format is flattened to match proto field names directly. This eliminates the nested `provider:`, `docker:`, `controller:`, and `timeouts:` sections.

**Before** (`examples/eu-west4.yaml`):
```yaml
provider:
  type: tpu
  project_id: hai-gcp-models
  region: europe-west4
  zone: europe-west4-b

docker:
  image: europe-west4-docker.pkg.dev/hai-gcp-models/marin/iris-worker:latest
  worker_port: 10001

controller:
  vm:
    enabled: true
    image: europe-west4-docker.pkg.dev/hai-gcp-models/marin/iris-controller:latest
    machine_type: n2-standard-4
    port: 10000

scale_groups:
  tpu_v5e_16:
    accelerator_type: v5litepod-16
    ...
```

**After** (`examples/eu-west4.yaml`):
```yaml
# Iris cluster configuration for europe-west4
# Flat structure matching IrisClusterConfig proto

provider_type: tpu
project_id: hai-gcp-models
region: europe-west4
zone: europe-west4-b

docker_image: europe-west4-docker.pkg.dev/hai-gcp-models/marin/iris-worker:latest
worker_port: 10001

controller_vm:
  gcp:
    image: europe-west4-docker.pkg.dev/hai-gcp-models/marin/iris-controller:latest
    machine_type: n2-standard-4
    port: 10000

scale_groups:
  tpu_v5e_16:
    accelerator_type: v5litepod-16
    runtime_version: v2-alpha-tpuv5-lite
    min_slices: 0
    max_slices: 2
    zones: [europe-west4-b]
    preemptible: true
    priority: 100
```

**After** (`examples/demo.yaml`):
```yaml
# Iris demo configuration
# Uses env var for controller address (external controller)

provider_type: tpu
project_id: hai-gcp-models
region: europe-west4
zone: europe-west4-b

docker_image: europe-west4-docker.pkg.dev/hai-gcp-models/marin/iris-worker:latest
worker_port: 10001

controller_address: "${IRIS_CONTROLLER_ADDRESS}"

scale_groups:
  tpu_v5e_16:
    accelerator_type: v5litepod-16
    runtime_version: v2-alpha-tpuv5-lite
    min_slices: 1
    max_slices: 2
    zones: [europe-west4-b]
    preemptible: true
```

**Verify**:
```bash
uv run scripts/gen_proto.sh  # Regenerate proto files
uv run pytest tests/cluster/vm/test_controller.py -v
uv run python -c "from iris.cluster.vm.config import load_config; c = load_config('examples/eu-west4.yaml'); print(c)"
```

---

## Step 4: Extract Duplicated Health Check Method

**Goal**: Extract `_wait_healthy_via_ssh` to module-level function.

**Files**:
- `src/iris/cluster/vm/controller.py`

**Changes**:

Extract to module-level:
```python
def wait_healthy_via_ssh(
    conn: SshConnection,
    port: int,
    timeout: float = HEALTH_CHECK_TIMEOUT_SECONDS,
) -> bool:
    """Poll health endpoint via SSH until healthy or timeout."""
    logger.info("Starting SSH-based health check (port=%d, timeout=%ds)", port, int(timeout))
    start_time = time.time()
    attempt = 0

    backoff = ExponentialBackoff(
        initial=HEALTH_CHECK_BACKOFF_INITIAL,
        maximum=HEALTH_CHECK_BACKOFF_MAX,
    )

    def check_with_logging() -> bool:
        nonlocal attempt
        attempt += 1
        result = check_health(conn, port)
        elapsed = time.time() - start_time
        if result:
            logger.info("SSH health check succeeded after %d attempts (%.1fs)", attempt, elapsed)
        else:
            logger.info("SSH health check attempt %d failed (%.1fs)", attempt, elapsed)
        return result

    success = backoff.wait_until(check_with_logging, timeout=timeout)
    if not success:
        logger.warning("SSH health check failed after %d attempts", attempt)
    return success
```

Then use in both classes:
```python
class GcpController:
    def start(self) -> str:
        ...
        if wait_healthy_via_ssh(conn, port):
            ...

class ManualController:
    def start(self) -> str:
        ...
        if not wait_healthy_via_ssh(conn, port):
            raise RuntimeError(...)
```

**Verify**: `uv run pytest tests/cluster/vm/test_controller.py -v`

---

## Step 5: Add GcpController Tests with Parameterized Pattern

**Goal**: Add test coverage for GcpController using fixtures/parameterization.

**Files**:
- `tests/cluster/vm/test_controller.py`

**Changes**:

Add parameterized tests that cover both controller types:
```python
@pytest.fixture
def mock_subprocess():
    """Mock subprocess.run for GCP commands."""
    with patch("iris.cluster.vm.controller.subprocess.run") as mock:
        yield mock


@pytest.fixture
def mock_ssh_health():
    """Mock SSH-based health checking."""
    with patch("iris.cluster.vm.controller.wait_healthy_via_ssh") as mock:
        mock.return_value = True
        yield mock


@pytest.mark.parametrize("controller_cls,config_fixture", [
    (GcpController, "gcp_config"),
    (ManualController, "ssh_bootstrap_config"),
])
def test_controller_start_checks_health(
    controller_cls,
    config_fixture,
    request,
    mock_subprocess,
    mock_ssh_health,
):
    """Both controller types check health after bootstrap."""
    config = request.getfixturevalue(config_fixture)

    # Setup mocks for successful VM creation
    mock_subprocess.return_value = subprocess.CompletedProcess(
        args=[], returncode=0, stdout='[{"networkInterfaces": [...]}]', stderr=""
    )

    controller = controller_cls(config)
    controller.start()

    mock_ssh_health.assert_called_once()
```

**Verify**: `uv run pytest tests/cluster/vm/test_controller.py -v`

---

## Implementation Order

| Step | Description | Effort | Dependencies |
|------|-------------|--------|--------------|
| 1 | Merge probe scripts | Medium | None |
| 2 | Remove /readiness endpoint | Low | None |
| 3 | Switch config to protobuf | Medium | None |
| 4 | Extract wait_healthy_via_ssh | Low | None |
| 5 | Add GcpController tests | Medium | Step 4 |

Steps 1-4 can be done in parallel. Step 5 depends on Step 4.

---

## Not Fixing

The following issues from the original review are intentionally not addressed:

- **Magic numbers in validate script**: These are debugging tools, exact values don't matter
- **Bootstrap timeout inconsistency**: Works in practice, not worth the complexity
