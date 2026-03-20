# Iris Service-Layer Violations Audit

## Section 1: Direct Kubectl/Gcloud Imports

### Production code (`lib/iris/src/`)

| File | Line | Import/Usage | Severity | Recommendation |
|------|------|-------------|----------|----------------|
| `src/iris/cluster/platform/coreweave.py` | 48-49 | `from iris.cluster.k8s import SubprocessK8s` + `from iris.cluster.k8s.kubectl import Kubectl` | **HIGH** | `coreweave.py` directly instantiates `Kubectl(...)` at line 149. It should accept a `SubprocessK8s` via constructor injection, not import and instantiate `Kubectl` itself. The `SubprocessK8s` import (the Protocol) is fine — only the `Kubectl` concrete class import is the violation. |
| `src/iris/cluster/config.py` | 1131 | `from iris.cluster.k8s.kubectl import Kubectl` (lazy import) | **HIGH** | `make_provider()` instantiates `Kubectl(namespace=..., kubeconfig_path=...)` at line 1138. Should use a factory or accept an injected `K8sService`/`SubprocessK8s`. |
| `src/iris/cluster/k8s/__init__.py` | 10 | `from iris.cluster.k8s.kubectl import Kubectl` | **MEDIUM** | Re-exports `Kubectl` in `__all__`. This makes `Kubectl` importable as `from iris.cluster.k8s import Kubectl`. The `__init__.py` should only re-export `K8sService`, `SubprocessK8s`, `K8sServiceImpl`, `ServiceMode`, and `KubectlError` — not `Kubectl`. |
| `src/iris/cluster/k8s/provider.py` | 25 | `from iris.cluster.k8s.k8s_service import K8sService` | **OK** | Imports the Protocol, not the concrete class. Clean. |
| `src/iris/cluster/controller/main.py` | 68 | `from iris.cluster.k8s.provider import KubernetesProvider` | **OK** | Imports the provider (which uses K8sService Protocol internally). |

### Test code (`lib/iris/tests/`)

| File | Line | Import/Usage | Severity | Recommendation |
|------|------|-------------|----------|----------------|
| `tests/kubernetes/test_kubectl.py` | 13 | `from iris.cluster.k8s.kubectl import Kubectl, _parse_k8s_cpu, _parse_k8s_memory` | **HIGH** | Directly imports concrete `Kubectl` and private parser functions. Tests instantiate `Kubectl(namespace="iris")` at lines 53, 64, 75, 82 and mock `kubectl.run`. Should either (a) test parsers as pure functions without `Kubectl` or (b) rewrite to use `K8sServiceImpl(DRY_RUN)`. |
| `tests/cluster/k8s/test_k8s_service_impl.py` | 10-12 | `from iris.cluster.k8s.k8s_service import K8sService`, etc. | **OK** | Tests the service impl itself — these imports are correct. |

### No `subprocess.run(["kubectl"` or `subprocess.run(["gcloud"` outside service impls
**CLEAN** — no hits found.

---

## Section 2: Mock Usage Audit

### K8s/GCP-related mocks (should use DRY_RUN instead)

| File | Lines | What's Mocked | Recommendation |
|------|-------|--------------|----------------|
| `tests/kubernetes/test_kubectl.py` | 56, 65, 76, 84 | `patch.object(kubectl, "run", ...)` — mocks the `subprocess.run` wrapper on a concrete `Kubectl` | **REPLACE**: These test pure parsing logic (`_parse_k8s_cpu`, `_parse_k8s_memory`) and `top_pod`. The parsers should be tested as standalone functions. `top_pod` tests should use `K8sServiceImpl(DRY_RUN)` if `top_pod` is added to the service, or just test the parser output. |
| `tests/cluster/platform/test_coreweave_platform.py` | 53-342 | `FakeKubectl` class + `patch("iris.cluster.k8s.kubectl.subprocess.run", fake)` | **BORDERLINE**: This is a stateful fake (not a mock), but it patches `subprocess.run` at the kubectl module level. Should be replaced with `K8sServiceImpl(DRY_RUN)` injected into `CoreweavePlatform`, but this requires `CoreweavePlatform` to accept a `SubprocessK8s` via DI instead of creating its own `Kubectl`. |
| `tests/cluster/platform/test_platform.py` | 421, 644, 680, 714, 751, 786, 828, 834 | `unittest.mock.patch("iris.cluster.platform.gcp.threading.Thread")`, `mock.patch.object(gcp_service, "vm_get_serial_port_output", ...)`, `mock.patch("iris.cluster.platform.gcp.urllib.request.urlopen", ...)` | **MIXED**: Threading patches are OK (test infra). Serial-port and urlopen patches mock GCP metadata — these are I/O boundary mocks that are appropriate. |
| `tests/cluster/controller/test_vm_lifecycle.py` | 200-296 | `@patch(f"{LIFECYCLE_MODULE}.build_controller_bootstrap_script_from_config", ...)`, `@patch(f"{LIFECYCLE_MODULE}.wait_healthy", ...)` | **OK**: Mocks internal lifecycle functions, not K8s/GCP services. |

### Non-K8s/GCP mocks (acceptable)

| File | What's Mocked | Status |
|------|--------------|--------|
| `tests/test_jax_init.py` | `jax.distributed.initialize`, `iris.runtime.jax_init.get_job_info`, `atexit` | **OK** — mocking JAX runtime, not K8s |
| `tests/test_jax_init_integration.py` | `MagicMock()` for jax init | **OK** |
| `tests/test_distributed_lock.py` | `GcsLease._read_with_generation`, `_delete` | **OK** — mocking GCS lease internals (I/O boundary) |
| `tests/test_marin_fs.py` | `urllib.request.urlopen` for zone metadata | **OK** — I/O boundary |
| `tests/cli/test_local_cluster.py` | `patch` for CLI tests | **OK** |
| `tests/cli/test_build_config_regions.py` | `patch` | **OK** |
| `tests/rpc/test_interceptors.py` | `Mock` for gRPC interceptors | **OK** |
| `tests/rpc/test_auth.py` | `Mock`/`MagicMock` for credentials | **OK** — I/O boundary |
| `tests/cluster/platform/test_scaling_group.py` | `MagicMock` for Platform/SliceHandle/WorkerHandle | **OK** — mocks the Platform protocol itself, not K8s directly |
| `tests/cluster/platform/test_bootstrap.py` | `MagicMock` for platform, `patch` for subprocess | **OK** |
| `tests/cluster/controller/test_dashboard.py` | `Mock` | **OK** |
| `tests/cluster/controller/test_autoscaler.py` | `MagicMock` for Platform/SliceHandle | **OK** — mocks Platform protocol |
| `tests/cluster/controller/conftest.py` | `Mock` | **OK** |
| `tests/cluster/worker/test_worker.py` | `Mock` | **OK** |
| `tests/cluster/worker/conftest.py` | `Mock` | **OK** |
| `tests/cluster/worker/test_dashboard.py` | `Mock` | **OK** |
| `tests/cluster/controller/test_api_keys.py` | `Mock` | **OK** |
| `tests/cluster/runtime/test_docker_runtime.py` | `Mock` | **OK** |
| `tests/cluster/client/test_bundle.py` | `patch` | **OK** |
| `tests/cluster/test_env_propagation.py` | `patch` | **OK** |
| `tests/e2e/test_smoke.py` | `patch` | **OK** |

---

## Section 3: Test Suite Status

```
1043 passed, 1 failed (flaky) in 62.52s
```

The single failure (`tests/cluster/test_attempt_logs.py::TestAttemptLogs::test_multiple_attempts_preserve_logs`) is **flaky** — it passes on re-run. Root cause: `OSError: [Errno 12] Cannot allocate memory` during subprocess fork in a test that launches local worker processes. Not related to service-layer violations.

---

## Section 4: Production Code Using Kubectl Directly

Two production files bypass the service layer:

### 1. `src/iris/cluster/platform/coreweave.py:149`
```python
self._kubectl: SubprocessK8s = Kubectl(
    namespace=self._namespace,
    kubeconfig_path=self._kubeconfig_path,
)
```
**Fix**: Accept `SubprocessK8s` as a constructor parameter. The factory (`create_platform`) should create and inject the `Kubectl` instance.

### 2. `src/iris/cluster/config.py:1131-1138`
```python
from iris.cluster.k8s.kubectl import Kubectl
kubectl=Kubectl(namespace=namespace, kubeconfig_path=kp.kubeconfig or None),
```
**Fix**: `make_provider()` should accept a `K8sService` parameter, or the caller should construct the kubectl and pass it in.

### Clean production files (use Protocol correctly):
- `src/iris/cluster/k8s/provider.py` — imports `K8sService` Protocol ✓
- `src/iris/cluster/controller/controller.py` — imports `KubernetesProvider` ✓

---

## Section 5: FakeGcloud / FakeKubectl Remnants

### `FakeKubectl` in `tests/cluster/platform/test_coreweave_platform.py`

**53 references** across this file. The `FakeKubectl` class (line 53) is a full stateful fake that intercepts `subprocess.run` calls at the kubectl module level:

```python
@pytest.fixture
def fake_kubectl() -> FakeKubectl:
    fake = FakeKubectl()
    with patch("iris.cluster.k8s.kubectl.subprocess.run", fake):
        yield fake
```

This is the **largest remaining violation**. It:
1. Patches `subprocess.run` inside the `kubectl` module — reaching past the service layer
2. Creates a parallel in-memory K8s implementation (duplicating `K8sServiceImpl(DRY_RUN)`)
3. Exists because `CoreweavePlatform` creates its own `Kubectl` internally, making DI impossible

**Fix**: Make `CoreweavePlatform` accept a `SubprocessK8s` via DI, then inject `K8sServiceImpl(DRY_RUN)` in tests. Delete `FakeKubectl` entirely.

### `FakeGcloud`
**No hits** — fully cleaned up.

### `fake_kubectl` fixture
Used only in `test_coreweave_platform.py` (line 339). All other test files are clean.

---

## Summary of Required Changes (Priority Order)

1. **`CoreweavePlatform` DI** — Accept `SubprocessK8s` as a constructor param instead of creating `Kubectl` internally. This unblocks eliminating `FakeKubectl` from tests.

2. **Delete `FakeKubectl`** in `test_coreweave_platform.py` — Replace with `K8sServiceImpl(DRY_RUN)` injected into platform. Note: `K8sServiceImpl` currently doesn't implement `SubprocessK8s.popen()` — it may need a stub or a separate `LocalSubprocessK8s` wrapper.

3. **`config.py:make_provider()`** — Accept a `K8sService` parameter instead of importing and instantiating `Kubectl`.

4. **`test_kubectl.py`** — Move pure parser tests to test the standalone functions. Either delete `top_pod` tests (tested via service layer) or rewrite to use `K8sServiceImpl`.

5. **`k8s/__init__.py`** — Remove `Kubectl` from `__all__` and the re-export. Only the service impl module should import it.

6. **Flaky test** — `test_multiple_attempts_preserve_logs` has a memory allocation issue under parallel test execution. Separate issue, not service-related.
