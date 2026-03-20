# K8s Service Cleanup: Unify K8sServiceImpl with CLOUD Mode

## Problem

`K8sServiceImpl` only handles `DRY_RUN`/`LOCAL` modes. `Kubectl` (the real kubectl wrapper) is a standalone dataclass that doesn't implement `K8sService`. Two factory sites directly import `Kubectl`:

- `lib/iris/src/iris/cluster/config.py:1131` — `make_provider()` constructs `Kubectl(...)` for `KubernetesProvider`
- `lib/iris/src/iris/cluster/platform/coreweave.py:148` — `CoreweavePlatform.__init__` fallback constructs `Kubectl(...)`

This means:
1. Tests can't swap in `K8sServiceImpl` at the factory level without bypassing the factory
2. `Kubectl` and `K8sServiceImpl` are parallel implementations of the same interface but disconnected
3. The GCP side is clean — `GcpServiceImpl` handles all 3 modes in one class — but K8s isn't

## Proposed Solution

Follow the `GcpServiceImpl` pattern exactly: make `K8sServiceImpl` handle all three `ServiceMode` values. In `CLOUD` mode, it delegates to an internal `Kubectl` instance. `Kubectl` becomes a private implementation detail, never imported outside `k8s_service_impl.py`.

### Why this approach

- **Consistency**: mirrors `GcpServiceImpl` which already works this way
- **Single entry point**: consumers construct `K8sServiceImpl(mode=ServiceMode.CLOUD, ...)` instead of choosing between `Kubectl` and `K8sServiceImpl`
- **Testability**: `DRY_RUN` mode validation runs the same paths as `CLOUD` minus subprocess calls
- **No new abstractions**: just extending an existing class with one more mode branch

### Core idea

```python
# k8s_service_impl.py — add CLOUD mode delegation

class K8sServiceImpl:
    def __init__(
        self,
        namespace: str = "iris",
        mode: ServiceMode = ServiceMode.DRY_RUN,
        available_node_pools: list[str] | None = None,
        # CLOUD mode params
        kubeconfig_path: str | None = None,
        timeout: float = 60.0,
    ):
        self._namespace = namespace
        self._mode = mode
        # ... existing DRY_RUN/LOCAL fields ...

        # CLOUD mode: create internal Kubectl
        self._kubectl: Kubectl | None = None
        if mode == ServiceMode.CLOUD:
            from iris.cluster.k8s.kubectl import Kubectl
            self._kubectl = Kubectl(
                namespace=namespace,
                kubeconfig_path=kubeconfig_path,
                timeout=timeout,
            )

    def apply_json(self, manifest: dict) -> None:
        self._check_failure("apply_json")
        if self._mode == ServiceMode.CLOUD:
            assert self._kubectl is not None
            self._kubectl.apply_json(manifest)
            return
        # existing DRY_RUN/LOCAL logic unchanged
        self._validate_manifest(manifest)
        ...
```

Each protocol method gains a `CLOUD` early-return that delegates to `self._kubectl`. The DRY_RUN/LOCAL paths stay unchanged.

## Implementation Plan

### Item 1: Add CLOUD mode to K8sServiceImpl

**Files**: `lib/iris/src/iris/cluster/k8s/k8s_service_impl.py`

**Changes**:
1. Add `kubeconfig_path: str | None = None` and `timeout: float = 60.0` params to `__init__`
2. Add `self._kubectl: Kubectl | None` field, initialized in CLOUD mode via lazy import
3. For each protocol method (`apply_json`, `get_json`, `list_json`, `delete`, `logs`, `stream_logs`, `exec`, `set_image`, `rollout_restart`, `rollout_status`, `get_events`, `top_pod`, `read_file`, `rm_files`, `port_forward`), add a CLOUD-mode early-return that delegates to `self._kubectl`
4. Keep the `_check_failure` call before the CLOUD delegation — this lets failure injection work in CLOUD mode too (useful for integration tests)
5. Add a `@property mode` to expose the mode (matches `GcpServiceImpl.mode`)

**Pattern per method** (same for all 16 methods):
```python
def apply_json(self, manifest: dict) -> None:
    self._check_failure("apply_json")
    if self._mode == ServiceMode.CLOUD:
        assert self._kubectl is not None
        self._kubectl.apply_json(manifest)
        return
    # ... existing DRY_RUN/LOCAL logic ...
```

**Tests**: Add a test in `lib/iris/tests/cluster/k8s/test_k8s_service_impl.py` that verifies:
- `K8sServiceImpl(mode=ServiceMode.CLOUD, ...)` constructs without error
- The `mode` property returns `ServiceMode.CLOUD`
- Failure injection works before CLOUD delegation (mock the internal `_kubectl` to verify delegation)

### Item 2: Update factory sites to use K8sServiceImpl

**Files**:
- `lib/iris/src/iris/cluster/config.py:1122-1147`
- `lib/iris/src/iris/cluster/platform/coreweave.py:133-154`

**config.py changes** (`make_provider` function, line 1131):
```python
# Before:
from iris.cluster.k8s.kubectl import Kubectl
...
kubectl=Kubectl(namespace=namespace, kubeconfig_path=kp.kubeconfig or None),

# After:
from iris.cluster.k8s.k8s_service_impl import K8sServiceImpl
from iris.cluster.service_mode import ServiceMode
...
kubectl=K8sServiceImpl(
    namespace=namespace,
    mode=ServiceMode.CLOUD,
    kubeconfig_path=kp.kubeconfig or None,
),
```

**coreweave.py changes** (`CoreweavePlatform.__init__`, line 148):
```python
# Before:
from iris.cluster.k8s.kubectl import Kubectl
self._kubectl = Kubectl(
    namespace=self._namespace,
    kubeconfig_path=...,
    timeout=_KUBECTL_TIMEOUT,
)

# After:
from iris.cluster.k8s.k8s_service_impl import K8sServiceImpl
from iris.cluster.service_mode import ServiceMode
self._kubectl = K8sServiceImpl(
    namespace=self._namespace,
    mode=ServiceMode.CLOUD,
    kubeconfig_path=...,
    timeout=_KUBECTL_TIMEOUT,
)
```

**Tests**: Existing tests for `KubernetesProvider` and `CoreweavePlatform` already use `K8sServiceImpl(mode=ServiceMode.DRY_RUN)` (see `conftest.py:17` and `test_coreweave_platform.py:40`). No test changes needed for this item — the tests already go through the protocol, not through the factory.

### Item 3: Make Kubectl private

**Files**:
- `lib/iris/src/iris/cluster/k8s/kubectl.py` — no rename needed, just ensure no external imports
- `lib/iris/src/iris/cluster/k8s/__init__.py` — already does NOT export Kubectl (good, no change)

**Verification**: After Items 1-2, `grep -r "from iris.cluster.k8s.kubectl import" lib/iris/` should return only `k8s_service_impl.py`. The two factory sites (config.py, coreweave.py) will have been updated in Item 2.

**Tests**: No new tests. Just verify the grep shows no remaining external imports.

---

### Item 4: Add `run` and `_popen` and `prefix` methods to K8sServiceImpl (CLOUD passthrough)

`KubernetesProvider` at `provider.py` does NOT use `kubectl.run()` or `kubectl._popen()` directly — it only uses the `K8sService` protocol methods. Verified by checking `provider.py` imports: it imports `K8sService`, not `Kubectl`.

`CoreweavePlatform` stores `self._kubectl` typed as `K8sService` (line 146). It also only calls protocol methods.

So `run()`, `_popen()`, and `prefix` on `Kubectl` are used internally by `Kubectl` itself and don't need to be exposed on `K8sServiceImpl` or `K8sService`. No action needed.

## Dependency Graph

```
Item 1 (add CLOUD mode)  →  Item 2 (update factories)  →  Item 3 (verify Kubectl is private)
```

Items 1→2→3 are sequential. Item 1 is the bulk of the work. Item 2 is a 4-line change per factory. Item 3 is verification.

These could be done as a single sub-issue since total change is ~60 lines of new code in k8s_service_impl.py plus ~10 lines changed in config.py + coreweave.py.

## Risks and Open Questions

1. **Kubectl `run()` method**: Some code path outside the K8sService protocol might call `kubectl.run()` directly. Checked: `provider.py` and `coreweave.py` only use protocol methods. If discovered during implementation, those calls should be migrated to use protocol methods or added to the K8sService protocol.

2. **Kubectl `prefix` property**: Used only internally by Kubectl for building commands. Not needed on K8sServiceImpl.

3. **`port_forward` in CLOUD mode**: `Kubectl.port_forward` uses `_popen` for subprocess management with reconnection. `K8sServiceImpl.port_forward` in DRY_RUN mode returns a fake URL. In CLOUD mode, we delegate to `Kubectl.port_forward` which handles the real subprocess lifecycle. This is the most complex method to delegate but it's straightforward — just `yield from self._kubectl.port_forward(...)`.

4. **Test isolation**: The existing `test_k8s_service_impl.py` tests DRY_RUN behavior extensively. CLOUD mode tests should verify delegation without actually calling kubectl (mock the internal `_kubectl` attribute or just test construction + mode property).
