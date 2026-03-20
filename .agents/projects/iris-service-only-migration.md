# Iris: Service-Only Migration Plan

## Problem

Four production/test files import `Kubectl` directly instead of going through the
`K8sService`/`SubprocessK8s` protocols, and `k8s/__init__.py` re-exports it. This
couples consumers to the subprocess implementation, making testing brittle (300-line
`FakeKubectl` class) and preventing clean DRY_RUN substitution.

Violations:

| # | File | Line | Issue |
|---|------|------|-------|
| 1 | `lib/iris/src/iris/cluster/platform/coreweave.py` | 149 | `Kubectl(...)` constructed in `__init__` |
| 2 | `lib/iris/src/iris/cluster/config.py` | 1131-1138 | `make_provider()` imports and constructs `Kubectl` |
| 3 | `lib/iris/tests/cluster/platform/test_coreweave_platform.py` | 53-342 | 300-line `FakeKubectl` patching `subprocess.run` |
| 4 | `lib/iris/tests/kubernetes/test_kubectl.py` | 13 | Imports `Kubectl`, `_parse_k8s_cpu`, `_parse_k8s_memory` |
| 5 | `lib/iris/src/iris/cluster/k8s/__init__.py` | 10-12 | Re-exports `Kubectl` in `__all__` |

## Proposed Solution

### Core Approach: Constructor Injection + popen() on K8sServiceImpl

**CoreweavePlatform** currently types `_kubectl` as `SubprocessK8s` (which extends
`K8sService` with `popen()`). The platform body only uses `K8sService` methods —
`popen()` is only used by `_coreweave_tunnel()`, a module-level function.

The key insight: `CoreweavePlatform.__init__` should accept a `SubprocessK8s` parameter
instead of creating `Kubectl` internally. For tests, inject `K8sServiceImpl`. For
the tunnel, `K8sServiceImpl` needs a `popen()` stub in DRY_RUN mode.

**Why not split popen() out of the platform?** The tunnel is called via
`platform.tunnel()` which delegates to `_coreweave_tunnel(kubectl=self._kubectl, ...)`.
Changing this call chain would require a separate tunnel factory, adding indirection
for no real gain. Simpler to add `popen()` to `K8sServiceImpl`.

### popen() in DRY_RUN mode

`K8sServiceImpl` gets a `popen()` method that returns a mock-like `subprocess.Popen`.
In DRY_RUN mode, it returns a `Popen` whose `poll()` returns `None` (alive),
`terminate()`/`kill()` are no-ops, and `communicate()` returns `("", "")`. This is
sufficient for tunnel tests since the tunnel just needs a process that stays alive
and then gets terminated in the finally block.

```python
# k8s_service_impl.py — new method on K8sServiceImpl
def popen(self, args: list[str], *, namespaced: bool = False, **kwargs: Any) -> subprocess.Popen:
    """DRY_RUN popen: returns a FakePopen that stays 'alive' until terminated."""
    self._check_failure("popen")
    return FakePopen()


class FakePopen:
    """Minimal subprocess.Popen stand-in for DRY_RUN mode."""
    pid = 0
    returncode: int | None = None
    stdin = None
    stdout = None
    stderr = None

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def communicate(self, input: bytes | str | None = None, timeout: float | None = None) -> tuple[str, str]:
        self.returncode = 0
        return ("", "")
```

This makes `K8sServiceImpl` satisfy `SubprocessK8s`, so it can be injected into
`CoreweavePlatform` directly.

### Alternative Considered: Separate Tunnel Protocol

Could define a `TunnelFactory` protocol and inject that separately. Rejected because:
- Adds a new abstraction for a single call site
- `SubprocessK8s` already exists and models exactly this need
- The tunnel function is tightly coupled to kubectl anyway

## Implementation Plan

### Task 1: Add `popen()` + `FakePopen` to `K8sServiceImpl` (no dependencies)

**Files:**
- `lib/iris/src/iris/cluster/k8s/k8s_service_impl.py:261` — add `FakePopen` class before `K8sServiceImpl`, add `popen()` method at end of class

**Behavior:** `popen()` checks for injected failures, then returns a `FakePopen()`.
`FakePopen` mimics a live process: `poll()` returns `None` until `terminate()`/`kill()` is called.

**Tests:** Add tests in `lib/iris/tests/kubernetes/test_k8s_service.py`:
- `test_popen_returns_fake_process` — verify poll/terminate/wait lifecycle
- `test_popen_failure_injection` — verify injected error is raised

### Task 2: Inject `SubprocessK8s` into `CoreweavePlatform` (depends on Task 1)

**Files:**
- `lib/iris/src/iris/cluster/platform/coreweave.py:138-154`

**Change:** Add `kubectl: SubprocessK8s | None = None` parameter to `__init__`.
If provided, use it; otherwise create `Kubectl(...)` as today (keeps backward
compat for production callers that don't pass it).

```python
def __init__(
    self,
    config: config_pb2.CoreweavePlatformConfig,
    label_prefix: str,
    poll_interval: float = _DEFAULT_POLL_INTERVAL,
    kubectl: SubprocessK8s | None = None,
):
    ...
    self._kubectl: SubprocessK8s = kubectl or Kubectl(
        namespace=self._namespace,
        kubeconfig_path=None if os.environ.get("KUBECONFIG") else (config.kubeconfig_path or None),
        timeout=_KUBECTL_TIMEOUT,
    )
```

Remove `from iris.cluster.k8s.kubectl import Kubectl` at the top of `coreweave.py:49`.
Move it to a lazy import inside `__init__` guarded by `if kubectl is None`.

**Tests:** Existing tests continue to work (they'll switch to injected service in Task 4).

### Task 3: Migrate `config.py:make_provider()` (no dependencies)

**Files:**
- `lib/iris/src/iris/cluster/config.py:1130-1138`

**Change:** `KubernetesProvider.kubectl` is already typed as `K8sService`. The
`Kubectl` import on line 1131 is only used to construct the instance passed to
`KubernetesProvider`. Keep the `Kubectl(...)` construction here — this is a
factory function, so concrete instantiation is appropriate. But move the import
to the top of the file, removing the deferred import, since `config.py` already
has many imports and `Kubectl` is the production implementation.

Actually — the goal is "no production code imports `Kubectl` directly except
`k8s_service_impl.py`." The issue description says `k8s_service_impl.py` is
the CLOUD impl, but `Kubectl` is the real subprocess impl, not `K8sServiceImpl`.

Re-reading the goal: the intent is that `Kubectl` is only imported by the one
module that wraps it. Currently that wrapper is `k8s_service_impl.py`... but
`K8sServiceImpl` is the DRY_RUN impl, not a wrapper around `Kubectl`.

The cleanest approach: `config.py` and `coreweave.py` are **factory sites** —
they're the only places that should instantiate `Kubectl`. That's acceptable.
The goal should be: no code *other than factory/wiring code* imports `Kubectl`.

For `make_provider()`, keep the `Kubectl(...)` construction — it's a factory.
Just keep the import local (deferred) since it's inside a branch. No change
needed here beyond cleanup.

Wait — re-reading the issue: "No production code imports `Kubectl` directly
except `k8s_service_impl.py`." This is aspirational. The actual `k8s_service_impl.py`
doesn't import `Kubectl` at all — it's the in-memory fake. The goal seems to
mean "only one place creates `Kubectl`."

**Revised approach for config.py:** Leave as-is. The deferred import of `Kubectl`
in `make_provider()` is a factory — it's the wiring point. If we want to
centralize `Kubectl` construction, we could add a factory function to
`k8s/__init__.py`, but that's over-engineering for two call sites. Instead:

- Keep the import in `make_provider()` as-is (it's a factory)
- Remove `Kubectl` from `k8s/__init__.py` re-exports (Task 5)
- Both `config.py:make_provider` and `coreweave.py:__init__` import from
  `iris.cluster.k8s.kubectl` directly — these are the two sanctioned factory sites

**Tests:** No new tests needed — `make_provider` is tested via integration tests.

### Task 4: Replace `FakeKubectl` in tests with `K8sServiceImpl` (depends on Tasks 1, 2)

**Files:**
- `lib/iris/tests/cluster/platform/test_coreweave_platform.py:35-342`

**Changes:**

1. Delete `FakeKubectl` class entirely (lines 53-335)
2. Delete the `fake_kubectl` fixture (lines 338-342)
3. Replace `_make_platform()` to inject `K8sServiceImpl`:

```python
def _make_platform(
    region: str = "LGA1",
    namespace: str = "iris",
    kubeconfig_path: str = "",
    label_prefix: str = "iris",
    k8s: K8sServiceImpl | None = None,
) -> tuple[CoreweavePlatform, K8sServiceImpl]:
    k8s = k8s or K8sServiceImpl(namespace=namespace)
    config = config_pb2.CoreweavePlatformConfig(
        region=region,
        namespace=namespace,
        kubeconfig_path=kubeconfig_path,
    )
    platform = CoreweavePlatform(config=config, label_prefix=label_prefix, poll_interval=0.05, kubectl=k8s)
    return platform, k8s
```

4. Update each test to use `K8sServiceImpl` methods instead of `FakeKubectl`:

| FakeKubectl pattern | K8sServiceImpl equivalent |
|---|---|
| `fake_kubectl._deployments["x"] = {...}` | `k8s.apply_json({"kind": "Deployment", "metadata": {"name": "x"}, ...})` |
| `fake_kubectl.make_deployment_available("x")` | Modify stored resource: `k8s._resources[("deployment", "x")]["status"]["availableReplicas"] = 1` — or add a helper method `set_deployment_available()` to K8sServiceImpl |
| `fake_kubectl._pods["x"] = {...}` | `k8s.apply_json({"kind": "Pod", "metadata": {"name": "x"}, ...})` |
| `fake_kubectl._pod_logs["x"] = "..."` | `k8s.set_logs("x", "...")` |
| `fake_kubectl._nodepools["x"] = {...}` | `k8s.apply_json({"kind": "NodePool", "metadata": {"name": "x"}, ...})` |
| `fake_kubectl._events.append(...)` | `k8s.add_event(...)` |
| `fake_kubectl._secrets["x"]` in assertions | `k8s.get_json("secret", "x") is not None` |

Key behavioral difference: `FakeKubectl` intercepts `subprocess.run` at the
kubectl module level, so it catches ALL kubectl calls from CoreweavePlatform.
With injected `K8sServiceImpl`, no patching is needed — the platform calls
methods directly on the injected service.

**Deployment readiness pattern:** Several tests use threading to auto-mark
deployments as available. With `K8sServiceImpl`, the pattern becomes:

```python
def auto_ready_deployment():
    _wait_for_condition(lambda: k8s.get_json("deployment", "iris-controller") is not None, timeout=10)
    manifest = k8s._resources[("deployment", "iris-controller")]
    manifest.setdefault("status", {})["availableReplicas"] = 1

t = threading.Thread(target=auto_ready_deployment, daemon=True)
t.start()
```

Consider adding a `set_deployment_available(name)` helper to `K8sServiceImpl`
to avoid reaching into `_resources` directly.

**Tests:** All existing test assertions should translate directly. The test count
stays the same.

### Task 5: Extract parsers and clean up `test_kubectl.py` (no dependencies)

**Files:**
- `lib/iris/src/iris/cluster/k8s/kubectl.py:402-426` — `_parse_k8s_cpu`, `_parse_k8s_memory`
- `lib/iris/tests/kubernetes/test_kubectl.py`

**Change:** The parser functions `_parse_k8s_cpu` and `_parse_k8s_memory` are
pure functions with no dependency on `Kubectl`. They're private (`_`-prefixed)
but tested directly. Options:

a) Make them public (remove `_` prefix), keep in `kubectl.py`, import from there
   in tests — but the goal says no test imports from `kubectl.py`.
b) Move them to a `k8s_parsers.py` module.
c) Leave them in `kubectl.py` and accept that `test_kubectl.py` tests the parsers
   from the module that defines them.

**Recommended: Option (a) with a twist.** Make `parse_k8s_cpu` and `parse_k8s_memory`
public, move them to `k8s_types.py` (or a new `k8s_parsers.py`), and re-export
from `k8s/__init__.py`. Then `test_kubectl.py` imports from `k8s_types` instead.

However, `k8s_service_impl.py` already has `_parse_k8s_quantity()` which is a
superset — it handles cpu millicore `m` suffix, binary suffixes, and SI suffixes.
The two parser functions in `kubectl.py` could be replaced by `_parse_k8s_quantity`
plus a scaling factor for CPU. But that's scope creep.

**Simplest path:** Move `_parse_k8s_cpu` and `_parse_k8s_memory` to `k8s_types.py`
as public functions. Update `kubectl.py` to import them from `k8s_types`. Update
`test_kubectl.py` to import from `k8s_types` and rename the test file to
`test_k8s_parsers.py`.

For the `top_pod` tests: these test `Kubectl.top_pod()` by mocking `kubectl.run()`.
Since `top_pod()` is part of the `K8sService` protocol and `K8sServiceImpl` already
implements it, these tests should either:
- Stay as-is (testing the concrete `Kubectl` implementation is valid for a unit test)
- Move to testing via `K8sServiceImpl.top_pod()` (already covered in service tests)

**Recommended:** Keep `top_pod` tests in `test_kubectl.py` since they test the real
subprocess-backed parser behavior. The goal "no test imports Kubectl" is aspirational
for integration tests — unit-testing the concrete impl is fine. But if strict
compliance is required, delete the `top_pod` tests from `test_kubectl.py` (they're
already covered by `K8sServiceImpl.set_top_pod()` + protocol-level tests).

### Task 6: Clean up `k8s/__init__.py` and delete audit artifact (depends on Tasks 1-5)

**Files:**
- `lib/iris/src/iris/cluster/k8s/__init__.py:10,12` — remove `Kubectl` import and `__all__` entry
- Repo root: delete `iris-service-violations-audit.md`

**Change:**
```python
# Before
from iris.cluster.k8s.kubectl import Kubectl
__all__ = ["K8sService", "K8sServiceImpl", "Kubectl", "KubectlError", "ServiceMode", "SubprocessK8s"]

# After
__all__ = ["K8sService", "K8sServiceImpl", "KubectlError", "ServiceMode", "SubprocessK8s"]
```

Remove line 10 (`from iris.cluster.k8s.kubectl import Kubectl`).

**Tests:** Run full test suite to verify nothing else imports `Kubectl` via the
package re-export.

## Task Dependency Graph

```
Task 1 (popen on K8sServiceImpl) ──┐
                                    ├──> Task 4 (rewrite CW tests)
Task 2 (inject into CW platform) ──┘
                                         │
Task 3 (config.py cleanup) ─────────────>│
                                         │
Task 5 (extract parsers) ──────────────>│
                                         v
                                    Task 6 (cleanup __init__, delete audit)
```

Tasks 1, 3, and 5 are independent and can run in parallel.
Task 2 depends on Task 1.
Task 4 depends on Tasks 1 and 2.
Task 6 depends on all prior tasks.

## Risks and Open Questions

1. **Thread-safety of `K8sServiceImpl._resources` in deployment-readiness tests.**
   Several tests spawn threads that mutate stored resources while the main thread
   polls `get_json`. `K8sServiceImpl` uses a plain dict with no locking. This works
   today because Python's GIL protects dict mutations, but it's fragile. Consider
   whether to add a lock or keep relying on the GIL.

2. **`FakePopen` fidelity.** The tunnel function reads `proc.stderr` on process
   exit (`stderr = proc.stderr.read()`). `FakePopen.stderr` is `None`, which will
   cause `AttributeError` if the tunnel's retry path is exercised. Need to set
   `stderr` to an `io.StringIO("")` or similar. Check all `_coreweave_tunnel`
   code paths that access popen attributes.

3. **`config.py` still imports `Kubectl`.** The stated goal says "no production
   code imports Kubectl directly except k8s_service_impl.py." But `config.py` is
   a factory — it *must* know the concrete type to instantiate it. Options:
   - Accept factory sites as exceptions to the rule
   - Add a `create_kubectl()` factory to `k8s/__init__.py` that hides the import
   - Use a registry pattern (over-engineered)

   Recommend: accept `config.py` and `coreweave.py` as sanctioned factory sites.
   The rule becomes: "only factory/wiring code imports Kubectl."

4. **`test_kubectl.py` top_pod tests.** Strict interpretation of "no test imports
   Kubectl" means deleting these. They do test real parsing logic that's worth
   keeping. If deleted, ensure `K8sServiceImpl`-based tests cover the same edge
   cases (empty output, multi-container, metrics unavailable).
