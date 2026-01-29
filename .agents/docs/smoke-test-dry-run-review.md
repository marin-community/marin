# Design Review: Smoke Test Dry‑Run Mode

## Overall Assessment
- **Feasibility:** **Feasible with moderate effort**, largely because `examples/demo_cluster.py` already proves the in‑process worker pattern can work. However, the proposed plan underestimates the wiring needed to make this a first‑class platform across config, controller, and smoke test flows.
- **Main risks:** API surface mismatches (proto/config changes), missing dependencies for in‑process workers (bundle/image providers, environment provider), and incomplete ControllerProtocol implementation.

---

## 1) Feasibility of Proposed Approach
**Feasible, but requires more integration than the doc suggests.**

**Evidence from codebase**
- `examples/demo_cluster.py` already runs an in‑process controller + autoscaler + worker stack, including TPU‑like attributes and coscheduling. This validates the core concept.
- `VmGroupProtocol` and `VmManagerProtocol` in `src/iris/cluster/vm/vm_platform.py` are simple enough to support a local implementation.

**Required work beyond the doc**
- The demo code depends on **local bundle/image/container providers** from `iris.cluster.client.local_client` and a **LocalEnvironmentProvider**; the proposal omits these dependencies.
- The local VM manager in the demo requires a **fake bundle path** to satisfy worker startup. The proposal doesn’t account for how the smoke test would provide bundles.

**Conclusion:** Core idea is solid. Implementation is not plug‑and‑play; a careful re‑plumbing of worker dependencies is needed.

---

## 2) Accuracy of Code Snippets vs Existing Code

### ✅ Accurate / Close
- `LocalVmGroup` and `_StubManagedVm` behavior matches `examples/demo_cluster.py`:
  - READY state without bootstrap delay
  - `ManagedVm` stub with minimal interface
- `LocalVmManager.create_vm_group()` logic roughly aligns:
  - Determines worker count based on TPU topology
  - Sets `tpu-name`, `tpu-worker-id`, and `tpu-topology` attributes
  - Creates workers immediately

### ❌ Inaccurate / Missing Details
- **`LocalVmManager` signature in doc omits `fake_bundle`**
  `examples/demo_cluster.py` requires `fake_bundle: Path` and constructs `_LocalBundleProvider` from it.
  The proposed `__init__` signature in the doc is incomplete.

- **Local worker wiring is absent from proposal**
  Demo uses:
  - `_LocalBundleProvider`
  - `_LocalImageProvider`
  - `_LocalContainerRuntime`
  - `LocalEnvironmentProvider`
  The doc shows none of these, but they are **required** for workers to boot.

- **LocalController snippet doesn’t match existing ControllerProtocol**
  `ControllerProtocol` in `src/iris/cluster/vm/controller.py` includes:
  - `restart`, `reload`, `discover`, `fetch_startup_logs`
  - `status()` expects `ControllerStatus` with proper `healthy` field
  The proposed `LocalController` only implements `start/stop/status`, so it would not conform.

- **Config integration snippet doesn’t match config.py behavior**
  `_get_provider_info` and `_create_manager_from_config` only accept `tpu` and `manual`. There is **no `local` provider** today. The proposal assumes it can be added without addressing proto changes and parsing behavior.

---

## 3) Missing Considerations / Gaps

### A) Worker bootstrap dependencies
- The local workers require bundle/image/runtime providers; currently only available via `iris.cluster.client.local_client`.
- The smoke test likely submits real jobs; a **fake bundle** (like demo) may not be enough unless smoke test uses purely inline entrypoints.

### B) Proto/config changes are larger than noted
- The plan includes `controller_vm.local` but doesn’t mention **`scale_groups.provider.local`** in the proto.
- `config.proto` changes mean regeneration and possible downstream changes in config loading.

### C) ControllerProtocol completeness
- LocalController must implement **all protocol methods** (`restart`, `reload`, `discover`, `fetch_startup_logs`).
- `status()` should reflect real health (likely via `ControllerServiceClientSync`), not a constant `healthy=True`.

### D) Port allocation and shared namespace
- Demo uses a **shared PortAllocator** across groups to avoid collisions in a single process. This should be explicitly required in the local platform.

### E) Cleanup and lifecycle
- `LocalVmGroup.terminate()` stops workers and unregisters VMs. The local controller should ensure autoscaler stop + vm cleanup is deterministic, especially in smoke test cleanup paths.

### F) Resource constraints
- Demo sets `cpu=1000`/`memory_gb=1000` for LocalEnvironmentProvider. That might mask scheduling/resource issues or exceed local machine limits.

---

## 4) Implementation Plan Completeness

### Missing or Under‑scoped Items
- **Proto updates for scale group provider** (`provider.local`) are not called out explicitly; only controller_vm.local is mentioned.
- **Controller factory updates** are needed:
  - `create_controller()` currently only supports `gcp` and `manual`.
- **Local controller config values** (port, bundle storage, cache paths) need a source of truth in config.
- **Smoke test bundle handling** is not defined:
  - If skipping image build, how does a worker acquire a runnable bundle?
- **Tests:** The plan says “unit tests for local_platform.py,” but does not specify integration coverage for smoke test flow. The most valuable test is a local smoke‑test run.

### Suggested missing steps
- Add a `LocalController` implementation that wraps **exactly** the demo’s in‑process controller/autoscaler pattern.
- Define how local mode supplies bundles:
  - Option 1: Bundle from workspace (preferred if smoke test uses real entrypoints)
  - Option 2: Fake bundle (if smoke test only uses inline callables)
- Add an explicit `local` provider in `config.proto` scale group provider oneof.
- Extend `config.py` to accept `local` provider without `project_id/hosts`.

---

## Final Verdict

### ✅ Feasibility
Yes, the approach is feasible and consistent with existing demo code, but it needs more integration detail.

### ⚠️ Accuracy
Several doc snippets do **not** reflect the actual code (notably the LocalVmManager signature and LocalController protocol compliance).

### ⚠️ Missing Gaps
Local worker dependency wiring, full ControllerProtocol coverage, bundle handling, and proto/provider changes are the primary gaps.

### ⚠️ Plan Completeness
The plan is **incomplete**: it underestimates proto changes and omits critical local‑mode wiring for workers and smoke‑test bundling.

---

## Concrete Recommendations
1. **Mirror demo_cluster wiring exactly** when extracting local platform (bundle/image/runtime/environment providers and port allocator).
2. **Update ControllerProtocol compliance** for LocalController (implement all methods, real health checks).
3. **Add `provider.local` to scale group proto**, not just controller_vm.local.
4. **Define bundle strategy for local mode** before implementing smoke‑test changes.
5. **Add an integration test** that runs the smoke test in local mode end‑to‑end.
