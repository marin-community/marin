# Rollout plan: GPU gang smoke through a real Iris controller (Kueue gang admission)

Status: DONE — validated on kind and on real CoreWeave H100s. Branch `iris-k8s-gang-admission`.

## Outcome (validated)

- **kind:** PASS — 3-pod CPU gang, atomic Kueue admission, bundle + `uv sync`,
  registry `initialize_jax`, transformer trains.
- **CoreWeave H100:** PASS — 3×8 = 24-GPU coscheduled gang admitted atomically in
  one IB leafgroup; NCCL collective ran over InfiniBand (`ibp0-7`); transformer
  trained to completion; `JOB_STATE_SUCCEEDED`. Teardown left 0 nodepools / 0 H100
  nodes.

Bring-up fixes found while validating on **fresh** CoreWeave capacity (none of
which kind could surface):
1. **Controller Deployment must tolerate `qos.coreweave.cloud/interruptable`** —
   freshly provisioned nodes carry it; task pods already tolerated it, the
   controller did not, so it sat `Pending` forever. Fixed in `controller.py`.
   (Product fix, not just smoke config — affects any clean controller bring-up on
   fresh CoreWeave capacity.)
2. **Dedicated `cpu-erapids` pool** so the controller never consumes an H100 (the
   GPU-taint toleration would otherwise let it land on one).
3. **Use canonical `start_controller`** (not a hand-rolled deploy) so the S3
   credentials Secret + PDB are created.
4. **Ephemeral `file://` controller state** — S3 state crash-looped the controller
   on a 403 (R2 token lacks `ListBucket` on the empty state prefix → 403 on the
   missing key, not 404). The state backend isn't under test.
5. **`NCCL_SOCKET_IFNAME` excludes IB/IPoIB/virtual NICs** (not the per-node exact
   pin from the plan): NCCL's OOB bootstrap otherwise grabbed `ibs0`'s link-local
   IPv6 and timed out. Exclusion is robust because the routable-ethernet NIC name
   varies per physical node.

Known follow-up (not blocking): on the **failure** path, teardown deletes the
NodePools out from under the gang pods, so Kueue never strips their
`kueue.x-k8s.io/managed` finalizer and the namespace wedges in `Terminating`
(cleared manually by patching finalizers). Teardown should delete the task pods /
namespace before the NodePools. The success path finalizes cleanly.

## Goal

Replace the controller-less `gpu_gang_smoke` (which drives `K8sTaskProvider`
directly and hand-rolls coordinator discovery + jax install) with a smoke that
submits a **normal Iris GPU job through a real controller**, so it exercises the
native paths this branch is actually about:

- the controller's Kueue **gang admission** for a coscheduled multi-replica GPU job,
- `EnvironmentSpec(extras=["gpu"])` → `uv sync --extra gpu` installing
  `jax[cuda13]==0.10.0` (the bundle's own mechanism — no hand-rolled `pip install`),
- `iris.runtime.jax_init.initialize_jax()` for multi-host JAX, using the
  controller's **endpoint registry** for coordinator discovery (no kubectl-exec hack),
- a real **basic model**: a small causal transformer trained data-parallel across
  the whole gang, whose gradient all-reduce is the cross-host NCCL/IB exercise.

Validate on **kind first**, then CoreWeave H100.

## Why the current standalone hacks go away

The in-flight edits to `gpu_gang_smoke.py` (driver writes the coordinator file via
`kubectl exec`; entrypoint `pip install`s jax) were the only way to do this WITHOUT
a controller. With a controller + workspace bundle:
- `initialize_jax()` is importable (iris is in the bundle) and its multi-task path
  works (`iris_ctx().registry/resolver` is the controller).
- `extras=["gpu"]` installs jax via `uv sync` (entrypoint.py:54,79) — no manual pip.

So those edits will be reverted as part of Phase 1.

## Confirmed mechanics (file refs, this branch)

- `KueueConfig` is in the proto: `kubernetes_provider.kueue` (config.proto:484) with
  `cluster_queue` (enables Kueue), `priority_classes`, `topologies`.
- `K8sControllerProvider.start_controller` (controller.py:270) runs `ensure_rbac` +
  `ensure_nodepools` + `ensure_kueue_queues` (creates the namespaced LocalQueue).
  It does **not** install the Kueue operator or the cluster-scoped
  ClusterQueue/ResourceFlavor/Topology — that's `scripts/install_kueue.py`
  (kind) / already admin-provisioned `iris-cq` (CoreWeave).
- `iris job run --gpu H100:8 --replicas 3 --extra gpu` auto-sets
  `coscheduling=group_by="pool"` for multi-replica GPU jobs (job.py:466) → the gang
  path. The pod manifest stamps the Kueue pod-group + LocalQueue when
  `coscheduling.group_by` is set and `kueue.cluster_queue` is configured (tasks.py:569).
- `gpu` extra = `jax[cuda13]==0.10.0` in `lib/marin` and `lib/levanter` pyproject.
- Task pods fetch the workspace bundle from `controller_address` via an init
  container (bundle_fetch.py); `controller_address` must be reachable from pods.

## The workload module (new): `lib/iris/scripts/gang_jax_smoke_workload.py`

Shipped in the bundle, so it can `import jax` (from the gpu extra) and
`from iris.runtime.jax_init import initialize_jax`. Pure JAX/NumPy, no marin deps.

```python
from iris.runtime.jax_init import initialize_jax
initialize_jax()                      # multi-host via controller registry
# mesh over jax.devices(); all-reduce sanity (sum over devices == device_count);
# tiny causal transformer (D=256,H=4,L=2,seq=128) trained data-parallel for ~20
# steps on one fixed synthetic batch; gradient all-reduce => cross-host NCCL/IB.
# multihost_utils.sync_global_devices("done"); print loss curve from process 0.
```

(The model code already drafted in the current `gpu_gang_smoke.py` edit is reused
verbatim; only the coordinator-file wait is dropped in favour of `initialize_jax()`.)

## Phase 1 — kind (full software path, no real GPUs, no cost)

kind has no GPUs, so the gang is **CPU-shaped** (`--extra` installs CPU jaxlib;
device omitted). This validates everything except real CUDA/IB: controller boot,
Kueue gang admission of a coscheduled pod-group, bundle delivery, `uv sync --extra`,
registry-based `initialize_jax`, and the model running multi-host on CPU.

Because the CLI only auto-enables coscheduling for GPU jobs, the smoke submits via
the **Python `IrisClient`** with explicit `coscheduling=CoschedulingConfig("pool")`
(cleaner than adding a CLI flag; the smoke is already a Python harness).

Controller bring-up — two options (recommend **A** for fidelity):

- **A. In-cluster controller (mirrors CoreWeave).** Build `iris-controller` +
  `iris-task` from this branch, `kind load docker-image` them, call
  `K8sControllerProvider.start_controller(config)` directly (bypasses the CLI's
  ghcr push), `kubectl port-forward svc/...-controller-svc 10000` for the client.
- **B. Local controller process (lighter, lower fidelity).** Run
  `iris cluster controller serve --config kind.yaml` on the host against the kind
  kubeconfig; set `controller_address` to the docker-bridge gateway so kind pods can
  fetch bundles from the host. No image builds. Wrinkle: host networking reachability.

Steps (Option A):
1. `kind create cluster` (3 workers), label workers into one synthetic CW leafgroup
   (reuse the existing `_label_nodes` logic).
2. `scripts/install_kueue.py --variant upstream --with-queues` (operator + iris-cq +
   ResourceFlavor + Topology) — already wired in the current smoke.
3. Build `iris-controller`/`iris-task` (this branch) → `kind load`.
4. New config `config/kind-controller-gpu-smoke.yaml`: `kubernetes_provider` with
   `kueue.cluster_queue: iris-cq`, `controller.coreweave` block (port/service/
   scale_group), a CPU scale group, `controller_address` = in-cluster svc DNS.
5. `start_controller(config)` → RBAC + LocalQueue + controller Deployment/Service.
6. Port-forward; `IrisClient.remote(tunnel, workspace=marin_root).submit(...)` with
   `replicas=3`, `coscheduling=pool`, `EnvironmentSpec(extras=[...])`,
   entrypoint `python -m scripts.gang_jax_smoke_workload`.
7. `job.wait()`; assert SUCCEEDED; teardown (delete job, controller, kind cluster).

Success: controller logs show one Kueue Workload admitted atomically for the
pod-group; all 3 tasks SUCCEEDED; loss decreases; all-reduce sum == device count.

## Phase 2 — CoreWeave H100 (real gang, real cost)

Mirrors Phase 1 with the real provider. Requires: cluster start + paid H100s
(your authorization).

1. Build `iris-controller`/`iris-task` from this branch, push to ghcr under a
   branch tag (needs `docker login ghcr.io`).
2. New config `config/coreweave-controller-gpu-smoke.yaml` in a **dedicated
   namespace** (e.g. `iris-gang-smoke`, NOT `iris-ci`): `kubernetes_provider` with
   `host_network: true`, `kueue.cluster_queue: iris-cq`,
   `kueue.topologies{pool->leafgroup}`, `task_env.NCCL_SOCKET_IFNAME==enp157s0np0`;
   an `h100-8x` scale group (instance `gd-8xh100ib-i128`).
3. `iris-cq` (ClusterQueue/ResourceFlavor/Topology) is already admin-provisioned and
   non-binding (prior session). The dedicated namespace's LocalQueue is created by
   `start_controller`.
4. `start_controller` → controller Deployment + NodePool. Manually set
   `targetNodes=3` (Kueue TAS can't scale from zero — see
   `project_kueue_tas_scale_from_zero`).
5. Submit: `iris --controller-url=<tunnel> job run --gpu H100:8 --replicas 3
   --extra gpu --cpu 32 --memory 512g --disk 100g -e <NCCL/HF/...> --
   python -m scripts.gang_jax_smoke_workload` (auto-coscheduled by pool).
6. Watch admission + the model train across 24 H100s; assert SUCCEEDED.
7. Teardown: delete job, controller, **NodePool** (stop H100 billing), verify 0
   H100 nodes and `iris-ci` untouched.

## Files

- ADD `lib/iris/scripts/gang_jax_smoke_workload.py` — the JAX model (bundle module).
- ADD `lib/iris/config/kind-controller-gpu-smoke.yaml`.
- ADD `lib/iris/config/coreweave-controller-gpu-smoke.yaml`.
- REWRITE `lib/iris/scripts/gpu_gang_smoke.py` — orchestrate controller bring-up +
  `IrisClient.submit` + teardown (drop the standalone `K8sTaskProvider.sync` path and
  the kubectl-exec coordinator injection). Keep kind node-labeling + nodepool/teardown
  helpers.
- REVERT the in-flight hand-rolled coordinator/install edits to `gpu_gang_smoke.py`.
- Maybe small product change: `start_controller` for kind may need an
  `imagePullPolicy: IfNotPresent` / local-tag path so kind-loaded images are used
  (verify in Phase 1; only touch product code if required).

## Open decisions for you

1. **Kind controller: Option A (in-cluster, faithful, builds images) or B (local
   process, lighter)?** Recommend A.
2. **Phase 2 cost/auth:** confirm OK to (a) build+push branch images to ghcr,
   (b) `start_controller` a dedicated-namespace controller, (c) provision 3×8 H100
   for the run. Dedicated namespace keeps `iris-ci` untouched.
3. **Workload size:** the basic transformer above, or also wire a couple of real
   knobs (steps/seq via env)? Default: the basic transformer, steps via env.

## Risks / notes

- kind image builds (controller has rust/node) take a few minutes — one-time.
- Bundle delivery on kind depends on `controller_address` reachability (drives the
  A/B choice).
- Bandwidth: `uv sync --extra gpu` pulls the CUDA13 wheel stack (~2-3GB) per H100
  pod from PyPI on Phase 2 — acceptable for an occasional smoke; flagged per repo
  cost guidance.
- Cost bounded by teardown in `finally` + a wall-clock timeout, as before.
