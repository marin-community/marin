# Selecting a node pool or CPU architecture for an Iris task

Proposal. Not implemented. Written after CPU-only jobs on `cw-us-east-08a`
repeatedly landed on ARM nodes and broke an x86-only workload.

## The problem, concretely

`experiments/build_pdf_source/` runs docling's layout model as an INT8 OpenVINO
graph. OpenVINO's ARM CPU plugin does not take the INT8 path, so on ARM the graph
is not merely slower — on an arm64 laptop it produced 300 identical detections per
page, and the run reports no speedup at all.

`cw-us-east-08a` has two scale groups:

| group | instance | vCPU / RAM / disk | arch | warm nodes |
|---|---|---|---|---|
| `cpu-erapids` | `cd-gp-i64-erapids` | 64 / 512GB / 15.36TB | x86_64 (AMX-INT8) | 4 |
| `gb200` | `gb200-4x` | 144 / 960GB / 30.72TB | aarch64 (Grace) | 216 |

Three CPU-only jobs, three `aarch64` landings. This is the default outcome, not
bad luck: `gb200` has more of *every* resource than `cpu-erapids`, so no resource
request can exclude it, and there are 54x more warm gb200 nodes.

Guarded for now by `_require_x86()` in
`experiments/build_pdf_source/_tune_layout_quantization.py`, which fails the job
rather than reporting numbers from the wrong instruction set. That is a bandage:
the extract step has the same requirement and cannot express it either.

## Why it cannot be expressed today

- `ResourceSpecProto` (`lib/iris/src/iris/rpc/job.proto:431`) carries only
  `cpu_millicores`, `memory_bytes`, `disk_bytes`, `device`.
- `_convert_device` (`lib/fray/src/fray/iris_backend.py`) maps a fray `CpuConfig`
  to `device=None` — a CPU request carries no constraint whatsoever.
- `ResourceConfig` once had a `pool` field. It was removed in `20b7003fe`; the
  residue is the stale comment at `lib/fray/src/fray/iris_backend.py:88`
  (*"pool no longer maps"*).

## What already exists

Most of the machinery is present and wired end to end:

- `IrisClient.submit(constraints=[...])` accepts arbitrary `Constraint`s
  (`lib/iris/src/iris/client/client.py:647`).
- Job constraints are stamped onto `RunTaskRequest.constraints`
  (`.../controller/projections/run_templates.py:81`, proto field 10 at
  `job.proto:653`).
- `pool` is already a recognised placement key:
  `_CONSTRAINT_KEY_TO_NODE_LABEL = {"pool": "iris.pool", "region": "iris.region"}`
  (`lib/iris/src/iris/cluster/backends/k8s/tasks.py:164`), turned into a pod
  `nodeSelector` by `_constraints_to_node_selector` (`:304-324`) and applied at
  `:966-970`.

**No proto change is needed.** The wire path already carries this.

### Why it does not work

Nothing ever produces a node label named `iris.pool`. CoreWeave NodePool
`spec.nodeLabels` come solely from `nodepool_node_labels()`
(`.../cluster/platforms/k8s/nodepool_manifests.py:19-25`), which emits exactly
two: `iris-{label_prefix}-managed` and `iris-{label_prefix}-scale-group`. The
`worker.attributes.pool` key in the cluster YAML feeds only the
worker-registration path (`lib/iris/src/iris/cluster/local_cluster.py:103-105`),
which does not exist on the k8s backend.

So a `pool` constraint today yields `nodeSelector: {iris.pool: cpu-erapids}` and
the pod pends forever.

Two further facts explain the ARM landings:

- Iris does not place tasks on this backend at all. `K8sTaskProvider.schedule()`
  (`.../backends/k8s/tasks.py:2417`) is an explicit no-op — Kueue owns placement.
  The only selector a CPU pod receives is `iris-cw-use08a-managed: true`, which
  **both** pools carry.
- The Kueue flavor that could narrow it is selector-less:
  `build_cpu_resource_flavor(node_label=None)` → `spec: {}`
  (`.../platforms/k8s/kueue_manifests.py:311-328`), and Pulumi calls it with no
  argument (`infra/pulumi/src/iac/coreweave/kueue.py:131`).

## Proposal

**Two complementary changes. Neither touches a `.proto`.**

### 1. Make `pool` mean "scale group", and add `arch`

Replace the module-level `_CONSTRAINT_KEY_TO_NODE_LABEL` constant with a mapping
derived per cluster from `label_prefix`, so `pool` resolves to
`iris-{label_prefix}-scale-group` (`Labels.iris_scale_group`) — the label nodes
actually carry. `label_prefix` already reaches this layer:
`composer.py:92-93` computes `managed_label` from it and `PodConfig.managed_label`
(`tasks.py:488`) carries it.

Then re-add `ResourceConfig.pool: str | None` and the `key="pool"` EQ constraint
in fray's `convert_constraints` — literally re-applying the `20b7003fe` diff.

**Add `arch` → `kubernetes.io/arch`** in the same mapping. That is a one-line
addition to a dict, uses a standard always-present Kubernetes node label, and
expresses the actual requirement ("must run on x86") without naming a pool or
hard-coding cluster topology. For this workload it is the highest-value line in
the whole change.

Because the value maps straight onto `kubernetes.io/arch`, the accepted values
are the Kubernetes/GOARCH label values — `amd64`, `arm64` — not `x86`,
`x86_64`, or `aarch64` as this document's prose and tables otherwise write
them. A pod with `kubernetes.io/arch=x86_64` pends forever. Either document
`amd64`/`arm64` as the only accepted values or normalize the common aliases at
the fray API boundary before the constraint is emitted.

### 2. Pin the Kueue CPU flavor on 08a

Pass `build_cpu_resource_flavor` a node label at
`infra/pulumi/src/iac/coreweave/kueue.py:131`. The knob already exists
(`lib/iris/scripts/install_kueue.py:449-455`, `--cpu-flavor-node-label`).

The pin must be **per-cluster**, not a literal in the shared code path:
`infra/pulumi/__main__.py` instantiates one `KueueAddon` for every CoreWeave
stack, so hard-coding `(Labels("cw-use08a").iris_scale_group, "cpu-erapids")`
there would inject a selector for a label no other cluster's nodes carry and
strand every CPU-only pod on those stacks. Key the pin on the 08a stack (or a
per-cluster config field) and derive the label from that cluster's own
`label_prefix`.

This makes every CPU-only pod on 08a land on erapids with no client-side change —
a safe default. It is *not* a substitute for (1): it is cluster-global and
all-or-nothing, cannot express "this task on pool X", and would strand CPU work
when the four erapids nodes saturate.

### Rejected

- **A new `pool`/`arch`/`node_selector` field on `ResourceSpecProto`.**
  Unnecessary — `Constraint` is already the typed, indexed, federation-aware
  placement vocabulary and already reaches the pod builder. A parallel field
  duplicates it and adds proto regeneration plus a coordinated redeploy.
- **Generalising the coscheduling `group_by` mechanism.** Wrong semantics:
  `tasks.py:912-920` resolves it against `config.kueue_topologies`, selects a
  Kueue *topology level*, forces gang admission, and raises on unmapped values.

## Change surface — small

| File | Change |
|---|---|
| `lib/iris/src/iris/cluster/backends/k8s/tasks.py` | derive the constraint→label map from `label_prefix`; add `arch`; thread into `PodConfig` |
| `lib/iris/src/iris/cluster/composer.py` | pass the derived map / prefix through |
| `lib/fray/src/fray/types.py` | re-add `ResourceConfig.pool` |
| `lib/fray/src/fray/iris_backend.py` | re-add the EQ constraint; delete the stale `:88` comment |
| `infra/pulumi/src/iac/coreweave/kueue.py` | pin the CPU flavor node label |
| tests, `lib/iris/docs/coreweave.md` | coverage and docs |

~5 source files, ~40 lines. No proto edit, no protobuf regeneration, no new
scheduler logic, no wire-format change, and prior art in git history for the fray
half.

## Landmines

1. **`iris.region` is broken the same way** and worse. `region_constraint` emits
   `IN` for multiple regions (`constraints.py:414-418`), and
   `_constraints_to_node_selector` **raises `PodManifestError`** on any non-EQ op
   for a known key (`tasks.py:319-323`) — a hard dispatch failure, not a pend.
   Fix or delete `region` in the same pass.
2. **`any_region_constraint()` is `region EXISTS`**, which would hit that same
   raise; it survives only because `IrisClient.submit` strips it before the
   controller (`constraints.py:26-33`). Do not disturb that stripping.
3. **Only EQ is supported for `nodeSelector`.** `pool IN [a, b]` will raise.
   Either restrict the API to a single value or emit `nodeAffinity`
   `matchExpressions` instead.
4. **Controller redeploy required**, since `_build_pod_manifest` runs in the
   controller — and the compatibility story differs by key. `pool` is already
   a known key, so an old controller receiving it writes a nodeSelector for a
   nonexistent label and fails closed with a pend. `arch` is **unknown** to an
   old controller: `_constraints_to_node_selector` skips unknown keys and adds
   only the managed selector, so an `arch=amd64` job sent before the mapping
   is deployed schedules anywhere — a silent mis-schedule, not a pend. Clients
   must not emit `arch` until the controllers understand it (or old
   controllers must reject unknown placement constraints); sequence the fray
   release after the controller rollout.
5. **The Kueue change needs a Pulumi apply, not a redeploy** — different blast
   radius and approval path. `ResourceFlavor` is cluster-scoped, so it affects
   every CPU pod on 08a including the controller's own
   (`controller.coreweave.scale_group: cpu-erapids`).
6. **Check the autoscaler routing.** `ConstraintIndex.matching_entities([])`
   returns *all* groups (`constraints.py:1180`), and CPU demand today carries no
   constraints. Verify a new `pool` key does not become a routing key that
   starves demand on clusters where `pool` is unadvertised.
7. **Federation is safe.** `route_jobs_to_backends` drops constraint keys no
   backend advertises (`meta_scheduler.py:129-131`), so a `pool` constraint will
   not make a job UNSCHEDULABLE on a peer without that pool.

## Verify first

The claim that no node carries an `iris.pool` label is **static inference** —
`nodepool_manifests.py:19-25` is the sole producer of NodePool `nodeLabels`. It
was not checked against the live cluster (`kubectl` unavailable in the
investigating environment). One `kubectl get nodes --show-labels` on 08a should
confirm it, and confirm that `kubernetes.io/arch` is present, before any code is
written.
