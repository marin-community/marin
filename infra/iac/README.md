# marin-iac

Infrastructure-as-code for the static substrate of Marin clusters, per the design in
`[.agents/projects/iac/](../../.agents/projects/iac/)`. Pulumi (Python). **CoreWeave first.**

This is the **minimal cut**: it provisions the ceded RBAC and the reserved NodePools for a
CoreWeave cluster — enough to run `pulumi preview` end-to-end. Kueue/Traefik/object-storage
and the CKS cluster object itself are the next slices (see the `TODO(iac)` in
`coreweave/cluster.py` and out-of-scope in `spec.md`).
[`gaps.md`](../../.agents/projects/iac/gaps.md) inventories every remaining prerequisite of a
live deployment (the deferred components plus the GCP-arm bits — egress IPs, signing-key
secrets) with the exact landing site for each.

Stacks: one per cluster — `cw-us-east-02a`, `cw-us-east-08a` (GB200), each a
`Pulumi.<cluster>.yaml` pointer to the cluster name.

## What it reads

Everything comes from the per-cluster Iris config (`lib/iris/config/<cluster>.yaml`):

- NodePools derive from `scale_groups` (`iac.nodepools.derive_nodepools`).
- Namespace from `kubernetes_provider.namespace`; ClusterQueue name from
`kubernetes_provider.kueue.cluster_queue`.
- The residual cluster facts (CKS cluster name, ResourceFlavor, ACME issuers, buckets) from the
new `provisioning:` section in that same file. Iris carries `provisioning:` as an opaque dict;
`iac.config` owns the typed schema. (The package is `infra/iac/src/iac/`, imported as
`iac` — a `src/<pkg>` layout mirroring `lib/*/src/<pkg>`.)



## Prerequisites

- The [Pulumi CLI](https://www.pulumi.com/docs/install/) (`brew install pulumi/tap/pulumi`).
- The repo's uv venv, synced so `iris`, `rigging`, `pulumi`, `pulumi_kubernetes` all import.
`marin-iac` is a member of the root uv workspace, so its deps live in the shared repo
`.venv`. Sync the **whole** workspace — a bare `uv sync` from a member dir prunes the venv
to just that member, so use `--all-packages`:
  ```bash
  uv sync --all-packages    # run from the repo root or anywhere in the tree
  ```
- For the k8s dry-run: the CoreWeave kubeconfig at the path in the cluster's
`platform.coreweave.kubeconfig_path` (read access to the live cluster).



## Running a preview

The committed `Pulumi.<cluster>.yaml` is only the stack's **config**, not the stack itself —
you create the stack once with `pulumi stack init` before it can be selected.

```bash
cd infra/iac

# 1. one-time: pick a state backend. --local keeps state in a local file (good for
#    experimentation); production uses gs://marin-iac-state (see spec.md §2 backend bootstrap).
pulumi login --local

# 2. no secretsprovider is configured yet (deferred to the KMS bootstrap), so pulumi uses a
#    passphrase. For a throwaway local preview, an empty passphrase is fine:
export PULUMI_CONFIG_PASSPHRASE=""

# 3. one-time: create the stack. This reads the committed Pulumi.cw-us-east-02a.yaml.
pulumi stack init cw-us-east-02a
#    (on later runs, just: pulumi stack select cw-us-east-02a)

# 4. preview
pulumi preview
```

Against an empty state this plans a Namespace, the `iris-controller` ServiceAccount, the
`iris-controller-iris` ClusterRole + ClusterRoleBinding, and one `NodePool` per CoreWeave
scale group (`cw-use02a-cpu-genoa`, `cw-use02a-h100-8x`) — all as **creates**, because
Pulumi's state doesn't yet know these already exist on the live cluster. To see the *real*
diff, adopt them first (next section).

## Adoption check — safe, read-only recon

The running CW clusters already hold Iris-created RBAC + NodePools, so before a first apply we
need to see whether Pulumi would *adopt* them (no-op/update) or *recreate* them (a NodePool
replace = deprovisioning the reserved fleet). This is read-only — the only command that mutates
the cluster is `pulumi up`, which you must **not** run until the preview is clean (`spec.md §4`).

Setting `marin-iac:import=true` makes the program stamp `import_=<live id>` on every resource, so
`pulumi preview` shows the true adoption diff — correctly parented and on the real `cw-k8s`
provider. Preview writes nothing; it reports per resource whether adoption is a no-op, an update,
or a **replace**. Do this against a **local** backend so state stays in a throwaway local file.

```bash
pulumi login --local
export PULUMI_CONFIG_PASSPHRASE=""
export KUBECONFIG=~/.kube/coreweave-iris-gpu   # read access to the live cluster
export PULUMI_K8S_ENABLE_PATCH_FORCE=true      # required — see caveat (1) below
pulumi stack select cw-us-east-02a

pulumi config set marin-iac:import true
pulumi preview                                 # read the adoption diff
pulumi config set marin-iac:import false       # (or `pulumi config rm marin-iac:import`)
```

Reading it: **no-change / update-in-place** on a NodePool → safe to adopt; **replace / delete** on
a NodePool → **stop.** A NodePool replace deprovisions the bare-metal fleet (a reserved 256-GPU
cluster); reconcile the program to match reality before going further, and never `pulumi up`
through a destructive NodePool diff.

Two caveats when reading diffs: (1) the live RBAC/NodePools were applied by Iris under the
server-side-apply field manager `iris`, so Pulumi's SSA dry-run reports a **field conflict**
(e.g. `conflict with "iris": .rules`) and the preview fails until you force ownership with
`PULUMI_K8S_ENABLE_PATCH_FORCE=true` (set above). This is still a dry-run (nothing is written); it
just lets Pulumi compute the merged result. Taking ownership for real happens only at `pulumi up`,
which has a hard ordering rule — see **Production adoption** below. (2) Fields the live object
carries but the program doesn't declare won't show as diffs (Pulumi only manages declared fields).

To discard the recon entirely, `pulumi stack rm cw-us-east-02a` (local backend) — it drops the
state file without touching the cluster.

## Production adoption (GCS state)

Recon (above) is all dry-run and safe to run anytime. The **real** adoption — `pulumi up` writing
the durable state in `gs://marin-iac-state` — is gated on an ordering constraint you cannot skip.

**The ordering constraint: cede *then* adopt.** Iris applies RBAC/NodePools with
`server_side_apply(field_manager="iris", force_conflicts=True)`
(`[service.py:257](../../lib/iris/src/iris/cluster/platforms/k8s/service.py#L257)`). Because it
**forces conflicts**, adopting while old Iris is still deployed does *not* fail — it flaps: every
`pulumi up` takes field ownership, and the next `iris cluster start` force-reclaims it, forever.
No outage (the specs match, so values don't change), but drift detection becomes meaningless and
the two managers ping-pong `managedFields` on every controller restart. So:

1. **Ship the cede first.** Land + roll out the Iris change that deletes `ensure_rbac()` and
  `ensure_nodepools()` from `start_controller()` (spec §4), so no running controller applies
   these resources anymore.
2. **Then adopt.** With no `iris` manager re-applying, Pulumi force-takes ownership once and stays
  the sole owner — clean.

Doing it in the other order (or adopting while any old controller can still restart) reintroduces
the flap until the ceded Iris is deployed everywhere.

**The adoption run itself** (per cluster, one-time):

```bash
# shared production backend + KMS secrets provider (NOT --local / passphrase)
pulumi login gs://marin-iac-state
pulumi stack init cw-us-east-02a \
  --secrets-provider gcpkms://projects/<proj>/locations/<loc>/keyRings/<ring>/cryptoKeys/pulumi
export KUBECONFIG=~/.kube/coreweave-iris-gpu
export PULUMI_K8S_ENABLE_PATCH_FORCE=true      # take ownership from the (now-retired) `iris` manager

pulumi config set marin-iac:import true
pulumi preview          # gate: every resource `import` + no-op/update; ANY NodePool replace/delete → STOP
pulumi up               # adopts live resources into GCS state; does not recreate them
pulumi config set marin-iac:import false        # import_ is ONE-SHOT — see below
```

`marin-iac:import` **is one-shot.** It tells Pulumi "adopt the existing object on the next `up`."
Once adopted, the resource is in state; leaving the flag on makes the *next* `up` try to import an
already-managed resource and error. Flow is strictly: set true → `up` once → set false.

**After adoption:** protect the fleet — add `protect=true` (or `retainOnDelete=true`) to the
NodePools so an accidental `pulumi destroy`/rename can't deprovision the reserved bare-metal
nodes. Then normal ops are plain `pulumi preview`/`up` with the flag off; state lives in GCS.

> **Keep local recon off the production stack.** Do recon under a throwaway stack name (e.g.
> `cw-us-east-02a-recon` on `--local`) so the KMS-backed production `cw-us-east-02a` stack config
> stays clean — don't mix a local passphrase `encryptionsalt` into it.

> **Do not** `pulumi up` **against a live cluster until the cede is deployed.** Recon (dry-run) is
> always fine; the real adoption is ordered. See `spec.md §4`.
