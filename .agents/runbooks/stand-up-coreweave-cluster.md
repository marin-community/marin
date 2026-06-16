---
name: stand-up-coreweave-cluster
description: Bring a CoreWeave GPU cluster from zero to a running job, run the multinode canary smoke, and clean up nodepools afterward.
---

# Runbook: Stand up and operate a CoreWeave GPU cluster

**When you're here:** You need a GPU/CoreWeave cluster running a job — a fresh
stand-up, the Grug MoE multinode canary smoke, or you just warmed/scaled a
nodepool and now have to tear it back down. CoreWeave bills per warm node, so the
trap is leaving GPU nodepools alive after you stop the cluster.

**TL;DR:**
- Install the kubeconfig + R2 creds, then `iris --cluster=<name> cluster start`
  (idempotent — reconciles RBAC, ConfigMap, nodepools, controller). Hello-world a
  CPU job, then a single-GPU JAX job.
- `cluster stop` deletes pods + controller but **leaves nodepools alive and
  billing.** Delete managed nodepools explicitly afterward.
- Reference (RBAC, config fields, instance-type naming, credential tables) lives
  in `lib/iris/docs/coreweave.md`. This runbook owns the *procedure*; link the
  tables, don't copy them.

## Before you touch anything

- **GPU cost is the load-bearing guardrail.** `iris cluster stop` deletes pods
  but **nodepools survive and keep billing** even at zero nodes
  (coreweave.md:212). After stopping a cluster you stood up, delete its managed
  nodepools (see Cleanup). Never warm a node (`targetNodes>0`) and walk away.
- **`iris-ci` / `coreweave-ci` is SHARED.** The daily GPU canary
  (`marin-canary-ferry-coreweave.yaml`) and `iris-smoke-coreweave.yaml` both run
  on the `iris-ci` controller and its H100 nodepool (concurrency group
  `iris-coreweave-ci-shared`, lib/iris/OPS.md:294). Do **not** bounce that
  controller, delete its nodepool, or leave `targetNodes` patched up without
  explicit approval — you will break PR CI.
- **H100 quota is account-wide.** A warm pool in one region eats the shared cap
  in another (coreweave.md:415). Check `kci get nodepools -A` before warming.
- **Cross-region/egress cost rule (AGENTS.md).** Don't move data across regions
  or to the open internet. Point each canary at a prefix + endpoint in the
  cluster's own region (the prefix/endpoint table is coreweave.md:314).
- **R2, not CoreWeave object storage, for JAX.** `cwobject.com`/`cwlota.com` use
  virtual-hosted addressing that JAX's S3 backend can't read; use Cloudflare R2
  for JAX data (coreweave.md:189).

## Steps — zero to a running job

The authority for the full quickstart is coreweave.md:5 (§0, `marin-gpu`). The
shape for any `--cluster=<name>`:

1. **Token + kubeconfig.** Create a token in the CoreWeave Tokens console,
   download its kubeconfig, install it (e.g. `~/.kube/coreweave-iris-gpu`), and
   `export KUBECONFIG=...`. `kubectl cluster-info` to sanity-check
   (coreweave.md:19).

2. **Controller extras + R2 creds.** `uv pip install 'marin-iris[controller]'`,
   then export `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY`. `cluster start` turns
   those into the `iris-s3-credentials` Secret (coreweave.md:186). Without the
   `[controller]` extras, auto-tunneled `--cluster=` commands fail with
   `ImportError: Install iris[controller] to use CloudK8sService` (coreweave.md:227).

3. **Start the cluster** (idempotent — safe to re-run):
   ```bash
   uv run iris --cluster=<name> cluster start
   ```
   This reconciles, in order: namespace + RBAC, S3 Secret, ConfigMap, nodepools,
   controller Deployment + Service (coreweave.md:203). Note: `cluster start`
   **always rebuilds and pushes images**, so it needs `docker login ghcr.io` with
   a `write:packages` PAT (coreweave.md:413). `--cluster=<name>` resolves the
   in-tree config under `lib/iris/config/` and opens a `kubectl port-forward` for
   you — it is the connection selector, a global flag before the subcommand
   (coreweave.md:217, lib/iris/OPS.md:11).

4. **Check status.** `uv run iris --cluster=<name> cluster status`. If
   `port-forward` returns 500, the `konnectivity-agent` pods aren't up yet
   (~18–30s after a node provisions) — wait and retry (coreweave.md:414).

5. **Hello-world.** CPU first, then one GPU to prove JAX sees the device
   (commands at coreweave.md:46):
   ```bash
   uv run iris --cluster=<name> job run --cpu 1 --memory 2GB --extra cpu \
     -- python -c "print('Hello from CoreWeave!')"
   uv run iris --cluster=<name> job run \
     --cpu 8 --memory 64GB --gpu H100x1 --enable-extra-resources --extra gpu \
     -- python -c "import jax; print(jax.devices())"
   ```
   Follow a detached job with `iris --cluster=<name> job logs <job-id> -f`. The
   per-target `--gpu` request and `nvidia-smi` name are in coreweave.md:243.

## Steps — Grug MoE multinode canary smoke

This is a **temporary warm-node** validation (cold-start/gang scheduling tracked
in #5480). The per-target settings (cluster, namespace, kubeconfig, nodepool,
`CANARY_GPU_*`, batch, `NCCL_SOCKET_IFNAME`) and the full submit command live in
coreweave.md:262 — that table is the authority; do not retype its values here.

1. **Warm-node preflight.** Confirm the pool is free, warm (not provisioning),
   and not already consumed by pods. The three checks (`job list --state
   running`, `kubectl get nodepool ... -o custom-columns=...`, `get pods -o
   wide`) are at coreweave.md:269.

2. **Warm the second node (H100 only).** The H100 pool defaults to one node;
   patch `targetNodes:2` before launch (coreweave.md:294):
   ```bash
   kci patch nodepool iris-ci-h100-8x --type merge -p '{"spec":{"targetNodes":2}}'
   ```
   If the controller restarts after warming, **recheck `targetNodes`** — startup
   reconcile can drag the pool back toward its single-node target
   (coreweave.md:304). The GitHub CoreWeave canary uses this same pool: run it and
   manual H100 validation **sequentially**, never concurrently.

3. **Submit with explicit `-e` env.** Iris containers do not inherit your shell
   env, so pass every var via `-e` (full command at coreweave.md:324). Use a
   durable `MARIN_PREFIX` + matching `AWS_ENDPOINT_URL`/credentials for the
   target's region (prefix/endpoint table coreweave.md:314); keep `AWS_REGION` /
   `AWS_DEFAULT_REGION` as `auto` for R2/CoreWeave S3, a real region only for AWS
   S3. `CANARY_TRACKER=json_logger` avoids needing W&B.

4. **Restore the pool after the run (H100):**
   ```bash
   kci patch nodepool iris-ci-h100-8x --type merge -p '{"spec":{"targetNodes":1}}'
   ```

(`kci` = `kubectl --kubeconfig ~/.kube/coreweave-iris`, coreweave.md:377.)

## Steps — nodepool scale / delete / stuck-deletion escape

There are no persistent workers on CoreWeave — the controller dispatches task
pods directly, so `list-workers` and the `workers` table are empty
(coreweave.md:363). Operate nodepools directly with `kci`.

- **Scale a pool:** patch `spec.targetNodes`, never `kubectl scale --replicas`
  (coreweave.md:383):
  ```bash
  kci patch nodepool <name> --type=merge -p '{"spec":{"targetNodes":N}}'
  ```
- **Cleanup after a stand-up you own** — stop, then delete managed pools so they
  stop billing (coreweave.md:400):
  ```bash
  iris cluster stop
  kci delete nodepool -l iris-<label_prefix>-managed=true
  ```
- **Stuck deletion** (autoscaler fights the delete, or a node is mid-delivery):
  scale the controller down so it stops reconciling the pool back up, disable
  autoscaling, strip the finalizer, then delete (coreweave.md:393):
  ```bash
  kci scale deployment iris-controller -n iris --replicas=0
  kci patch nodepool <name> --type=merge -p '{"spec":{"autoscaling":false,"targetNodes":0}}'
  kci patch nodepool <name> --type=json  -p '[{"op":"remove","path":"/metadata/finalizers"}]'
  kci delete nodepool <name>
  ```
  Removing the finalizer is the escape hatch — only reach for it once the pool is
  scaled to zero and autoscaling is off, or it can leak nodes.

## Diagnose — pod/nodepool failures

- **Nodepool won't scale:** `kci get nodepool` (TARGET vs CURRENT),
  `kci describe nodepool <name>` — `Valid: False` means the instance type/config
  was rejected (coreweave.md:783). Instance-type naming is coreweave.md:493.
- **Pod stuck Pending:** `kci describe pod <name> -n iris` + `kci get events -n
  iris --sort-by=.lastTimestamp` — usually node still provisioning, resource
  limits, or missing tolerations (coreweave.md:792).
- **Early-fail signatures** the platform surfaces before the full timeout:
  `ErrImagePull`/`ImagePullBackOff` (verify the image is public),
  `CreateContainerConfigError` (missing Secret/ConfigMap), `CrashLoopBackOff`
  after 2+ restarts (reports last 30 log lines; `kci logs <pod> -n iris
  --previous`), `FailedMount` (coreweave.md:573, 802).
- **Disk full / eviction:** if `cache_dir` isn't `/mnt/local/...`, the ~15 GB
  root RAM disk fills instantly — fix in config and redeploy (coreweave.md:820).
- **`NotTriggerScaleUp: max node group size reached`:** the account-wide H100 cap
  is consumed elsewhere — `kci get nodepools -A` (coreweave.md:415).

## Verify

- **Cluster up:** `iris --cluster=<name> cluster status` returns; CPU and GPU
  hello-world jobs reach `JOB_STATE_SUCCEEDED` and the GPU job's
  `jax.devices()` lists the accelerator.
- **Canary smoke green:** both replicas report JAX, both enter `initialize_jax`
  with `IRIS_NUM_TASKS=2`, the step reaches 20/20, a checkpoint commits, and the
  parent job exits `JOB_STATE_SUCCEEDED` (coreweave.md:356). H100 batch 128 OOMs
  the current model — use batch 64 for functional validation.
- **Cleanup actually freed the GPUs:** `kci get nodepools` shows your managed
  pools gone (not merely scaled to zero). For a shared pool (`iris-ci-h100-8x`),
  confirm `targetNodes` is back to `1` — a stranded `2` keeps a node billing and
  can starve the canary.

## Why this happens / Notes

CoreWeave runs on CKS (bare-metal Kubernetes) with a shared-NodePool model: each
Iris scale group maps to one CoreWeave NodePool with autoscaling, CoreWeave
provisions/deprovisions nodes, and Iris manages only Pods (coreweave.md:63). That
split is why nodepools outlive `cluster stop` — Iris deletes its pods and
controller but the NodePool object is CoreWeave's, and a warm node bills whether
or not a pod is on it. The stuck-deletion dance exists because the autoscaler and
a NodePool finalizer can both keep a pool alive against your delete; you have to
quiet the controller and the autoscaler first, then drop the finalizer.

Cold-start is slow by design: ~14 min for a CPU node, ~20 min for an H100
bare-metal node, ~25–30 min to a first training step from zero (coreweave.md:417).
A pod sitting Pending for minutes after a scale-up is the autoscaler provisioning,
not a fault — don't redrive it.

## See also

- `lib/iris/docs/coreweave.md` — the reference half this runbook leans on: RBAC
  table (§5, :425), config fields (§6, :446), instance-type naming (§7, :493),
  credential summary (§10/§14, :584/:750), timeouts (§11, :609). Read it before
  operating a new region.
- lib/iris/OPS.md:285 "CoreWeave (GPU) Operations" — the one-line pointer + the
  `coreweave-*.yaml` config note; lib/iris/OPS.md:289 "CI Workflows" for the
  shared `iris-ci` canary/smoke rows.
- `run-ferries` and `triage-canary` skills — the GPU canary lane that submits and
  triages the smoke this runbook stands the cluster up for.
- [deploy-controller-fix](deploy-controller-fix.md) — if the controller is
  running stale code after a merged fix (the `:latest` trap), not a stand-up
  problem.
