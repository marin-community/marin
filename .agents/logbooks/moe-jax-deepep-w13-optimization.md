# JAX DeepEP `w13` Optimization: Research Logbook

## Scope
- Goal: reduce the JAX `w13` local expert path on the authoritative exact-cap H100x8 cell and determine how much of that win survives into `local_compute_only`, `deepep_transport_capped_prewarmed`, and `current`.
- Primary metric(s): `time_s` / `tokens_per_s` for `deepep_transport_w13_only_probe`, `deepep_transport_local_compute_only_probe`, `deepep_transport_capped_prewarmed`, and `current`; profile deltas for the dominant `w13` kernels when needed.
- Constraints:
  - start from sealed root-cause evidence in #3752 / tag `moe-jax-megatron-gap-root-cause-seal-20260318`
  - treat gate, early transport, capped warmup, isolated collapse, isolated combine, and `w2`-first as ruled down unless new evidence forces a reopen
  - keep the first tranche stage-gated inside `w13`, not a broad rewrite
  - commit and push any benchmark-code change before launching remote pods
  - post to the GitHub issue only for major milestones / discoveries
- GitHub issue: https://github.com/marin-community/marin/issues/3821
- Prior sealed issue: https://github.com/marin-community/marin/issues/3752
- Experiment ID prefix: `W13-OPT`

## Baseline
- Date: 2026-03-18
- Code refs:
  - `lib/levanter/scripts/bench/bench_moe_hillclimb.py`
  - `lib/levanter/src/levanter/grug/grug_moe.py`
  - `.agents/scripts/deepep_jax_krt_bench.py`
- Fixed baseline case:
  - hardware: H100x8 on CoreWeave
  - `tokens=262144`
  - `topk=2`
  - `bench_pass=forward`
  - `ep=8`
  - `distribution=random`
  - `shared_expert_dim=0`
  - `warmup=5`
  - `iters=20`
- Sealed baseline numbers carried forward from #3752:
  - `deepep_transport_w13_only_probe`: `26,051,875.76 tok/s` (`10.062 ms`)
  - `deepep_transport_local_compute_only_probe`: `15,625,801.24 tok/s` (`16.776 ms`)
  - `deepep_transport_capped_prewarmed`: `12,085,257.93 tok/s` (`21.691 ms`)
  - `current`: `10,186,545.45 tok/s` (`25.735 ms`)
  - matched Megatron exact-cap forward: `33,094,416.78 tok/s` (`7.921 ms`)
- Trustworthy sealed facts:
  - `w13_only` is already slower than full matched Megatron exact-cap forward on the authoritative cell.
  - `w13_only + w2_only` almost reconstructs `local_compute_only`.
  - the pure `w13_only` profile is dominated by `nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NNN` plus `loop_transpose_fusion`.
  - the four `gpt5p4_pro_next_*` reviews agree on `w13`-first ordering, but treat `layout/transpose first` as a hypothesis to test, not a settled sub-root-cause.
- Initial stop / pivot criteria:
  - stop the first micro-branch once one `w13`-targeted change produces a replicated `w13_only` win that propagates into `local_compute_only` and `deepep_transport_capped_prewarmed`.
  - pivot away from `transpose/layout first` if two plausible `w13` sub-branches fail to move `w13_only` materially, or if `w13_only` moves but `deepep_transport_capped_prewarmed` does not.
- First experiment matrix:
  - `W13-OPT-002`: inspect the current `w13` implementation and existing `w13only-262144` / `localcompute-262144` profile summaries to choose one narrow `w13` candidate.
  - `W13-OPT-003`: if the first candidate is a layout/materialization change, validate in order: `w13_only`, `local_compute_only`, `capped_prewarmed`, then `current` only if warranted.
  - `W13-OPT-004`: if the first layout/materialization candidate does not move `w13_only`, pivot to a `ragged_dot` lowering / kernel-path hypothesis rather than iterating broadly on layout.

## Experiment Log
### 2026-03-18 16:05 UTC - Kickoff from the sealed root-cause state
- Experiment ID: `W13-OPT-001`
- Hypothesis:
  - the highest-information next tranche is a narrow `w13` optimization loop on the authoritative `262144` exact-cap cell, with `w13_only` as the first post-change acceptance gate.
- Command:
  - admin/scaffolding only; no new benchmark command yet
- Config:
  - worktree: `/home/ubuntu/dev/marin-wt/moe-jax-megatron-gap-root-cause`
  - branch: `research/moe-jax-deepep-w13-optimization`
  - starting commit: `c258e9fca2f94ec03a6ec1611d033bb513707bfc`
  - sealed tag at start: `moe-jax-megatron-gap-root-cause-seal-20260318`
  - GitHub issue: `#3821`
- Result:
  - verified the local branch/worktree state matches the handoff: `HEAD` is the sealed commit and the untracked files are the expected research artifacts, not stray benchmark changes.
  - read the handoff, the sealed logbook, the sealed issue, the plan-strength brief, the four `gpt5p4_pro_next_*` reviews, the earlier Pro answer batches, and the cautionary negative-result issue `#3665`.
  - created a fresh experiment branch and a fresh GitHub issue rather than continuing the sealed root-cause issue.
- Interpretation:
  - the branch choice is already solved by sealed evidence: `w13` first is high-confidence.
  - the first inner-loop choice is still open: `layout/transpose first` is plausible, but it needs a stage-local `w13_only` win before assuming it is the right micro-branch.
- Next action:
  - inspect the exact `w13` code path and baseline profile artifacts, choose one narrow candidate change, and only then run the required validation ladder on the authoritative cell.

### 2026-03-18 16:38 UTC - First candidate: opt-in `w13` out-first weight layout
- Experiment ID: `W13-OPT-002`
- Hypothesis:
  - the large pure-`w13_only` `loop_transpose_fusion` may be partly caused by the current `w_up_gate` storage order (`[E, D, 2M]`), and an out-first layout (`[E, 2M, D]`) may let XLA lower the `w13` ragged dot with less transpose/materialization work.
- Command:

```bash
uv run pytest lib/haliax/tests/test_ragged_dot_dispatch.py
uv run python -m py_compile \
  lib/haliax/src/haliax/nn/ragged_dot.py \
  lib/levanter/src/levanter/grug/grug_moe.py \
  lib/levanter/scripts/bench/bench_moe_hillclimb.py \
  .agents/scripts/deepep_jax_krt_bench.py
uv run python lib/levanter/scripts/bench/bench_moe_hillclimb.py --help | rg 'w13-out-first'
```

- Config:
  - code change is opt-in only via `--w13-out-first`
  - `ragged_dot` now supports contracting against either rhs axis `1` or rhs axis `2`
  - benchmark harness can now synthesize `w_up_gate` as either `[E, D, 2M]` or `[E, 2M, D]`
- Result:
  - added contract-axis support to `haliax.nn.ragged_dot`
  - added a targeted unit test for out-first rhs layout
  - added `--w13-out-first` to both the benchmark script and the CoreWeave wrapper
  - local verification passed:
    - `lib/haliax/tests/test_ragged_dot_dispatch.py`: `2 passed`
    - `py_compile` on all changed Python files: passed
    - `bench_moe_hillclimb.py --help` shows `--w13-out-first`
- Interpretation:
  - this is a narrow layout/materialization hypothesis test inside `w13`, not a default behavioral change
  - the candidate is now ready for cluster validation using the required ladder order on the authoritative cell
- Next action:
  - commit and push the benchmark candidate, then run `deepep_transport_w13_only_probe` first with `--w13-out-first` before spending any pods on broader follow-up kernels.

### 2026-03-18 17:40 UTC - First `w13_only` launch blocked by H100x8 scheduling
- Experiment ID: `W13-OPT-003`
- Hypothesis:
  - the opt-in `--w13-out-first` candidate should be validated on `deepep_transport_w13_only_probe` before any broader rung.
- Command:

```bash
KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3677.yaml \
  --repo-ref 85df6509d996b1b49a06ee8e525d44960d414531 \
  --task-id w13opt-outfirst-w13only-t262144-20260318-174013 \
  --tokens 262144 \
  --shared-expert-dim 0 \
  --kernels deepep_transport_w13_only_probe \
  --topk-list 2 \
  --distributions random \
  --bench-pass forward \
  --ep-list 8 \
  --warmup 5 \
  --iters 20 \
  --per-bench-timeout-seconds 1200 \
  --per-bench-kill-after-seconds 20 \
  --build-with-torch-extension \
  --load-as-python-module \
  --skip-smoke \
  --skip-cleanup \
  --w13-out-first
```

- Config:
  - candidate commit: `85df6509d996b1b49a06ee8e525d44960d414531`
  - namespace: `iris-3677-jax`
  - initial pod: `iris-task-ffe1a7c9aca0`
- Result:
  - the launch reached pod creation, but no benchmark execution started
  - the pod stayed `Pending` for about 10 minutes with repeated scheduler events:
    - `Insufficient nvidia.com/gpu`
    - one transient event mentioning an untolerated taint on a newly appearing node
  - after waiting for autoscaler progress and confirming there was still no runnable 8-GPU slot, I deleted the pending pod and interrupted the local wrapper so the queued job would not start later unattended
- Interpretation:
  - this is an operational capacity blocker on the CoreWeave H100x8 lane, not a numerical result and not evidence for or against the out-first `w13` candidate
  - the validation order remains unchanged; the next real run should still be `deepep_transport_w13_only_probe`
- Next action:
  - rerun the same pinned `w13_only` command once an H100x8 lane is schedulable, then continue to `local_compute_only` only if `w13_only` moves materially.

### 2026-03-18 18:29 UTC - Rebase the research branch onto current `main` before redeploying a fresh lane
- Experiment ID: `W13-OPT-004`
- Hypothesis:
  - before spending more cluster time, replay the sealed root-cause + `w13` experiment stack onto current `origin/main` so any fresh-lane rerun tests the candidate on an updated branch base rather than on the older sealed-base branch.
- Command:

```bash
git fetch origin main
git branch backup/w13opt-pre-main-rebase-20260318-1825
git rebase --rebase-merges --onto origin/main 69d30f1c11fcb3ad3349f594f59766b7081370b0
uv run pytest lib/haliax/tests/test_ragged_dot_dispatch.py
uv run python -m py_compile \
  lib/haliax/src/haliax/nn/ragged_dot.py \
  lib/levanter/src/levanter/grug/grug_moe.py \
  lib/levanter/scripts/bench/bench_moe_hillclimb.py \
  .agents/scripts/deepep_jax_krt_bench.py
uv run python lib/levanter/scripts/bench/bench_moe_hillclimb.py --help | rg 'w13-out-first'
```

- Config:
  - requested by the user after confirming the previous branch base was not recent
  - previous merge-base against `origin/main`: `69d30f1c11fcb3ad3349f594f59766b7081370b0`
  - new merge-base against `origin/main`: `1ec601a29e2d3fb2101f6d3934cd8e1e75ad538f`
  - backup ref before rewriting history: `backup/w13opt-pre-main-rebase-20260318-1825`
  - one non-code conflict was resolved by keeping the sealed root-cause research logbook in-branch
  - one code conflict in `lib/levanter/src/levanter/grug/grug_moe.py` was resolved by merging the `_w13_ragged_dot(...)` helper into the newer mainline file
- Result:
  - the research branch now sits directly on current `origin/main`
  - local verification passed again after the rebase:
    - `lib/haliax/tests/test_ragged_dot_dispatch.py`: `2 passed`
    - `py_compile` on the changed Python files: passed
    - `bench_moe_hillclimb.py --help` still exposes `--w13-out-first`
  - I also cleaned up the partial fresh-lane artifacts created before the user redirected to the rebase:
    - deleted the old `i3677jax-cpu-erapids` / `i3677jax-h100-8x` node pools
    - deleted `iris-3677-jax`
    - deleted the aborted partial `iris-3821-jax` namespace / RBAC before recreating it from the rebased branch
- Interpretation:
  - the `w13` candidate survives replay onto current main and is still locally runnable
  - the next fresh-lane rerun should use the rebased branch tip, not the sealed-base tip
  - the numerical validation ladder is unchanged: `w13_only` first, then `local_compute_only`, then `capped_prewarmed`, then `current` only if warranted
- Next action:
  - commit the fresh `#3821` CoreWeave lane config, push the rebased branch, redeploy a fresh H100x8 node pool on that branch base, and rerun `deepep_transport_w13_only_probe`.

### 2026-03-18 18:31 UTC - Fresh rebased H100 pool creation is blocked by repeated `NodeConfigInternalError`
- Experiment ID: `W13-OPT-005`
- Hypothesis:
  - after rebasing onto current main and deleting the old 3677 lane, a newly created isolated H100x8 pool should either validate and provision normally or reveal whether the stale-lane problem reproduces on a fresh pool.
- Command:

```bash
KUBECONFIG=/home/ubuntu/.kube/coreweave-iris kubectl create namespace iris-3821-jax --dry-run=client -o yaml | kubectl apply -f -

# attempt 1: prewarm target=1
cat <<'EOF' | KUBECONFIG=/home/ubuntu/.kube/coreweave-iris kubectl apply -f -
apiVersion: compute.coreweave.com/v1alpha1
kind: NodePool
metadata:
  name: i3821jax-h100-8x
  labels:
    iris-i3821jax-managed: "true"
    iris-i3821jax-scale-group: h100-8x
spec:
  computeClass: default
  instanceType: gd-8xh100ib-i128
  autoscaling: true
  minNodes: 0
  maxNodes: 1
  targetNodes: 1
  nodeLabels:
    iris-i3821jax-managed: "true"
    iris-i3821jax-scale-group: h100-8x
EOF

KUBECONFIG=/home/ubuntu/.kube/coreweave-iris kubectl describe nodepool i3821jax-h100-8x
KUBECONFIG=/home/ubuntu/.kube/coreweave-iris kubectl delete nodepool i3821jax-h100-8x --wait=false

# attempt 2: clean recreate, same target=1
cat <<'EOF' | KUBECONFIG=/home/ubuntu/.kube/coreweave-iris kubectl apply -f -
apiVersion: compute.coreweave.com/v1alpha1
kind: NodePool
metadata:
  name: i3821jax-h100-8x
  labels:
    iris-i3821jax-managed: "true"
    iris-i3821jax-scale-group: h100-8x
spec:
  computeClass: default
  instanceType: gd-8xh100ib-i128
  autoscaling: true
  minNodes: 0
  maxNodes: 1
  targetNodes: 1
  nodeLabels:
    iris-i3821jax-managed: "true"
    iris-i3821jax-scale-group: h100-8x
EOF

# attempt 3: create the way Iris normally does first, target=0
cat <<'EOF' | KUBECONFIG=/home/ubuntu/.kube/coreweave-iris kubectl apply -f -
apiVersion: compute.coreweave.com/v1alpha1
kind: NodePool
metadata:
  name: i3821jax-h100-8x
  labels:
    iris-i3821jax-managed: "true"
    iris-i3821jax-scale-group: h100-8x
spec:
  computeClass: default
  instanceType: gd-8xh100ib-i128
  autoscaling: true
  minNodes: 0
  maxNodes: 1
  targetNodes: 0
  nodeLabels:
    iris-i3821jax-managed: "true"
    iris-i3821jax-scale-group: h100-8x
EOF

KUBECONFIG=/home/ubuntu/.kube/coreweave-iris kubectl get nodepools --all-namespaces
KUBECONFIG=/home/ubuntu/.kube/coreweave-iris kubectl get nodepool iris-canary-mh-h100-16x -o yaml
KUBECONFIG=/home/ubuntu/.kube/coreweave-iris kubectl delete nodepool i3821jax-h100-8x --wait=false
```

- Config:
  - fresh lane namespace: `iris-3821-jax`
  - fresh lane label prefix / pool name: `i3821jax-h100-8x`
  - branch tip used for this redeploy: `dd50748e8`
  - reference healthy comparison pool on the same cluster: `iris-canary-mh-h100-16x`
- Result:
  - all three fresh creates of `i3821jax-h100-8x` failed within seconds with the same condition and events:
    - `status.conditions[type=Validated].status=False`
    - `reason=NodeConfigInternalError`
    - message: `failed to generate node configuration due to internal error`
    - repeating event reason: `CWNodeConfigInternalError`
  - the failure reproduced both when the pool was created with `targetNodes=1` and when it was created with `targetNodes=0`
  - no node ever appeared for the fresh pool: `currentNodes=0`, `queuedNodes=0`, `inProgress=0`
  - another H100 pool on the same cluster was healthy at the same time:
    - `iris-canary-mh-h100-16x`: `Validated=True`, `Capacity=Sufficient`, `currentNodes=1`, `inProgress=1`
  - I deleted the failing fresh pool after the third reproduce so the cluster was not left with a broken dangling NodePool object
- Interpretation:
  - the fresh-lane retry did not reproduce the old stale-pool symptom; it hit a different earlier control-plane failure during node-configuration generation
  - because another H100 pool with the same instance type is healthy, this does not look like a blanket H100 outage
  - this blocks the `w13_only` rerun before any benchmark pod can be scheduled, so there is still no new numerical result for or against `--w13-out-first`
- Next action:
  - treat fresh-pool creation as the current blocker for the rebased branch and only continue after an infra workaround or operator-side retry path is chosen.

### 2026-03-18 22:20 UTC - Rebase again onto newer `main`, clear `iris-3821-jax`, and restart via full `iris cluster start`
- Experiment ID: `W13-OPT-006`
- Hypothesis:
  - the cleanest way to rule out ad hoc pool-creation drift is to rebase onto the latest `origin/main`, fully clear the `iris-3821-jax` namespace, and let current Iris recreate the namespace, RBAC, secrets, controller, and both NodePools from the checked-in config.
- Command:

```bash
git fetch origin main
git branch backup/w13opt-pre-main-rebase-20260318-2118
GIT_EDITOR=true git rebase --rebase-merges --onto origin/main 1ec601a29e2d3fb2101f6d3934cd8e1e75ad538f
git push --force-with-lease origin research/moe-jax-deepep-w13-optimization

KUBECONFIG=/home/ubuntu/.kube/coreweave-iris kubectl delete namespace iris-3821-jax --wait=false
KUBECONFIG=/home/ubuntu/.kube/coreweave-iris kubectl delete clusterrolebinding iris-controller-iris-3821-jax --ignore-not-found
KUBECONFIG=/home/ubuntu/.kube/coreweave-iris kubectl delete clusterrole iris-controller-iris-3821-jax --ignore-not-found

. /home/ubuntu/.config/yoblin/env
KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run iris \
  --config lib/iris/examples/coreweave-moe-jax-3821.yaml \
  cluster start
```

- Config:
  - `origin/main` moved again before restart; new base is `43ca35177a7898f2d08526e0d7907fe04445d4c8`
  - new branch tip after the rebase: `7d7672f0c21a3cd82ed41de2d5f3ac01e49e8f3f`
  - restart needed `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY`; on this box they come from `. /home/ubuntu/.config/yoblin/env`
  - cluster config used: `lib/iris/examples/coreweave-moe-jax-3821.yaml`
- Result:
  - there is no `origin/master` in this repo; only `main`, so the rebase target was current `origin/main`
  - the rebase onto `43ca35177a7898f2d08526e0d7907fe04445d4c8` completed cleanly and the branch was force-pushed
  - the first `cluster start` attempt failed only because the shell lacked exported R2 env vars; after sourcing `~/.config/yoblin/env`, the second attempt recreated the lane successfully up to controller wait
  - current post-restart cluster state:
    - `i3821jax-h100-8x`: `Validated=True`, `Target=0`, `Current=0`, `Capacity=Sufficient`, `Quota=Under`
    - `i3821jax-cpu-erapids`: `Validated=True`, `Target=1`, `InProgress=1`, `Current=0`
    - controller pods were recreated in `iris-3821-jax` and are currently `Pending` only because the fresh CPU node has not finished coming online yet
  - importantly, the H100 pool no longer shows `NodeConfigInternalError` when created by the current full Iris restart path
- Interpretation:
  - the earlier manual fresh-pool failure was not reproduced by the current full restart path on newer `main`
  - the cluster is restarted at the resource layer; the remaining wait is controller scheduling on the fresh CPU pool
  - this materially raises confidence that the full current Iris flow is the right way to recreate the lane, rather than manually applying the H100 NodePool first
- Next action:
  - once the controller pod binds and becomes healthy, continue with the normal `w13` validation ladder on the restarted `iris-3821-jax` lane.

### 2026-03-18 22:45 UTC - First restarted `w13_only` launch bound to a shared H100 node; add explicit node selection before the pinned rerun
- Experiment ID: `W13-OPT-007`
- Hypothesis:
  - the raw `KubernetesRuntime` wrapper needs an explicit node-placement knob; otherwise even a fresh `iris-3821-jax` lane can leak onto unrelated H100 nodes when a shared 8-GPU slot becomes free.
- Command:

```bash
KUBECONFIG=/home/ubuntu/.kube/coreweave-iris kubectl patch nodepool i3821jax-h100-8x \
  --type merge -p '{"spec":{"targetNodes":1}}'

KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3821.yaml \
  --repo-ref dbf62c30646a9407af07a1757ba4d76a0649ab93 \
  --task-id w13opt-outfirst-w13only-t262144-20260318-2241 \
  --tokens 262144 \
  --shared-expert-dim 0 \
  --kernels deepep_transport_w13_only_probe \
  --topk-list 2 \
  --distributions random \
  --bench-pass forward \
  --ep-list 8 \
  --warmup 5 \
  --iters 20 \
  --per-bench-timeout-seconds 1200 \
  --per-bench-kill-after-seconds 20 \
  --build-with-torch-extension \
  --load-as-python-module \
  --skip-smoke \
  --skip-cleanup \
  --w13-out-first

uv run pytest lib/iris/tests/cluster/runtime/test_kubernetes_runtime.py -q
uv run python -m py_compile \
  lib/iris/src/iris/cluster/runtime/types.py \
  lib/iris/src/iris/cluster/runtime/kubernetes.py \
  .agents/scripts/deepep_jax_krt_bench.py
uv run python .agents/scripts/deepep_jax_krt_bench.py --help | rg 'node-selector|w13-out-first'
```

- Config:
  - fresh H100 pool: `i3821jax-h100-8x`, `targetNodes=1`
  - assigned fresh H100 node: `g11ed54` (still booting; not yet visible in `kubectl get nodes` when the task was launched)
  - restarted `w13_only` task pod: `iris-task-436f138f98b4`
  - unexpected bound node: `gd92f4a`
  - shared-node label on the bound node: `iris-iris-canary-mh-scale-group=h100-16x`
- Result:
  - the first restarted `w13_only` pod did not wait for the fresh `i3821jax` node; as soon as a shared H100x8 slot opened, Kubernetes scheduled the pod onto `gd92f4a`
  - this is because `.agents/scripts/deepep_jax_krt_bench.py` uses `KubernetesRuntime`, and the runtime-generated Pod spec had GPU requests and tolerations but no `nodeSelector` / placement constraint
  - I added a minimal placement path for future reruns:
    - `ContainerConfig.node_selector`
    - `KubernetesRuntime` propagation to `spec.nodeSelector`
    - repeatable `--node-selector KEY=VALUE` support in `.agents/scripts/deepep_jax_krt_bench.py`
    - a unit test covering node-selector manifest generation
  - local verification passed:
    - `lib/iris/tests/cluster/runtime/test_kubernetes_runtime.py`: `22 passed`
    - `py_compile` on the changed runtime / wrapper files: passed
    - `.agents/scripts/deepep_jax_krt_bench.py --help` now exposes `--node-selector`
  - I am letting `iris-task-436f138f98b4` continue as an exploratory shared-node datapoint since it already reached the actual `deepep_transport_w13_only_probe` bench command
- Interpretation:
  - without explicit placement, the fresh-lane restart is not enough to guarantee that the benchmark actually executes on the fresh `i3821jax-h100-8x` pool
  - this does not invalidate the shared-node run as exploratory performance evidence, but it does mean the clean fresh-lane confirmation must be rerun with an explicit node selector once `g11ed54` is ready
- Next action:
  - commit and push the placement patch, then rerun `deepep_transport_w13_only_probe` with `--node-selector iris-i3821jax-scale-group=h100-8x` as soon as the fresh H100 node is schedulable.

### 2026-03-18 22:48 UTC - Exploratory shared-node `w13_only` result is slower than the sealed baseline and exits with noisy DeepEP/CUDA teardown
- Experiment ID: `W13-OPT-008`
- Hypothesis:
  - even though the first restarted launch landed on the wrong H100 node, its result can still provide a directional sanity check on whether `--w13-out-first` is obviously promising before the clean pinned rerun.
- Command:

```bash
KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3821.yaml \
  --repo-ref dbf62c30646a9407af07a1757ba4d76a0649ab93 \
  --task-id w13opt-outfirst-w13only-t262144-20260318-2241 \
  --tokens 262144 \
  --shared-expert-dim 0 \
  --kernels deepep_transport_w13_only_probe \
  --topk-list 2 \
  --distributions random \
  --bench-pass forward \
  --ep-list 8 \
  --warmup 5 \
  --iters 20 \
  --per-bench-timeout-seconds 1200 \
  --per-bench-kill-after-seconds 20 \
  --build-with-torch-extension \
  --load-as-python-module \
  --skip-smoke \
  --skip-cleanup \
  --w13-out-first
```

- Config:
  - pod: `iris-task-436f138f98b4`
  - node: `gd92f4a`
  - node label: `iris-iris-canary-mh-scale-group=h100-16x`
  - code ref used by the pod: `dbf62c30646a9407af07a1757ba4d76a0649ab93` (pre-node-selector commit)
  - sealed baseline for comparison: `26,051,875.76 tok/s` (`10.062 ms`)
- Result:
  - `deepep_transport_w13_only_probe`: `24,832,377.50 tok/s` (`10.557 ms`)
  - delta vs sealed baseline:
    - throughput: `-4.68%`
    - time: `+4.92%`
  - the task still reached `BENCH_END` and the wrapper returned `EXIT_CODE=0`
  - immediately after the `RESULT` line, the logs emitted noisy shutdown failures:
    - `DeepEP timeout check failed: rank = 0, thread = 0..7`
    - many `CUDA_ERROR_LAUNCH_FAILED: unspecified launch failure` teardown lines from XLA/CUDA cleanup
- Interpretation:
  - this shared-node run does not support the `w13-out-first` hypothesis; directionally it is worse than the sealed `w13_only` baseline
  - because the pod ran on the wrong node and ended with noisy DeepEP/CUDA teardown, this is not a clean acceptance/rejection datapoint for the fresh-lane experiment thread
  - the main value is negative triage: there is no sign here of an obvious large `w13_only` win worth broadening before the pinned rerun
- Next action:
  - wait for `g11ed54` to join `i3821jax-h100-8x`, then rerun the same `w13_only` command from commit `1e24699c6` with `--node-selector iris-i3821jax-scale-group=h100-8x`.

### 2026-03-18 23:04 UTC - Clean fresh-node pinned `w13_only` rerun rules down the first `w13-out-first` layout candidate
- Experiment ID: `W13-OPT-009`
- Hypothesis:
  - the exploratory shared-node slowdown was not just a placement artifact; a clean rerun pinned to the fresh `i3821jax-h100-8x` lane should confirm whether `--w13-out-first` helps the isolated `w13` stage on the authoritative exact-cap cell.
- Command:

```bash
KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3821.yaml \
  --repo-ref 1e24699c6 \
  --task-id w13opt-outfirst-w13only-pinned-t262144-20260318-225850 \
  --tokens 262144 \
  --shared-expert-dim 0 \
  --kernels deepep_transport_w13_only_probe \
  --topk-list 2 \
  --distributions random \
  --bench-pass forward \
  --ep-list 8 \
  --warmup 5 \
  --iters 20 \
  --per-bench-timeout-seconds 1200 \
  --per-bench-kill-after-seconds 20 \
  --build-with-torch-extension \
  --load-as-python-module \
  --skip-smoke \
  --skip-cleanup \
  --w13-out-first \
  --node-selector iris-i3821jax-scale-group=h100-8x
```

- Config:
  - pod: `iris-task-a0b32440d47c`
  - node: `g11ed54`
  - node label: `iris-i3821jax-scale-group=h100-8x`
  - code ref used by the pod: `1e24699c6`
  - exact-cap metadata printed by the run: `max_recv_tokens=61952`, `max_local_assignments=65920`
  - sealed baseline for comparison: `26,051,875.76 tok/s` (`10.062 ms`)
- Result:
  - `deepep_transport_w13_only_probe`: `24,266,057.38 tok/s` (`10.803 ms`)
  - delta vs sealed baseline:
    - throughput: `-6.85%`
    - time: `+7.36%`
  - the task reached `BENCH_END` and the wrapper returned `EXIT_CODE=0`
  - the same noisy shutdown failures appeared after the result line:
    - `DeepEP timeout check failed: rank = 0, thread = 0..7`
    - many `CUDA_ERROR_LAUNCH_FAILED: unspecified launch failure` teardown lines from XLA/CUDA cleanup
- Interpretation:
  - the first `w13` layout/storage-order micro-branch is cleanly ruled down on the authoritative rung-1 check
  - because the pinned fresh-node rerun is slower by a material margin, do not advance this candidate to `local_compute_only`, `capped_prewarmed`, or `current`
  - the repeated teardown noise looks operationally annoying but not numerically decisive here, because both negative runs reached `RESULT`, `BENCH_END`, and `EXIT_CODE=0`
- Next action:
  - pivot away from layout-first and try a narrower `w13` lowering / kernel-path candidate, still validated on `deepep_transport_w13_only_probe` first.

### 2026-03-18 23:19 UTC - Implement an opt-in expert-padded `w13` FC1 kernel-path candidate for the next rung-1 test
- Experiment ID: `W13-OPT-010`
- Hypothesis:
  - the first local expert FC1 is paying for the current XLA ragged lowering on the compact `[assignments, hidden]` pack, and using a static exact-cap per-expert buffer with an explicit batched expert GEMM may beat the current `ragged_dot` lowering even if the storage-order tweak did not.
- Code changes:
  - extend the exact-cap metadata helper to compute and print `max_local_expert_assignments`
  - add `_ragged_dot_expert_padded_batched(...)` to the bench harness:
    - scatter the compact expert-major rows into a static `[local_experts, local_expert_capacity, hidden]` buffer
    - run a batched expert GEMM with `jax.lax.dot_general`
    - gather the outputs back to the original compact row order
  - wire the helper behind a new opt-in `--w13-expert-padded` flag for:
    - `deepep_transport_w13_only_probe`
    - `deepep_transport_local_compute_only_probe` (the `w13` stage only)
  - plumb the new flag through `.agents/scripts/deepep_jax_krt_bench.py`
- Local validation:
  - `uv run python -m py_compile lib/levanter/scripts/bench/bench_moe_hillclimb.py .agents/scripts/deepep_jax_krt_bench.py`
  - toy numerical equivalence check against `haliax.nn.ragged_dot.ragged_dot`:
    - compact weight layout `[E, H, O]`: `max_abs=0.0`
    - out-first weight layout `[E, O, H]`: `max_abs=0.0`
  - `.agents/scripts/deepep_jax_krt_bench.py --help` now exposes `--w13-expert-padded`
- Interpretation:
  - this is the first real post-layout pivot inside the `w13` branch: not another transpose tweak, but an alternate FC1 lowering that keeps the rest of the probe structure intact
  - the candidate is still narrow enough to test cleanly on `w13_only` before deciding whether it deserves the broader validation ladder
- Next action:
  - commit and push the expert-padded FC1 candidate, then rerun `deepep_transport_w13_only_probe` on the fresh `iris-i3821jax-scale-group=h100-8x` node with `--w13-expert-padded`.

### 2026-03-18 23:25 UTC - Expert-padded FC1 candidate produces a major rung-1 `w13_only` win on the fresh H100 lane
- Experiment ID: `W13-OPT-011`
- Hypothesis:
  - the compact `ragged_dot` lowering is the main FC1 problem inside `w13`, and an explicit expert-padded batched FC1 can beat it materially on the authoritative exact-cap cell.
- Command:

```bash
KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3821.yaml \
  --repo-ref f9734d0b32bb761c2cc7f89b9048544294554e03 \
  --task-id w13opt-expertpadded-w13only-pinned-t262144-20260318-232052 \
  --tokens 262144 \
  --shared-expert-dim 0 \
  --kernels deepep_transport_w13_only_probe \
  --topk-list 2 \
  --distributions random \
  --bench-pass forward \
  --ep-list 8 \
  --warmup 5 \
  --iters 20 \
  --per-bench-timeout-seconds 1200 \
  --per-bench-kill-after-seconds 20 \
  --build-with-torch-extension \
  --load-as-python-module \
  --skip-smoke \
  --skip-cleanup \
  --w13-expert-padded \
  --node-selector iris-i3821jax-scale-group=h100-8x
```

- Config:
  - pod: `iris-task-e17654400523`
  - node: `g11ed54`
  - node label: `iris-i3821jax-scale-group=h100-8x`
  - code ref used by the pod: `f9734d0b32bb761c2cc7f89b9048544294554e03`
  - exact-cap metadata printed by the run:
    - `max_recv_tokens=61952`
    - `max_local_assignments=65920`
    - `max_local_expert_assignments=4352`
  - sealed baseline for comparison: `26,051,875.76 tok/s` (`10.062 ms`)
- Result:
  - `deepep_transport_w13_only_probe`: `107,976,854.82 tok/s` (`2.428 ms`)
  - delta vs sealed baseline:
    - throughput: `+314.47%`
    - time: `-75.87%`
  - the task reached `BENCH_END` and the wrapper returned `EXIT_CODE=0`
  - the same noisy shutdown failures still appeared after the result line:
    - `DeepEP timeout check failed: rank = 0, thread = 0..7`
    - many `CUDA_ERROR_LAUNCH_FAILED: unspecified launch failure` teardown lines from XLA/CUDA cleanup
- Interpretation:
  - this is the first major positive result in the new `w13` optimization thread
  - the stage-local `w13` budget is highly sensitive to the FC1 lowering / packing choice; the compact `ragged_dot` lowering appears to be the wrong shape for this cell
  - because the win is very large and was measured on the fresh pinned lane, the candidate must now be validated immediately on `deepep_transport_local_compute_only_probe`
- Next action:
  - run the same fresh-lane pinned benchmark with `--w13-expert-padded` on `deepep_transport_local_compute_only_probe`.

### 2026-03-18 23:31 UTC - The expert-padded `w13` candidate survives rung 2 and cuts `local_compute_only` almost in half
- Experiment ID: `W13-OPT-012`
- Hypothesis:
  - the large rung-1 `w13_only` win is real enough to survive reintegrating `gate -> w2` inside the isolated local-compute probe, rather than disappearing once the rest of local expert compute is restored.
- Command:

```bash
KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3821.yaml \
  --repo-ref f9734d0b32bb761c2cc7f89b9048544294554e03 \
  --task-id w13opt-expertpadded-localcompute-pinned-t262144-20260318-232636 \
  --tokens 262144 \
  --shared-expert-dim 0 \
  --kernels deepep_transport_local_compute_only_probe \
  --topk-list 2 \
  --distributions random \
  --bench-pass forward \
  --ep-list 8 \
  --warmup 5 \
  --iters 20 \
  --per-bench-timeout-seconds 1200 \
  --per-bench-kill-after-seconds 20 \
  --build-with-torch-extension \
  --load-as-python-module \
  --skip-smoke \
  --skip-cleanup \
  --w13-expert-padded \
  --node-selector iris-i3821jax-scale-group=h100-8x
```

- Config:
  - pod: `iris-task-df3aa4531e70`
  - node: `g11ed54`
  - node label: `iris-i3821jax-scale-group=h100-8x`
  - code ref used by the pod: `f9734d0b32bb761c2cc7f89b9048544294554e03`
  - exact-cap metadata printed by the run:
    - `max_recv_tokens=61952`
    - `max_local_assignments=65920`
    - `max_local_expert_assignments=4352`
  - sealed baseline for comparison: `15,625,801.24 tok/s` (`16.776 ms`)
- Result:
  - `deepep_transport_local_compute_only_probe`: `29,461,265.86 tok/s` (`8.898 ms`)
  - delta vs sealed baseline:
    - throughput: `+88.54%`
    - time: `-46.96%`
  - the task reached `BENCH_END` and the wrapper returned `EXIT_CODE=0`
  - the same noisy shutdown failures still appeared after the result line:
    - `DeepEP timeout check failed: rank = 0, thread = 0..7`
    - many `CUDA_ERROR_LAUNCH_FAILED: unspecified launch failure` teardown lines from XLA/CUDA cleanup
- Interpretation:
  - the FC1 candidate is not just a probe-only artifact; a large fraction of the rung-1 `w13` gain survives once the rest of local expert compute is restored
  - the remaining `local_compute_only` time (`8.898 ms`) is now close enough to the sealed full matched Megatron forward (`7.921 ms`) that this branch is clearly worth carrying into the exact-cap integrated path
  - the next validation step is now mandatory: wire the same candidate into the actual capped/prewarmed forward path and measure `deepep_transport_capped_prewarmed`
- Next action:
  - extend the opt-in expert-padded `w13` path into the actual capped/prewarmed DeepEP forward, commit and push, then run `deepep_transport_capped_prewarmed` on the same fresh H100 lane.

### 2026-03-18 23:37 UTC - The expert-padded `w13` candidate survives rung 3 and materially improves the authoritative exact-cap forward path
- Experiment ID: `W13-OPT-013`
- Hypothesis:
  - the `w13` FC1 gain is not just a probe/local-compute artifact; once carried into the actual capped/prewarmed DeepEP forward, it should still produce a meaningful exact-cap end-to-end win on the authoritative cell.
- Commands:

```bash
# First capped/prewarmed launch failed operationally before benchmarking because
# I passed an incorrect full repo SHA; the pod 404'ed while downloading the tarball.
KUBECONFIG=/home/ubuntu/.kube/coreweave-iris kubectl -n iris-3821-jax delete pod iris-task-23c2efa97627 --wait=false

KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3821.yaml \
  --repo-ref e21bcd8a5d4515e37fad14927e092f3bee7b6814 \
  --task-id w13opt-expertpadded-cappedprewarmed-pinned-t262144-20260318-233417 \
  --tokens 262144 \
  --shared-expert-dim 0 \
  --kernels deepep_transport_capped_prewarmed \
  --topk-list 2 \
  --distributions random \
  --bench-pass forward \
  --ep-list 8 \
  --warmup 5 \
  --iters 20 \
  --per-bench-timeout-seconds 1200 \
  --per-bench-kill-after-seconds 20 \
  --build-with-torch-extension \
  --load-as-python-module \
  --skip-smoke \
  --skip-cleanup \
  --w13-expert-padded \
  --node-selector iris-i3821jax-scale-group=h100-8x
```

- Config:
  - failed operational pod: `iris-task-23c2efa97627`
    - failure mode: `urllib.error.HTTPError: HTTP Error 404: Not Found` while downloading the repo archive
    - cause: I passed the wrong full repo SHA to the launcher; there was no benchmark result from that attempt
  - successful pod: `iris-task-1cde04ece586`
  - node: `g11ed54`
  - node label: `iris-i3821jax-scale-group=h100-8x`
  - code ref used by the successful pod: `e21bcd8a5d4515e37fad14927e092f3bee7b6814`
  - exact-cap metadata printed by the run:
    - `max_recv_tokens=61952`
    - `max_local_assignments=65920`
    - `max_local_expert_assignments=4352`
  - sealed baseline for comparison: `12,085,257.93 tok/s` (`21.691 ms`)
- Result:
  - `deepep_transport_capped_prewarmed`: `21,763,265.78 tok/s` (`12.045 ms`)
  - delta vs sealed baseline:
    - throughput: `+80.08%`
    - time: `-44.47%`
  - the successful task reached `BENCH_END` and the wrapper returned `EXIT_CODE=0`
  - the same noisy shutdown failures still appeared after the result line:
    - `DeepEP timeout check failed: rank = 0, thread = 0..7`
    - many `CUDA_ERROR_LAUNCH_FAILED: unspecified launch failure` teardown lines from XLA/CUDA cleanup
- Interpretation:
  - the candidate is now clearly validated through the authoritative exact-cap forward rung
  - the `w13` FC1 lowering change propagates far enough that the exact-cap prewarmed path drops by almost half, which is much stronger than a stage-local-only win
  - the remaining gap to the sealed exact-cap prewarmed path (`12.045 ms`) is now much smaller, and the next question is whether enough of that gain can survive all the way into `current`
- Next action:
  - post this as a major milestone on issue `#3821`, then decide whether to integrate the same expert-padded FC1 idea into the actual `current` path for the final rung.

### 2026-03-18 23:47 UTC - The expert-padded `w13` candidate survives the final `current` rung on the fresh pinned lane
- Experiment ID: `W13-OPT-014`
- Hypothesis:
  - if the `w13` exact-cap FC1 gain is the real dominant residual inside the live ring path, then threading the same expert-padded capacity into the actual `current` path should materially reduce the authoritative full forward time rather than stalling out at `capped_prewarmed`.
- Command:

```bash
KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3821.yaml \
  --repo-ref 7c3c5176d3d6aaa093dbd8359c7c965c4ac10610 \
  --task-id w13opt-expertpadded-current-pinned-t262144-20260318-234531 \
  --tokens 262144 \
  --shared-expert-dim 0 \
  --kernels current \
  --topk-list 2 \
  --distributions random \
  --bench-pass forward \
  --ep-list 8 \
  --warmup 5 \
  --iters 20 \
  --per-bench-timeout-seconds 1200 \
  --per-bench-kill-after-seconds 20 \
  --build-with-torch-extension \
  --load-as-python-module \
  --skip-smoke \
  --skip-cleanup \
  --w13-expert-padded \
  --node-selector iris-i3821jax-scale-group=h100-8x
```

- Config:
  - pod: `iris-task-116ce90b3b3e`
  - node: `g11ed54`
  - node label: `iris-i3821jax-scale-group=h100-8x`
  - code ref used by the pod: `7c3c5176d3d6aaa093dbd8359c7c965c4ac10610`
  - current-path exact-cap metadata printed by the run:
    - `CURRENT_RING_W13_CAPS local_capacity=81920 max_local_expert_assignments=4352`
  - sealed baseline for comparison: `10,186,545.45 tok/s` (`25.735 ms`)
- Result:
  - `current`: `17,120,042.64 tok/s` (`15.312 ms`)
  - delta vs sealed baseline:
    - throughput: `+68.07%`
    - time: `-40.50%`
  - the task reached `BENCH_END` and the wrapper returned `EXIT_CODE=0`
- Interpretation:
  - the `w13` exact-cap FC1 candidate is not just a probe-path or prewarmed-path win; it survives into the live `current` ring forward as well
  - the `current - capped_prewarmed` residual also shrank, from `4.044 ms` in the sealed baseline to `3.267 ms` here, so the ring-path integration preserved most of the rung-3 gain instead of reintroducing the old gap
  - this completes the required validation ladder for the `w13`-first tranche on the authoritative cell with a material win at every rung
- Next action:
  - post the full-rung milestone on issue `#3821`, then decide whether the next thread should be a confirmation/profile pass on the remaining `current - capped_prewarmed` residual or broaderization beyond the single authoritative cell.

### 2026-03-18 23:52 UTC - A fresh `current` profile shows the remaining residual is no longer the old `w13` transpose path
- Experiment ID: `W13-OPT-015`
- Hypothesis:
  - after the expert-padded `w13` fix lands in `current`, the next bottleneck should move away from the old `w13` transpose-heavy ragged path; a fresh `current` trace should reveal whether the residual is now compute (`w2`-like) or communication/overlap dominated.
- Commands:

```bash
KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3821.yaml \
  --repo-ref 7c3c5176d3d6aaa093dbd8359c7c965c4ac10610 \
  --task-id w13opt-expertpadded-current-profile-t262144-20260318-235000 \
  --tokens 262144 \
  --shared-expert-dim 0 \
  --kernels current \
  --topk-list 2 \
  --distributions random \
  --bench-pass forward \
  --ep-list 8 \
  --warmup 2 \
  --iters 3 \
  --profile-root /tmp/current-expertpadded-262144 \
  --post-bench-sleep-seconds 1800 \
  --per-bench-timeout-seconds 1800 \
  --per-bench-kill-after-seconds 20 \
  --build-with-torch-extension \
  --load-as-python-module \
  --skip-smoke \
  --skip-cleanup \
  --w13-expert-padded \
  --node-selector iris-i3821jax-scale-group=h100-8x

KUBECONFIG=/home/ubuntu/.kube/coreweave-iris kubectl -n iris-3821-jax cp \
  iris-task-b4a60f903086:/tmp/current-expertpadded-262144/. \
  scratch/profiles/current-expertpadded-262144

uv run python -m marin.profiling.cli summarize \
  --profile-dir scratch/profiles/current-expertpadded-262144 \
  --warmup-steps 0 \
  --output scratch/profiles/current-expertpadded-262144-summary.json

uv run python -m marin.profiling.cli report \
  --summary scratch/profiles/current-expertpadded-262144-summary.json \
  --output scratch/profiles/current-expertpadded-262144-report.md
```

- Config:
  - profiling pod: `iris-task-b4a60f903086`
  - node: `g11ed54`
  - node label: `iris-i3821jax-scale-group=h100-8x`
  - code ref used by the pod: `7c3c5176d3d6aaa093dbd8359c7c965c4ac10610`
  - trace artifact copied locally to:
    - `scratch/profiles/current-expertpadded-262144`
    - `scratch/profiles/current-expertpadded-262144-summary.json`
    - `scratch/profiles/current-expertpadded-262144-report.md`
  - note: the profiled `RESULT kernel=current ... time_s=0.058428` is tracing overhead and is not comparable to the timing-only rung result
- Result:
  - top exclusive compute op:
    - `nvjet_tst_256x128_64x4_1x2_h_bz_coopA_NNT`: `115934.245` exclusive (`21.9%` share)
  - communication share:
    - `25.11%` of exclusive profiled duration
  - top collectives:
    - `all-gather`: `67140.606` exclusive across `48` calls
    - `reduce-scatter`: `66014.115` exclusive across `24` calls
  - largest pre-op idle gaps:
    - before `reduce-scatter`: `229021.318` total gap (`24` occurrences, `9542.555` avg)
    - before `all-gather`: `175005.335` total gap (`24` occurrences, `7291.889` avg)
  - the old pure-`w13` top kernel from the sealed probe summaries, `nvjet_tst_192x192_64x3_2x1_v_bz_coopB_NNN`, is no longer the dominant `current` hot op in this post-fix trace
- Interpretation:
  - the `w13` exact-cap optimization did what it needed to do: the remaining `current` residual is no longer centered on the old `w13` ragged FC1 transpose-heavy path
  - the next branch point is now between:
    - reopening an isolated local compute path around the new top GEMM (`w2`-like residual)
    - or attacking the ring-path communication / synchronization residual, since collectives plus pre-collective gaps now dominate a large share of the trace
- Next action:
  - run a pinned `deepep_transport_w2_only_probe` on the same commit to decide whether the new top compute kernel is worth reopening `w2`, or whether the next tranche should pivot directly to communication/overlap.

### 2026-03-18 23:58 UTC - Fresh pinned `w2_only` is effectively flat, so the next tranche should pivot to ring communication / synchronization
- Experiment ID: `W13-OPT-016`
- Hypothesis:
  - if the new top `current` compute kernel in the post-fix profile is really the next dominant local-compute branch, then a fresh pinned `deepep_transport_w2_only_probe` should show a meaningful shift relative to the sealed baseline; if it stays flat, the next tranche should target the ring residual instead.
- Command:

```bash
KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3821.yaml \
  --repo-ref 7c3c5176d3d6aaa093dbd8359c7c965c4ac10610 \
  --task-id w13opt-residualtriage-w2only-pinned-t262144-20260318-235356 \
  --tokens 262144 \
  --shared-expert-dim 0 \
  --kernels deepep_transport_w2_only_probe \
  --topk-list 2 \
  --distributions random \
  --bench-pass forward \
  --ep-list 8 \
  --warmup 5 \
  --iters 20 \
  --per-bench-timeout-seconds 1200 \
  --per-bench-kill-after-seconds 20 \
  --build-with-torch-extension \
  --load-as-python-module \
  --skip-smoke \
  --skip-cleanup \
  --node-selector iris-i3821jax-scale-group=h100-8x
```

- Config:
  - pod: `iris-task-7c5cef1aff4b`
  - node: `g11ed54`
  - node label: `iris-i3821jax-scale-group=h100-8x`
  - code ref used by the pod: `7c3c5176d3d6aaa093dbd8359c7c965c4ac10610`
  - exact-cap metadata printed by the run:
    - `max_recv_tokens=61952`
    - `max_local_assignments=65920`
    - `max_local_expert_assignments=4352`
  - sealed baseline for comparison: `51,574,551.62 tok/s` (`5.083 ms`)
  - operational note:
    - the pod lingered after the `RESULT` line without promptly emitting `BENCH_END` / `EXIT_CODE`; after the numerical result was captured I deleted the pod to free the fresh node
- Result:
  - `deepep_transport_w2_only_probe`: `51,933,311.55 tok/s` (`5.048 ms`)
  - delta vs sealed baseline:
    - throughput: `+0.70%`
    - time: `-0.69%`
- Interpretation:
  - this is effectively flat relative to the sealed baseline, so the post-fix `current` residual is not pointing to a newly promising `w2`-first local-compute branch
  - combined with the fresh `current` profile, the next tranche should pivot away from `w2` and toward the ring-path residual:
    - `all-gather` / `reduce-scatter`
    - and the large pre-collective idle gaps immediately before those collectives
- Next action:
  - start a new residual-focused thread on `current - capped_prewarmed`, centered on communication volume / overlap / synchronization rather than reopening `w2`.

### 2026-03-19 00:36 UTC - Reduced `#3717` two-column rerun kickoff: JAX DeepEP `32768` row moves up materially with `w13_expert_padded`, but post-result teardown is still noisy
- Experiment ID: `W13-OPT-017`
- Hypothesis:
  - if the `w13` layout fix also matters in the fixed-shape `forward_backward` regime from `#3717`, then rerunning only the JAX DeepEP and Megatron DeepEP columns with light sampling should show the JAX column move upward, especially on the larger-token rows where the old gap was most severe.
- Benchmark harness change:
  - patched `lib/levanter/scripts/bench/bench_moe_hillclimb.py` on commit `b3f3126e5887b41d2fa56705c14221513c44e329` so `deepep_transport_capped_prewarmed` threads `--w13-expert-padded` through the `forward_backward` path instead of only the forward-only path.
- Command:

```bash
KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3821.yaml \
  --repo-ref b3f3126e5887b41d2fa56705c14221513c44e329 \
  --task-id 3717rerun-jax-fb-t32768-20260319-0028 \
  --tokens 32768 \
  --shared-expert-dim 2048 \
  --kernels deepep_transport_capped_prewarmed \
  --topk-list 2,8 \
  --distributions random \
  --bench-pass forward_backward \
  --ep-list 8 \
  --warmup 1 \
  --iters 2 \
  --per-bench-timeout-seconds 3600 \
  --per-bench-kill-after-seconds 30 \
  --build-with-torch-extension \
  --load-as-python-module \
  --skip-smoke \
  --w13-expert-padded \
  --node-selector iris-i3821jax-scale-group=h100-8x
```

- Config:
  - pod: `iris-task-7a403a8469c5`
  - node: `g11ed54`
  - code ref used by the pod: `b3f3126e5887b41d2fa56705c14221513c44e329`
  - output log: `scratch/3717-rerun/jax-fb-t32768.log`
- Result:
  - `32768, topk=2`: `4,722,170.08 tok/s` (`6.939 ms`)
    - old `#3717` JAX DeepEP value: `3,312,032 tok/s`
    - delta: `+42.6%`
  - `32768, topk=8`: `1,811,121.75 tok/s` (`18.093 ms`)
    - old `#3717` JAX DeepEP value: `1,115,553 tok/s`
    - delta: `+62.4%`
- Operational note:
  - both subcases emitted `DeepEP timeout check failed` plus `CUDA_ERROR_LAUNCH_FAILED` teardown noise after their `RESULT` lines, but the pod still completed with `EXIT_CODE=0` and the second subcase ran after the first.
- Interpretation:
  - the reduced rerun is already consistent with the earlier forward-only finding: the fixed-shape JAX DeepEP column does move upward materially once the `w13` layout change is actually threaded into the `forward_backward` path.
  - this does not by itself imply parity with Megatron, but it is enough signal to continue the reduced two-column sweep before drawing conclusions.
- Next action:
  - continue the JAX DeepEP rerun for `65536`, `131072`, and `262144`, then run the Megatron DeepEP column with the same `2`-iteration lightweight settings for a direct side-by-side against the old `#3717` table.

### 2026-03-19 00:43 UTC - Reduced `#3717` rerun: JAX DeepEP `65536` row also improves materially, with the same reproducible post-result DeepEP teardown noise
- Experiment ID: `W13-OPT-018`
- Command:

```bash
KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3821.yaml \
  --repo-ref b3f3126e5887b41d2fa56705c14221513c44e329 \
  --task-id 3717rerun-jax-fb-t65536-20260319-0037 \
  --tokens 65536 \
  --shared-expert-dim 2048 \
  --kernels deepep_transport_capped_prewarmed \
  --topk-list 2,8 \
  --distributions random \
  --bench-pass forward_backward \
  --ep-list 8 \
  --warmup 1 \
  --iters 2 \
  --per-bench-timeout-seconds 3600 \
  --per-bench-kill-after-seconds 30 \
  --build-with-torch-extension \
  --load-as-python-module \
  --skip-smoke \
  --w13-expert-padded \
  --node-selector iris-i3821jax-scale-group=h100-8x
```

- Config:
  - pod: `iris-task-10fb691de691`
  - node: `g11ed54`
  - code ref used by the pod: `b3f3126e5887b41d2fa56705c14221513c44e329`
  - output log: `scratch/3717-rerun/jax-fb-t65536.log`
- Result:
  - `65536, topk=2`: `5,423,119.32 tok/s` (`12.085 ms`)
    - old `#3717` JAX DeepEP value: `3,752,354 tok/s`
    - delta: `+44.5%`
  - `65536, topk=8`: `1,840,043.05 tok/s` (`35.617 ms`)
    - old `#3717` JAX DeepEP value: `1,116,306 tok/s`
    - delta: `+64.8%`
- Operational note:
  - exactly like `32768`, both subcases emitted `DeepEP timeout check failed` followed by many `CUDA_ERROR_LAUNCH_FAILED` teardown lines after `RESULT`, but the pod still emitted `BENCH_END` and exited `0`.
- Interpretation:
  - the reduced rerun is not just improving the smallest row; the uplift persists at `65536` for both `topk=2` and `topk=8`.
  - the repeated teardown signature looks systematic enough to log as a separate operational issue if it blocks broader sweeps, but it is not invalidating the measured throughput lines in these runs.
- Next action:
  - continue the JAX DeepEP rerun at `131072` and `262144`, where the old table’s JAX-vs-Megatron gap was largest and therefore most important to the original question.

### 2026-03-19 00:50 UTC - Reduced `#3717` rerun: `131072` confirms the JAX DeepEP uplift carries into the high-token regime, but it is still well short of Megatron parity
- Experiment ID: `W13-OPT-019`
- Command:

```bash
KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3821.yaml \
  --repo-ref b3f3126e5887b41d2fa56705c14221513c44e329 \
  --task-id 3717rerun-jax-fb-t131072-20260319-0044 \
  --tokens 131072 \
  --shared-expert-dim 2048 \
  --kernels deepep_transport_capped_prewarmed \
  --topk-list 2,8 \
  --distributions random \
  --bench-pass forward_backward \
  --ep-list 8 \
  --warmup 1 \
  --iters 2 \
  --per-bench-timeout-seconds 3600 \
  --per-bench-kill-after-seconds 30 \
  --build-with-torch-extension \
  --load-as-python-module \
  --skip-smoke \
  --w13-expert-padded \
  --node-selector iris-i3821jax-scale-group=h100-8x
```

- Config:
  - pod: `iris-task-3317f006bfce`
  - node: `g11ed54`
  - code ref used by the pod: `b3f3126e5887b41d2fa56705c14221513c44e329`
  - output log: `scratch/3717-rerun/jax-fb-t131072.log`
- Result:
  - `131072, topk=2`: `5,812,763.83 tok/s` (`22.549 ms`)
    - old `#3717` JAX DeepEP value: `3,824,437 tok/s`
    - delta: `+52.0%`
    - still well below old Megatron DeepEP `12,712,443 tok/s`
  - `131072, topk=8`: `1,910,248.60 tok/s` (`68.615 ms`)
    - old `#3717` JAX DeepEP value: `1,057,153 tok/s`
    - delta: `+80.7%`
    - still well below old Megatron DeepEP `6,577,851 tok/s`
- Operational note:
  - the same post-`RESULT` DeepEP timeout / `CUDA_ERROR_LAUNCH_FAILED` teardown signature appeared again, but the pod advanced through both subcases and exited `0`.
- Interpretation:
  - this is the first directly relevant high-token row against the original `#3717` question, and it makes the answer sharper:
    - the `w13` fix materially improves JAX DeepEP in this regime,
    - but it does not make JAX “keep up” with the old Megatron DeepEP column at `131072`.
- Next action:
  - finish the last JAX row at `262144`, then run the reduced Megatron DeepEP column so the final side-by-side is measured on the same fresh lane and same lightweight iteration settings.

### 2026-03-19 00:57 UTC - Reduced `#3717` rerun: JAX DeepEP `262144` row also moves up, but still does not approach the old Megatron DeepEP throughput
- Experiment ID: `W13-OPT-020`
- Command:

```bash
KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3821.yaml \
  --repo-ref b3f3126e5887b41d2fa56705c14221513c44e329 \
  --task-id 3717rerun-jax-fb-t262144-20260319-0051 \
  --tokens 262144 \
  --shared-expert-dim 2048 \
  --kernels deepep_transport_capped_prewarmed \
  --topk-list 2,8 \
  --distributions random \
  --bench-pass forward_backward \
  --ep-list 8 \
  --warmup 1 \
  --iters 2 \
  --per-bench-timeout-seconds 3600 \
  --per-bench-kill-after-seconds 30 \
  --build-with-torch-extension \
  --load-as-python-module \
  --skip-smoke \
  --w13-expert-padded \
  --node-selector iris-i3821jax-scale-group=h100-8x
```

- Config:
  - pod: `iris-task-7e4ee57484da`
  - node: `g11ed54`
  - code ref used by the pod: `b3f3126e5887b41d2fa56705c14221513c44e329`
  - output log: `scratch/3717-rerun/jax-fb-t262144.log`
- Result:
  - `262144, topk=2`: `5,902,299.01 tok/s` (`44.414 ms`)
    - old `#3717` JAX DeepEP value: `3,821,071 tok/s`
    - delta: `+54.5%`
    - still well below old Megatron DeepEP `15,830,875 tok/s`
  - `262144, topk=8`: `1,790,803.36 tok/s` (`146.383 ms`)
    - old `#3717` JAX DeepEP value: `985,350 tok/s`
    - delta: `+81.7%`
    - still well below old Megatron DeepEP `7,652,599 tok/s`
- Operational note:
  - the same reproducible post-`RESULT` DeepEP timeout / `CUDA_ERROR_LAUNCH_FAILED` teardown signature appeared again, but the pod still completed and exited `0`.
- Interpretation:
  - the reduced JAX DeepEP column is now complete and the conclusion is clearer than the earlier projection:
    - the `w13` fix materially improves every row we checked,
    - but the high-token rows still remain far from the old Megatron DeepEP column, so this fix alone does not make JAX DeepEP “keep up” in the sense asked in `#3717`.
- Next action:
  - run the reduced Megatron DeepEP column with the same lightweight iteration settings on the same fresh H100x8 lane, then compile the direct side-by-side table and post a milestone update.

### 2026-03-19 01:06 UTC - Reduced two-column rerun complete: JAX DeepEP improves on every row, but still does not broadly keep up with Megatron DeepEP on the high-token regime
- Experiment ID: `W13-OPT-021`
- Command:

```bash
KUBECONFIG=/home/ubuntu/.kube/coreweave-iris uv run .agents/scripts/megatron_qwen_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3821.yaml \
  --worktree /home/ubuntu/dev/marin-wt/moe-jax-megatron-gap-root-cause \
  --task-id 3717rerun-megatron-deepep-20260319-0058 \
  --cases marin_3633_topk_2,marin_3633_topk_8,marin_3633_topk_2_mb2,marin_3633_topk_8_mb2,marin_3633_topk_2_mb4,marin_3633_topk_8_mb4,marin_3633_topk_2_mb8,marin_3633_topk_8_mb8 \
  --dispatchers deepep \
  --warmup-iters 1 \
  --measure-iters 2
```

- Config:
  - pod: `iris-task-a16d2846edcd`
  - node: `g11ed54`
  - output log: `scratch/3717-rerun/megatron-deepep.log`
  - comparison method:
    - JAX `bench_pass=forward_backward` already reports full-step `tokens/s`
    - for Megatron I computed comparable full-step throughput as:
      - `global_tokens / ((forward_ms + backward_ms) / 1000)`
- Direct reduced comparison:

| tokens | topk | old `#3717` JAX DeepEP | new JAX DeepEP | old `#3717` Megatron DeepEP | new Megatron DeepEP (combined) | JAX / new-Meg ratio |
|---|---|---|---|---|---|---|
| 32768 | 2 | 3,312,032 | 4,722,170 | 2,644,307 | 934,083 | 5.06 |
| 32768 | 8 | 1,115,553 | 1,811,122 | 1,965,951 | 812,600 | 2.23 |
| 65536 | 2 | 3,752,354 | 5,423,119 | 2,660,216 | 897,613 | 6.04 |
| 65536 | 8 | 1,116,306 | 1,840,043 | 3,088,961 | 2,927,885 | 0.63 |
| 131072 | 2 | 3,824,437 | 5,812,764 | 12,712,443 | 5,607,280 | 1.04 |
| 131072 | 8 | 1,057,153 | 1,910,249 | 6,577,851 | 5,053,786 | 0.38 |
| 262144 | 2 | 3,821,071 | 5,902,299 | 15,830,875 | 12,996,676 | 0.45 |
| 262144 | 8 | 985,350 | 1,790,803 | 7,652,599 | 3,783,723 | 0.47 |

- Interpretation:
  - the rerun decisively validates the directional claim:
    - the `w13`-driven JAX fix improves the exact-cap DeepEP column on every row we remeasured
    - but it does **not** make JAX DeepEP broadly “keep up” with Megatron DeepEP on the high-token regime that motivated `#3717`
  - at the most relevant large-token points:
    - `131072, topk=2`: JAX moved from `3.82M` to `5.81M tok/s`, which is a real gain but still far short of the old sealed Megatron anchor `12.71M tok/s`; the new `2`-iteration Megatron rerun happened to land near parity here, but that run is noisy enough that I would not treat that single near-match as stronger evidence than the broader pattern
    - `262144, topk=2`: JAX moved from `3.82M` to `5.90M tok/s`, still clearly below both the old sealed Megatron anchor `15.83M tok/s` and the new direct Megatron rerun `13.00M tok/s`
    - `topk=8` remains clearly behind at both `131072` and `262144`
- Caveat:
  - the user requested only `2` measurement iterations, and the Megatron side is visibly noisier under that setting than the JAX side, especially on the smaller-token rows; I trust the directional conclusion and the large-row separation more than the exact small-row ratios from this reduced rerun.
- Next action:
  - treat this as a confirmed “large partial fix, not full closure” result and keep the next optimization tranche focused on the post-`w13` residual in `current` / exact-cap:
    - ring communication
    - synchronization
    - overlap
