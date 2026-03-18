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
