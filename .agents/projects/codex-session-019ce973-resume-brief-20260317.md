# Codex Session `019ce973-283f-7a20-8a01-972be62bf497`: Resume Brief

Checkpoint note:

- This brief was added after the frozen session so the recovery state is preserved in the research branch instead of only in local Codex artifacts.
- The branch-state section below describes the pre-checkpoint recovery snapshot, not the post-checkpoint git status after this brief was committed.

## Purpose

This brief reconstructs the exact state of the frozen Codex session `019ce973-283f-7a20-8a01-972be62bf497` so the research thread can be resumed without re-deriving context.

The main goals are:

- identify the exact branch and worktree state at the end of the session
- separate trusted measurements from misleading intermediate ones
- explain why the benchmark harness changed twice in the last ten minutes of work
- state the next experiment the agent was logically about to run
- give a concrete resume command and a safer fallback if the same pod instability returns

## Executive Summary

The session had already moved past the earlier "does JAX DeepEP help at all?" question. By the end, it was in a new root-cause thread focused on why JAX DeepEP was still much slower than Megatron DeepEP on the same H100x8 fixed-shape MoE regime, especially at large token counts.

The decisive late-session finding was that the first phase-decomposition ladder was partially confounded: the local-assignment probe kernels were timing inflated uncapped buffers rather than the exact-cap shapes used by the real fast path. The agent fixed that in two commits, reran the stage ladder, and recovered two trustworthy exact-cap numbers before the session stopped producing useful output:

- `deepep_transport_assignments_identity`: `46,402,960.98 tok/s`
- `deepep_transport_first_ragged_dot_probe`: `15,392,690.83 tok/s`

That changed the conclusion materially. The residual gap was no longer credibly in transport or basic assignment packing. It had moved into or beyond the first local expert compute stage.

The session then degraded into compaction/retry churn. After the last real technical output, later turns kept completing with `last_agent_message=null`, so there is no hidden saved conclusion after the exact-cap rerun.

## Authoritative Artifacts

### Primary local evidence

- Session transcript:
  - `/Users/romain/.codex/sessions/2026/03/13/rollout-2026-03-13T16-06-04-019ce973-283f-7a20-8a01-972be62bf497.jsonl`
- Global prompt history:
  - `/Users/romain/.codex/history.jsonl`
- Research worktree:
  - `/Users/romain/marin-wt/moe-jax-megatron-gap-root-cause`
- Experiment issue draft:
  - `/Users/romain/marin-wt/moe-jax-megatron-gap-root-cause/.agents/projects/moe-jax-megatron-gap-root-cause-issue.md`
- Research logbook:
  - `/Users/romain/marin-wt/moe-jax-megatron-gap-root-cause/.agents/logbooks/moe-jax-megatron-gap-root-cause.md`
- Main benchmark file patched in the final phase:
  - `/Users/romain/marin-wt/moe-jax-megatron-gap-root-cause/lib/levanter/scripts/bench/bench_moe_hillclimb.py`

### Sensitive artifact

There is also a shell snapshot for this session under `~/.codex/shell_snapshots/`, but it contains exported secrets. Do not quote or circulate it. This brief intentionally does not depend on it.

## Branch And Worktree State At Recovery Time

As reconstructed on March 17, 2026:

- worktree: `/Users/romain/marin-wt/moe-jax-megatron-gap-root-cause`
- branch: `research/moe-jax-megatron-gap-root-cause`
- upstream: `origin/research/moe-jax-megatron-gap-root-cause`
- branch divergence: none visible; local `HEAD` matches `origin`
- dirty files:
  - `.agents/logbooks/moe-jax-megatron-gap-root-cause.md`

Recent commits:

```text
03df173ef (HEAD -> research/moe-jax-megatron-gap-root-cause, origin/research/moe-jax-megatron-gap-root-cause) Time DeepEP probes with precomputed exact caps
1403fc03d Cap DeepEP probe kernels to real local shapes
8106c1897 Seed JAX vs Megatron DeepEP gap experiment baseline
```

Interpretation:

- The end-of-session code changes are not lost.
- Both late fixes were committed and pushed.
- The only unsaved local state in the worktree is the logbook, and that logbook does not contain the final exact-cap findings.

## Research Context At Freeze Time

This was no longer the earlier reintegration thread. It was a new root-cause experiment for issue `#3752`, driven from:

- issue draft:
  - `.agents/projects/moe-jax-megatron-gap-root-cause-issue.md`
- logbook:
  - `.agents/logbooks/moe-jax-megatron-gap-root-cause.md`

The explicit question in that issue draft was:

> why does JAX DeepEP still underperform Megatron DeepEP on the same fixed-shape MoE block regime, especially as global tokens scale up?

The issue draft preserved the sealed baseline:

```text
tokens=131072, topk=2: jax_deepep=3.82M tok/s vs megatron_deepep=12.71M tok/s
tokens=262144, topk=2: jax_deepep=3.82M tok/s vs megatron_deepep=15.83M tok/s
```

So the new thread was not asking whether DeepEP helped JAX relative to JAX `current`. That had already been established. The remaining task was attribution of the residual JAX-vs-Megatron gap.

## Fixed Comparison Cell The Session Settled On

By the end, the session had standardized on one authoritative "large-gap" cell for the forward decomposition:

- hardware: H100x8 on CoreWeave
- `tokens=131072`
- `hidden=2048`
- `mlp_dim=768`
- `experts=128`
- `topk=2`
- `shared_expert_dim=0`
- `distribution=random`
- `bench_pass=forward`
- `ep-list=8`
- dtype: `bfloat16`
- capacity factor: `1.25`

This cell mattered because:

- it removed the shared-expert mismatch from the JAX side
- it still showed a large JAX-vs-Megatron gap
- it was small enough to iterate quickly on staged probes

## Trusted Findings Before The Late Probe Fixes

These findings were already solid before the exact-cap rerun and should be treated as part of the stable context.

### 1. Shared experts were real but not the main cause

From the local logbook:

```text
shared_expert_dim=0:
  current: 0.037423 s, 3,502,411.86 tok/s
  deepep_transport_capped_prewarmed: 0.030084 s, 4,356,856.14 tok/s

sealed shared_expert_dim=2048 baseline:
  current: 0.040598 s, 3,228,500.34 tok/s
  deepep_transport_capped_prewarmed: 0.034272 s, 3,824,436.67 tok/s
```

Interpretation:

- removing the shared-expert branch improved JAX `current` by about `1.08x`
- removing the shared-expert branch improved JAX DeepEP by about `1.14x`
- this is meaningful, but nowhere near enough to explain the much larger JAX-vs-Megatron delta

### 2. The residual gap already existed in forward

From the local logbook:

```text
shared_expert_dim=0, forward:
  current: 0.013146 s, 9,970,631.70 tok/s
  deepep_transport_capped_prewarmed: 0.010900 s, 12,025,297.24 tok/s

comparison point from sealed Megatron baseline:
  megatron_deepep forward: 0.005116 s, about 25.62M tok/s
```

Interpretation:

- JAX DeepEP forward on the shared-free large-gap cell was still about `2.13x` slower than Megatron DeepEP forward
- so the gap was not "just backward"

This `12.03M tok/s` number is important because it anchors later probe interpretation.

## Timeline Of The Final Working Phase

All times below are local Pacific time on March 16, 2026 unless stated otherwise.

### 16:51:39 to 16:51:50 PDT - The first staged ladder pointed at post-transport compute

Saved assistant commentary:

```text
The phase ladder has reached the middle of the pipeline. The first hard evidence is already useful: transport-only is very fast, while the gate/up stage alone is already slower than Megatron’s full forward on the same cell, so the gap is moving decisively into post-transport JAX compute.
```

Then more explicitly:

```text
The gap is not "DeepEP transport on JAX"; the ladder is showing transport-only at ~64.7M tok/s, while the gate/up probe alone is down at ~2.74M tok/s on a cell where Megatron DeepEP forward is ~25.6M tok/s.
```

Status of those numbers:

- useful directionally
- later superseded quantitatively because the staged local-assignment probes were not yet exact-cap

### 17:00:54 PDT - First "clean" staged number before the confound was fully recognized

Saved assistant commentary:

```text
The first clean stage number is in: transport plus assignment packing/collapse/combine is ~17.4M tok/s on the large-gap cell.
```

Status:

- directionally useful
- later superseded by exact-cap timing

### 17:03:17 PDT - First ragged-dot probe looked catastrophically bad

Saved assistant commentary:

```text
The first ragged-dot number is in, and it is essentially the same as the gate probe: ~2.73M vs ~2.74M tok/s. That means the first local expert GEMM is already where the forward path falls off a cliff; the gate/up elementwise stage is basically noise on top of it.
```

Status:

- directionally useful
- quantitatively invalid for attribution after the shape confound was found

### 17:04:22 to 17:05:48 PDT - The agent recognized the probe-shape confound

Saved commentary before editing:

```text
The probes need to share the exact-cap dispatch/packing setup that the real fast path uses; otherwise they keep timing an intentionally oversized local-assignment shape and cannot answer the residual-gap question.
```

Then:

```text
I’m patching the probe kernels now so they use the same exact-cap dispatch and local-assignment limits as the real capped DeepEP path.
```

The first patch became:

- commit: `1403fc03d`
- timestamp: `2026-03-16 17:05:48 PDT`
- message: `Cap DeepEP probe kernels to real local shapes`

What that commit did:

- added `max_recv_tokens` to the DeepEP transport identity path
- added `max_recv_tokens` and `max_local_assignments` to:
  - `deepep_transport_assignments_identity`
  - `deepep_transport_first_ragged_dot_probe`
  - `deepep_transport_gate_probe`
  - `deepep_transport_second_ragged_dot_probe`
- threaded those exact-cap limits into the local dispatch and assignment-packing helpers

Why it mattered:

- the earlier probe kernels were timing inflated local buffers
- that made the ladder overstate the cost of local compute after transport

### 17:06:13 to 17:06:56 PDT - First exact-cap rerun launched from commit `1403fc03d`

Saved commentary:

```text
The probe-cap patch is pushed. I’m rerunning a short capped breakdown bundle now so I can get the whole exact-cap stage ladder in one pod and stop paying setup on every single probe.
```

The transcript captured the pod metadata:

```text
POD_NAME=iris-task-e8fc72427ad9
TASK_ID=jax-gap-breakdown-capped-fwd-t131072-topk2-20260316-170732
REPO_REF=1403fc03dd0d3c80819e2f25b09c0d1b1ab63290
DEEPEP_REF=7febc6e25660af0f54d95dd781ecdcd62265ecca
```

Operational note:

- the launcher fetches the repo via GitHub tarball for the specific ref
- this is why the agent committed and pushed before rerunning
- uncommitted local edits would not have reached the pod

### 17:09:03 to 17:09:44 PDT - First exact-cap patch failed under JIT tracing

The rerun did not finish cleanly. It failed with:

```text
jax.errors.TracerArrayConversionError
...
max_recv_tokens, max_local_assignments = _deepep_transport_exact_caps(...)
...
num_tokens_per_rank_host = np.asarray(jax.device_get(num_tokens_per_rank), dtype=np.int32)
```

Saved assistant interpretation:

```text
The patch surfaced a clean boundary: exact-cap counts cannot be computed inside the jitted `_forward` path because they call `device_get` on traced values.
```

Then:

```text
I’m fixing the probe-cap implementation properly now: the probes will take precomputed caps from the timing layer, not compute them inside a traced function.
```

### 17:10:31 PDT - Second and final harness fix

The second patch became:

- commit: `03df173ef`
- timestamp: `2026-03-16 17:10:31 PDT`
- message: `Time DeepEP probes with precomputed exact caps`

What that commit did:

- removed exact-cap computation from the jitted probe forward path
- added a new helper:
  - `_time_deepep_transport_probe_forward(...)`
- computed `max_recv_tokens` and `max_local_assignments` once in that timing helper
- passed the precomputed caps into the probe functions as explicit arguments
- routed the forward-only probe kernels through that new timing helper from `main()`

This was the final code state of the session.

### 17:10:58 to 17:12:18 PDT - Second exact-cap rerun launched from commit `03df173ef`

Saved commentary:

```text
The capped probe timing path is fixed and pushed. I’m rerunning the exact same short breakdown bundle now; this one should finally measure the real exact-cap stages instead of either oversized probes or traced-cap failures.
```

The transcript captured the second pod metadata:

```text
POD_NAME=iris-task-dd804ef6b0ff
TASK_ID=jax-gap-breakdown-capped2-fwd-t131072-topk2-20260316-171515
REPO_REF=03df173ef7dfb5d31320a7aba9659a7f659fe9bc
DEEPEP_REF=7febc6e25660af0f54d95dd781ecdcd62265ecca
```

This is the key run to resume from conceptually.

## Last Trustworthy Measurements Recovered From The Corrected Exact-Cap Run

These are the most important numbers recovered from the session transcript.

### Exact-cap metadata for the authoritative cell

Raw transcript:

```text
DEEPEP_EXACT_CAPS max_recv_tokens=30976 max_local_assignments=33024 recv_factor=4.231405 assign_factor=7.937984
```

Interpretation:

- receive buffers were still inflated by about `4.23x` relative to the average token load
- local-assignment buffers were still inflated by about `7.94x`
- this matches the broader earlier exact-cap diagnosis: local assignment shape inflation was the dominant structural overprovision

### Exact-cap stage 1: transport plus assignment pack/collapse/combine

Raw transcript:

```text
RESULT kernel=deepep_transport_assignments_identity ep=8 pass=forward time_s=0.002825 tokens_per_s=46402960.98
```

Saved assistant interpretation:

```text
The first exact-cap probe landed, and it is a major shift: transport plus capped assignment packing/collapse/combine is now ~46.4M tok/s on the large-gap cell. That pushes the residual JAX-vs-Megatron gap decisively past transport and basic packing.
```

Why this mattered:

- it invalidated the earlier `~17.4M tok/s` assignment-stage reading as a quantitative attribution result
- it showed that transport plus basic packing was not the limiting stage on this cell

Useful comparisons:

- `46.40M tok/s` is about `3.86x` the measured JAX full forward throughput on the same shared-free cell (`12.03M`)
- `46.40M tok/s` is about `1.81x` the Megatron forward comparison point (`25.62M`)

This does not mean those phases are "free", but it does mean they are no longer a credible explanation for the full residual gap.

### Exact-cap stage 2: first ragged-dot probe

Raw transcript:

```text
DEEPEP_EXACT_CAPS max_recv_tokens=30976 max_local_assignments=33024 recv_factor=4.231405 assign_factor=7.937984
RESULT kernel=deepep_transport_first_ragged_dot_probe ep=8 pass=forward time_s=0.008515 tokens_per_s=15392690.83
```

This number was not followed by a saved assistant interpretation before the session degraded, but its significance is straightforward:

- it is much slower than `46.40M tok/s`, so the first local expert matmul reintroduces a large cost
- it is far faster than the earlier uncapped `~2.73M tok/s` probe, so the earlier cliff was partly an artifact of oversized local shapes
- it is still well below the Megatron forward comparison point (`25.62M tok/s`)
- it is still above the measured full JAX forward throughput on the same cell (`12.03M tok/s`)

Practical implication:

- the first ragged GEMM is now a plausible major contributor
- but it is not, by itself, enough to explain the entire gap between the `15.39M` first-ragged probe and the `12.03M` full JAX forward
- therefore the remaining exact-cap stages still mattered

That is exactly why the next logical experiment was to finish the exact-cap ladder.

## What Was And Was Not Trustworthy At The End

### Trustworthy

- branch `research/moe-jax-megatron-gap-root-cause` at `03df173ef`
- worktree state showing only the logbook as dirty
- shared-expert mismatch result
- shared-free forward-only JAX result (`12.03M tok/s`)
- sealed Megatron forward comparison point (`~25.62M tok/s`)
- exact-cap metadata:
  - `max_recv_tokens=30976`
  - `max_local_assignments=33024`
  - `recv_factor=4.231405`
  - `assign_factor=7.937984`
- exact-cap `deepep_transport_assignments_identity` result:
  - `46,402,960.98 tok/s`
- exact-cap `deepep_transport_first_ragged_dot_probe` result:
  - `15,392,690.83 tok/s`

### Directionally useful but quantitatively superseded

- uncapped transport-only reading: `~64.7M tok/s`
- uncapped assignment-stage reading: `~17.4M tok/s`
- uncapped gate probe reading: `~2.74M tok/s`
- uncapped first-ragged probe reading: `~2.73M tok/s`

Reason:

- those values were produced before the probe kernels were forced to use the same exact-cap shapes as the real capped path
- they are still useful for understanding how the agent changed its hypothesis
- they should not be used as final attribution numbers

### Not recovered before the session stopped producing real output

From the second exact-cap rerun, the transcript did not preserve final interpreted results for:

- `deepep_transport_gate_probe`
- `deepep_transport_second_ragged_dot_probe`
- `deepep_transport_capped_prewarmed`
- `current`

Those were the missing pieces.

## The Most Likely Immediate Next Interpretation

Based on the recovered numbers, the agent was almost certainly about to say something like this:

1. The earlier "first ragged dot is catastrophic at ~2.7M" conclusion was too pessimistic because it was confounded by uncapped local shapes.
2. After correcting the probe shapes, transport and basic packing became too fast to explain the residual gap.
3. The first ragged dot remained slower than Megatron's full forward, so the residual gap was now plausibly centered in:
   - the first ragged GEMM itself
   - later local expert compute
   - or remaining full-path overhead between the probe boundary and the actual end-to-end forward
4. The next measurement needed was the rest of the exact-cap ladder, not another new hypothesis branch.

## The Next Experiment The Session Was Logically About To Run

### Best continuity experiment

Resume the corrected exact-cap ladder on the same authoritative shared-free forward cell, starting with the stages that were still missing from the corrected run:

- `deepep_transport_gate_probe`
- `deepep_transport_second_ragged_dot_probe`
- `deepep_transport_capped_prewarmed`
- `current`

Why this is the right next step:

- `46.40M` already rules out transport/basic packing as the dominant residual bottleneck
- `15.39M` for the first ragged dot says the first local expert matmul is significant, but not sufficient to explain the final `12.03M` end-to-end forward throughput
- the missing corrected stages are exactly what would tell whether the remaining throughput loss sits mostly in:
  - gate/up elementwise work
  - second ragged GEMM
  - combine/full-path overhead
  - or "everything after first ragged dot" more broadly

### Exact command to resume the same breakdown bundle

This is the command pattern the session had already launched successfully from commit `03df173ef`:

```bash
KUBECONFIG=/Users/romain/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3677.yaml \
  --task-id jax-gap-breakdown-capped2-fwd-t131072-topk2-20260316-171515-rerun \
  --tokens 131072 \
  --shared-expert-dim 0 \
  --kernels deepep_transport_gate_probe,deepep_transport_second_ragged_dot_probe,deepep_transport_capped_prewarmed,current \
  --topk-list 2 \
  --distributions random \
  --bench-pass forward \
  --ep-list 8 \
  --warmup 1 \
  --iters 3 \
  --per-bench-timeout-seconds 420 \
  --per-bench-kill-after-seconds 20 \
  --build-with-torch-extension \
  --load-as-python-module \
  --skip-smoke \
  --skip-cleanup
```

Run this from:

- `/Users/romain/marin-wt/moe-jax-megatron-gap-root-cause`

### Safer fallback if the teardown noise makes the bundle unreliable again

Split the missing kernels into two shorter pods:

Pod A:

```bash
KUBECONFIG=/Users/romain/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3677.yaml \
  --task-id jax-gap-gate-seconddot-fwd-t131072-topk2-rerun \
  --tokens 131072 \
  --shared-expert-dim 0 \
  --kernels deepep_transport_gate_probe,deepep_transport_second_ragged_dot_probe \
  --topk-list 2 \
  --distributions random \
  --bench-pass forward \
  --ep-list 8 \
  --warmup 1 \
  --iters 3 \
  --per-bench-timeout-seconds 420 \
  --per-bench-kill-after-seconds 20 \
  --build-with-torch-extension \
  --load-as-python-module \
  --skip-smoke \
  --skip-cleanup
```

Pod B:

```bash
KUBECONFIG=/Users/romain/.kube/coreweave-iris uv run .agents/scripts/deepep_jax_krt_bench.py \
  --config lib/iris/examples/coreweave-moe-jax-3677.yaml \
  --task-id jax-gap-fullpath-controls-fwd-t131072-topk2-rerun \
  --tokens 131072 \
  --shared-expert-dim 0 \
  --kernels deepep_transport_capped_prewarmed,current \
  --topk-list 2 \
  --distributions random \
  --bench-pass forward \
  --ep-list 8 \
  --warmup 1 \
  --iters 3 \
  --per-bench-timeout-seconds 420 \
  --per-bench-kill-after-seconds 20 \
  --build-with-torch-extension \
  --load-as-python-module \
  --skip-smoke \
  --skip-cleanup
```

Rationale:

- the pod logs repeatedly showed heavy XLA/CUDA teardown noise after a stage finished
- shorter pods reduce the chance of losing later-stage results in the same bundle
- the cost is slightly more setup overhead, but the session had already learned that setup overhead was preferable to losing a whole ladder late in the pod

## Operational Notes That Matter For Any Resume

### 1. Commit and push before changing the harness again

The launcher does not use uncommitted local files. It downloads a GitHub tarball for the specified ref. So if `bench_moe_hillclimb.py` changes again:

- commit the change
- push the branch
- rerun from the pushed ref

### 2. Expect teardown noise that is not the benchmark result

The exact-cap rerun logs contained large blocks of:

- `CUDA_ERROR_LAUNCH_FAILED: unspecified launch failure`
- stream destroy failures
- event destroy failures
- memory free failures during cleanup

Important nuance:

- those errors appeared after real stage outputs were already printed
- the bundle also showed `bench_status=0` before advancing to the next kernel
- so the session did not interpret the teardown tail itself as the primary benchmark signal

### 3. The logbook is incomplete relative to the transcript

The current uncommitted logbook stops after:

- shared-expert mismatch
- shared-free forward split

It does not contain:

- commit `1403fc03d`
- commit `03df173ef`
- the exact-cap rerun metadata
- the recovered `46.40M` and `15.39M` numbers

If this thread is resumed seriously, the first housekeeping step should be to append those findings to the logbook so future sessions do not need to reconstruct them from the JSONL.

## Session Failure Pattern After The Last Real Result

The session did not end with a final technical conclusion. Instead it degraded into retries, interrupted turns, and silent completions.

### User-side prompts captured in global history

Relevant prompts after the last substantive benchmark work:

- `2026-03-16 17:20:11 PDT`
  - `Every time you experimence a compaction error, simply retry. Don't forget that.`
- `2026-03-16 17:25:32 PDT`
  - `Try again`
- `2026-03-16 17:36:46 PDT`
  - `Compaction worked, resume where you left off.`
- `2026-03-16 17:42:58 PDT`
  - `Resume where you left off`
- `2026-03-16 17:48:23 PDT`
  - `What's going on?`
- `2026-03-16 17:48:59 PDT`
  - `Resume where you left off.`
- `2026-03-16 18:26:53 PDT`
  - `Resume where you left off`
- `2026-03-16 18:35:19 PDT`
  - `Resume where you left off`
- `2026-03-17 09:41:07 PDT`
  - `Resume where you left off`

### Explicit interrupted turns captured in the session transcript

- `2026-03-16 17:36:35 PDT`
  - `turn_aborted`
  - turn id: `019cf939-01aa-7020-8c97-4efe7a5d6472`
- `2026-03-16 17:48:27 PDT`
  - `turn_aborted`
  - turn id: `019cf943-e83c-7d52-93d7-865e120bc052`

### Silent completions that indicate the session was no longer producing useful output

After the exact-cap results, the transcript showed repeated `task_complete` events with `last_agent_message=null`. That included:

- `2026-03-16 17:19:38 PDT`
- `2026-03-16 17:25:17 PDT`
- `2026-03-16 17:30:37 PDT`
- `2026-03-16 17:35:59 PDT`
- `2026-03-16 17:41:52 PDT`
- `2026-03-16 17:48:04 PDT`
- `2026-03-16 17:54:05 PDT`
- `2026-03-16 18:18:55 PDT`
- `2026-03-16 18:31:59 PDT`
- `2026-03-16 18:40:25 PDT`
- `2026-03-17 09:46:14 PDT`

The strongest end-state signature is not "the last turn never completed." It is more specific:

- later turns did complete
- but they completed without any recoverable assistant message

That means there is no evidence of a hidden final conclusion after the recovered exact-cap numbers.

## Practical Resume Checklist

1. Start from worktree `/Users/romain/marin-wt/moe-jax-megatron-gap-root-cause`.
2. Verify branch is still `research/moe-jax-megatron-gap-root-cause` at `03df173ef`.
3. Do not discard the dirty logbook; it is the only local unsaved research artifact.
4. Before new experimentation, append the recovered exact-cap findings to the logbook.
5. Resume with the missing exact-cap stages on the same authoritative cell:
   - `gate_probe`
   - `second_ragged_dot_probe`
   - `deepep_transport_capped_prewarmed`
   - `current`
6. If pod instability recurs, split the run into two shorter pods rather than a six-kernel bundle.
7. If you edit the harness again, commit and push before relaunching because the pod fetches code by repo ref.

## Bottom Line

The final state of the frozen session was not ambiguous.

The branch tip already contained the final harness fix. The last trustworthy numerical conclusion was that exact-cap transport plus basic packing ran at `46.40M tok/s`, while the first exact-cap ragged-dot probe ran at `15.39M tok/s` on the shared-free `tokens=131072, topk=2` forward cell. That pushed the real remaining gap past transport/basic packing and into or beyond the first local expert compute stage.

The session froze before it could finish the corrected ladder. The next correct move is to finish that ladder, not to restart the investigation from scratch.
