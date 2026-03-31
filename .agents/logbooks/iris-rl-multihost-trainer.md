# Iris RL Multihost Trainer Logbook

Source threads:
- [iris-rl-codex.md](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/.agents/logbooks/iris-rl-codex.md)
- [iris-rl-claude.md](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/.agents/logbooks/iris-rl-claude.md)

Purpose:
- keep a focused history for the `v6e` / multi-host RL trainer thread
- isolate trainer-side weight export, multi-host Arrow Flight, and trainer sharding questions
- avoid further clutter in the main Iris RL migration logbook

## Scope

Current question:

- why does the RL trainer/export path fail or become memory-tight on multi-host
  `v6e-16` while ordinary Levanter multi-host training is expected to work?
- what is the right path to make `v6e-16` trainer + `v6e-8` rollout stable with
  per-step weight sync?
- if needed, how should a future `v6e-32` layout be partitioned so that it
  actually reduces per-chip export pressure?

Hard constraints carried into this thread:

1. Weight sync stays at every trainer step.
2. Cross-region artifact access is forbidden in practice; model, checkpoints,
   and rollout storage must be region-local.
3. Root CPU coordinator may run anywhere; TPU children may need explicit zone
   pinning.
4. We care about the RL trainer/export path, not just steady-state Levanter
   training in isolation.

## What Was Already Solved Before The Multihost Thread

Carry-over from [iris-rl-claude.md](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/.agents/logbooks/iris-rl-claude.md):

- The original Iris slowdown was mostly rollout/eval cadence, not trainer
  compute:
  - `r4` was slow because `n_prompts=16` caused `4` rollout batches per trainer
    step, and `eval_frequency=1` accidentally triggered `4` full evals per
    trainer step.
- Matching the on-demand rollout shape with `n_prompts=64` fixed the major
  slowdown:
  - `e4par` reached about `2.3 min/step`
  - `e4ms2` with `2` rollout workers reached about `1.7 min/step`
- Retry/resume was fixed by splitting stable run identity from per-attempt
  instance identity:
  - checkpoints resume
  - trainer W&B resumes
  - rollout storage resumes

That matters here because the multihost trainer thread is **not** trying to
rediscover the old Iris performance bug. It is a later, narrower thread about
multi-host trainer/export correctness on `v6e`.

## Why The `v6e` Thread Started

The original motivation was to reproduce the late trainer failure from the long
`v5p-8` RL run on a different TPU generation and zone.

The first blocker was regional artifacts:

- the old launcher hardcoded `gs://marin-us-central1`
- east-coast TPU workers then hit Iris cross-region transfer-budget limits

This led to the region-aware launcher introduced in
[xp_iris_rl_regression_direct_gcs_prod.py](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/experiments/xp_iris_rl_regression_direct_gcs_prod.py),
which derives:

- model path from `--region`
- checkpoint prefix from `--region`
- rollout prefix from `--region`

This was required before any real `v6e` investigation could even start.

## Multihost Attempt Chronology

### Attempt A: `v6e-8` trainer + `v6e-8` rollout

Relevant run family:

- `/ahmed/iris-rl-e4ms2-500-0328-v6e8-e1d-v2`

Observed result:

- region-local artifacts worked
- trainer failed immediately in Arrow Flight bootstrap weight export

Observed failure:

```text
RESOURCE_EXHAUSTED: Error loading program 'jit_copy_and_flatten':
Attempting to reserve 9.62G at the bottom of memory.
That was not possible. There are 7.14G free.
```

Interpretation:

- this was not generic "8B cannot run on v6e"
- it was specifically trainer-side weight export pressure during
  `copy_and_flatten(...)`

### Attempt B: `v6e-16` trainer + `v6e-8` rollout, initial multi-host bring-up

Relevant run family:

- `/ahmed/iris-rl-v6e-e1d-0328`
- `/ahmed/iris-rl-v6e-e1d-0328-v2`

Observed result:

- JAX multi-host init succeeded
- trainer hit barrier timeouts during weight serve

Initial diagnosis was wrong:

- the mesh config itself was not the real blocker

Correct diagnosis from [iris-rl-claude.md](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/.agents/logbooks/iris-rl-claude.md):

- only process `0` was doing `copy_and_flatten(...)` and `device_get(...)`
- those operations trigger cross-host JAX collectives on a multi-host trainer
- other processes skipped ahead to the post-serve barrier
- result: classic distributed deadlock

Fix that landed:

- all trainer processes participate in the collective part of `serve_weights()`
- only process `0` does the Arrow Flight serving / coordinator update

This was the first major multi-host-specific RL trainer/export bug.

### Attempt C: multi-host materialization bug after the collective fix

Relevant debug note:

- [debug-log-v6e-multihost-weight-transfer-materialization.md](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/docs/debug-log-v6e-multihost-weight-transfer-materialization.md)

Observed result:

- the trainer could get past the original deadlock
- but weight serving still failed when the code tried to `device_get` a
  non-addressable global array directly

Fix that landed:

- per-leaf host materialization now uses
  `multihost_utils.process_allgather(..., tiled=True)` for non-fully-addressable
  global arrays
- fully addressable leaves still use `np.asarray(...)`

This moved the trainer/export path forward again.

### Attempt D: rollout-side `v6e-8` hot-reload failure

Relevant runs:

- `/ahmed/iris-rl-v6e-e1d-0328-v4`
- `/ahmed/iris-rl-v6e-e5b-0328-v4`

Observed result:

- trainer bootstrap serve worked
- rollout received bootstrap weights
- rollout then died in `reload_model -> sync_weights`

Observed failure shape:

- `jax.device_put -> _multi_slice -> RESOURCE_EXHAUSTED`
- failure landed while handling the `lm_head` path

Interpretation:

- not generic inference fit failure
- hot-reload on live `v6e-8` rollout workers needed more contiguous HBM than
  the default cache reservation left available

### Attempt E: rollout bootstrap fix on `v6e-8`

Relevant probes:

- `ROL-V6E8-002`: TP=`8`, `gpu_memory_utilization=0.90`
- `ROL-V6E8-003`: TP=`8`, `gpu_memory_utilization=0.60`

Outcome:

- TP=`8` alone did not fix the rollout-side failure
- TP=`8` plus `gpu_memory_utilization=0.60` did

Key result:

- rollout bootstrap on `v6e-8` can work
- so `v6e-8` rollout is viable if cache pressure is reduced

### Attempt F: trainer export on `v6e-16` is still on the HBM cliff

Carry-over from
[iris-rl-codex.md](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/.agents/logbooks/iris-rl-codex.md):

- once rollout bootstrap was fixed, the trainer-side export path became the main
  remaining pressure point
- allocator misses matched full bf16 leaf sizes rather than steady-state shard
  sizes:
  - `112 MiB` = MLP projection
  - `32 MiB` = attention `q_proj` / `o_proj`
  - `1002 MiB` = `lm_head`

The important point is that these warnings matched the export path, not ordinary
training compute.

## Current Understanding Of The Trainer Side

### The current `v6e-16` trainer is not "obviously sharded wrong"

From the quantitative audit in
[iris-rl-codex.md](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/.agents/logbooks/iris-rl-codex.md):

- trainer mesh currently resolves to `data=16`, `model=1`, `context=1`
- large transformer weights carrying the `embed` axis already get a `16x`
  steady-state shard factor through `embed -> data`

Estimated bf16 sizes:

- full model transfer payload: about `14.96 GiB`
- MLP projection leaf: `112.00 MiB`
- attention `q_proj` / `o_proj`: `32.00 MiB`
- `lm_head`: `1002.00 MiB`

Ideal steady-state shard sizes on current `data=16`:

- MLP projection: about `7.00 MiB`
- attention `q_proj` / `o_proj`: about `2.00 MiB`
- `lm_head`: about `62.62 MiB`

Implication:

- the main export problem is not that the steady-state trainer shard is too big
- the main export problem is that the export path appears to amplify leaves back
  toward their full unsharded bf16 sizes during serialization/materialization

### Why Levanter multi-host training working does not contradict this bug

Ordinary multi-host Levanter training and RL trainer weight export are not the
same operation.

Ordinary training path:

- keeps the model sharded
- does forward/backward/all-reduce style distributed compute
- does **not** require eagerly walking the whole model and materializing a full
  host-visible transfer tree on one process

RL trainer export path:

- must serialize model state for rollout workers every step
- goes through Haliax state-dict conversion and host materialization
- is therefore much more sensitive to:
  - eager Python iteration over sharded arrays
  - host materialization semantics
  - temporary bf16 export buffers
  - collective participation rules

So "Levanter multi-host training works" and "our RL export path is broken on
multi-host" are completely compatible statements.

## The Two Distinct Multi-Host Export Bugs

### 1. Old `tree_jit` path: distributed-safe enough, but memory-tight

Current baseline path:

```python
@jax.jit
def copy_and_flatten(model, convert_to_bfloat16=True):
    state_dict = hsd.to_state_dict(model)
    ...
```

Properties:

- after the collective-participation fix, this path is compatible with the
  multi-host trainer
- it got far enough to prove trainer bootstrap serving and later trainer/export
  progress
- but it appears to create large temporary export buffers and sits on the HBM
  cliff

### 2. New `sequential_host_flatten` path: lower-memory idea, but currently invalid on sharded trainers

Experimental path added during the export-pressure investigation:

```python
def _export_weights_sequential_host_flatten(model, config):
    state_dict = hsd.to_state_dict(model)
    for name, leaf in state_dict.items():
        ...
```

Observed failure on both TPU probes:

- `/ahmed/iris-rl-v6e-e5b-exportdbg-001`
- `/ahmed/iris-rl-v6e-e1d-exportdbg-001`

Exact failure:

- bootstrap `serve_weights(-1, state.model)` entered the sequential path
- `hsd.to_state_dict(model)` ran eagerly on the concrete sharded trainer model
- Haliax scan `_unstack_state_dict(...)` tried to do:
  - `for i, v_i in enumerate(v):`
- JAX raised:

```text
AssertionError
assert self.is_fully_replicated or self.is_fully_addressable
```

Why this happens:

- the scanned-layer unstack path is iterating a concrete global `jax.Array`
- on a multi-host trainer, those leaves are not fully addressable from one
  process
- eager Python iteration over such arrays is illegal

Why the old `tree_jit` path did not fail the same way:

- `copy_and_flatten(...)` wraps the same `hsd.to_state_dict(model)` call in
  `jax.jit`
- that means the state-dict walk happens under JAX tracing rather than as eager
  Python iteration over concrete global arrays
- so the current sequential experiment did **not** discover that multi-host
  Levanter is broken; it discovered that our new eager serialization variant
  violated an assumption that the jitted path had been hiding

## Did We Actually Test RL On Multi-Host Before This?

Yes, but only partially and only after several fixes.

What had already been exercised:

- multi-host trainer initialization on `v6e-16`
- multi-host Arrow Flight collective participation after the deadlock fix
- multi-host-safe per-leaf materialization after the non-addressable-array fix
- trainer bootstrap serving under the older `tree_jit` export path

What had **not** been validated before this new experiment:

- eagerly calling `hsd.to_state_dict(model)` on the concrete sharded trainer and
  then iterating it leaf-by-leaf in Python

So the precise answer is:

- multi-host RL trainer/export had been tested enough to expose and fix two real
  bugs already
- the new `sequential_host_flatten` path introduced a third, different,
  multi-host-specific bug

## Current Jobs / Signals

The two short export-debug TPU probes are both dead:

- `/ahmed/iris-rl-v6e-e5b-exportdbg-001`
- `/ahmed/iris-rl-v6e-e1d-exportdbg-001`

Both failed in the same place:

- first bootstrap serve
- inside `_export_weights_sequential_host_flatten(...)`
- before the experiment could even test whether the new path improves the old
  trainer allocator misses

This means the current blocker is now sharper:

- `tree_jit` path: correctness is much better, but it is memory-tight
- `sequential_host_flatten` path: memory idea is interesting, but the current
  implementation is invalid for multi-host sharded trainers

## Focused Debug Plan

### Phase 1: isolate the semantic difference between the two export paths

Goal:

- prove exactly why `tree_jit` works far enough to run while eager
  `sequential_host_flatten` dies immediately

Planned work:

1. Add a focused regression test around the two paths on a scanned model:
   - `copy_and_flatten(...)` under `jax.jit`
   - eager `hsd.to_state_dict(model)` on the same sharded/scanned structure
2. Capture the smallest reproducer that still hits:
   - Haliax scan `_unstack_state_dict(...)`
   - non-fully-addressable eager array iteration
3. Decide whether the safe path is:
   - keep all state-dict conversion inside JIT
   - or avoid Haliax eager state-dict conversion entirely

Key snippet to debug:

```python
state_dict = hsd.to_state_dict(model)
for name, leaf in state_dict.items():
    ...
```

versus:

```python
@jax.jit
def copy_and_flatten(model, convert_to_bfloat16=True):
    state_dict = hsd.to_state_dict(model)
    ...
```

### Phase 2: recover a distributed-safe low-peak export design

Goal:

- reduce trainer export peak memory without leaving the distributed-safe
  semantics of the old path

Candidate directions:

1. Preserve the old jitted `copy_and_flatten(...)` front half, but stop doing
   whole-tree host materialization / serialization afterward.
2. Derive a jitted flat export tree first, then:
   - materialize leaves sequentially on host
   - Arrow-serialize leaves sequentially
   - discard each leaf before moving to the next
3. Avoid eager `hsd.to_state_dict(model)` on the concrete trainer object
   entirely.

Non-goal:

- do **not** reduce `sync_interval_steps`

### Phase 3: re-measure the old trainer HBM pressure once the path is valid again

Goal:

- get back to the original question: does lower-peak export remove the old
  `32 MiB` / `112 MiB` trainer allocator misses?

Planned probe shape:

- trainer: `v6e-16`
- rollout: `v6e-8`
- rollout TP: `8`
- rollout `gpu_memory_utilization`: `0.60`
- low eval cadence
- weight-transfer debug on

Success criteria:

- bootstrap `serve_weights(-1)` succeeds
- at least a few later step-level serves succeed
- export metrics show lower peak materialization pressure
- old allocator misses either disappear or become much rarer

### Phase 4: only then revisit `v6e-32`

Goal:

- decide whether `v6e-32` is needed, and if so, whether it helps for the right
  reason

Important rule:

- a larger slice only helps if the trainer layout actually reduces per-chip
  state and per-chip export buffer size

Work to do before any `v6e-32` claim:

1. compute per-chip shard sizes for the major leaves under a candidate
   `v6e-32` layout
2. compare those with current `v6e-16`
3. verify whether the export path itself still recreates near-full-leaf buffers

If the export path is still materializing near-full bf16 leaves, `v6e-32`
without a real repartition may not be the right fix.

## Pointers To Relevant Detailed Notes

- [debug-log-v6e-multihost-weight-transfer-materialization.md](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/docs/debug-log-v6e-multihost-weight-transfer-materialization.md)
- [debug-log-v6e-rollout-sync-oom.md](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/docs/debug-log-v6e-rollout-sync-oom.md)
- [debug-log-iris-rl-e4ms2-500-final-failure.md](/Users/ahmed/code/marin/.claude/worktrees/precious-squishing-melody/docs/debug-log-iris-rl-e4ms2-500-final-failure.md)
