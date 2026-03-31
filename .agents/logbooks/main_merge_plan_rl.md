# RL Merge Plan for `origin/main`

Date: 2026-03-30

Status: merge in progress on branch `iris_rl`

## Problem

We attempted to merge `origin/main` into `iris_rl` after landing the Iris RL
checkpointing and logbook work. The merge stopped with four conflicts:

1. `lib/marin/src/marin/rl/rl_job.py`
2. `lib/marin/src/marin/rl/curriculum.py`
3. `lib/marin/src/marin/rl/scripts/test_llama_small.py`
4. `uv.lock`

Two of these are mechanically uninteresting. Two are real RL architecture
conflicts.

The central question is not "how do we clear conflict markers?" It is:

How do we merge `main` without regressing the Iris RL migration back toward the
old Fray v1 / implicit-actor / inline-launcher model?

This note is the answer.

## Context

### Branch split

- `iris_rl` diverged from `main` at merge base `62ca4ad827844db5c900ad8e7f45d2b146d49050`
- Since that split, the substantive RL changes on `iris_rl` were all authored by
  Ahmed Ahmed
- Since that split, the fresh conflicting RL-side touch on `origin/main` is
  almost entirely one repo-wide refactor:
  - `bae3fe109` by Russell Power
  - subject: `[rigging] Extract shared utilities from iris into new rigging package (#3245)`

### Architectural source of truth

The forward-looking RL architecture is already expressed in stable, non-conflict
files:

- `lib/marin/src/marin/rl/orchestration.py:4`
  - states that RL uses a Fray v2 coordinator job with child jobs underneath it
- `lib/marin/src/marin/rl/orchestration.py:80`
  - `submit_rl_job()` submits a single coordinator job
- `lib/marin/src/marin/rl/orchestration.py:109`
  - `_run_rl_coordinator()` creates runtime actors, worker environments, and
    child jobs inside the cluster
- `lib/marin/src/marin/rl/orchestration.py:158`
  - worker resources now honor slice count, region, zone, and host RAM
- `lib/marin/src/marin/rl/runtime.py:25`
  - `RLRuntimeHandles` is the contract for explicit runtime handle passing
- `lib/marin/src/marin/rl/runtime.py:30`
  - explicitly says workers should never call `get_default_job_ctx()` or
    discover actors by name

That is the design we should preserve through the merge.

## Goals

- Keep the RL migration on the Fray v2 / Iris coordinator path
- Avoid reintroducing Fray v1 cluster launching in RL
- Avoid reintroducing implicit actor discovery by name
- Take safe repo-wide refactors from `main`, especially the `rigging` package
  extraction
- Resolve the merge in a way that future agents can understand and defend

## Non-goals

- Backwards compatibility with the old RL launcher shape
- Preserving dead or legacy helper scripts if they no longer match the mainline
  RL architecture
- Making this merge a full RL cleanup sweep beyond the actual conflicts
- Resolving every remaining `iris.*` helper import in the repo during the merge

## What Changed on Each Side

### What `iris_rl` changed

In the conflicted RL files, `iris_rl` introduced the migration-critical changes:

- moved `RLJob.run()` away from inline `current_cluster().launch(...)`
- added the Fray v2 coordinator submission path
- introduced stable `run_id` plus volatile `instance_id`
- added region / zone / host-RAM aware worker resource controls
- removed Fray v1 from the core RL pipeline
- removed implicit curriculum actor discovery from the core RL path

These are not cosmetic edits. They are the migration.

### What `origin/main` changed

`origin/main` brought in two different kinds of changes:

1. A legitimate repo-wide refactor:
   - shared helpers like `marin_fs` moved from `iris.*` to `rigging.*`
   - this is a good change and should be absorbed where applicable

2. Legacy RL code that still reflects the older execution model:
   - `fray.v1.cluster.current_cluster()`
   - inline TPU worker launching in `RLJob.run()`
   - `get_default_job_ctx()` in curriculum actor creation

Those RL semantics should not be revived.

## Conflict Inventory

| File | Conflict type | Significance | Planned resolution |
|---|---|---|---|
| `lib/marin/src/marin/rl/rl_job.py` | semantic architecture conflict | high | keep `iris_rl` logic, do not restore v1 launcher path |
| `lib/marin/src/marin/rl/curriculum.py` | mixed import refactor + old actor helper | medium | keep `iris_rl` behavior, but switch filesystem import to `rigging.filesystem` |
| `lib/marin/src/marin/rl/scripts/test_llama_small.py` | modify/delete | low | keep deleted |
| `uv.lock` | lockfile churn | low | take `main` or regenerate after merge |

## Proposed Solution

### Core rule

Use `iris_rl` as the source of truth for RL architecture. Use `origin/main` as
the source of truth for generic package moves and lockfile normalization.

In other words:

- keep RL semantics from `iris_rl`
- take generic helper relocation from `main`
- do not mix in old launcher behavior just because it arrives through a merge

### Why this is the right rule

The current codebase already documents the intended direction:

- `lib/marin/src/marin/rl/runtime.py:29-30`
  - runtime handles are passed explicitly
  - workers should not discover actors implicitly
- `lib/marin/src/marin/rl/orchestration.py:80-106`
  - the coordinator is the unit of submission
- `lib/marin/src/marin/rl/orchestration.py:189-220`
  - trainer and rollout workers are child jobs beneath that coordinator

If we took `main`'s `rl_job.py` wholesale, we would directly contradict those
files and regress the migration.

## File-by-File Resolution Plan

### 1. `lib/marin/src/marin/rl/rl_job.py`

#### Decision

Keep the `iris_rl` version of the file's behavior.

#### What to preserve from `iris_rl`

- `from fray.v2 import JobHandle`
- `load_tokenizer(...)` instead of `AutoTokenizer.from_pretrained(...)`
- support for `PackedvLLMInferenceContextConfig`
- `RunConfig.train_ram`
- `RunConfig.inference_ram`
- `RunConfig.regions`
- `RunConfig.zone`
- `RLJobConfig.instance_id`
- `RLJobConfig.resolved_instance_id`
- `RLJob.run()` delegating to `submit_rl_job(...)`

These are all aligned with the v2 coordinator design and are already consumed
by `orchestration.py`.

#### What not to restore from `main`

- `fray.v1.cluster.Entrypoint`
- `EnvironmentConfig`
- `JobRequest`
- `ResourceConfig`
- `current_cluster()`
- inline `TrainWorker` / `RolloutWorker` TPU launching inside `RLJob.run()`
- `remove_tpu_lockfile_on_exit()` and `configure_logging()` inside the old
  inline launcher
- `AutoTokenizer.from_pretrained(...)` as the main tokenizer loader

Those belong to the older RL execution path. They are not compatible with the
direction already embodied in `orchestration.py`.

#### One optional thing from `main`

`main` adds `RunConfig.env_vars: dict[str, str]`.

This is the only part of `main`'s `RunConfig` that looks potentially useful and
not inherently regressive. But it is not currently required for the merge:

- current orchestration already constructs explicit worker envs in
  `lib/marin/src/marin/rl/orchestration.py:134-156`
- there is no existing RL call path depending on `RunConfig.env_vars`

Recommendation:

- do not complicate the merge by half-porting `env_vars`
- if desired, add it later in a small follow-up patch with explicit wiring and
  tests

#### Intended shape after resolution

```python
from fray.v2 import JobHandle
from levanter.compat.hf_checkpoints import load_tokenizer

def run(self, name: str) -> JobHandle:
    from marin.rl.orchestration import submit_rl_job

    handle = submit_rl_job(self.config)
    handle.wait(raise_on_failure=True)
    return handle
```

That is the architectural heart of the file. Keep it.

### 2. `lib/marin/src/marin/rl/curriculum.py`

#### Decision

Keep `iris_rl` semantics, but adopt the `rigging.filesystem` import move.

#### What to preserve from `iris_rl`

- no `get_or_create_curriculum_actor(...)`
- no `get_default_job_ctx()` in the core RL path
- `micro_eval_frequency: int | None`
- the current full-eval semantics and documentation aligned with trainer-step
  based triggering

The `micro_eval_frequency: int | None` shape matches current rollout code:

- `lib/marin/src/marin/rl/rollout_worker.py:421`
  - accepts `micro_eval_frequency: int | None`
- `lib/marin/src/marin/rl/rollout_worker.py:425`
  - explicitly handles `None`

So `main`'s attempt to narrow that field back to plain `int` is not a clear
forward improvement.

#### What to take from `main`

Take the filesystem helper import rename:

```python
from rigging.filesystem import url_to_fs
```

This is a real repo-wide package move, not an RL design disagreement.

#### What not to restore from `main`

- `from fray.v1.job import get_default_job_ctx`
- `get_or_create_curriculum_actor(...)`

This would directly violate the explicit runtime-handle rule in
`lib/marin/src/marin/rl/runtime.py:29-30`.

#### Intended shape after resolution

```python
import numpy as np
from rigging.filesystem import url_to_fs

class Curriculum:
    ...
```

No implicit actor lookup helper should return to this file.

### 3. `lib/marin/src/marin/rl/scripts/test_llama_small.py`

#### Decision

Keep the file deleted.

#### Why

`main`'s surviving copy is still rooted in older assumptions:

- `RayConfig`
- direct `TrainWorker` / `RolloutWorker` imports
- older local test-launch patterns

This file is not part of the production RL control plane. Reviving it during
this merge would add noise and preserve stale execution assumptions.

If a small RL smoke script is still useful later, it should be recreated in a
way that matches the v2 coordinator model rather than carried forward
accidentally through a modify/delete merge.

### 4. `uv.lock`

#### Decision

Take `main`'s side or regenerate after merge.

#### Why

This conflict is mechanical. The specific hunk is the `dupekit` asset hash.

Current `pyproject.toml:27` already points to:

- `dupekit-0.1.0-40ac799`

That matches `origin/main`, not the older `iris_rl` lock entry.

So the consistent outcome is either:

- accept `main`'s `uv.lock`, or
- regenerate the lockfile from the merged dependency state

There is no RL-specific reason to preserve the old lock entry.

## Why We Should Not "Just Take Main"

Because `main` is not merely newer. It is newer in one dimension and older in
another.

It is newer on:

- package extraction to `rigging.*`
- lockfile and dependency churn

It is older on:

- RL orchestration model
- actor-discovery model
- tokenizer loading choice
- worker submission path

If we "just take main" for `rl_job.py` and `curriculum.py`, we would be
silently undoing the migration while keeping its surrounding docs and helper
layers. That would create a structurally inconsistent codebase:

- `runtime.py` would say "workers never call `get_default_job_ctx()`"
- `curriculum.py` would do exactly that
- `orchestration.py` would define the coordinator topology
- `rl_job.py` would bypass it and launch workers directly with Fray v1

That is the worst possible merge outcome: conflict markers gone, architecture
regressed, inconsistency hidden.

## Implementation Outline

1. Resolve `rl_job.py` in favor of `iris_rl` semantics, only considering small
   non-semantic salvage from `main`
2. Resolve `curriculum.py` in favor of `iris_rl` semantics while switching
   `url_to_fs` to `rigging.filesystem`
3. Resolve `test_llama_small.py` by keeping the deletion
4. Resolve `uv.lock` by taking `main` or regenerating from the merged
   dependency state
5. Run `make fix`
6. Run a targeted RL validation pass to ensure the merged imports and RL
   configuration surface still work

## Validation Notes

After conflict resolution, validation should answer two questions:

1. Did we preserve the intended architecture?
2. Did we absorb the safe repo-wide refactor cleanly?

Minimum checks:

- `make fix`
- import / compile sanity for RL modules
- targeted RL tests touching config and orchestration surfaces

Suggested focused checks:

- `lib/marin/src/marin/rl/rl_job.py` imports cleanly
- `lib/marin/src/marin/rl/curriculum.py` imports cleanly
- no remaining accidental conflict markers in RL files
- no accidental reintroduction of `get_default_job_ctx()` in the core RL path

Useful grep after merge:

- search RL for `from iris.marin_fs`
- search RL for `get_default_job_ctx()`
- search RL for `current_cluster()`

The expected result is:

- generic filesystem helpers migrate toward `rigging.filesystem`
- the core RL pipeline remains on the explicit-runtime / coordinator path

## Notes

### On `rigging.filesystem`

This is not an RL invention. It is the new canonical home for the old
`iris.marin_fs` helpers:

- `url_to_fs`
- `open_url`
- `marin_prefix`
- `marin_region`
- `REGION_TO_DATA_BUCKET`

The move came from `bae3fe109` and is effectively a package relocation, not a
semantic redesign of filesystem behavior.

So we should adopt it where the helper is truly generic.

### On `iris.*` imports

Not every `iris.*` import should be purged blindly.

Examples:

- `iris.logging` may still be appropriate for Iris-specific logging behavior
- truly generic storage helpers should move to `rigging.filesystem`

The merge should not become a random search-and-replace campaign. It should
apply the package move where it is clearly correct.

### On historical authorship

The conflicting old RL code was not authored only by one person historically,
but the recent migration-side divergence is clean:

- your recent branch-side RL changes in the conflicted files are yours
- the fresh `main`-side collision is mostly Russell's `rigging` extraction

That is useful because it means this merge is not reconciling two simultaneous
competing RL redesigns. It is reconciling:

- one current RL migration branch
- one repo-wide infrastructure refactor

## Future Work

- add `RunConfig.env_vars` later if we actually want user-configurable worker
  env injection in the v2 path
- do a separate cleanup pass for remaining RL imports that still use
  `iris.marin_fs` where `rigging.filesystem` is more appropriate
- decide whether a replacement for `test_llama_small.py` is still valuable under
  the v2 coordinator architecture
- once the merge lands, document the resolved policy briefly in
  `.agents/logbooks/iris-rl-codex.md` so future agents do not have to rediscover
  this reasoning

## Final Recommendation

Merge `main` into `iris_rl` with this policy:

- preserve the RL v2 / Iris coordinator architecture from `iris_rl`
- absorb the `rigging.filesystem` package move from `main`
- keep stale RL helper scripts deleted
- treat `uv.lock` as mechanical

For the two real RL conflicts, there is no good forward-looking reason to take
`main`'s semantics over the branch's current design.

The right merge is not "main wins because it is newer."

The right merge is:

`main` for generic refactors, `iris_rl` for RL architecture.
