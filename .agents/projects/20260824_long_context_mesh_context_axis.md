# Spec: `context` axis in the Grug mesh + sequence-aware partition helpers

**Branch:** `long-context/mesh-context-axis` (worktree `~/projects/marin.long-context-mesh-context-axis`)
**Item:** "Grug's compact mesh contains only `(replica_dcn, data, expert, model)`" — from
`.agents/projects/20260824_535b_long_context_cooldown.md` (branch `docs/535b-long-context-summary`)
and #8374. **Do not open PRs/issues or post anything to GitHub for this work. Commit locally on
this branch only; do not push.**

## Prior art (use it)

Commit `b2cf4fe1aa` on `origin/june_tpu_67b_a2b` ("grug moe: add context-parallel via new
'context' mesh axis + 262k launcher") already implemented this for the TPU 67B variant, and a
262K CP4 TPU run trained to completion with it. Its `lib/levanter/src/levanter/grug/sharding.py`
hunks are the reference implementation. Port those hunks (plus the `GrugTrainerConfig.context_axis_size`
plumbing pattern from the same commit and `4c98e7a918`) onto current `main`. Beware the
backport hazard: `git apply` can land hunks in the wrong function when context lines repeat —
verify each hunk landed on the intended symbol.

## Pinned interface (consumed by sibling branches — do not rename)

- Axis name: `"context"`.
- Mesh order: `("replica_dcn", "data", "context", "expert", "model")` — matches the trained TPU
  prior art. `expert` stays innermost-but-model so EP collectives (the dominant traffic) remain
  contiguous device runs; a CP≤4 group is still confined to one NVL72 rack at one-rack scale.
- `compact_grug_mesh(..., context_axis_size: int = 1)` — keyword-only, validated positive,
  folded into the divisibility check; `data` remains the residual axis. Length-1 axes are kept,
  so specs may name `"context"` unconditionally.
- `_batch_axes(mesh)` in `sharding.py` becomes the flat-token-axis tuple: subset of
  `("replica_dcn", "data", "expert", "context")` present in the mesh (exactly as in
  `b2cf4fe1aa`, including its docstring explaining token-space psums). Consumers that need
  batch-dim-only axes keep their own local `_BATCH_AXES` tuples (they already do).
- Sibling branches: `long-context/fa4-sequence-sharding` (FA4/CuTe CP), `long-context/te-cp-backend`
  (TE backend), `long-context/moe-context-sharding` (hero-model token-axis policy). They will
  `git merge long-context/mesh-context-axis` — keep this branch minimal and stable.

## Scope

1. Port the `sharding.py` diff of `b2cf4fe1aa`: `_GRUG_MESH_AXIS_NAMES`,
   `_compact_grug_mesh_shape`, `compact_grug_mesh`, `_batch_axes` (current file:
   `lib/levanter/src/levanter/grug/sharding.py:43-170`).
2. Add `context_axis_size: int = 1` to each trainer config that owns mesh sizing and thread it
   to `compact_grug_mesh`: `experiments/grug/moe/train.py` (`GrugTrainerConfig`, `_run_grug_local`),
   `experiments/grug/moe_hero_ep/train.py` (both mesh builds, :694 and the dropless-eval mesh
   :763), `experiments/grug/moe_hero_fsdp/train.py:512`, `experiments/grug/base/train.py:448`,
   `experiments/grug/recovery/train.py:89`, `experiments/june_tpu_67b_a2b/moe/train.py:515`.
   Default 1 everywhere; no launcher flag changes in this branch.
3. Audit remaining `compact_grug_mesh` call sites (tests, vllm backends,
   `experiments/grug/moe/standalone/compile_probe.py`) — defaults keep them working; fix any
   code that hard-codes the 4-tuple of axis names or builds specs positionally
   (search `_GRUG_MESH_AXIS_NAMES`, `mesh.axis_names`, axis-count asserts).
4. Audit consumers of `sharding.py:_batch_axes`/`_batch_spec`/`_batch_spec_from_x`
   (`grug_moe.py`, `loss.py`, FA4's separate `_BATCH_AXES` in `_fa4_cute.py`) and confirm
   behavior is unchanged when `context_axis_size == 1` (a size-1 axis in a spec is a no-op).
   Note `experiments/grug/moe/model.py`'s local `_mesh_axis_size` raises on a *missing* axis —
   after this change all meshes built by `compact_grug_mesh` carry `context`, but meshes built
   by tests/other tools may not; keep helpers tolerant of absent axes (as prior art does).

## Out of scope

Attention backends, MoE/loss/model changes (sibling branches), launcher flags, any behavior
change at `context_axis_size == 1`.

## Verification

- Unit tests (CPU, `XLA_FLAGS=--xla_force_host_platform_device_count=8`): mesh shape/order for
  several `(replica, context, expert, model)` combinations; divisibility error mentions
  context; `_batch_axes` with and without a context axis; default-args mesh preserves today's
  device order (reshaped with the extra size-1 dim). Read `TESTING.md` and the nearest module
  TESTING.md first; extend `tests/test_grug_variant_contracts.py` the way `b2cf4fe1aa` did if
  that fits better than a new file.
- `uv run pyrefly check`; `./infra/pre-commit.py --all-files --fix`. The repo-global gitignore
  silently excludes new files under `lib/` — use `git add -f` and verify with `git status`.
- `uv run --no-project infra/ci/run_tests.py` must pass.

## Risks

- Checkpoint sharding metadata: confirm whether saved shardings serialize mesh axis names such
  that a 4-axis-mesh checkpoint restores onto the 5-axis mesh (the hero 6k restore will do
  exactly this). Read `experiments/grug/moe_hero_ep/train.py:856-891` (`_init_restore_template`,
  `restore_template_from`) and document the finding in the code or tests. This is the one place
  a "no behavior change" claim could silently fail.
- QB-beta local-token arithmetic (`experiments/grug/moe/model.py:601-605`) divides by
  `prod(mesh.shape[a] for a in token axes)`; at size 1 it is unchanged, but confirm in the audit.
