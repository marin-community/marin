# Spec: MoE + loss token-axis policy under a context-sharded sequence (hero model)

**Branch:** `long-context/moe-context-sharding` (worktree
`~/projects/marin.long-context-moe-context-sharding`)
**Item:** "The MoE path has no defined policy for activations sharded over both expert/data
axes and a context axis" — from `.agents/projects/20260824_535b_long_context_cooldown.md` and
#8374. **Do not open PRs/issues or post anything to GitHub. Commit locally on this branch only;
do not push.** First step: `git merge long-context/mesh-context-axis` (local branch; provides
the `context` mesh axis and extends `sharding.py:_batch_axes` to include it — see its spec).

## Policy (the decision this branch encodes)

When the sequence dim of `[B, S, D]` activations is sharded over `context`, the flattened
`(b s) d` token axis is sharded over the composite tuple
`("replica_dcn", "data", "expert", "context")`. Each context shard routes its own tokens
independently: MoE collectives keep `axis_name="expert"` and therefore run per context slice
(CP multiplies the number of independent routing shards, shrinking tokens per shard — exactly
the dropping risk #8374 wants measured). Capacity is computed from local shard token counts as
today. Reductions over "all tokens" (loss, QB beta, drop accounting) must psum/pmean over the
full token-axis tuple including `context`.

## Prior art (this is largely a port)

Commit `b2cf4fe1aa` on `origin/june_tpu_67b_a2b` implemented exactly this policy for the TPU
variant `experiments/grug/moe/model.py`: `_seq_axis` / `_token_axes` / `_seq_spec_3d` /
`_seq_spec_4d` / `_token_spec` helpers, `_batch_reshard` → seq-aware, DenseMLP (shared expert)
`out_sharding=_token_spec()` + reshard-after-unflatten comment, QB beta pmean over token axes
with corrected `local_tokens` arithmetic. Read that diff first and mirror its shape. Note its
`_kv_spec_4d`/attention resharding parts belong to the sibling branch
`long-context/fa4-sequence-sharding` — do NOT port those; keep your edits to MoE/loss/metrics
code paths so the branches merge cleanly.

## Scope

Primary target is the production model `experiments/grug/moe_hero_ep/model.py` (+ the shared
library); the TPU variant already has its own version on its branch.

1. **Hero model token-axis helpers** (`experiments/grug/moe_hero_ep/model.py`): port the
   `_seq_axis`/`_token_axes`/`_token_spec`/seq-spec helper pattern; update `MoEMLP.__call__`
   (:962+) flatten/unflatten and reshards, DenseMLP/shared-expert output sharding, and the QB
   beta shard_map (`in_specs=(P(token_axes, None),)`, `pmean` over token axes, `local_tokens =
   T // prod(mesh.shape[a] for a in token_axes)`).
2. **Library MoE** (`lib/levanter/src/levanter/grug/grug_moe.py`, `_moe/*.py`): after the
   mesh-branch merge, `sharding.py:_batch_axes` already includes `context`, so
   `moe_mlp`'s shard_map specs and drop-count psums pick it up. Audit every EP body for
   hidden assumptions that the token axis tuple has exactly its old members:
   - psums/pmeans over `_batch_axes(...)` (`ep_ring.py:116`, `ep_ragged_all_to_all.py:123-124`,
     `ep_fixed_all_to_all.py:180`, `ep_fixed_pooled_wave_all_to_all.py:596-599`,
     `grug_moe.py:337-339`) — must reduce over `context` too;
   - `_batch_spec_from_x` (`sharding.py:54-59`) reads `x.sharding.spec[0]` of the fused token
     axis — confirm a `("replica_dcn","data","expert","context")`-sharded token axis round-trips;
   - capacity arithmetic in `ep_fixed_pooled_wave_all_to_all.py` (pool/receiver capacities,
     wave striping, interleaved receiver ranks) derives from local shapes — confirm no global
     token-count constant leaks in.
3. **Loss**: `lib/levanter/src/levanter/grug/loss.py` fused CE psums numerator/denominator over
   the batch-axis names (:151-155) — must include `context` (it should inherit via
   `_batch_axes`; verify and test). Same for router z-loss aggregation if it reduces over
   tokens.
4. **Metrics**: `_summarize_router_metrics` (`moe_hero_ep/model.py` analog of
   `moe/model.py:486-506`) and `_drop_metrics` (`moe_hero_ep/train.py:475-505`) use
   `total_assignments = batch * seq * top_k * layers` — still correct globally; verify the
   summed drop counters are global (post-psum) not per-shard, so rates stay comparable across
   CP degrees.
5. **Data loader**: extend the hero train.py batch `axis_resources` mapping only if required
   for correctness of the flatten (tokens arrive `[B, S]` sharded batch-only today; the model
   reshards). Prefer leaving loader untouched and resharding in-model like the TPU port did.

## Out of scope

Attention backends and attention-boundary reshards (sibling branches), mesh construction
(merged in), launcher flags, changing capacity-factor defaults, any quality tuning.

## Verification

- Unit tests (CPU, `XLA_FLAGS=--xla_force_host_platform_device_count=8`): with mesh
  `context_axis_size=2` (and expert>1 where feasible):
  - routed-MoE forward parity: identical inputs through the local backend on a context=1 mesh
    vs context=2 mesh, outputs match (tolerance per TESTING.md guidance); include top-k>1 and
    a case with drops (capacity_factor small) checking global drop counts match;
  - loss parity: fused CE identical unsharded vs context-sharded;
  - QB beta parity: pmean'd beta identical across CP degrees for the same token population;
  - grad parity on a small end-to-end layer (shared+routed) — silent psum omissions show up
    only in gradients.
  - Vacuous-pass check: each parity test must fail when the corresponding psum/spec is
    deliberately broken.
- `uv run pyrefly check`; `./infra/pre-commit.py --all-files --fix`; `git add -f` for new
  files under `lib/` (gitignore trap); `uv run --no-project infra/ci/run_tests.py` passes.

## Risks

- A missing `context` in any psum silently under-reduces (wrong loss normalization, wrong beta,
  under-counted drops). Gradient/metric parity tests across CP degrees are the mitigation.
- The `expert` axis's dual role (EP collective axis AND member of the token-axis tuple) is
  preserved by this policy; document it where the tuple is defined so reviewers don't "fix" it.
- Shared-checkout hazard: other sessions write to `/home/marin/projects/marin` — work only in
  this worktree and verify staged content before committing.
