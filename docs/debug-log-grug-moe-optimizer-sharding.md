# Debugging log for Grug MoE optimizer sharding

Find where optimizer intermediates for expert 3D weights may lose their named
sharding and collapse the `expert` axis, causing large unexpected allocations.

## Initial status

The branch already had a low-noise `NamedSharding.spec` comparison helper in
`experiments/grug/moe/optimizer_sharding.py`. It fails specifically when 3D
expert leaves lose `expert` on the leading pspec axis, and also catches general
pspec drift.

## Hypothesis 1

AdamH and MuonH intermediates can drift after an internal update boundary even
if final updates are resharded before application.

## Changes made

Added assertion coverage after AdamH's `mu` dtype cast and around MuonH's
hyperball helper after its input sharding match and after the hyperball update.

## Results

`uv run pytest experiments/grug/moe/test_optimizer.py` passed.

## Future work

- [ ] If a run trips one of these assertions, add a minimal reshard at the
      exact failing boundary rather than adding broader resharding.

## Hypothesis 2

Grouped MuonH updates can be restored to each FSDP leaf and immediately consumed
by `optax.apply_updates`, avoiding a simultaneously live FSDP-shaped update tree
and reducing grouped-to-FSDP boundary collectives.

## Changes made

Added `expert_fsdp_grouped_updates_muonh_direct_apply` to
`experiments/grug/moe/muon_update_bench.py`. The benchmark computes grouped
MuonH in the NS-friendly layout, restores each split update to the corresponding
FSDP param sharding, and applies it directly. Added focused lowering coverage in
`experiments/grug/moe/test_muon_update_bench.py`.

## Results

The focused grouped-MuonH tests passed. A tiny forced-host runtime compile for
L4/G4/R2D2E2M1 showed direct apply and the existing restore-then-apply path both
compiled to two all-to-alls. Direct apply was slightly slower on that tiny host
case, so this harness is useful evidence but not yet a performance win.

## Future work

- [ ] If testing on CoreWeave, compare direct apply, restore-then-apply, and
      persistent grouped apply in one job and require compiled HLO collective
      counts plus timing before declaring the boundary improved.
