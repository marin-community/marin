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
