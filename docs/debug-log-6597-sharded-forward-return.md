# Debugging log for 6597 sharded semantic forward return

Diagnose nonfinite output from `forward_return_expert_major_prepacked_sharded_compare`.

## Initial status

Target H100 run reported finite JAX reference output but Pallas sharded return
output with large nonfinite counts in both `y` and `route_by_slot`.

## Hypothesis 1

The local sharded return call receives `route_y_expert` with destination
dimension `1`, but `_source_push_semantic_forward_return_expert_major_pallas_call`
used the global destination count from `token_ids` for its Pallas grid. That
made each device launch programs for non-local destinations, indexing past the
local route buffer and computing out-of-range global destination IDs.

## Changes

Changed the Pallas return grid to use `route_y_expert.shape[0]`, the local
destination count. Added a CPU-interpreted regression that passes a single local
destination with `dst_offset=1` and compares the result to the independent JAX
reference with only that destination populated.

## Results

Focused semantic W2 Pallas tests pass locally:
`uv run --package marin-levanter --group test pytest --tb=short -q lib/levanter/tests/grug/test_source_push_semantic_w2_pallas.py`
reported `8 passed`.

## Future work

- [ ] Re-run the target H100 sharded return compare under babysitting to confirm
      the nonfinite counters are zero at production shape.
