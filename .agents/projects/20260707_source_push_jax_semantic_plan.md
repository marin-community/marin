# Source-Push JAX Semantic Plan Sketch

This note describes the slot-free source-push metadata model. Slots, inbox
semaphores, and queue rounds are lowering details layered below this plan.

## Current Implementation Status

- JAX semantic metadata/reference paths exist and are used as the correctness
  boundary inside the benchmark harness.
- Source-driven expert-major W13 and direct x-pack Pallas kernels use explicit
  Mosaic GPU WGMMA / GMEM loads. The `/dlwh/iris-run-bench_source_push_semantic_plan-20260709-013856`
  retry proved that plain JAX `w13_expert_major_pack` costs `0.029065911s`,
  essentially explaining the prior integrated saved-x fwd+bwd gap. Direct
  Pallas x-pack now writes expert-major rows directly and `/014800` proved that
  it runs at `0.022163098s`, improving full saved-x direct-pack fwd+bwd to
  `0.073667202s` (`104.998798` useful / `131.248497` rounded TFLOP/s/rank).
  After fixing the semantic bench W13-backward row-block default to the
  WGMMA-safe value, `/dlwh/bench_source_push_semantic_plan-20260709-direct-pack-decomp-current`
  measured current direct-pack fwd+bwd at `0.070555656s` (`109.629307`
  useful / `137.036633` rounded TFLOP/s/rank). A follow-up direct x-pack tile
  sweep `/dlwh/semantic-direct-pack-tile-sweep-20260709-033100` found
  `gather_row_block=16`, `gather_hidden_block=512` at `0.006939658s`, a
  `3.1877x` isolated x-pack speedup over the old `0.022121539s` row. The
  integrated/decomposition check
  `/dlwh/bench_source_push_semantic_plan-20260709-direct-pack-h512-integrated`
  succeeded with integrated fwd+bwd `0.055450854s` (`139.492310` useful /
  `174.365387` rounded TFLOP/s/rank), a `21.408%` full-path reduction over the
  prior `0.070555656s` baseline. Treat direct Pallas x-pack with the H512 tile
  as the current production-relevant path.
- Source-driven W13 remains a correctness/performance idea, but current
  Pallas/Mosaic Lane+Warpgroup lowering blocks the in-kernel source gather to
  WGMMA SMEM. H100 retries moved through these blockers:
  `.at[row_lane]` scatter, 1-row WGMMA SMEM store tile size, `stack`,
  `dynamic_update_slice`, iota layout inference, captured non-scalar constants,
  `concatenate`, helper `memref<8xi32>` transfer minimum, non-tiled layout
  slicing, and WGMMA layout shape incompatibility (`(8, 128)` is not legal for
  WGMMA tiling). Stop spending blind retries on this exact lowering. Resume only
  with a structural design change, such as a legal 64-row source tile, a
  separate pack kernel feeding WGMMA, or a Pallas/Mosaic feature that supports
  the needed row assembly.
- Expert-major W2 forward uses explicit WGMMA. The masked standalone path is
  correct but slow; the production semantic path uses the assume-zero-invalid
  W2 variant only when upstream W13 has already zeroed invalid H rows.
  A separate W2-to-compact-pair fork produced a correct isolated direct writer
  (`route_y_pair_max_abs_diff=6.866455078125e-05`), but its full pair-return
  timing row is not integration-safe: the compare row showed
  `route_y_pair_max_abs_diff=154.81640625` and
  `y_w2_return_only_max_abs_diff=26895.7421875`. Do not promote that full
  pair-return path until a sharded compare validates both `route_y_pair` and
  `y`.
- Expert-major W2 backward now has explicit WGMMA kernels for both
  `dh = dy @ W2.T` and `dw2 = H.T @ dy`. Target H100 isolated validation is
  correct at `0.005404372s`, `317.888377` useful /
  `397.360471` rounded TFLOP/s/rank.
- W13 backward has explicit WGMMA kernels for `dx_route` and `dw13`. The
  bounded H100 tile sweep `/dlwh/semantic-w13-backward-tile-sweep-20260709`
  found the best combined default at `row_block=64`, `hidden_block=256`,
  `output_block=64`, `lowering=warpgroup`, median `0.010037962s`. This is a
  small `0.56%` win over the previous `0.010094299s` default; output block
  `256` regressed badly and should not be promoted. A follow-up structural
  cleanup removed the redundant hidden-sized `dx_route` post-mask by pre-zeroing
  invalid `dz13` rows before WGMMA. H100 job
  `/dlwh/semantic-w13-backward-mask-elide-fixed-20260709T061640Z` measured W13
  backward at `0.009141506s` (`375.865` useful / `469.831` rounded
  TFLOP/s/rank), an `8.93%` W13-backward win. The same job measured integrated
  y-owner fwd+bwd at `0.044225975s` (`174.896` useful / `218.621` rounded),
  a `2.78%` full-path reduction over `0.045490726s`. Promote this W13
  mask-elision path.
- Forward/dx return-combine direct-sum paths are target-correct with fp32
  atomics, but copy-only diagnostics show the cost is primarily the
  atomic/source-scatter combine rather than metadata traversal:
  forward copy-only `0.002276297s` vs direct-sum `0.013127315s`,
  dx copy-only `0.002122613s` vs direct-sum `0.012950074s`.
  Exact non-atomic route-slot materialization is only marginally better than
  atomics and still far above copy-only:
  forward slot-reduce `0.012756550s` vs direct-sum `0.013111857s`,
  dx slot-reduce `0.012593363s` vs direct-sum `0.012955240s`.
  Source-owned reverse metadata
  `[src, token, route_slot] -> (dst, local_expert, expert_row)` exists as
  `source_push_semantic_reverse_route_jax(plan)` and supports source-owned
  gather diagnostics. Forward return source-gather was tested in the integrated
  direct-pack path and is not a promotion candidate: the same v2 job measured
  direct-pack+source-gather fwd+bwd at `0.070875834s`, `0.195ms` slower than
  direct-pack+sum-return (`+0.276%`) with matching checksum. Keep the current
  direct-pack + sum-return path unless a structurally different combine design
  emerges. A follow-up direct source-token wrapper cleanup was tested and
  reverted after target H100 results showed it still hit the sharded-ref slicing
  blocker for the standalone path and remained slower in the integrated path.
- Remaining `pl.dot` usages in the semantic Pallas modules are pair-flat
  scaffold/debug modes. Production-relevant expert-major matmul paths should
  continue to use explicit `mgpu.wgmma`.
- The current integrated saved-x semantic fwd+bwd target row is correct at
  `0.053676363s`, `144.103796` useful / `180.129745` rounded TFLOP/s/rank.
  This is the promoted W13-backward + dx-return result from
  `/dlwh/bench_source_push_semantic_plan-20260709-promoted-w13-dx-integrated`,
  a `3.20%` wall-time reduction over the previous H512 baseline
  `0.055450854s`. The visible isolated stage medians sum to `0.055078326s`, so
  the latest decomposition shows no positive hidden integrated overhead;
  integrated is `0.001401963s` faster than the naive sum of visible isolated
  medians. The current H512 decomposition rows are:
  - forward return sum: `0.013161402s` (return sweep best: `0.012819453s`
    at row block `64`, hidden block `256`; copy-only floor `0.001367112s`)
  - backward W13: `0.010037962s` after promoting `hidden_block=256`,
    `output_block=64`
  - dx return source-gather: `0.006557451s` after promoting
    `row_block=64`, `hidden_block=512`; copy-only floor at the same tile is
    `0.001374360s`
  - direct x-pack: `0.006932546s`
  - W13 prepacked: `0.005900210s`
  - backward W2: `0.005373897s`
  - backward source-expand: `0.004871634s`
  - W2 prepacked zero-invalid: `0.002584281s`
  Source-expand is therefore not the current bottleneck. The immediate
  performance path is improving return-combine structure, rather than chasing
  hidden graph overhead. The forward return tile sweep gave only a small
  `2.58%` production win and still leaves a `~11.45ms` gap to copy-only; the dx
  return sweep gave a larger tile win but still leaves a `~5.18ms` gap to
  copy-only. Further return work probably needs a structural combine change
  rather than more tile tuning. A later isolated dx-return check rejected
  remote dx source-gather as a performance path: target H100 measured normal dx
  source-gather `0.006539781s`, owner-sharded dx source-gather `0.004962953s`,
  remote dx source-gather `0.018706967s`, and dx copy-only `0.001364567s`,
  with remote compare `dx_max_abs_diff=0.0078125`.
- Owner-sharded return is the current structural combine candidate. Standalone
  owner-sharded source-token return improved forward return from
  `0.012826364s` to `0.009512028s` and dx return from `0.006595757s` to
  `0.004988354s`. Early monolithic attempts failed because the W2 output was
  re-constrained to `P('expert', None, None, None)` before the owner-sharded
  return shard_map, producing a `P('expert')` vs `P(None)` mismatch for
  `float32[8,32,4096,2560]`. After removing that extra constraint, target
  integrated split diagnostics succeeded:
  `/dlwh/bench_source_push_semantic_plan-20260709-owner-sharded-split-v2`
  measured direct baseline `0.053716708s` (`143.996` useful /
  `179.994` rounded TFLOP/s/rank), owner-sharded y-return `0.045490726s`
  (`170.034` useful / `212.542` rounded), and owner-sharded dx-return
  `0.049590468s` (`155.977` useful / `194.971` rounded), with no errors,
  drops, or metadata overflow. The combined owner-sharded y+dx run
  `/dlwh/bench_source_push_semantic_plan-20260709-owner-sharded-both`
  measured `0.045818861s` (`168.816` useful / `211.020` rounded), also with no
  production Pallas errors, drops, or metadata overflow. The target-shape
  compare row OOMed in the XLA reference/autotune path, while tiny interpreted
  compare passed locally. With the W13 backward mask-elision patch, y-only
  owner-sharded return remains the current fastest integrated semantic fwd+bwd
  structure at `0.044225975s`; dx-owner does not stack additively in the
  monolithic graph.
- The harness now has aliases for the promoted path:
  `current_best_fwd_bwd` maps to the y-only owner-sharded semantic fwd+bwd
  mode, and `current_best_fwd_bwd_with_metadata` rebuilds the JAX semantic plan
  from `selected_experts` and `route_weights` inside the jitted callable before
  running the same promoted kernels. The row capacity remains a static
  shape-specialization parameter, as required by the expert-major buffers. H100
  job `/dlwh/bench_source_push_semantic_plan-20260709-current-best-with-metadata`
  measured the target-shape overhead of this in-JIT metadata boundary: prebuilt
  metadata `0.044244650s` (`174.823` useful / `218.528` rounded TFLOP/s/rank),
  in-JIT JAX metadata `0.045761111s` (`169.029` useful / `211.287` rounded),
  with no errors, drops, or metadata overflow. The JAX metadata-in-JIT tax is
  therefore `0.001516460s` (`3.43%`) on the target shape. The harness also has
  `current_best_fwd_bwd_with_pallas_metadata`, which uses the existing Pallas
  histogram/scatter metadata builder inside the same fwd+bwd JIT boundary. H100
  job `/dlwh/bench_source_push_semantic_plan-20260709-pallas-metadata-injit` is
  proved that Pallas metadata does not recover the JAX metadata tax: JAX
  metadata-in-JIT remeasured at `0.045691938s` (`169.285` useful /
  `211.606` rounded), while Pallas metadata-in-JIT measured `0.049698620s`
  (`155.637` useful / `194.547` rounded). Keep
  `current_best_fwd_bwd_with_metadata` as the production-relevant
  metadata-in-JIT alias for now. The current-best decomposition replacement
  `/dlwh/bench_source_push_semantic_plan-20260709-current-best-decomp-fixed-inline`
  succeeded; the earlier fixed-env attempt failed before benchmark rows because
  `scratch/` scripts are not staged into the remote task environment.
  The latest current-best decomposition row is `0.044151736s`, `175.191`
  useful / `218.988` rounded TFLOP/s/rank. Current isolated stage medians are
  direct x-pack `0.006940519s`, W13 prepacked `0.005926196s`, W2
  zero-invalid `0.002586844s`, owner-sharded forward return `0.009542040s`,
  source-expand `0.004894563s`, backward W2 `0.005369381s`, backward W13
  `0.009132540s`, and dx source-gather return `0.006552185s`. The largest
  remaining isolated stage is forward return; W2 forward is no longer the
  obvious bottleneck.
  A focused lookup-return experiment added expert-major row lookup metadata
  `(source, token, weight, valid)` and a lookup owner-sharded return-sum kernel
  to remove the per-row source-rank scan. The first H100 run failed lookup modes
  because the metadata remained replicated; changing the metadata constraint to
  `jax.sharding.reshard(...)` fixed the `shard_map` input sharding mismatch.
  The corrected target run
  `/dlwh/bench_source_push_semantic_plan-20260709-lookup-forward-return-reshard`
  measured current owner-sharded return at `0.009500383s` and lookup
  owner-sharded return at `0.009634218s`, with lookup compare
  `y_max_abs_diff=0.0`. This falsifies lookup metadata as a speedup path: the
  forward-return tax is not the source-scan metadata traversal, but the
  atomic/source-token combine structure.
  A follow-up source-owned remote-gather return diagnostic avoids both atomics
  and destination-partial reduction by having each source shard pull remote
  `route_y_expert` rows with `mgpu.remote_ref(...)` and sum `K` locally. H100
  job `/dlwh/bench_source_push_semantic_plan-20260709-remote-source-gather-return`
  measured current owner-sharded return at `0.009515191s`, lookup return at
  `0.009608690s`, and remote source-gather return at `0.004319881s`; the remote
  compare row had `y_max_abs_diff=0.0`. This was the first structural isolated
  forward-return win after owner-sharding, but it does not compose into the
  integrated fwd+bwd graph. Job
  `/dlwh/bench-source-push-semantic-no-y-split-boundary-20260709T151614Z`
  measured owner-sharded y fwd+bwd at `0.044111914s`, no-y fwd+bwd at
  `0.040276926s`, isolated remote-y at `0.004422509s`, and monolithic remote-y
  at `0.047670704s`. The split estimate `0.044699435s` is still `0.5875ms`
  slower than owner-sharded y, so keep owner-sharded forward return in the
  current-best path. Owner-sharded route-slot reduce was subsequently exact but
  slower in isolation (`0.013399676s` versus `0.009531809s`) and is rejected.
  Source-driven W13 has no target timing row because current Pallas/Mosaic
  lowering blocks the needed row assembly.
- Random-routing target reruns exposed a blanket capacity-padding regression:
  all expert-major modes reserved a full extra row tile even though only the
  experimental direct-queue TMA path needs that tail guard. Splitting ordinary
  tile rounding from direct-queue guarding improved the three-repeat prebuilt
  median from `0.052072022s` to `0.051239784s` and metadata-in-JIT from
  `0.053500867s` to `0.052532735s`. The new rows are `150.956` and `147.241`
  useful TFLOP/s/rank respectively, with zero errors, drops, or metadata
  overflow. The metadata-in-JIT tax is now `0.001292952s` (`2.523%`).
- Direct remote return queues are not promotion candidates. An EP8/T4096 random
  full-stage compare localized the first forward failure to `return_y`
  (`151552` max, `2473.827` mean absolute difference); direct dX independently
  corrupts destination ordinals `2`, `3`, `4`, and `6`. Branch lowering,
  per-tile semaphores, aligned reverse-route bases, and an SMEM visibility
  commit did not change the dX signature. Keep direct-y/direct-dX queues as
  diagnostics only.
- Expert-major SwiGLU backward now has a reusable JAX semantic boundary,
  `source_push_semantic_swiglu_backward_expert_major_jax`, with fp32 internal
  math, invalid-row masking, autodiff parity, and pair-flat parity tests. The
  benchmark uses this shared function instead of a local duplicate.

## Metadata

Canonical semantic layout:

```text
assignment_id[s, d, r] = token * topk + route_slot
valid[s, d, r] = assignment_id >= 0
xcounts[s, d, e] = valid row count
pair_expert_base[s, d, e] = exclusive_cumsum_e(xcounts[s, d, :])[e]
rows_per_local_expert[d, e] = sum_s xcounts[s, d, e]
expert_base[d, e] = exclusive_cumsum_e(rows_per_local_expert[d, :])
src_base_by_expert[d, s, e] = exclusive_cumsum_s(xcounts[:, d, e])[s]
pair row interval for expert e:
  pair_r in [pair_expert_base[s, d, e], pair_expert_base[s, d, e] + xcounts[s, d, e])
local_r = pair_r - pair_expert_base[s, d, e]
expert_row[d, s, e, local_r] = expert_base[d, e] + src_base_by_expert[d, s, e] + local_r
```

`assignment_id` is the reverse handle for source-side return/combine:

```text
token = assignment_id // topk
route_slot = assignment_id % topk
```

### Capacity Decision

Use pair-flat capacity per `(source, destination)` rather than fixed capacity
per `(source, destination, expert)`.

Rationale:

- JAX still needs static output shapes, but a per-pair row buffer avoids the
  worst artificial overflow from routing skew across local experts.
- Expert identity does not need a hot-path `expert_id[s, d, r]` array as long as
  rows inside each pair buffer are grouped by local expert and `xcounts` plus
  `pair_expert_base` define the intervals.
- `expert_id[s, d, r]` can remain a debug/reference derived value, but the
  production kernels should recover expert id from the current expert loop or
  expert-group loop.

Overflow accounting is therefore:

```text
routing_dropped_routes:
  routes dropped by receiver-capacity clipping
metadata_overflow_routes:
  routes accepted by router capacity but not stored because
  sum_e xcounts[s, d, e] > rows_per_src_dst_capacity or because
  sum_s xcounts[s, d, e] > rows_per_expert_capacity
```

For exact diagnostics set `rows_per_src_dst_capacity = tokens_per_source *
topk`. For training and target performance use an explicit capacity factor and
report `metadata_overflow_routes` separately from router-capacity drops.

## Forward W13 Source Push

Semantic work items:

```text
send_chunk(s, d, e, send_row_start, send_rows)
compute_w13(d, s, e, compute_row_start, n_tile)
```

Prefer:

```text
send_m = compute_m * compute_blocks_per_send
```

Pseudocode:

```text
for send_chunk in live_chunks(s, d, e):
  pair_rows = pair_expert_base[s, d, e] + send_row_start : ... + send_rows
  tokens = assignment_id[s, d, pair_rows] // topk
  copy x[s, tokens, :] to destination staging[d, s, e, rows, :]
  publish chunk-ready

for compute_chunk in compute_chunks_inside_ready_send_chunk:
  expert_row_start = expert_base[d, e] + src_base_by_expert[d, s, e] + compute_row_start
  for n_tile in W13 output tiles:
    acc_gate, acc_up = 0
    for k_tile in hidden tiles:
      acc_gate += staging[compute_rows, k_tile] @ w13[d, e, k_tile, gate_n_tile]
      acc_up += staging[compute_rows, k_tile] @ w13[d, e, k_tile, up_n_tile]
    z13[d, e, expert_row_start:..., gate_n_tile] = acc_gate
    z13[d, e, expert_row_start:..., up_n_tile] = acc_up
    h[d, e, expert_row_start:..., n_tile] = silu(acc_gate) * acc_up
```

## Forward W2 And Return

```text
for d, e, row_block, n_out_tile:
  y_route = h[d, e, row_block, :] @ w2[d, e, :, n_out_tile]
  optional y_route *= route_weight[s, d, pair_r]
  return y_route to source using assignment_id[s, d, pair_r]

for source s:
  route_y[s, token, route_slot, :] = returned_y
  y[s, token, :] = sum_route route_y[s, token, route_slot, :]
```

If W2 return is preweighted, source combine is only the final sum.

## Backward Source Expansion

Source owns token gradients and route identities:

```text
for s, token, route_slot:
  assignment_id = token * topk + route_slot
  dy_route = dy_token[s, token, :] * combine_weight[s, token, route_slot]
  dcombine[s, token, route_slot] = dot(dy_token[s, token, :], y_route[s, token, route_slot, :])
  route dy_route to (d, e, r) using the same semantic plan
```

## Backward W2

Destination expert-major layout:

```text
h[d, e, row, i]
dy_route[d, e, row, o]
```

Pseudocode:

```text
for d, e:
  dw2[d, e] = h[d, e].T @ dy_route[d, e]
  dh[d, e] = dy_route[d, e] @ w2[d, e].T
```

This is a standard grouped matmul/reduction over expert-major rows.

## SwiGLU Backward

Forward saved:

```text
z_gate[d, e, row, i]
z_up[d, e, row, i]
h = silu(z_gate) * z_up
```

Pseudocode:

```text
dz_up = dh * silu(z_gate)
dz_gate = dh * z_up * d_silu(z_gate)
dz13 = concat(dz_gate, dz_up)
```

## Backward W13

Use the same expert-major row layout:

```text
x_expert[d, e, row, h]
dz13[d, e, row, 2i]
```

Pseudocode:

```text
for d, e:
  dw13[d, e] = x_expert[d, e].T @ dz13[d, e]
  dx_route[d, e] = dz13[d, e] @ w13[d, e].T
```

`x_expert` can be materialized by the forward W13 transport or rematerialized
from `assignment_id` by gathering source tokens into expert-major rows.

## Backward DX Return And Combine

```text
for s, d, e, r:
  pair_r = pair_expert_base[s, d, e] + r
  assignment_id = assignment_id[s, d, pair_r]
  token = assignment_id // topk
  route_slot = assignment_id % topk
  return dx_route[d, e, expert_row[d, s, e, r], :] to source route buffer

for source s:
  dx[s, token, :] = sum_route dx_route[s, token, route_slot, :]
```

This uses `assignment_id` directly; no `rev_src_pos = argwhere(...)` tensor is
needed unless a specific lowering wants it as a precomputed gather table.

## Source-Expand Transport Split

Current semantic source-expand is too slow relative to the forward
send/permute-W13 path. The likely issue is structural: the existing
expert-major source-expand path computes dy-route by destination-pulling over
expert-major rows, then separately computes dcombine. That is not the same
traffic pattern as the inbox sender.

The next diagnostic split is:

```text
dy-route only:
  source rank owns dy/source token metadata
  source rank writes dy * route_weight directly into destination expert-major rows

dcombine only:
  destination owns route_y_expert rows
  compute dot(dy[token], route_y_expert[row]) into source route slots

source-expand composed:
  dy-route source-push + existing expert-major dcombine
```

Harness modes added for this split:

```text
backward_dy_route_source_push_expert_major_pallas
backward_dy_route_source_push_expert_major_compare
backward_source_expand_from_expert_major_source_push_pallas
backward_source_expand_from_expert_major_source_push_compare
```

H100 diagnostic result at the target shape:

```text
job: /dlwh/iris-run-job-20260709-011354
backward_source_expand_from_expert_major_pallas:               0.004885257s
backward_dy_route_source_push_expert_major_pallas:             0.009460379s
backward_source_expand_from_expert_major_source_push_pallas:   0.012183732s
backward_dcombine_source_gather_expert_major_pallas:           0.015904189s
backward_source_expand_from_expert_major_source_gather_pallas: 0.025108179s
```

After fixing harness sharding, the existing destination-owned expert-major
source-expand is the fast path. The source-owned dy-route and source-gather
dcombine variants remain useful diagnostics, but they are not current
performance paths. Do not chase source-expand further unless a later integrated
profile contradicts this result.

## Metadata Construction Plan

Initial target:

```text
selected_experts, route_weights
  -> JAX-native semantic plan builder
  -> source-push lowering
```

The current JAX builder is intended as the correctness/reference boundary. It is
simple, jittable, and keeps routing semantics separate from the inbox queue. It
may still be too expensive at target shape because it uses a source-local sort
over `T * topk` assignments.

The semantic benchmark harness now exposes this choice explicitly:

```text
--plan-builder jax     # default, current staged kernel baseline
--plan-builder pallas  # build the implicit plan for stage modes through the
                       # Pallas histogram/scatter facade
```

This means ordinary stage modes such as `gather_x`, `w13`, `w2`, forward-return,
and fwd+bwd can be run against Pallas-built metadata without relying on
`metadata_pallas` appearing earlier in the mode list. Use `metadata_pallas` and
`metadata_tile_pallas` when measuring metadata construction itself.

Later target:

```text
selected_experts, route_weights
  -> Pallas metadata kernel(s)
  -> same semantic plan contract
```

The Pallas path should follow the Sonic-style tiled metadata construction rather
than a global sort:

```text
kernel 1:
  for each source/tile:
    histogram routes by (destination, local_expert)
    write tile_counts[source, tile, destination, expert]

JAX prefix boundary:
  xcounts[source, destination, expert] = sum_tile tile_counts
  tile_base[source, tile, destination, expert] = exclusive_cumsum_tile(tile_counts)
  rows_per_local_expert[destination, expert] = sum_source xcounts
  expert_base[destination, expert] = exclusive_cumsum_expert(rows_per_local_expert)
  src_base_by_expert[destination, source, expert] = exclusive_cumsum_source(xcounts)

kernel 2:
  for each source/tile:
    locally group the tile's route assignments by (destination, local_expert)
    compute within_tile_rank for each route
    local_row = tile_base[source, tile, destination, expert] + within_tile_rank
    pair_row = pair_expert_base[source, destination, expert] + local_row
    scatter assignment_id/source route weight to [source, destination, pair_row]
```

This is separable from the W13/W2/backward kernels. The same semantic plan
contract should feed both the reference JAX path and the Pallas metadata path.
That lets us first use the JAX builder to simplify forward/backward ownership,
then replace only metadata construction when its cost is measured at target
shape.

## Remote Source-Gather Forward Return

The owner-sharded forward-return stage was the largest visible remaining
isolated forward-stage tax in the promoted fwd+bwd decomposition. A remote
source-gather return variant is exact in isolation and much faster at target
shape:

```text
owner-sharded return:       0.009515191s
remote source-gather return: 0.004319881s
diff:                       0.005195309s
```

The integrated harness now has a narrow mode,
`forward_backward_expert_major_saved_x_direct_pack_remote_y_pallas`, which
changes only the y-return leg from owner-sharded sum to remote source-gather.
All other current-best fwd+bwd stages remain unchanged.

Validation before H100:

```text
py_compile: passed
focused pytest: 4 passed, 38 deselected
```

Target comparison job:

```text
/dlwh/bench_source_push_semantic_plan-20260709-remote-y-integrated
```

Result:

```text
owner-sharded y return stage: 0.009513554s
remote source-gather y stage: 0.004498466s

owner-sharded y integrated fwd+bwd: 0.044403770s, 174.196 useful / 217.745 rounded TFLOP/s/rank
remote y integrated fwd+bwd:        0.047381830s, 163.248 useful / 204.059 rounded TFLOP/s/rank
```

Do not promote remote-y integrated. The isolated stage win is real, but the
full graph regresses by `2.978ms`. Treat this as a graph/sharding/materialization
problem around the `route_y_expert` handoff. The next diagnostic should stop
the integrated graph immediately after y-return, or return enough
intermediates to see whether the remote-y path forces extra materialization
before the backward stages.

Y-stop diagnostic:

```text
forward_expert_major_direct_pack_owner_sharded_y_stop_pallas: 0.024226625s
forward_expert_major_direct_pack_remote_y_stop_pallas:        0.021286676s

isolated owner-sharded return: 0.009518982s
isolated remote return:        0.004393393s
```

Remote-y remains faster before backward even when `route_y_expert` is returned
and kept live. Therefore the integrated regression is not caused simply by the
remote y-return stage or by returning the W2 output. The conflict appears when
backward source-expand is in the same graph and also consumes `route_y_expert`.

Next diagnostic target: y-return plus source-expand only. If that regresses,
focus on scheduling/lifetime between remote source-gather y-return's
source-sharded metadata/read pattern and source-expand's destination-owned
expert-major read pattern. If it does not regress, the issue is farther
downstream in W2/W13 backward scheduling.

Y-return plus source-expand result:

```text
forward_source_expand_direct_pack_owner_sharded_y_pallas: 0.027304641s
forward_source_expand_direct_pack_remote_y_pallas:        0.025940024s
```

Remote-y remains faster after source-expand (`1.365ms` faster). The full
fwd+bwd remote-y regression therefore appears later than source-expand, after
adding W2 backward, SWIGLU backward, W13 backward, and dx-return. Next boundary
diagnostic should add W2 backward only and stop before W13 backward/dx-return.

W2-backward boundary result:

```text
forward_w2_backward_direct_pack_owner_sharded_y_pallas: 0.032487318s
forward_w2_backward_direct_pack_remote_y_pallas:        0.030910849s
```

Remote-y remains faster after W2 backward (`1.576ms` faster). W2 backward is
not the crossover point. The integrated remote-y regression must appear when
adding SWIGLU/W13 backward, dx-return, or full output tuple/live-range pressure.
Next boundary diagnostic should include W13 backward and stop before dx-return.

W13-backward boundary result:

```text
forward_w2_backward_direct_pack_owner_sharded_y_pallas:  0.032496134s
forward_w2_backward_direct_pack_remote_y_pallas:         0.030950892s

forward_w13_backward_direct_pack_owner_sharded_y_pallas: 0.038785406s, 199.430 useful / 249.287 rounded TFLOP/s/rank
forward_w13_backward_direct_pack_remote_y_pallas:        0.041013731s, 188.595 useful / 235.743 rounded TFLOP/s/rank
```

Remote-y remains faster through W2 backward, then becomes slower when SWIGLU/W13
backward is included. That makes W13 backward scheduling/live ranges, or the
larger output tuple required by that boundary diagnostic, the next suspect. W2
backward is not the owner-vs-remote-y crossover point, so pre-zeroing W2 is not
the current high-leverage explanation.

W13 scalar-digest boundary result:

```text
full tuple owner-y:  0.038788109s, 199.416 useful / 249.270 rounded TFLOP/s/rank
full tuple remote-y: 0.040684683s, 190.120 useful / 237.650 rounded TFLOP/s/rank

digest owner-y:      0.040536039s, 190.817 useful / 238.521 rounded TFLOP/s/rank
digest remote-y:     0.042681157s, 181.227 useful / 226.533 rounded TFLOP/s/rank
```

The digest mode returns a scalar reduction instead of the large boundary tuple.
The fixed digest uses full reductions to avoid illegal slicing across the
expert-sharded mesh, so it is not a pure zero-cost output-pressure test. Even
so, the remote-y regression persists. Treat the crossover as a layout/liveness
interaction entering SWIGLU/W13 backward, not merely host-visible tuple size.
Next isolate SWIGLU-only, W13 `dx_route`, and W13 `dw13` after owner-y vs
remote-y.

W13 subsplit boundary result:

```text
swiglu owner-y:  0.033116363s, 129.815 useful / 162.268 rounded TFLOP/s/rank
swiglu remote-y: 0.032122028s, 133.833 useful / 167.292 rounded TFLOP/s/rank

w13 dx owner-y:  0.036129243s, 166.540 useful / 208.176 rounded TFLOP/s/rank
w13 dx remote-y: 0.038622822s, 155.788 useful / 194.735 rounded TFLOP/s/rank

dw13 owner-y:    0.036021365s, 167.039 useful / 208.799 rounded TFLOP/s/rank
dw13 remote-y:   0.037941300s, 158.587 useful / 198.233 rounded TFLOP/s/rank
```

Remote-y is still faster through SWIGLU backward, so the handoff through W2 and
SwiGLU is not the regression. Both individual W13 WGMMA subkernels regress when
they consume the remote-y live set. Next test explicit materialization barriers
or resharding of `x_expert`/`dz13` before W13 backward in the remote-y path.

W13 input-barrier result:

```text
digest owner-y:         0.040293613s, 191.965 useful / 239.956 rounded TFLOP/s/rank
digest remote-y:        0.042421286s, 182.337 useful / 227.921 rounded TFLOP/s/rank

barrier digest owner-y: 0.040969003s, 188.800 useful / 236.001 rounded TFLOP/s/rank
barrier digest remote-y:0.043105143s, 179.444 useful / 224.305 rounded TFLOP/s/rank
```

Explicit `jax.lax.optimization_barrier` on `x_expert`, `dz13`, `w_gate_up`, and
`valid` before W13 backward slows both owner-y and remote-y and leaves the
owner/remote gap essentially unchanged. Simple W13 input materialization is not
the fix. Keep the y-only owner-sharded return as the promoted integrated path;
further remote-y integration work requires a structural layout change rather
than barriers.

Delayed remote-y diagnostic:

The remote source-gather y-return does not feed backward algebra directly; W13
backward consumes `route_y_expert` through source-expand/W2/SwiGLU. Added
`forward_backward_expert_major_saved_x_direct_pack_remote_y_delayed_pallas`,
which keeps the same direct-pack/W13/W2/source-expand/W2-backward/W13-backward
path but delays the remote source-gather y-return until after dx-return. This
tests whether the isolated remote-y return win can be recovered by removing the
eager remote-y call from the live range entering W13 backward.

Local validation:

```text
py_compile: passed
focused pytest: 17 passed, 4 deselected
```

First H100 launch failed before benchmark rows because the command accidentally
included unsupported `--no-progress-events`. Retry without that flag is active:

```text
/dlwh/bench-source-push-delayed-remote-y-retry-20260709
```

That retry succeeded at the Iris level but all modes emitted error rows because
random routing produced a non-tile-aligned live expert capacity:

```text
rows_per_expert_capacity=4249, row_block=16
```

The harness now rounds expert-major scratch capacity up to the LCM of the
active row-block requirements and masks padded rows as invalid. Local validation
after the fix: `18 passed, 4 deselected`. The active H100 retry is:

```text
/dlwh/bench-source-push-delayed-remote-y-rounded-20260709
```

Rounded-capacity H100 result:

```text
owner-y integrated:          0.051166218s, 151.173 useful / 188.967 rounded TFLOP/s/rank
eager remote-y integrated:   0.054603593s, 141.657 useful / 177.071 rounded TFLOP/s/rank
delayed remote-y integrated: 0.054596049s, 141.676 useful / 177.095 rounded TFLOP/s/rank
```

All three modes had zero error rows, zero drops, and zero metadata overflow.
Capacity rounding fixed the rough/random-routing harness failure, but delayed
remote-y is effectively identical to eager remote-y and still slower than
owner-y by `3.430ms`. Do not promote delayed remote-y. The isolated remote
source-gather return win is still real, but it does not survive the integrated
W13-backward live set through call reordering. Keep y-only owner-sharded return
as the promoted integrated path; further return work needs a different
return/combine structure.

Compare modes:

```text
forward_backward_expert_major_saved_x_direct_pack_owner_sharded_y_pallas
forward_backward_expert_major_saved_x_direct_pack_remote_y_pallas
forward_backward_expert_major_saved_x_direct_pack_remote_y_delayed_pallas
```

Decision rule: promote delayed remote-y only if it beats owner-y in the full
integrated fwd+bwd row. If delayed remote-y still regresses, treat the remaining
return opportunity as requiring a different forward-return structure rather
than call reordering or barriers.

## Package-Private Semantic MLP Boundary

The production-relevant semantic graph now has a dedicated package-private API
in `source_push_semantic_mlp.py`:

```text
source_push_moe_mlp_semantic_pallas_mgpu(
    selected_experts, x, route_weights, w13, w2,
    *, capacity, capacity_factor, mesh,
) -> (y, dropped_routes)
```

Metadata is built with JAX inside the custom-VJP forward. Both pair capacity and
destination-local expert capacity are static specialization parameters. Expert
capacity clipping is earlier-source-wins and contributes to
`metadata_overflow_routes`; all row offsets and reverse metadata derive from the
final clipped counts. This closes the previous possibility that dynamic routing
could exceed the expert-major buffer and disappear without drop accounting.
The Pallas histogram/scatter facade now accepts the same optional expert
capacity and rebuilds its tile bases, offsets, reverse routes, and overflow
counts from the final clipped counts. Interpret and JIT parity tests cover a
cross-source skew case against the JAX plan. JAX remains the production metadata
path because the measured Pallas metadata path is slower.

The fixed profile matches the promoted graph rather than exposing benchmark
knobs:

```text
direct x-pack:       M16 x H512
W13:                 M64 x K128 x N128
W2 forward:          M64 x K128 x N128
owner-sharded y:     M64 x H256
source-expand:       M128 x H128
W2 backward:         M64 x K128 x N128
W13 backward:        M64 x K256 x N64, Warpgroup lowering
dX source-gather:    M64 x H256
```

Only forward `y` uses the owner-sharded return. dX uses the ordinary promoted
source-gather. The correct destination-pull source-expand requires replicated
`dy`; the package API therefore makes the source-sharded-cotangent all-gather
explicit before that kernel. The owner-sharded-dcombine wrapper does not avoid
this communication: it shards only the output and has the same replicated
`dy` input contract.

CPU interpreted JIT value and custom-VJP gradient parity passes against a dense
MoE reference, including bf16 values/gradients, pair overflow, and cross-source
expert overflow. The
benchmark alias `current_best_fwd_bwd_with_metadata` now calls this API rather
than reconstructing a parallel stage graph. Target H100 job
`/dlwh/bench-semantic-mlp-api-dy-replicated-repeat3-20260710-104218` validated
that exact custom-VJP boundary end to end: median `63.038497ms`, range
`62.915588-63.185462ms`, `122.702` useful / `153.378` rounded TFLOP/s/rank,
with zero errors, drops, routing drops, or metadata overflow. The hand-written
benchmark graph in the same job was `51.465903ms`; the honest source-sharded
custom-VJP boundary therefore exposes an `11.572594ms` dy-replication tax.
An owner-sharded-dcombine retry failed at the same `P('expert')` dy versus
`P(None)` input contract, confirming that it is not a source-sharded substitute.

The random-routing capacity-fix decomposition is the current optimization
baseline: integrated fwd+bwd `51.404270ms` (`150.473` useful / `188.092`
rounded TFLOP/s/rank), metadata-inclusive `52.771839ms` (`146.574` /
`183.217`), with zero errors/drops/overflow. Visible isolated stages sum to
`57.778959ms`; integration recovers `6.374688ms`. Communication-heavy stages
sum to `33.342436ms`, led by direct x-pack `10.109811ms`, owner return
`9.540068ms`, dX source-gather `7.213923ms`, and source-expand `6.478634ms`.
Another broad tile sweep is not justified by the current evidence.

That fusion diagnostic reached the same current Mosaic boundary as
source-driven W13. `mgpu.remote_ref` can select the source peer, but
`mgpu.copy_gmem_to_smem` only accepts rectangular slices; it cannot gather the
arbitrary `(source, token)` rows into a WGMMA-tiled lhs. Register assembly then
hits the tiled row-slice/layout constraint. An interpret-only scalar prototype
was not retained as a production API. Resume this fusion only if transport
first provides a legal prepacked row tile or Mosaic gains indexed GMEM-to-SMEM
gather support.

The 250 useful TFLOP/s/rank target would require approximately `30.94ms` for
the fixed useful work. Even before metadata and dy replication, pairing the
current isolated stages under idealized perfect overlap gives an optimistic
split-stage floor of about `35.85ms`:

```text
max(x-pack 10.11, W13 6.28)
+ max(W2 2.74, y-return 9.54)
+ max(source-expand 6.48, W2-backward 5.70)
+ max(W13-backward 9.72, dX-return 7.21)
= 35.85ms
```

The production custom-VJP must additionally pay approximately `1.3ms` for JAX
metadata and `11.57ms` for dy replication. The source-push/source-gather
source-expand alternatives avoid the all-gather but measured `12.18-25.11ms`
for that stage and are not a win. Reaching 250 therefore requires a different
communication contract: a correct source-owned route buffer/direct queue plus
true producer/consumer overlap. Current direct remote queues are numerically
incorrect, and current Mosaic cannot assemble arbitrary remote token rows into
WGMMA-tiled SMEM. Treat this as the present architectural ceiling, not a tile
tuning gap.
