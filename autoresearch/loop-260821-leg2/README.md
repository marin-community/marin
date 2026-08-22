# Ragged EP64 tuning loop, leg 2 (2026-08-21)

Continuation of `../loop-260821-1430` with a fresh 20-iteration budget. Leg 1 closed the
launch-flag space; this leg goes after the structural residuals its trace anatomy named:
~0.62 s/step permutation compute, ~0.37 s/step Symk-slowed ordinary collectives,
~0.7 s/step exposure. Bigger design hops are explicitly in scope.

Protocol, metric, guards, and reproducibility rules are unchanged from leg 1's README
(compile + 15 min cap, watchdog + iris timeout, mean `throughput/mfu` over steps 5-15,
loss-match + drop guards, 4-GPU `ragged_ep_check.py` for any change touching offsets or
kernels). Leg-1 frontier under this protocol: ragged 21.32, pooled 22.31.

## Why the permutation compute exists (and why pooled doesn't pay it)

The pooled backend never sorts activations: it computes int32 linear indices once and
moves rows with single gathers whose custom VJPs are one-to-one gathers. The ragged
backend materializes the 8x-expanded buffer (`jnp.repeat`), permutes it by argsort,
compacts it by a second argsort, regroups the received buffer by a third, inverts that on
combine, and expands the returned buffer -- roughly five full [TK,H]/[C,H] passes per
layer per direction beyond the transport itself, each with an argsort-based VJP.

Three structural eliminations, exploitable because `ragged_all_to_all` takes arbitrary
per-update offsets (`dst = i // slices_per_device`):

1. **Unclipped offsets** -- accepted rows are a prefix of each expert group, so the a2a
   can read at unclipped group offsets with clipped sizes; `_compact_by_keep_mask` and
   `_expand_from_keep_mask` disappear (the return a2a writes valid prefixes back to
   unclipped positions; dropped rows keep the output buffer's zeros).
2. **Expert-major receiver layout** -- issue updates at (peer x local-expert) granularity
   with receiver offsets computed from the already-all-gathered `clipped_group_sizes`;
   rows arrive grouped by local expert, so `_local_permute_from_counts` and the
   combine-side inverse sort disappear.
3. **Fused sender gather** -- `repeat(x, topk)` followed by a gather is
   `x_local[sorted_indices // topk]`: one gather, no 8x materialization.

## Hypothesis queue

| id | hypothesis | mechanism | est. stake |
|---|---|---|---|
| L2-i01 | lean sender/return path (3 above + 1) | kills ~3 full-buffer passes/layer/dir | part of 0.62 s |
| L2-i02 | expert-major receiver layout (2) | kills ~2 more passes/layer/dir | part of 0.62 s |
| L2-i03 | device kernel re-test | `use_device_kernel` allocates kCollective barrier memory, not NCCL windows -> ordinary RS/AG leave Symk | 0.37 s |
| L2-i04 | LHS retry at reduced footprint | leg-1 OOMs may have been the extra permutation temporaries | up to 0.7 s |
| L2-i05+ | NCCL_SYM_KERNEL forcing (if i03 fails); sender dedup (send once per token,shard); chunked transport/MLP pipelining | | |

Leg-1's device-kernel tie (dk05 21.20 vs dk06 21.44, old campaign) is reinterpreted, not
retired: if the device kernel already ran ring collectives, its transport is slower and
tuning may fix it; if it still paid the Symk tax, the tie masked a transport win.

## Files

Same tooling as leg 1 (`arm.sh`, `watchdog.sh`, `score.py`), copied here so the leg keeps
its own `arms.tsv` / `results.tsv`. Iteration 0 baseline = leg-1 splits-32 keep
(`c3f71b2fc9` code state).
