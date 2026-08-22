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

## Results so far (2026-08-22)

**Keep: the lean data path + 2 sequential expert chunks + splits 64 measures 22.80 MFU
against a same-night old-code baseline of 21.16 (+1.64 points, +7.7% throughput),
dropless, loss bit-consistent with the leg-1 band. It beats the pooled-wave reference
(22.31 under this protocol), which was the goal's throughput criterion.**

The road there and the arms after it:

- The lean path alone cannot run at the hero shape: XLA never rematerializes collectives,
  and the a2a outputs it exposes directly pin 192-194 GiB across four launch postures
  (fractions 0.75/0.70/0.68, slop 85/80, LHS on/off) -- NCCL's alltoall OOMs at step 1.
  Chunking is what made it land, by halving the pinned transport buffers.
- Guard rebuilt before any arm ran: the 4-GPU check's `ring` "ground truth" is itself
  broken (0.75-4.5x deviation from exact fp32 dense, NaN grads, garbage values to 2.3e28;
  filed as #8578, never caught because the check had never been run). The check now gates
  on a dense fp32 arbiter; the lean path passes it at 0.4-0.5% (bf16 noise) with clean
  gradient medians, and is split-invariant at forced 30% drops.
- New-keep anatomy (i04 trace, scaled to the 17.4 s step): compute 14.52 (now *below*
  pooled's 14.97 -- the permutation win landed at ~1.1 s/step), transport 1.77 busy/1.61
  exposed, other NCCL 0.75 busy (Symk tax halved). The remaining gap to 25% is the ~2.4
  s/step of fully exposed collectives.
- Exposure levers are closed with mechanisms: LHS adds ~9 GiB of scheduled liveness and
  OOMs at chunks 2 and 3; chunk pipelining fits in memory but ties (SM contention);
  command buffers tie (leg-1's crash and -0.25 both gone); removing the chunk barrier
  regresses (-0.11). Device kernel ties (was -3 pre-#47263).
- Splits bracket under chunking: 64 == 128, both ~+0.1 over 32; default moved to 64.

## Small-scale ablation (the goal's validation step)

`small_scale_abl_launch.py` grew a `ragged` flavor. d768 on gb200-rack (EP64, faithful
sender gate), harrier 2026.08.17.1 mix, 10.8k steps:

- `l2-abl-d768-ragged`: finished, final train loss 1.9393, steady-state drops 0.031%
  mean / 0.2% max over the last 500 steps -- the <2% criterion with ~60x margin, under a
  trained router and the stricter per-chunk capacity gate.
- `l2-abl-d768-ep` (pooled-wave control, identical settings): running.

## Files

Same tooling as leg 1 (`arm.sh`, `watchdog.sh`, `score.py`), copied here so the leg keeps
its own `arms.tsv` / `results.tsv`. Iteration 0 baseline = leg-1 splits-32 keep
(`c3f71b2fc9` code state).
