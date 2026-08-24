# Leg 3: dk-native branch (issue #8317)

Goal (user, 2026-08-23): bring the ragged all-to-all DEVICE KERNEL to parity with (or above)
the one-shot on unpatched XLA main; final verification at hero shape restored from the hero
step-6000 checkpoint. This branch starts CLEAN from repo main and ports one-shot-branch
optimizations one at a time, each validated by an arm — including the ones we are confident
about. PR #8549 (the one-shot lean branch) is frozen as the fallback and must not change.

Reference numbers on the SAME main-vintage compiler (leg-2b, `loop-260821-leg2` rows i17-i28):
- one-shot on kmax128-only main: 22.77 MFU (its only patch is the one-line kMaxPeers bump)
- dk on g8x128-grid main: 21.69 MFU with the LEAN structure; per-launch parity with the
  one-shot (4.56 vs 4.85 ms); residual gap = backward transport recompute (576 vs 288
  launches/step), which follows the dk flag's operand registration, not compiler vintage
- dk grid axis: stock 19.37 / 4x256 21.02 / 8x128 21.69 (peak) / 16x128 21.11

Branch base: origin/main e7c34f396f. Tooling ported as-is (transport-agnostic): hero launcher
(launch_mfu_test.py, train.py, model.py), pjrt_wheel.py override, arm/watchdog/score harness,
wheel build recipes. lib/levanter stays at main (the OLD ragged backend: compact/expand +
local permutes, splits supported, no chunking).

Standard runtime: the g8x128 wheel (clean main e5d008bb03 + 8x-SM/128-thread dk grid patch),
`s3://marin-us-east-02a/marin/research/mcwitt-ra2a/pjrt-mainpatch-g8x128-20260823/`.
dk engagement: `--xla_gpu_experimental_ragged_all_to_all_use_device_kernel` +
`--xla_enable_nccl_symmetric_buffers_for_collectives=raggedalltoall` (scoped registration;
global symbuf measured equivalent).

Port queue (one arm each; ledger below decides keep/drop):
1. dkn-00 baseline: main's old ragged backend + dk flags (posture ported with the launcher —
   re-ablate posture explicitly as its own arm).
2. Lean data path (expert-granular params, unclipped-offset senders, expert-major receiver,
   fused dispatch gather) — the +1.5 one-shot win; transport-agnostic in principle.
3. Chunking: OFF by default here (arena operands sit outside the pool); test 1 vs 2 chunks
   under dk before adopting (leg-2 i28 measures this on the lean structure).
4. splits_per_peer bracket under dk (one-shot optimum was 64; old-backend leg-1 optimum 32).
5. Scheduling posture A/B: ported (LHS off, overlap 1) vs main defaults.
6. Backward-recompute elimination (slim transport vjp) — targets the 2x-launch residual.
7. QuACK grouped GEMMs / cuDNN wgrad path if not already active via lib main.

Every arm: synthetic data, hero shape, one rack, mean mfu steps 5-15 relative to first,
per-arm rows in arms.tsv (reissuable), results in results.tsv.

## Conclusion (2026-08-24): goal met

- Synthetic hero shape, same compiler: dk 22.62 vs one-shot 22.77 (delta 0.15, within
  same-night noise; one-shot-branch splits=64 actively hurts the dk, splits=1 is the keep).
- Final verification, hero shape restored from the hero step-6000 checkpoint on mixture data,
  transport the only variable: **dk 21.43 vs one-shot 21.25 (tie, dk nominally ahead)**;
  drops 0.017% vs 0.018%; identical loss 1.4727@6019. The skew-weakness hypothesis is refuted.
- Runtime: XLA main e5d008bb03 + the g8x128 device-kernel grid patch (5 lines: 8x-SM grid of
  128-thread CTAs + matching barrier registration) + two flags (dk + scoped raggedalltoall
  registration). The one-shot control instead needs the kMaxPeers patch. Neither transport runs
  on truly flag-only stock main today; the dk's remaining diff is small and upstreamable.
- Known cost on this branch: the main-vintage DataLoader depresses restore MFU ~1 vs the tune
  branch's loader for BOTH transports (plateau ~21.7 with dips; fetch_batch_size=256 mapping).
  Loader tuning is a follow-up port, orthogonal to the transport.

## Final standings (2026-08-24, splits sweep closed)

Splits is transport- AND regime-dependent:
- one-shot: 64 >> 1 in both regimes (+1.2 at hero shape; the old +2.4 was priced on the
  pre-lean backend)
- dk synthetic (uniform payloads): 1 > 32 > 64 (22.62 / 22.41 / ~21.7)
- dk restore (trained-router skew): 64 > 128 > 1 (21.98 / 21.49 / 21.43) — splitting
  balances skewed updates, which the dk's unrotated unit striping needs

Hero 6k restore, each transport at its best: one-shot 22.46-22.60, pooled 22.52,
dk 21.98 — the dk trails by ~0.5 MFU (~2%) under realistic conditions; synthetic is a tie
(22.62 vs 22.77). The earlier "dk parity at restore" compared both transports at splits=1,
which handicaps only the one-shot. Remaining known dk headroom: the copy kernel still runs
~1.8x slower per byte than the one-shot at equal payload (in-kernel work, upstreamable);
closing that would flip the restore standings.

The cross-branch "drift" investigation (dkn-05..09 + replay) concluded: no code drift at
all — lib-swap hybrid null, siblings null, wheel vintage null, epoch null; every delta was
the splits default. dkn-10 first attempt crashed on a mis-submitted bisect tree (pyc-blocked
checkout); resubmitted clean as dkn-10b.
