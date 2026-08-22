# Ragged EP64 tuning loop (2026-08-21)

Goal: make the ragged all-to-all transport beat main's `fixed_pooled_wave_all_to_all` hero default
at the EP64 hero shape. Under the previous protocol the two measured 21.88% and 22.69% MFU.

Branch `research/mcwitt/8317-ragged-tune`, forked from the PR branch head (`4c1c3b7152`,
draft PR #8549) so the loop starts from the best known ragged configuration. The PR branch stays
untouched; wins get consolidated back into it afterwards.

## Protocol

Every arm: one NVL72 rack on `cw-us-east-08a`, **production** priority, one rack at a time,
synthetic data, no profiler, no checkpoints, no eval, `--master-params disabled`, capacity 1.15,
4 processes per node, the kmax128 PJRT wheel.

**Hard runtime cap: compile time + 15 minutes** (changed after iteration 3; through iteration 3 the
cap was a flat 15 minutes from submission). Enforced in two layers:

1. `watchdog.sh`, a detached out-of-process poll loop that runs `iris job cancel` on the coord path
   (Iris cancels descendants). It cancels 900 s after the first step it observes, or 1200 s after
   submission if no step ever appears, and it releases the rack early as soon as W&B shows the last
   step the scoring window needs. The step budget starts at the first step so a slow *compile*
   cannot eat the measurement window, while a change that makes *steps* slow still gets no extra
   rack time.
2. `iris job run --timeout 2100` on the coord job, the backstop that fires only if the watchdog
   process is dead.

`systemd-run` and `crontab` are both unavailable in this sandbox — the classifier blocks the
former and NixOS ships no `crontab` — so layer 2 is a detached shell loop rather than a timer unit.

## Regime

Iterations 0-2 ran from scratch on synthetic data. That regime turned out not to test the thing
the goal is about: at step 15 the router is still near-uniform, so capacity never clips and both
transports measured dropless (pooled 7.4e-6, ragged 0.0). The hero in production sits near 4.9%.

Restoring a trained checkpoint was tried and abandoned. The hero writes permanent checkpoints only
every 6000 steps and is at ~4000, so the only trained d6144 state that exists is an hourly rolling
temporary the hero deletes as soon as the next one commits. An arm restored from it could not be
re-run against the same state, which makes its number unfalsifiable the next day. Reproducibility
wins over regime fidelity here.

The machinery stays in the tree (`--restore-from`, `--restore-master-params`) for when a permanent
checkpoint exists at step 6000. Neither restore arm produced a measurement: `r01` failed fast on
the wrong path, `r02` died on task 0 at 8 minutes with only a coordination-service cascade in its
diagnostic, so whether the restore itself works is still unestablished.

**What this regime does not measure.** Both transports are dropless from scratch, so the loop
optimizes transport speed at balanced routing only. The goal's drop-rate claim -- comparable
throughput at a lower drop rate in the trained-router regime -- is not tested by any number here
and needs the permanent checkpoint or `small_scale_abl_launch.py` to settle.

## Reproducibility

Every arm is reissuable from its row in `arms.tsv`: commit, artifact version, backend, data mode,
parameter storage, schedule length, timeout, and any extra launcher arguments. The inputs that
could otherwise drift are pinned -- the synthetic batch is a pure function of the seed, the PJRT
wheel is an immutable object URL, and the schedule length fixes the learning rate at every step.

Budget: 20 iterations.

## Metric and guards

Metric: mean `throughput/mfu` over **steps 5-15**, higher is better. `score.py` reports
`window_complete: false` rather than averaging a window with holes in it.

The window comes from what the baseline arm actually reached. Compile, not the step rate, is what
eats the budget: step 1 landed at 3:21 elapsed and step 2 at 5:56 (a second XLA compile sits
between them), after which steps settle at ~20.7 s. The baseline reached step 22 by the cap, so a
5-19 window is reachable but with no margin; 5-15 leaves about 80 s of slack for a slower compile.
MFU is flat across that window (stdev 0.14), so the shorter window costs nothing in precision.

Numbers here are not comparable to the pre-loop campaign's: that scored 5-19 on real data with the
profiler on. The baseline measures 18.96% where the same code measured 21.88% under the old
protocol. Only within-loop comparisons mean anything.

Drop fraction is reported alongside MFU on every arm, as an outcome rather than only a guard: the
ragged transport exists to move tokens the pooled one clips, so a throughput win bought with drops
is not a win.

Guards, all three of which must pass or the iteration is discarded regardless of MFU:

- `moe/drop_fraction` does not rise above the arm it is being compared against.
- `train/loss` at the window's last step matches the baseline to four decimals. Synthetic data is a
  deterministic batch, so any real change to the model or the transport's arithmetic shows up here.
- The 4-GPU `ragged_all_to_all` vs `ring` forward and gradient check passes, for any change that
  touches transport offsets or expert kernels.

## Conclusion (2026-08-22, stopped at 10 of 20 iterations)

The loop stopped early: after the splits-32 keep, every arm measured within [21.18, 21.32] MFU
-- protocol noise -- and every mechanism the i04 trace anatomy identified is closed:

- **Splits per peer 32** is the one keep (+2.36 over the PR's old default of 1) and the bracket
  24/32/48 is flat, so 32 stands. Already consolidated into PR #8549.
- **The 1.0-pt gap to pooled decomposes as**: ~0.62 s/step extra permutation compute (sorts,
  scatter fusions, dispatch copies), ~0.37 s/step slower ordinary collectives, ~0.7 s/step worse
  exposure. The ragged transport kernel itself is *faster* than pooled's all-to-all (1.60 vs
  2.23 s/step busy, 1.50 vs 1.83 exposed).
- **Ordinary collectives run on NCCL symmetric kernels** (LDMC reduce-scatter 2.75x slower than
  pooled's ring) because the transport's window registration makes their buffers Symk-eligible.
  `NCCL_ALGO=Ring` cannot reach that selection path (null, confirmed against NCCL source);
  `NCCL_SYM_CTAS=32` measured slightly negative. No runtime knob deselects Symk without
  `NCCL_WIN_ENABLE=0`, which the transport needs.
- **The latency-hiding scheduler is memory-walled**: at overlap 4, overlap 1, and overlap 1 with
  fraction 0.70 / slop 80, NCCL's alltoall hits CUDA OOM on the first step. Hiding the 1.3 s/step
  of exposed NCCL needs a different memory architecture, not a flag.
- Memory slop 87, command buffers, and collective overlap 2 are nulls or regressions.

Frontier under this protocol: **ragged 21.32 vs pooled 22.31 MFU**, both dropless from-scratch.
The remaining gap is structural (fused permutation + transport work of the marin-ep line), which
is out of scope for launch-flag arms. The remaining 10 iterations were declined rather than spent
mining noise.

## Files

| file | what it does |
|---|---|
| `arm.sh` | submit one arm |
| `watchdog.sh` | hard cap + early release for one arm |
| `score.py` | metric and guard quantities from W&B |
| `results.tsv` | one row per iteration |
