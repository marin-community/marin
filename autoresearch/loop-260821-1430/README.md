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

**Hard 15-minute runtime cap**, enforced in two layers:

1. `iris job run --timeout 900` on the coord job.
2. `watchdog.sh`, a detached out-of-process poll loop that runs `iris job cancel` on the coord path
   (Iris cancels descendants) at the deadline. It also releases the rack early, as soon as W&B
   shows the last step the scoring window needs.

`systemd-run` and `crontab` are both unavailable in this sandbox — the classifier blocks the
former and NixOS ships no `crontab` — so layer 2 is a detached shell loop rather than a timer unit.

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

Guards, all three of which must pass or the iteration is discarded regardless of MFU:

- `moe/drop_fraction` stays at the baseline's level (the baseline is dropless).
- `train/loss` at the window's last step matches the baseline to four decimals. Synthetic data is a
  deterministic batch, so any real change to the model or the transport's arithmetic shows up here.
- The 4-GPU `ragged_all_to_all` vs `ring` forward and gradient check passes, for any change that
  touches transport offsets or expert kernels.

## Files

| file | what it does |
|---|---|
| `arm.sh` | submit one arm |
| `watchdog.sh` | hard cap + early release for one arm |
| `score.py` | metric and guard quantities from W&B |
| `results.tsv` | one row per iteration |
