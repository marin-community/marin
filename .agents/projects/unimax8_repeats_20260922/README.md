# UniMax-8 trainer-seed repeats at Delphi 3e18 (2026-09-22)

Calvin asked (2026-09-22 15:20 PDT) for two more UniMax-8 runs at the Qwen3 360M/1.6B scale (3e18 FLOPs) with
OlmoBaseEval Easy (Table 9) evaluations, so Table 2 and the other 3e18 comparisons can carry a mean ± std over
three trainer seeds for UniMax-8 as they do for MARINER, Olmix and proportional.

## Launch

- Launcher `experiments/domain_phase_mix/launch_delphi_baseline_mixtures.py`, extended today with `--trainer-seeds`
  (one run per seed; seeds other than 0 add `_t<seed>` to the run name) and `--data-seed` (one fixed data seed for
  every run). Defaults reproduce the previous behaviour (trainer seed 0, data seed = run id). The manifest step
  carries the seeds too. Strict graph build verified locally (`CI=1 MARIN_EXECUTOR_STRICT=1`: 2 training + 2
  evaluation steps); dry-run specs in `dry_run/`.
- Runs: `unimax8_3e18_t1` (run id 660740, trainer seed 1) and `unimax8_3e18_t2` (660741, seed 2), both at data seed
  660700, the data seed of the ladder's `unimax8_3e18-cb3b49` (trainer seed 0). v5p-8 in us-east5-a, batch 128,
  3,007 steps, HF export at step 3006; Table 9 evaluations on v6e-8 in us-east5-b (`t9_unimax8_3e18_t1/_t2`).
- Iris parent `/calvinxu/dm-delphi-unimax8-repeats-3e18-20260922`, submitted 15:51 PDT from `launch.sh` (east5
  guard passed, bundle 17.3 MB); Fieldbook job_01m35n2dntmmxpway8s2r320fd under exp_01kvvvv6zxrf0j7tkp4f7k6y66.
  Watch: `watch.sh` (detached) -> `watch.log`.
- ETA from this morning's sweep: training about 80 minutes (landing about 17:15 PDT); Table 9 evaluations 20 to
  30 minutes more when a v6e-8 slice is free (a stockout added 50 minutes this morning).

## When it lands

Collect Uncheatable (byte-weighted macro from `checkpoints/eval_metrics.jsonl`, the ladder weighting) and the
Table 9 macro/components from `gs://marin-us-east5/evaluation/olmo_base_eval_table9/t9_<run>-<hash>/`, as the cap
sweep README did. Then update: Table 2's 3e18 UniMax-8 cells (mean ± std over the three trainer seeds; caption
already says "three trainer seeds"), Figure 6's first-rung UniMax-8 points and error bars, the UniMax cap table
(`tab:a-unimax-cap`, cap-8 row), the ladder appendix table if it lists 3e18 UniMax-8, and any compute-equivalence
fit that uses the 3e18 UniMax-8 value. Outline notes and the revision record follow the usual paper workflow.

## Training landed (17:05 PDT); Table 9 evaluations pending on v6e-8 (checked 18:47 PDT)

Both runs succeeded with HF exports at step 3006 (`unimax8_3e18_t1-4e768b`, `unimax8_3e18_t2-b05dff`). Uncheatable from
the final `eval_metrics.jsonl` row, frozen seven-component weighting (the new inline evaluator's byte-pooled parent
aggregate `eval/uncheatable_eval/bpb` is a different estimator and is not the paper's metric; see the 13 September
midpoint README's metric-comparability section):

| run | trainer seed | Uncheatable (frozen weighted) | byte-pooled parent (not used) |
|---|---:|---:|---:|
| unimax8_3e18-cb3b49 (ladder) | 0 | 1.0223 | 1.0223 (old schema) |
| unimax8_3e18_t1-4e768b | 1 | 1.0191 | 1.0253 |
| unimax8_3e18_t2-b05dff | 2 | 1.0227 | 1.0289 |

Mean 1.0214, std 0.0020 over the three trainer seeds (MARINER's three-seed std at 3e18 is 0.001).

The two Table 9 evaluations (`olmo-base-eval-t9-unimax8-3e18-t1/_t2`) have been pending since 17:05 PDT. Cause
(`rpc controller list-backends`, `last_routing_decision`): their demand (v6e-8, us-east5-b) is
`tier_blocked: blocked by quota-pool tier monotonicity`. In `routing.py`, a quota pool is blocked from the lowest tier
whose group is in BACKOFF or QUOTA_EXCEEDED; `tpu_v6e-preemptible_4-us-east5-b` (tier 1 of pool
`v6e-preemptible/us-east5-b`) is in backoff ("degraded (health=0.19)", five recent slice failures, last attempts
18:30 to 18:36 PDT), so the v6e-8 group (tier 2, itself "available", three failures at 18:30 to 18:37 PDT) is blocked
with it. Two evaluations of the tpp40 east5 batch are blocked the same way. Effectively a v6e stockout in us-east5-b
expressed through the tier rule; it clears when the tier-1 group's AIMD health recovers and slices succeed. Fallback
approved this morning but not yet used: run the evaluations on v5p-8 in us-east5-a with proportional_3e18 and the
seed-0 UniMax-8 re-scored as hardware controls (needs a `--table9-tpu-type` option in the launcher).

## Table 9 evaluations landed (seed 1 at 19:40 PDT, seed 2 at 23:03 PDT)

The v6e-8 tier block cleared on its own; seed 2's evaluation was preempted four times and resumed from its per-task
progress each time (103 of 104 tasks were already scored before the last wait). OlmoBaseEval Easy macro: seed 0
1.1372 (ladder), seed 1 1.1285, seed 2 1.1339; mean 1.133, std 0.004. Collected by
`collect_unimax8_repeats_3e18_20260922.py` into `reference_outputs/delphi_unimax8_repeats_3e18_20260922/`; the
scaling-figure builder reads the summary for UniMax-8's first rung. Paper: Table 2 cells 1.021 ± 0.002 and
1.133 ± 0.004, Figure 6 error bars, Table 3 unchanged at one decimal, cap-table caption points to the three-seed means.
Parent job succeeded.
