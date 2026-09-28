# UniMax epoch-cap sweep at Delphi 3e18 (2026-09-22)

Calvin asked (2026-09-22 03:00 PDT) for a sweep of the UniMax epoch cap at the Qwen3 360M/1.6B (3e18 FLOPs) scale,
caps 1, 4 and 12 beside the ladder's UniMax-8, with Table 9 (OlmoBaseEval Easy) BPB evaluations. Motivation: the
paper's UniMax-8 has no recorded rationale for the cap; the UniMax paper (Chung et al., 2023) uses N = 1 and ablates
N in {1, 5, 10} with "the effect is small"; at 3e18 our only other UniMax point was the swarm's cap-1 baseline
(1.029 Uncheatable / 1.175 OlmoBaseEval Easy against UniMax-8's 1.022 / 1.137).

## Launch

- Launcher: `experiments/domain_phase_mix/launch_delphi_baseline_mixtures.py` (issue #6607 baseline launcher,
  v5p-8 in us-east5-a, batch 128, 3,007 steps, HF export at step 3006), extended today with mixtures `unimax1`,
  `unimax4`, `unimax12` (`UNIMAX_EPOCH_CAPS`), `--run-id-base` (660730-660732, also the data seeds) and
  `--with-table9-eval`, which chains `olmo_base_eval_step` (v6e-8, us-east5-b, `--table9-tpu-zone`) after each
  training step, as the swarm launchers do. Dry-run specs: `reference_outputs/delphi_baseline_mixtures_issue6607_20260623/run_specs.json`.
- Materialized epochs at the fixed 6.3T target budget: cap 1 binds on 38 buckets (largest CC bucket takes the rest,
  weight 0.104); cap 4 on 11; cap 8 on 4; cap 12 on 2.
- Iris parent `/calvinxu/dm-delphi-unimax-cap-sweep-3e18-20260922` (interactive band, us-east5-a), submitted
  03:06 PDT from `launch.sh` (east5 guard passed; bundle 17.3 MB); Fieldbook job_01m349aszgqqxashzk64yb8kem under
  the baselines' experiment exp_01kvvvv6zxrf0j7tkp4f7k6y66. Watch: `watch.sh` (detached) -> `watch.log`.
- Outputs: training under `gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_baseline_mixtures_issue6607_20260623/<run>-<hash>/`
  (Uncheatable in `checkpoints/eval_metrics.jsonl`), Table 9 under `gs://marin-us-east5/evaluation/olmo_base_eval_table9/t9_<run>-<hash>/`.

## When it lands

Collect Uncheatable (byte-weighted macro from eval_metrics.jsonl, as the ladder collect procedure does) and the
Table 9 macro/components from the eval results JSON; compare with proportional_3e18 (1.038 / 1.198), the swarm's
cap-1 baseline and unimax8_3e18 (1.022 / 1.137); then decide whether the paper's UniMax baseline changes and add
the defining sentence for UniMax-8 in Section 4 (currently absent).

## Attempt 1 failed at the parent (03:17 PDT), retry1 submitted (03:18 PDT)

The parent ran with `MARIN_EXECUTOR_STRICT=1` (the ladder's command shape) and the baseline launcher built its
steps outside `executor_context()`, which strict mode turns into an error after 54 s, before any TPU was
allocated. The launcher now builds the graph inside `executor_context()` like the ladder and swarm launchers
(strict build verified locally). Resubmitted unchanged otherwise as
`/calvinxu/dm-delphi-unimax-cap-sweep-3e18-20260922-retry1` (Fieldbook job_01m349zxpfae1fd6e1651xsp5g, retry of
job_01m349aszgqqxashzk64yb8kem); `watch.sh` follows the new parent.

## Training landed (04:39 PDT); Uncheatable from the final eval_metrics (byte-weighted, ladder weighting)

| run | dir | Uncheatable | vs UniMax-8 |
|---|---|---:|---:|
| proportional_3e18 (ladder) | proportional_3e18-ebc4aa | 1.0383 | +0.016 |
| unimax1_3e18 | unimax1_3e18-76174c | 1.0364 | +0.014 |
| unimax4_3e18 | unimax4_3e18-b8991a | 1.0238 | +0.002 |
| unimax8_3e18 (ladder) | unimax8_3e18-cb3b49 | 1.0223 | -- |
| unimax12_3e18 | unimax12_3e18-44b7a5 | 1.0251 | +0.003 |

Single runs (proportional repeat SD 0.001). Cap 8 is the best on Uncheatable, cap 4 within 0.002, cap 12 slightly
worse, cap 1 close to proportional. The launcher's cap-1 mixture (largest CC bucket takes the remainder, weight
0.104) differs from the swarm's cap-1 baseline (built at the phase budget, 1.25 materialized epochs, 1.029),
which is why it scores worse than that earlier point. Table 9 evaluations pending on v6e-8 (us-east5-b).

## Table 9 evaluations waited on a v6e-8 stockout (04:31 to about 05:20 PDT)

The three evaluations (v6e-8, us-east5-b, the site's conventional evaluator hardware) sat pending in the interactive
band for ~50 minutes: the cluster had no v6e-8 worker in us-east5-b and both v6e-8 scaling groups there recorded
"no more capacity in the zone" (the 2026-09-16 stockout pattern). Calvin approved moving the evaluations to v5p-8 in
us-east5-a with proportional_3e18 and unimax8_3e18 re-scored as hardware controls, but a preemptible v6e-8 slice
appeared before the resubmission, unimax1's evaluation started on it, and the plan was dropped: the evaluations stay
on v6e-8 and no control re-scoring is needed. Lesson for the launcher: the Table 9 TPU type should be an option
(v5p-8 in us-east5-a is the fallback when v6e-8 is stocked out), with a same-hardware control when switching.

## Results (all three evaluations landed by 06:19 PDT; `table9_results.json` holds the 51 components per run)

| cap | Uncheatable | OlmoBaseEval Easy | MT MBPP (17) | Minerva (7) | Basic Skills (6) | code (2) | MMLU (4) | QA (15) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| proportional | 1.0383 | 1.1987 | 0.997 | 1.099 | 1.573 | 1.088 | 1.380 | 1.291 |
| 1 | 1.0364 | 1.1902 | 0.977 | 1.086 | 1.586 | 1.065 | 1.374 | 1.290 |
| 4 | 1.0238 | 1.1399 | 0.909 | 0.970 | 1.469 | 0.957 | 1.375 | 1.311 |
| 8 (ladder) | 1.0223 | 1.1372 | 0.904 | 0.935 | 1.460 | 0.939 | 1.391 | 1.326 |
| 12 | 1.0251 | 1.1357 | 0.900 | 0.936 | 1.443 | 0.954 | 1.389 | 1.330 |

Single runs; proportional repeat SDs are 0.001 (Uncheatable) and 0.004 (OlmoBaseEval Easy). Caps 4, 8 and 12 lie
within 0.003 BPB of one another on both objectives: 8 is best on Uncheatable, 12 nominally best on OlmoBaseEval Easy
(0.0015 below 8, inside seed noise), 4 within 0.003 of both. Cap 1 sits next to proportional. The groups show the
trade-off the cap sets: raising it lowers BPB on the repeated small buckets' tasks (MT MBPP, Minerva, Basic Skills,
code) and raises it on QA and MMLU, whose data is the large Common Crawl mass that a higher cap dilutes. UniMax-8 is
a defensible baseline cap and the ladder result is not sensitive to it between 4 and 12.
