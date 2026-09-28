# Scaling-ladder babysitting handoff — 15 September 2026, 16:30 PDT

You own monitoring and recovery of the seven remaining Qwen3 scaling-ladder training runs (MARINER and matched Olmix at 3e20 and 1e21 FLOPs) until each finishes and its native Table-9 eval child succeeds. You do not edit the paper; report results and the other session updates Figure 6, Table 2 and the abstract.

## Jobs (Iris config `lib/iris/config/marin.yaml`)

| Run | Rung / TPU | Latest step | Progress | Last activity (PDT) | Failures used | Parent job |
|---|---|---|---|---|---|---|
| MARINER Uncheatable 1e21 | v6e-64 | 17,142 / 22,057 | 78% | Sep 15 09:03 | 3 of 4 | `/calvinxu/dm-delphi-frozen-unconstrained-scaling-v6e-20260908-retry6` |
| MARINER suite 3e20 | v6e-32 | 19,142 / 23,531 | 81% | Sep 15 14:23 | 2 of 4 | `...-20260908-retry7` |
| MARINER suite 1e21 | v6e-64 | 11,485 / 22,057 | 52% | Sep 15 09:03 | 2 of 4 | `...-20260908-retry5` |
| Olmix Uncheatable 3e20 | v6e-32 | 20,642 / 23,531 | 88% | Sep 15 11:39 | 3 of 4 | `/calvinxu/dm-delphi-matched-olmix-scaling-v6e-20260910` |
| Olmix suite 3e20 | v6e-32 | 9,865 / 23,531 | 42% | Sep 15 14:23 | 0 of 4 | `...-20260910` |
| Olmix Uncheatable 1e21 | v6e-64 | 7,385 / 22,057 | 33% | Sep 15 09:03 | 0 of 12 | `...-20260910-retry1` |
| Olmix suite 1e21 | v6e-64 | 6,756 / 22,057 | 31% | Sep 15 09:03 | 0 of 12 | `...-20260910-retry1` |

Total steps: 23,531 at 3e20 (batch 256), 22,057 at 1e21 (batch 512), 4,096-token sequences. Children run on preemptible v6e in us-east5-b; the pool is churning (up to 27 preemptions per child) and schedules in bursts, so all tasks pending for hours is normal, not a failure.

## Failure mode and recovery

A host preemption kills the sibling hosts with SIGSEGV (`client.h:82 Terminating process because the JAX distributed service detected failure`, exit 139), and Iris counts each as a failure. When a child's cumulative failures exceed `max_task_failures` (4 on the older parents, 12 on retry1) the child fails with `Cumulative failed task attempts exceeded max_task_failures` and the parent ends. Recovery is a new scoped parent with the same executor steps, which resumes from the latest checkpoint:

```bash
/Users/calvinxu/Projects/Work/Marin/marin/.agents/projects/ladder_watch_20260915/retry.sh mariner retry8 12 fpu_1e21 u_snc_cap06_1e21
/Users/calvinxu/Projects/Work/Marin/marin/.agents/projects/ladder_watch_20260915/retry.sh olmix retry2 12 mou_3e20 u_kl0p05_cap04_3e20
```

`retry.sh <mariner|olmix> <retry-suffix> <budget> <run-only regex>...` builds the recorded 2026-09-10/09-13 command shape, validates it with `east5_launch_safety --expected-child-zone us-east5-b`, writes `submission_<suffix>/` under the family's `reference_outputs` directory, submits from a subshell that sources `~/.zshrc.secrets` (output redacted through sed; never print the key), and registers the parent in Fieldbook. Regexes: MARINER eval steps `fpu_1e21`, `fpt_3e20`, `fpt_1e21` with training steps `u_snc_cap06_1e21`, `t9_snc_cap08_3e20`, `t9_snc_cap08_1e21`; Olmix `mou_1e21`, `mot_3e20`, `mot_1e21` with `u_kl0p05_cap04_1e21`, `t9_kl0p005_cap04_3e20`, `t9_kl0p005_cap04_1e21`. Use the next unused suffix (MARINER retry8+, Olmix retry2+) and always both the eval and training regex for the rung. Check the parent's dry-run manifest is unchanged (`launch_delphi_matched_olmix_scaling --dry-run` byte-identical to `launch_dry_run/run_manifest.json`) if any launcher file changed. One owner only: never retry a child that is still `running`, and never run two parents for the same step (the executor lock collides).

Never restart or bounce the Iris cluster. Do not change checkpoint intervals or launcher configs. gsutil needs reauthentication; use fsspec/gcsfs for reads.

## Progress signals (do not trust the permanent keeps alone)

- Permanent checkpoints every 5,000 steps under `<output path>/checkpoints/step-N`.
- Rolling temporary checkpoints every 10 minutes under `gs://marin-us-east5/tmp/ttl=14d/checkpoints-temp/marin-us-east5/<output path without bucket>/checkpoints/step-N`; the run resumes from these.
- `<output path>/checkpoints/eval_metrics.jsonl` gets a line every 1,000 steps.

Output paths: `gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_frozen_procedure_scaling_v6e_20260908/<run>-<hash>` and `.../delphi_matched_olmix_scaling_v6e_20260910/<run>-<hash>` (hashes: 4da478, ab3b1e, e8e9d7, c7fc1d, ccb282, 54791f, 3f95f2 in the table's order).

`snapshot.py --gcs` in the same directory prints one line per child (state, failures, preemptions, tasks, latest checkpoint step across both locations); `watch.sh` loops it every 10 minutes and prints only changes plus a heartbeat every 3 hours. Edit the `PARENTS` list in `snapshot.py` when you create a retry parent.

## When a run finishes

The parent then runs the native Table-9 eval child (`olmo-base-eval-t9-fp{u,t}-<rung>-s<seed>` or `...-mo{u,t}-...`). When that succeeds, run from the repo root:

```bash
uv run --offline --no-sync python experiments/domain_phase_mix/exploratory/two_phase_many/collect_delphi_frozen_procedure_scaling_20260908.py
uv run --offline --no-sync python experiments/domain_phase_mix/exploratory/two_phase_many/collect_delphi_frozen_procedure_scaling_20260908.py --ladder matched_olmix
```

Report the new `measured_results.csv` rows (Uncheatable BPB and OlmoBaseEval Easy macro BPB) and the final permanent checkpoint step; the paper session takes it from there. The MARINER and Olmix 1e21 final HF exports also unlock the accuracy evaluations (`prepare_table9_checkpoint`), which that session coordinates.

## Records

Fieldbook: MARINER ladder `exp_01m1zy8yths4dqp5bgc0ffzztp`, matched Olmix `exp_01m21dyb8mxjg2aqhtbjncejpc` (retry1 parent job_01m2jbzqbd9t34cps5z974yrdf). Record every retry and every landed rung there (`fieldbook job add`, `fieldbook note add --type checkpoint`). Submission records live under `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_{frozen_procedure_scaling_v6e_20260908,matched_olmix_scaling_v6e_20260910}/submission_*`.
