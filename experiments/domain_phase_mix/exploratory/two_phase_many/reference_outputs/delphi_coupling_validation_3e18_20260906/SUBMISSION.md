# Delphi coupled WSPU validation at 3e18

Submitted on 2026-09-06. Iris acknowledged the parent and all nine training
children. At 12:40:21 UTC the parent was running and all children were pending,
with zero failures. The scheduler reported 90 requested CPU cores versus 80
available on matching workers; the autoscaler reported a quota-pool tier block.

[Iris parent](https://iris.oa.dev/#/job/%2Fcalvinxu%2Fdm-delphi-3e18-cwspu-kappa-v6e8-20260906)

| Optimization target | Epoch cap | Kappa | Data seed | Training runs |
| --- | ---: | --- | ---: | ---: |
| Uncheatable | 6 | 0.25, 0.5, 1 | 666200 | 3 |
| Table 9 | 6 | 0.25, 0.5, 1 | 662009 | 3 |
| Table 9 | 8 | 0.25, 0.5, 1 | 662009 | 3 |

KL coefficient is zero and trainer seed is zero throughout. Each training uses
the existing 3e18 recipe: 358,304,128 total parameters, batch 128, sequence 4096,
3007 steps and 1,576,534,016 tokens. The final HF checkpoint is step 3006.
Both phases use identical mixture weights. Every candidate receives inline
Uncheatable evaluation and native Table 9 evaluation after final HF export.

The CPU parent is in us-east5-a; training and evaluation children request v6e-8
in us-east5-b. All data, checkpoint and executor paths use gs://marin-us-east5.
The launcher releases all nine rows with max_concurrent=9. Resume uses the
same command, run identities and durable output paths; check Iris and
Fieldbook before any resubmission.

The surrogate parameters and anchor remain fitted to the canonical 280 rows.
The requested kappa values were fixed before this validation. No running
cross-scale ladder results informed the policies. Continuous optima were
converted to exact counts/2048 and improved with the existing one-count
exchange procedure evaluated on the frozen coupled predictor. This establishes
a local integer optimum, not a global integer certificate. All nine mixtures
are distinct and respect their caps. Maximum allocation transfer from the
continuous optimum is 0.3039%; maximum predicted BPB increase is 0.000110.
The same materialization reproduces the historical kappa-zero comparator
counts exactly for Uncheatable cap 6 and Table 9 caps 6 and 8.

The prospective comparison is each frozen policy's realized macro BPB against
its target/cap matched-seed kappa-zero control, with the full component changes
and predicted-versus-realized gain reported for all nine cells. Report both
evaluation suites for every run. These single-seed comparisons do not establish
seed robustness or justify selecting a successor from the best observed cell.

Reproduction and submission evidence:

- `offline_materialization/`: nine continuous policies, 45 restart records,
  source hashes, protocol, and numerical validation. All nine selected policies
  converged; 44 of 45 restarts converged.
- `runtime_materialization/`: exact candidate table, policy mapping, cap checks,
  historical control parity, materializer, and manifest.
- `launch_dry_run/`: target-specific resolved training manifests and run specs.
- `submission/launch_command.sh`: exact submitted command.
- `submission/launch_source_hashes.json` and `submission/source_snapshot/`:
  submitted launcher, shared recipe, environment lock and cluster configuration.
- `submission/preflight.json`: graph construction, lint, types, nine shared
  launcher tests, and region-local launch checks passed. The materializers also
  passed their numerical checks. No training smoke runs were used.
- `submission/submit.log`, `submission/iris_child_snapshot.json`, and
  `submission/fieldbook_children.json`: acknowledgement, observed queue state,
  and exact run-to-job mapping.

Candidate CSV SHA256:
`e72d1f2c9b3acc6ffde470dbd05db7ed2cf50d5a8c28fddfcad0d5d547c0f241`.

Fieldbook experiment: `exp_01m1vbajfrdebbsatntj4sm00z`.
Fieldbook parent job: `job_01m1vbkbfhwbm8709xyt4cfzhk`.
Iris parent: `/calvinxu/dm-delphi-3e18-cwspu-kappa-v6e8-20260906`.
