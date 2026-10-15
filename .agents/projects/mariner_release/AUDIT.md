# Paper table audit

`experiments/domain_phase_mix/release/verify_paper_numbers.py` compares every number printed in the paper's 20 tables
with the dataset build. The registry `experiments/domain_phase_mix/release/paper_cells.csv` maps each number to a query
over the build tables. The audit reports differences and never edits the LaTeX.

Result on 2026-10-15: the 20 tables print 1,768 numbers that need a check (1,758 outside the row-label column, plus
the ten registered row labels of `tab:a-heteroskedastic`). 1,656 are traced to the build: 1,655 match at the paper's
printed decimals and one differs. 88 are registered skips: 54 configuration constants and 34 numbers of the two
deferred two-phase tables. 24 numbers of `tab:a-unimax-cap` are untraced, because the build does not hold the UniMax cap
sweep's cap-1, cap-4 and cap-12 runs. The audit therefore exits 1: one difference, and 24 untraced numbers under
`--fail-on-untraced`.

## Run

```bash
uv run python -m experiments.domain_phase_mix.release.export_dataset
uv run python -m experiments.domain_phase_mix.release.verify_paper_numbers --build release_build \
  --sections "<paper folder>/sections" --fail-on-untraced --allow-registered-skips
```

- Build: 34 tables, MANIFEST.json `commit` 6bfb359981, built 2026-10-15T10:33Z, 0 consistency problems.
- Paper: `sections/` of the paper folder on Google Drive. The tables are in `results.tex` (modified 2026-10-12 20:13,
  SHA-256 prefix e2a47c370e100b1e), `surrogate.tex` (2026-10-13 02:17, 81cc91596e2a38ea), `appendix.tex` (2026-10-13
  18:21, 56d9771c55d2d188) and `appendix_analysis.tex` (2026-10-14 01:25, 61aeead7375b3d31), local times.
- Output: `1 differences, 24 untraced cells, 88 registered skips`, exit code 1.
- `tests/domain_phase_mix/release/test_verify_paper_numbers.py::test_registry_checks_every_number_of_the_paper` runs
  the same audit and pins the known difference.

## Per table

A number is traced when the registry gives it a query. "Differ" counts traced numbers whose query value rounds to
another value than the paper prints.

| Table | Numbers | Traced | Constant | Deferred | Untraced | Differ | Build tables queried |
|---|---:|---:|---:|---:|---:|---:|---|
| tab:scaling-main | 44 | 44 | | | | 0 | ladder, ladder_accuracy |
| tab:ablations | 10 | 10 | | | | 0 | validation_3e18 |
| tab:fit | 45 | 45 | | | | 0 | fits_reliability |
| tab:a-snr | 364 | 364 | | | | 0 | fits_reliability |
| tab:a-snr-subset | 36 | 36 | | | | 0 | validation_3e18, fits_reliability, fits_proposal_predictions |
| tab:a-heteroskedastic | 40 | 40 | | | | 0 | starcoder_repeat_noise |
| tab:a-factorial-predictions | 36 | 36 | | | | 0 | validation_3e18, heldout, fits_proposal_predictions |
| tab:a-matched-swarms | 18 | | | 18 | | | |
| tab:a-phase-fiber | 16 | | | 16 | | | |
| tab:a-buckets | 80 | 80 | | | | 0 | partition |
| tab:a-hyperparameters | 53 | 24 | 29 | | | 0 | ladder, swarm_*, heldout |
| tab:a-scaling-configurations | 16 | 16 | | | | 0 | ladder |
| tab:a-olmix-cap-kl | 92 | 92 | | | | 0 | validation_3e18 |
| tab:a-unimax-cap | 40 | 16 | | | 24 | 1 | ladder, tasks |
| tab:a-regmix-argmin | 32 | 32 | | | | 0 | validation_3e18, fits_proposal_predictions |
| tab:a-kl-ablation | 38 | 38 | | | | 0 | validation_3e18, fits_proposal_predictions |
| tab:a-ablations-full | 193 | 168 | 25 | | | 0 | fits_surrogate_benchmark |
| tab:a-baseline-validation | 84 | 84 | | | | 0 | validation_3e18, swarm_qwen3_360m_1p6b |
| tab:a-compute-equivalence | 72 | 72 | | | | 0 | ladder_compute_equivalence |
| tab:a-accuracy-components | 459 | 459 | | | | 0 | ladder_accuracy |
| Total | 1,768 | 1,656 | 54 | 34 | 24 | 1 | |

## Differences

`tab:a-unimax-cap`, row Proportional, column Math (7): the paper prints 1.099. The build gives 1.098485, which rounds
to 1.098. The query is `ladder[method=='proportional' & compute_flops==3e18 & data_seed==660700]
.mean_of(tasks[task_id.str.startswith('minerva_math_')].task_id).value`, the mean of the June proportional run's seven
Minerva MATH components. The paper's table matches the results table of the cap sweep's README
(`.agents/projects/unimax_cap_sweep_20260922/README.md`), which also prints 1.099. The README's source,
`table9_results.json`, gives the same 1.098485 as the build. The printed value matches rounding 1.0985 half up, so the
README likely rounded a four-decimal intermediate. The correction is 1.098.

The same pattern affects one untraced number of that table. Cap 12's Math (7) prints 0.936; `table9_results.json`
gives 0.935454, which rounds to 0.935. The other 20 OlmoBaseEval Easy numbers of the cap-1, cap-4 and cap-12 rows
match that file. If this cell is corrected too, caps 8 and 12 both print 0.935 in that column (0.935380 and 0.935454
at full precision). The bold mark on cap 8, the lowest entry, stays correct.

## Registered skips

Constants (54). The registry rows have an empty query and the note `constant`.

- `tab:a-hyperparameters`, for each of the three swarms: layers, hidden size, MLP intermediate size, query and
  key-value heads (two numbers), head dimension, context length, batch size, peak learning rate, and the Qwen3
  schedule's warmup and decay percentages (two numbers). No build table records architecture or optimizer settings.
  The traced rows of this table (parameter counts, tokens, tokens per parameter, FLOPs per run, fit-swarm,
  repeat and held-out run counts) query the swarm tables, `heldout` and the ladder's 3e18 rung, which trains the
  Qwen3 swarm's model on its token budget.
- `tab:a-ablations-full`, column Params: the nominal number of response parameters per task, a property of each model
  form (for example 2M + 5 = 83 for MARINER with M = 39 buckets). The observatory records the fitted design columns,
  which differ from these nominal counts.

Deferred two-phase numbers (34), note `deferred_two_phase`: all of `tab:a-matched-swarms` (18) and
`tab:a-phase-fiber` (16). This release omits the two-phase swarms.

Untraced numbers (24), note `untraced: ...`: the cap-1, cap-4 and cap-12 rows of `tab:a-unimax-cap`. The 21
OlmoBaseEval Easy numbers come from the sweep's local `table9_results.json`. The three Uncheatable values (1.036,
1.024, 1.025) exist only in the README's table and in each run's `checkpoints/eval_metrics.jsonl` on GCS, under
`gs://marin-us-east5/.../delphi_baseline_mixtures_issue6607_20260623/<run>-<hash>/`. The runs' hashed names appear
only in that README. A partial row in `validation_3e18` would break that table's rule that a run has all 58 per-task
values or none. To trace the 24 numbers, pull the three runs' final Uncheatable components and add the launch to
`validation_manifest.csv` as `unimax_cap_sweep`. The Proportional and cap-8 rows are traced; they are ladder runs.

## Tables added to the build

The audit needed three sources that no build table held. Each was added to an existing config with a test:

- `starcoder_repeat_noise` (config `starcoder`, 50 rows): the repeat-noise panel behind `tab:a-heteroskedastic`, ten
  two-phase Nemotron-CC/StarCoder mixtures, five data seeds each, on the 60M-parameter proxy for 1.0B tokens. Per run:
  mixture label, StarCoder share of each phase, data seed, final programming-languages and C4 BPB. Sources:
  `experiments/domain_phase_mix/exploratory/reference_outputs/starcoder_heteroskedastic_snr_20260523/`
  (`collected_train_only_metrics_live.csv`, checked against the launch's `training_manifest.csv`). The test checks that
  the per-mixture mean and std reproduce the heteroskedasticity analysis (`anchor_metric_noise_features.csv`) to 1e-12.
- `ladder_compute_equivalence` (own config folder, 24 rows): the power laws and compute multiples of
  `tab:a-compute-equivalence`, from `RO/compute_equivalence_20260915/` (`compute_equivalence.csv`, `fits.json`). The
  test checks the copy against the ladder runs: MARINER's losses are the ladder's rung means, each law's residual over
  its baseline's four rung means equals the recorded RMSE, and each law reaches MARINER's loss at the matched compute.
- `fits_surrogate_benchmark` (config `fits`, 168 rows): out-of-fold metrics of the 28 models of
  `tab:a-ablations-full` on the three swarms, each from the single-phase observatory run that the paper's table reports
  (`fits.SURROGATE_BENCHMARK`). Runs of 2026-09-08 and later re-fit MARINER, Olmix, the linear model and the
  comparators bit for bit. Earlier runs of the additive response, Olmix and the linear model, and the first two
  2026-09-16 runs of the two convex per-bucket models, report other values; the paper uses the third, `20260916c`.

`figures_tables.csv` now lists configs for these three tables in place of `untraced` (the Params column of
`tab:a-ablations-full` adds `constant`).

## Strength of the checks

- Several build tables copy the outputs of the scripts that produced the paper's tables (`fits_reliability`,
  `fits_proposal_predictions`, `fits_surrogate_benchmark`, `ladder_compute_equivalence`). For these numbers the audit
  confirms the transcription and the copy. Only the compute-equivalence test ties its copy to the run tables; the
  other copies are checked against their source files.
- `ladder_accuracy.n_examples` was read from the accuracy builder's `rows.tex`, so the $n$ column of
  `tab:a-accuracy-components` checks transcription only.
- The registry identifies the rows of `tab:a-heteroskedastic` by their printed StarCoder shares. Those two columns
  confirm the identification; the mean and std columns are independent checks.
- Italic entries of `tab:a-olmix-cap-kl` point to the trained run whose loss they repeat. The audit does not test the
  caption's rule that such a policy lies within total variation 0.01 of the trained one.
- The standard errors of `tab:a-factorial-predictions` use the 20 data-seed repeats of the centre in `heldout`
  (`proposal_method == 'center_control'`, OBE Easy objective). Their mixture differs from the factorial centre's by at
  most 2.4e-4 in any bucket weight.
- `tab:a-snr-subset` defines its subsets as the components with SNR at least 5 (32) and at least 8 (17) in
  `fits_reliability`; all subset means and paired differences match.
- The last row of `tab:a-baseline-validation` combines the swarm's run `run_00125` (trainer seed 0) with its two
  retrains in `validation_3e18` (`table5_repeats`, trainer seeds 1 and 2, the same data seed).
- The total share of `tab:a-buckets` (100.00) is a sum over itself and cannot differ.
- Bold, italics and daggers are not checked.
- A perturbation check (every float of the build scaled by 1.03 and shifted by 0.01) makes 1,406 of the 1,656 traced
  numbers differ. The other 250 read integer columns, which the check left alone, or are standard deviations,
  standard errors, paired differences and ratios whose printed digits survive the change.

## Maintaining the registry

The registry has one row per printed number: `label, row, col, part, query, decimals, note` (the module docstring of
`verify_paper_numbers.py` defines the columns and the query language). When the paper changes, a registry row whose
cell no longer prints a number, or prints other decimals, makes the audit raise and name the row, and a new number is
reported as untraced. After the paper is corrected, remove the fixed cell from `KNOWN_DIFFERENCES` in
`test_verify_paper_numbers.py`.
