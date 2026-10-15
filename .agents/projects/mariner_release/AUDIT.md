# Paper table audit

`experiments/domain_phase_mix/release/verify_paper_numbers.py` compares every number printed in the paper's 20 tables
with the dataset build. The registry `experiments/domain_phase_mix/release/paper_cells.csv` maps each number to a query
over the build tables. The audit reports differences and never edits the LaTeX.

Result on 2026-10-15, after the paper corrected two cells of `tab:a-unimax-cap` and the build gained the Uncheatable
components of the UniMax cap sweep's runs: the 20 tables print 1,768 numbers that need a check (1,758 outside the
row-label column, plus the ten registered row labels of `tab:a-heteroskedastic`). 1,680 are traced to the build, and all
of them match at the paper's printed decimals. The other 88 are registered skips: 54 configuration constants and 34
numbers of the two deferred two-phase tables. No number is untraced, so the audit exits 0 under `--fail-on-untraced
--allow-registered-skips`.

## Run

```bash
uv run python -m experiments.domain_phase_mix.release.export_dataset
uv run python -m experiments.domain_phase_mix.release.verify_paper_numbers --build release_build \
  --sections "<paper folder>/sections" --fail-on-untraced --allow-registered-skips
```

- Build: 35 tables, MANIFEST.json `commit` 772b1a5be9, built 2026-10-15T23:19Z, 0 consistency problems.
- Paper: `sections/` of the paper folder on Google Drive. The tables are in `results.tex` (modified 2026-10-12 20:13,
  SHA-256 prefix e2a47c370e100b1e), `surrogate.tex` (2026-10-13 02:17, 81cc91596e2a38ea), `appendix.tex` (2026-10-15
  14:15, a6b17d4f7fffadd1) and `appendix_analysis.tex` (2026-10-14 01:25, 61aeead7375b3d31), local times.
- Output: `0 differences, 0 untraced cells, 88 registered skips`, exit code 0: `--allow-registered-skips` accepts the
  constant and deferred two-phase cells, the only skips.
- `tests/domain_phase_mix/release/test_verify_paper_numbers.py::test_registry_checks_every_number_of_the_paper` runs
  the same audit and expects no differences (`KNOWN_DIFFERENCES` is empty) and no skips other than constants and
  deferred two-phase numbers.

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
| tab:a-unimax-cap | 40 | 40 | | | | 0 | ladder, ladder_unimax_cap_sweep, tasks |
| tab:a-regmix-argmin | 32 | 32 | | | | 0 | validation_3e18, fits_proposal_predictions |
| tab:a-kl-ablation | 38 | 38 | | | | 0 | validation_3e18, fits_proposal_predictions |
| tab:a-ablations-full | 193 | 168 | 25 | | | 0 | fits_surrogate_benchmark |
| tab:a-baseline-validation | 84 | 84 | | | | 0 | validation_3e18, swarm_qwen3_360m_1p6b |
| tab:a-compute-equivalence | 72 | 72 | | | | 0 | ladder_compute_equivalence |
| tab:a-accuracy-components | 459 | 459 | | | | 0 | ladder_accuracy |
| Total | 1,768 | 1,680 | 54 | 34 | 0 | 0 | |

## Differences (corrected in the paper on 2026-10-15; the audit now reports 0 differences)

Both differences are in `tab:a-unimax-cap`, column Math (7), the mean of a run's seven Minerva MATH components. The
paper's rows match the results table of the cap sweep's README (`.agents/projects/unimax_cap_sweep_20260922/README.md`),
which prints the same two values. The README's source, `table9_results.json`, gives the build's values.

| Row | Paper | Build | Correct rounding | Query |
|---|---:|---:|---:|---|
| Proportional (r0 c4) | 1.099 | 1.098485 | 1.098 | `ladder[method=='proportional' & compute_flops==3e18 & data_seed==660700].mean_of(...)` |
| Cap 12 (r4 c4) | 0.936 | 0.935454 | 0.935 | `ladder_unimax_cap_sweep[epoch_cap==12].mean_of(...)` |

Both printed values match rounding a four-decimal intermediate (1.0985, 0.9355) half up, so the README likely rounded
twice. The table's other 38 numbers, including the other 20 OlmoBaseEval Easy numbers and the three Uncheatable
numbers of the sweep's runs, match. With both corrections, caps 8 and 12 print 0.935 in that column (0.935380 and
0.935454 at full precision); the bold mark on cap 8, the lowest entry, stays correct.

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

## Tables added to the build

The audit needed four sources that no build table held. Each was added to an existing exporter with a test:

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
- `ladder_unimax_cap_sweep` (own config folder, 3 rows): the UniMax cap sweep's runs at caps 1, 4 and 12, from
  `.agents/projects/unimax_cap_sweep_20260922/table9_results.json`, the launch's `run_specs.json` and `summary.json`
  under `RO/delphi_baseline_mixtures_issue6607_20260623/`, and the runs' W&B summaries and configs, read once on
  2026-10-15 and frozen in
  `experiments/domain_phase_mix/release/configs/unimax_cap_sweep_wandb_components_20261015.csv`. Per run: cap, seeds,
  the 58 components, `components_available` and both aggregates. The OlmoBaseEval Easy mean equals the recorded mean to
  1e-12. The fixed byte shares of the seven pulled Uncheatable components give 1.036420, 1.023785 and 1.025084; the
  sweep's README, which weighted each run's evaluation log, prints 1.0364, 1.0238 and 1.0251. The exporter joins each
  pulled run to its run spec by name (the W&B id is the run name with its hash), requires one pulled run per spec, no
  unused pull and the spec's data seed, and checks each run's model size and token count against the ladder's 3e18 rung.
  The table has no observation ids, because its three runs appear in no other table; the cross-table consistency check
  skips tables without ids. The tests check the three caps, that the same results file scores the ladder's UniMax-8
  seed-0 run with the ladder's components (to 1e-15), that each cap's row holds its pulled run's components and data
  seed, and that its Uncheatable aggregate rounds to the README's value.

`figures_tables.csv` lists every config that each table's registry queries read; a test in
`test_verify_paper_numbers.py` enforces this. The Params column of `tab:a-ablations-full` adds `constant`.

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
- A perturbation check (every float of the build scaled by 1.03 and shifted by 0.01, each query selecting its rows on
  the unperturbed tables) makes 1,422 of the 1,680 traced numbers differ, the cap sweep's three Uncheatable numbers
  among them. The other 258 read integer columns, which the check leaves alone (example counts, pool tokens, parameter
  and token counts, run counts), or are standard deviations, standard errors, differences, and five small values
  printed to one decimal, whose printed digits survive the change. The earlier record, 1,427 of 1,677, came from a
  variant of the check that it does not describe; the check above gives 1,419 of 1,677 on the build at fbe103a76f.

## Maintaining the registry

The registry has one row per printed number: `label, row, col, part, query, decimals, note` (the module docstring of
`verify_paper_numbers.py` defines the columns and the query language). Regenerate it with
`uv run python -m experiments.domain_phase_mix.release.gen_registry`, which maps each table's row labels and columns
to queries and reads the row and column indices from the parsed paper. When the paper changes, a registry row whose
cell no longer prints a number, or prints other decimals, makes the audit raise and name the row, and a new number is
reported as untraced. After the paper is corrected, remove the fixed cells from `KNOWN_DIFFERENCES` in
`test_verify_paper_numbers.py`.


## Update 2026-10-15

Both tab:a-unimax-cap cells were corrected in sections/appendix.tex (1.099 -> 1.098; 0.936 -> 0.935), and the audit
reported 0 differences.

The build at 3681ab32b3 replaces `aggregate_version` and the `_source` and `_fixed` aggregate columns with
`bpb__uncheatable` and `bpb__obe_easy` under one rule, and the registry queries those two columns. The tab:scaling-main
values, which now read the ladder's computed aggregates, moved by at most 7.7e-8.

The build at 772b1a5be9 adds the seven Uncheatable components of the cap sweep's cap-1, cap-4 and cap-12 runs, frozen
from their W&B run summaries, and the registry traces the table's three Uncheatable cells, the audit's last untraced
numbers. The other 34 tables are byte-identical to the build at fbe103a76f. The audit reports `0 differences, 0
untraced cells, 88 registered skips` and exits 0 under `--fail-on-untraced --allow-registered-skips`.
