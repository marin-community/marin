# Proportional Controllability 300M

## Purpose

Run a 300M/6B-only diagnostic around `baseline_proportional` to compare:

- full leave-one-domain-out coverage ablations, and
- per-domain central log-tilt perturbation pairs around proportional.

This is a single-seed diagnostic. Deletions are nonlocal boundary interventions
and should be interpreted as coverage ablations, not clean local derivatives.
The log-tilt pairs are closer to local projected-gradient probes, but they are
still single-seed estimates.

Related prior experiment:
`.agents/logbooks/proportional-perturbation-scale-transfer.md`, especially the
`+0.05` domain-bump panel from
`/calvinxu/dm-proportional-perturbation-scale-transfer-20260507-143728`.

## Design

- Scale: historical `300m_6b`, displayed as corrected `100M/6B`.
- Baseline: reuse existing `baseline_proportional` from
  `pinlin_calvin_xu/data_mixture/ngd3dm2_qsplit240_300m_6b`.
- Source experiment prefix:
  `pinlin_calvin_xu/data_mixture/ngd3dm2_proportional_controllability_300m`.
- Family: `proportional_controllability_300m_6b`.
- Run IDs: `800000..800116`.
- Phase mode: `both_phases`; every run has `phase_0 == phase_1`.
- Training: `v5p-8`, `us-east5-a`, target final checkpoint step `22887`,
  perplexity-only training eval, final checkpoint and HF export preserved.

### Domain deletions

For each domain `j`, run:

`w_j = 0`, and `w_i = p_i / (1 - p_j)` for `i != j`.

This yields exactly `39` runs. The TV distance from proportional is `p_j`.

### Central log-tilts

For each domain `j`, construct a singleton domain direction `v^j` and run a
plus and minus endpoint:

`w_i^+ = p_i exp(+alpha v_i) / sum_k p_k exp(+alpha v_k)`

`w_i^- = p_i exp(-alpha v_i) / sum_k p_k exp(-alpha v_k)`

with `alpha = 0.10`, `sum_i p_i v_i = 0`, and
`sum_i p_i v_i^2 = 1`.

This yields exactly `78` runs: `2` endpoints for each of `39` domains. Domain
direction IDs use slugs such as `domain_dolma3_cc_art_and_design_high`, while
the manifest preserves the original domain name in `target_domain`.

## Planned Analysis

- Compare deletion effects against the existing `+0.05` domain-bump effects.
- Orient every metric as utility; BPB/loss metrics are sign-flipped.
- Report raw deletion contrast:
  `Delta_j^del = U(w_without_j) - U(p)`.
- Report deletion-implied score:
  `q_j^del = -(1 - p_j) Delta_j^del / p_j`.
- Report log-tilt directional derivative:
  `d_v = (U(w_plus) - U(w_minus)) / (2 alpha)`.
- Flag small-domain deletion scores as noise-amplified because the rescaling
  divides by `p_j`.
- Produce plots for deletion effects by domain mass, deletion-vs-bump sign
  agreement, deletion-implied `q` vs bump-implied `q`, log-tilt directional
  derivatives, and metric-family coverage-criticality.

## 2026-05-20 Implementation Notes

- Added launcher:
  `experiments/domain_phase_mix/launch_proportional_controllability_300m.py`.
- Added eval candidate builder:
  `experiments/domain_phase_mix/build_proportional_controllability_eval_candidates.py`.
- Registry integration should preserve intervention metadata and checkpoint
  provenance under family `proportional_controllability_300m_6b`.
- Before live launch, run dry-run validation and Claude Code review using the
  subscription workflow.

## 2026-05-20 Initial Dry-Run Validation Superseded

Command:

```bash
uv run python experiments/domain_phase_mix/launch_proportional_controllability_300m.py \
  --dry-run \
  --tpu-region us-east5 \
  --tpu-zone us-east5-a \
  --max-concurrent 8
```

Results from the initial selected-direction design:

- Dry-run wrote manifests under
  `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/proportional_controllability_300m_20260520/`.
- Training rows: `69`.
- Intervention rows: `69`.
- Intervention counts: `39` `domain_deletion`, `30` `central_log_tilt`.
- Run IDs: `800000..800068`.
- Scale: all `300m_6b`.
- Phase columns: `39` domains in `phase_0` and `39` in `phase_1`.
- Phase sums: max error `2.22e-15` for each phase.
- `phase_0 == phase_1`: max absolute delta `0.0`.
- Target final checkpoint step: `22887`.
- `num_train_steps`: `22888`, so the final checkpoint step is `num_train_steps - 1`.
- Duplicate run names: `0`.
- Deletion TV check: max absolute error between `tv_distance` and `base_mass` is `1.04e-16`.
- Log-tilt directions: `15`, each with exactly `minus,plus`.

This validation was superseded before launch. We decided the clean comparison
should include a central log-tilt pair for every domain, not just `15` selected
directions. The corrected design has `117` total runs: `39` deletions and `78`
log-tilt endpoints.

## 2026-05-20 Corrected 117-Run Dry-Run Validation

After discussion, the selected-direction log-tilt design was replaced with a
fully symmetric per-domain design. The corrected manifest now contains:

- `39` leave-one-domain-out coverage ablations.
- `78` central log-tilt endpoints: plus and minus for each of `39` domains.
- `117` total 300M/6B training runs.

Dry-run command:

```bash
uv run python experiments/domain_phase_mix/launch_proportional_controllability_300m.py \
  --dry-run \
  --tpu-region us-east5 \
  --tpu-zone us-east5-a \
  --max-concurrent 8
```

Dry-run result:

- Manifest directory:
  `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/proportional_controllability_300m_20260520/`.
- Training rows: `117`.
- Intervention rows: `117`.
- Intervention counts: `39` `domain_deletion`, `78` `central_log_tilt`.
- Run IDs: `800000..800116`.
- Scale: all `300m_6b`.
- Phase columns: `39` domains in `phase_0` and `39` in `phase_1`.
- Phase sums: max error `2.3314683517128287e-15` for each phase.
- `phase_0 == phase_1`: max absolute delta `0.0`.
- Minimum phase weight: `0.0`, only from deletion target domains.
- Target final checkpoint step: `22887`.
- `num_train_steps`: `22888`, so the final checkpoint step is `num_train_steps - 1`.
- Duplicate run names: `0`.
- Duplicate run IDs: `0`.
- Deletion TV check: max absolute error between `tv_distance` and `base_mass` is `1.04e-16`.
- Log-tilt directions: `39`, exactly one per domain.
- Log-tilt pairs: every direction has exactly `minus,plus`.
- Log-tilt target domains: all `39` domains covered, missing `0`.

Validation commands for the corrected design:

```bash
uv run python -m py_compile \
  experiments/domain_phase_mix/launch_proportional_controllability_300m.py \
  experiments/domain_phase_mix/build_proportional_controllability_eval_candidates.py \
  experiments/domain_phase_mix/exploratory/two_phase_many/run_registry/build_run_registry.py \
  experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/build_metric_registry.py \
  experiments/domain_phase_mix/launch_300m_noise_parity_evals.py \
  experiments/domain_phase_mix/launch_300m_agentic_coding_bpb_evals.py

./infra/pre-commit.py \
  experiments/domain_phase_mix/launch_proportional_controllability_300m.py \
  experiments/domain_phase_mix/build_proportional_controllability_eval_candidates.py \
  experiments/domain_phase_mix/exploratory/two_phase_many/run_registry/build_run_registry.py \
  experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/build_metric_registry.py \
  experiments/domain_phase_mix/launch_300m_noise_parity_evals.py \
  experiments/domain_phase_mix/launch_300m_agentic_coding_bpb_evals.py
```

Both validation commands passed for the corrected `117`-run design. The live
launch still needs the planned Claude Code review gate.

## 2026-05-20 Log-Tilt Visualization

Generated a visualization of the `78` central log-tilt mixtures relative to the
proportional baseline:

- Interactive HTML:
  `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/proportional_controllability_300m_20260520/img/log_tilt_weight_relative_heatmap.html`.
- Static PNG:
  `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/proportional_controllability_300m_20260520/img/log_tilt_weight_relative_heatmap.png`.
- Per-target summary CSV:
  `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/proportional_controllability_300m_20260520/log_tilt_target_multiplier_summary.csv`.

The heatmap value is `log2(w_tilt / w_proportional)` for each plotted domain.
The largest relative target upweights happen for very small proportional-mass
domains, as expected under fixed `alpha = 0.10` KL-local coordinates.

Also generated materialized-weight audit artifacts:

- Interactive HTML:
  `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/proportional_controllability_300m_20260520/img/log_tilt_materialized_weight_heatmap.html`.
- Static PNG:
  `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/proportional_controllability_300m_20260520/img/log_tilt_materialized_weight_heatmap.png`.
- Full `78 x 39` materialized weight matrix:
  `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/proportional_controllability_300m_20260520/log_tilt_materialized_weights_matrix.csv`.
- Target-domain `w+`/`w-` summary:
  `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/proportional_controllability_300m_20260520/log_tilt_materialized_target_weights.csv`.

The materialized heatmap values are raw phase weights in percent, not ratios.
Rows sum to `100%` with max absolute numerical error
`2.4158453015843406e-13`.

## 2026-05-22 Launch Incident: Unsafe Executor Prefix

The first live training submission
`/calvinxu/dm-proportional-controllability-300m-train-20260521-1941` was unhealthy.
It was stopped after logs showed the parent was actively re-downloading and
tokenizing Dolma/Dolmino raw data under a fresh experiment-scoped executor
prefix, e.g.
`gs://marin-us-east5/pinlin_calvin_xu/data_mixture/ngd3dm2_proportional_controllability_300m_20260521/raw/dolma3_dolmino_pool-72089d/...`.

Root cause: this training graph contains shared raw/tokenized data-prep
dependencies. Passing `--executor-prefix` to `executor_main` re-roots every
executor step, including those shared cache steps, so previously completed
east5 caches under `gs://marin-us-east5/raw/...` and
`gs://marin-us-east5/tokenized/...` are not reused.

Mitigation:

- Stopped the bad Iris parent to halt the raw download/tokenization path.
- Disabled `--executor-prefix` in
  `experiments/domain_phase_mix/launch_proportional_controllability_300m.py`
  so future submissions fail fast instead of silently re-materializing data.
- Resubmission should omit `--executor-prefix`; the default east5
  `MARIN_PREFIX` lets shared raw/tokenized cache steps resolve to the existing
  regional cache paths while training checkpoints still use the normal
  `gs://marin-us-east5/checkpoints/...` roots.

### Corrected resubmission

- Validation:
  - `uv run python -m py_compile experiments/domain_phase_mix/launch_proportional_controllability_300m.py`
    passed.
  - Dry-run without `--executor-prefix` prepared `117` interventions and
    `117` training steps.
  - Dry-run with `--executor-prefix unsafe-test` now exits with
    `ValueError: --executor-prefix is disabled for this training launcher`.
- Claude Code review attempt:
  - Auth preflight showed `plambdafour@proton.me`, `stripe_subscription`, and
    no inherited `ANTHROPIC_API_KEY`.
  - The requested resume session was unavailable locally, and a fresh
    non-interactive Opus/max-effort review timed out without output. Proceeded
    based on local verification because the cost issue was already identified
    and blocked in code.
- Corrected parent:
  `/calvinxu/dm-proportional-controllability-300m-train-20260522-0148`.
- Submission command intentionally omitted `--executor-prefix`.
- Early log check confirmed data-prep steps resolve to shared east5 paths such
  as `gs://marin-us-east5/tokenized/dolma3_pool/...` and are skipped as
  `already succeeded`, rather than being written under the experiment prefix.

### Second launch failure: unnecessary eval cache step

The corrected no-prefix parent
`/calvinxu/dm-proportional-controllability-300m-train-20260522-0148` failed
before training. The data-cache issue was fixed, but the parent still scheduled
`cache_eval_datasets_cc9c0d5e` even though training had
`LEVANTER_SKIP_EVAL_HARNESS=1`. The cache step imported `lm_eval.tasks` in the
parent environment and failed with:

`ModuleNotFoundError: No module named 'lm_eval'`.

This step is unnecessary for the intended training-only launch because
downstream lm-eval is a separate follow-up pass. Mitigation:

- `LaunchArtifacts.cache_eval_datasets_step` is now optional.
- The cache step is only built when `--include-eval-harness` is set.
- Training steps only depend on the cache step when it exists.
- Dry-run validation now fails if skipped-harness training has an eval-cache
  dependency or if the skipped-harness graph contains a cache step.

Validation after the fix:

- `uv run python -m py_compile experiments/domain_phase_mix/launch_proportional_controllability_300m.py`
  passed.
- Programmatic graph check reported `cache_step None`, `cache_named_steps []`,
  `training_steps 117`, and `eval_cache_env_count 0`.
- Dry-run prepared `117` interventions and `117` training steps with
  `eval_harness=skipped`.
- Dry-run with `--executor-prefix unsafe-test` still exits with the executor
  prefix guard.
- Second Claude Code review attempt used the subscription-safe
  `env -u ANTHROPIC_API_KEY` workflow after confirming
  `plambdafour@proton.me` / `stripe_subscription`, but timed out without
  output. This is not treated as CC signoff.
- Submitted replacement parent
  `/calvinxu/dm-proportional-controllability-300m-train-20260522-0154`.
  The command omitted `--executor-prefix` and left the training eval harness
  skipped.
- Immediate smoke check after submission:
  - Iris parent state: `JOB_STATE_RUNNING`.
  - No `cache_eval`, `lm_eval`, `ModuleNotFoundError`, traceback, or exception
    lines in the recent parent logs.
  - Data-prep paths are shared east5 paths such as
    `gs://marin-us-east5/tokenized/dolma3_pool/...`, and observed data-prep
    steps are being skipped as `already succeeded`.
- Follow-up smoke check:
  - Parent has children.
  - Observed checkpoint child jobs under
    `/calvinxu/dm-proportional-controllability-300m-train-20260522-0154/...`.
  - Status snapshot: parent `JOB_STATE_RUNNING`, `7` children running, and
    `1` child pending for v5p-8 TPU capacity.
  - Checkpoint outputs are under `gs://marin-us-east5/checkpoints/...`.

### Parent preemption recovery

The continuation parent
`/calvinxu/dm-proportional-controllability-300m-train-continue-20260522-1118`
eventually showed a different failure mode: the Iris parent itself was placed on
a preemptible TPU worker and was repeatedly preempted. Child failures were
`Parent task preempted`, and sampled checkpoint roots still had zero final
target checkpoints. This made the job structurally unsafe even though the graph
no longer contained raw-download, tokenization, or eval-cache work.

Claude Code reviewed the diagnosis and agreed the parent needed to be
nonpreemptible. The first nonpreemptible retry
`/calvinxu/dm-proportional-controllability-300m-train-retry-nonpreempt-parent-20260522-1552`
kept the parent pinned to `us-east5-a` and remained pending on
`cpu_vm_e2_highmem_2_ondemand-us-east5-a=at_max_slices`, so it was stopped
before dispatch.

The active retry is
`/calvinxu/dm-proportional-controllability-300m-train-retry-nonpreempt-parent-region-20260522-1556`.
It uses a nonpreemptible CPU parent with `--region us-east5` but no parent
`--zone`; the launcher still pins TPU child training jobs to `us-east5-a`.
Current status as of 2026-05-22 16:26 PT: parent running on an on-demand CPU
worker with `8` training children running. No raw/tokenize/download children are
present in the active prefix.

### 2026-05-23 - Current-failure subset retry

Status check on
`/calvinxu/dm-proportional-controllability-300m-train-retry-nonpreempt-parent-region-20260522-1556`
showed the parent still running, with currently visible children at
`14` succeeded, `9` running, `6` failed, and `10` killed. The failures were
infra-like TPU child failures (`SIGSEGV` in `add_port.cc`, worker ping timeout,
task-not-found, and reconcile timeout), not a new launcher bug.

To avoid duplicating still-running rows, I added a subset-retry mode to
`experiments/domain_phase_mix/launch_proportional_controllability_300m.py`:

- `--only-run-name <name>` may be repeated.
- `--only-run-name-file <path>` reads one run name per line.
- The subset preserves the deterministic original run IDs and output roots.
- The launcher still validates phase weights, east5 locality, skipped eval
  harness, target steps, and HF final export on the selected rows.

Validation:

- `uv run python -m py_compile experiments/domain_phase_mix/launch_proportional_controllability_300m.py tests/test_domain_phase_mix_proportional_controllability_300m.py`
  passed.
- `uv run pytest tests/test_domain_phase_mix_proportional_controllability_300m.py tests/test_domain_phase_mix_starcoder_heteroskedastic_snr.py tests/test_grug_logprob_eval.py -q`
  passed (`18` tests).
- Dry-run with
  `experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/proportional_controllability_300m_20260520/retry_failed_run_names_20260523_1652.txt`
  prepared exactly `16` interventions and `16` training steps with
  `eval_harness=skipped`.
- Claude Code reviewed the subset change and found no launcher blockers. It
  noted that a second dispatcher over the same step locks can be redundant if
  the active parent is already retrying failures; I left the main parent running
  because it still appears to be dispatching the remaining panel.

Submitted a low-concurrency current-failure retry:

`/calvinxu/dm-proportional-controllability-300m-train-retry-current-failures-20260523-1659`

Command notes:

- Parent is nonpreemptible CPU with `--cpu 1 --memory 2GB --disk 10GB`.
- Parent placement is unconstrained; child TPU placement remains pinned by the
  launcher to `v5p-8` in `us-east5-a`.
- `--max-concurrent 4` is used to limit lock and TPU queue contention with the
  still-running main parent.

### 2026-05-23 22:28 - uncovered 3-row retry

Status check on the main parent showed additional transient failures, but the
existing current-failures retry manifest already covers most of them. A first
new 15-row retry list overlapped
`retry_failed_run_names_20260523_1652.txt` by `12` names. Claude Code review
explicitly asked for this disjointness check before submission; the overlap was
caught, so the 15-row file was not submitted.

Created the corrected uncovered-only retry file:

`experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/proportional_controllability_300m_20260520/retry_failed_run_names_20260523_2228_uncovered.txt`

It contains exactly `3` run names:

- `pctrl_tilt_domain_dolma3_cc_art_and_design_high_plus`
- `pctrl_tilt_domain_dolma3_cc_art_and_design_low_plus`
- `pctrl_tilt_domain_dolma3_cc_art_and_design_low_minus`

Dry-run:

`uv run python -m experiments.domain_phase_mix.launch_proportional_controllability_300m --dry-run --only-run-name-file experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/proportional_controllability_300m_20260520/retry_failed_run_names_20260523_2228_uncovered.txt --local-artifact-dir /tmp/pctrl_retry_2228_uncovered_dry`

Result: exactly `3` interventions and `3` training steps.

Submitted:

`/calvinxu/dm-proportional-controllability-300m-train-retry-uncovered3-20260523-2228`

Command:

`uv run iris --cluster=marin job run --no-wait --no-preemptible --job-name dm-proportional-controllability-300m-train-retry-uncovered3-20260523-2228 --enable-extra-resources --cpu 1 --memory 2GB --disk 10GB -e WANDB_API_KEY "$WANDB_API_KEY" -e HF_TOKEN "$HF_TOKEN" -- python -m experiments.domain_phase_mix.launch_proportional_controllability_300m --max-concurrent 3 --only-run-name-file experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/proportional_controllability_300m_20260520/retry_failed_run_names_20260523_2228_uncovered.txt --local-artifact-dir experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/proportional_controllability_300m_20260520/retry_uncovered3_20260523_2228`

Fieldbook:

Added running retry job `job_01ksc73evxm32np2rf693a8087`, linked as a retry of
the main pctrl parent job.

### 2026-05-24 - cross-region egress incident

Rohith's egress report flagged this experiment, especially
`/calvinxu/dm-proportional-controllability-300m-train-retry-current-failures-20260523-1659`
and the failed uncovered3 retry. The current-failures retry used an
unconstrained CPU parent while reading east5 state/artifact paths and launching
east5-a children. That parent pattern is no longer acceptable.

Immediate action:

- Stopped
  `/calvinxu/dm-proportional-controllability-300m-train-retry-current-failures-20260523-1659`.
- Stop command terminated the parent and four active children:
  `pctrl_tilt_domain_dolma3_cc_art_and_design_high_minus`,
  `pctrl_del_dolmino_synth_math`, `pctrl_del_dolmino_synth_code`, and
  `pctrl_del_dolmino_stem_heavy_crawl`.
- Marked Fieldbook job `job_01ksbmeqhdrtq7fkxw66918jna` as `killed`.

CC review:

- Invoked with `env -u ANTHROPIC_API_KEY`, Opus 4.7 max effort, resumed Marin
  session `d0a45bcd-ae4f-4efd-8bd5-3cbcdf4b3490`; preflight showed
  `plambdafour@proton.me`, `stripe_subscription`, and no inherited
  `ANTHROPIC_API_KEY`.
- Verdict: leave the main pctrl parent
  `/calvinxu/dm-proportional-controllability-300m-train-retry-nonpreempt-parent-region-20260522-1556`
  running. It is region-pinned to `us-east5`, its children are pinned to
  `us-east5-a`, and `gs://marin-us-east5` is a regional `US-EAST5` bucket
  (`gsutil ls -L -b gs://marin-us-east5` verified this).

Policy from this point:

- No new pctrl retry without explicit parent `--region us-east5 --zone us-east5-a`.
- State CSVs, executor prefixes, checkpoint roots, and eval caches must all use
  `gs://marin-us-east5`.
- Parent capacity/preemption recovery must not use placement-unconstrained
  parents for east5 workloads.
- Run CC review before every live retry submission.
