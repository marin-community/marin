# Baseline Scaling Trajectories: Research Logbook

## Scope
- Goal: build the central paper-plot workflow for 1x Chinchilla scaling
  trajectories across corrected `20M/2.6B`, `60M/1.2B`, `100M/6B`,
  `340M/10.4B`, and `900M/24B` scales.
- Primary metric: `eval/uncheatable_eval/bpb`.
- Methods: Proportional, Olmix, Uniform, UniMax, and GRP no-L2.
- Constraint: lm-eval harness is deferred; target cells need final checkpoints
  and perplexity/next-token metrics only.

## Baseline
- Date: 2026-04-24
- Code refs:
  - `experiments/domain_phase_mix/exploratory/two_phase_many/analysis_dataset/nd_scale_runs.csv`
  - `experiments/domain_phase_mix/exploratory/two_phase_many/run_registry/logical_runs.csv`
  - `experiments/domain_phase_mix/exploratory/paper_plots/baseline_scaling_trajectories.py`
- Baseline issue: historical one-off plotting mixed target-ready rows with stale
  packet metrics and registry latest metrics. The central plot now marks that
  distinction explicitly.

## Experiment Log
### 2026-04-24 - Central Plot Bootstrap
- Hypothesis: the first useful paper plot should prioritize provenance clarity
  over drawing a fully connected curve from stale rows.
- Command:
  ```bash
  uv run experiments/domain_phase_mix/exploratory/paper_plots/baseline_scaling_trajectories.py
  ```
- Config: 25 canonical cells, `target_budget_multiplier=1.0`, corrected
  non-embedding scale labels.
- Result: rendered
  `experiments/domain_phase_mix/exploratory/paper_plots/img/baseline_scaling_trajectories.html`
  and `.png`. Current audit status is 12 target-ready cells, 8 diagnostic-only
  cells, and 5 missing/relaunch cells.
- Interpretation: solid markers are target-ready; hollow markers identify
  diagnostic rows that must be relaunched or have target-step perplexity
  recovered before presentation.
- Next action: use `baseline_scaling_trajectories_manifest.csv` to submit only
  missing/perplexity-only cells. Launchers now support the simple
  `--perplexity-only` / `--skip-eval-harness` alias, which sets
  `LEVANTER_SKIP_EVAL_HARNESS=1` on training steps.

### 2026-04-24 - Perplexity-Only Launch Plumbing
- Change: added the `--perplexity-only` / `--skip-eval-harness` alias to the
  strong-tier, single-cell relaunch, stratified, 1.2B pilot, and GRP raw-optimum
  launchers. The alias keeps normal validation/perplexity and checkpointing, but
  sets `LEVANTER_SKIP_EVAL_HARNESS=1` to skip lm-eval harness work.
- Change: pinned 1.2B default launch recipes to `v5p-64` in `us-east5-a`, and
  updated registry resubmit hints to include the same TPU type, region, zone,
  and perplexity-only flag.
- Current audit: the central manifest has 25 canonical cells. Of these, 12 are
  target-ready, 8 are diagnostic-only, 4 are missing, and 1 needs relaunch.
- No live training jobs were submitted in this pass. The current output is the
  audit/plot workflow plus launch-flag plumbing; the remaining missing cells
  should be submitted from the manifest once the exact launch recipes are
  selected.
