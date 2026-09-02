# Debugging log for HumanEval scaling plot

Investigate whether exact overlap in `baseline_scaling_humaneval_10shot.png`
indicates a plot/data bug.

## Initial Status

The HumanEval trajectory plot has several exactly overlapping points/segments,
especially at small scales.

## Hypothesis 1: duplicated result source rows caused overlapping trajectories

Checked `baseline_scaling_downstream_eval_metrics_merged.csv` and raw
`humaneval_10shot` result JSONs for representative overlapping rows.

## Results

No duplicate-output bug found.

- The task config is `EvalTaskConfig(name="humaneval", num_fewshot=10, task_alias="humaneval_10shot")`.
- The metric is `pass@1,create_test`.
- Each HumanEval result reports `n-samples = {"original": 164, "effective": 164}`.
- The apparent overlaps correspond to identical integer pass counts over 164 tasks:
  - `0.000000` = `0/164`
  - `0.006098` = `1/164`
  - `0.024390` = `4/164`
  - `0.079268` = `13/164`
  - `0.085366` = `14/164`
- Example overlapping rows use different result paths:
  - Proportional `900M/24B`: `4/164`
  - UniMax `900M/24B`: `4/164`
  - GRP no-L2 `60M/1.2B`: `1/164`
  - Olmix `60M/1.2B`: `1/164`

## Interpretation

The plot is mostly behaving correctly, but HumanEval is a very coarse metric in
this regime. At these small pass rates, one task is `0.6098` percentage points,
so ties are common and exact overlaps are expected. Treat the plot as a coarse
sanity check rather than a fine ranking signal.

## Changes

- Updated the HumanEval y-axis label to state that one task is `0.61` percentage
  points.
- Updated hover text to show the estimated number of correct tasks out of 164.
