# Factor-DSP Gradio Dashboard

Agent-first interactive dashboard for the current factor-DSP mixture recommendation workflow.

Launch from the Marin repo root:

```bash
uv run --script experiments/domain_phase_mix/exploratory/two_phase_many/factor_dsp_gradio_dashboard/app.py --server-port 7860 --no-share
```

Then open `http://127.0.0.1:7860`.

## Interaction Model

- Every task slider starts at `0`, meaning proportional baseline performance.
- Task sliders use standardized oriented-delta units and are bounded to `[-5, +5]` with `0.1` steps.
- Moving a task slider records that value as a minimum target constraint for that task.
- Checking a task's `Lock` checkbox records the current slider value as a constraint, including exactly `0`.
- The app searches the precomputed candidate library and picks the best feasible candidate by lower-confidence-bound aggregate gain, subject to global filters and active task constraints.
- The current recommendation panel explicitly reports the matched precomputed candidate row, including source, score kind, predicted aggregate gain, nearest observed TV, and task-prediction availability.
- After a candidate is selected, all task sliders are programmatically moved to that candidate's predicted task deltas.
- The locked constraints table preserves the thresholds that were requested, because the sliders now show candidate outcomes rather than target thresholds.
- If no cached candidate satisfies the active constraints, the recommendation panel states that explicitly and locked sliders remain at their requested thresholds.
- Each task slider label starts with its readiness and train Pearson, e.g. `[usable r=0.62]`. Low-readiness metrics should not be treated as reliable steering objectives.
- The starting mixture dropdown lists only curated named baselines and DSP/path optima, not all precomputed cache rows.
- Selecting a starting mixture clears task constraints and moves sliders to that candidate's predicted deltas.
- The dropdown remains the selected starting mixture even when constraints match a different cached recommendation.
- Automatic recommendation is always on once task constraints are set; there is no manual/off mode.
- `Clear task constraints` clears task thresholds and returns sliders to the selected starting mixture.

## Caveats

The aggregate score is the current `y_factor` factor aggregate. Task deltas come from a local ridge surrogate over phase weights. Slider readiness should be treated as part of the UI: weakly fitted tasks are not reliable direct steering controls and are better interpreted as guardrails. The full task surrogate quality table remains available in its audit tab.
