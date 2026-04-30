# GLM-5.1 sampling-noise control

5 independent reruns of `write_cross_tier_rubrics_v2_glm51.py` on the **base spec, no edits**, at temperature=0.2 (the writer's default). Pairwise per-field text-change (`1 - difflib.SequenceMatcher.ratio`) is computed across all 10 run-pairs × 22 (pair_id, tension_point) tasks = **220 datapoints per field**.

- Baseline (single run): `experiments/posttrain/stage3_output/cross_tier_rubrics_v2_glm51.jsonl`
- With-edits R1 (self-edits, single run): `experiments/posttrain/stage3_output/cross_tier_rubrics_v2_glm51_with_self_edits.jsonl`
- Resample files: `cross_tier_rubrics_v2_glm51_resample_{1..5}.jsonl`

## No-edit sampling-noise floor (pairwise across 5 reruns)

| Field | n | mean | p25 | p50 | p75 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dominant_rubric.GOOD` | 220 | 0.778 | 0.687 | 0.818 | 0.897 | 0.952 | 0.987 |
| `dominant_rubric.BAD` | 220 | 0.856 | 0.819 | 0.879 | 0.917 | 0.956 | 0.985 |
| `rationale.alternative_readings_rejected` | 220 | 0.908 | 0.885 | 0.932 | 0.958 | 0.978 | 0.990 |
| `worked_example.spec_compliant` | 220 | 0.769 | 0.676 | 0.807 | 0.895 | 0.964 | 0.989 |

## With-edits Δ (GLM-5.1 baseline vs R1 self-edits, single run each)

| Field | n | mean | p25 | p50 | p75 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dominant_rubric.GOOD` | 22 | 0.776 | 0.691 | 0.838 | 0.909 | 0.952 | 0.955 |
| `dominant_rubric.BAD` | 22 | 0.886 | 0.867 | 0.897 | 0.930 | 0.963 | 0.968 |
| `rationale.alternative_readings_rejected` | 22 | 0.931 | 0.915 | 0.948 | 0.956 | 0.980 | 0.990 |
| `worked_example.spec_compliant` | 22 | 0.821 | 0.762 | 0.854 | 0.925 | 0.969 | 0.992 |

## Signal vs noise: edit-Δ minus noise-floor

| Field | edit-Δ mean | noise mean | gap (edit − noise) | edit-Δ p50 | noise p50 | gap (p50) | ratio (edit-mean / noise-mean) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dominant_rubric.GOOD` | 0.776 | 0.778 | -0.002 | 0.838 | 0.818 | +0.020 | 1.00× |
| `dominant_rubric.BAD` | 0.886 | 0.856 | +0.031 | 0.897 | 0.879 | +0.018 | 1.04× |
| `rationale.alternative_readings_rejected` | 0.931 | 0.908 | +0.023 | 0.948 | 0.932 | +0.016 | 1.03× |
| `worked_example.spec_compliant` | 0.821 | 0.769 | +0.053 | 0.854 | 0.807 | +0.047 | 1.07× |

## Interpretation

The no-edit noise floor measures how much rubric text shifts purely from stochastic decoding (temperature=0.2) when the writer is given the **same** prompt and base spec. The with-edits Δ measures the change between baseline and R1 (self-edits). The gap (edit − noise) is the **edit-attributable signal**; the ratio tells you how many noise floors the with-edits change clears.

If the ratio is near 1× the field is dominated by sampling noise; if it is ≥2× the edit is shifting text well beyond what re-rolling the dice would produce.

