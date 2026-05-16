# Source info — rubric-default-style scenarios

## Generation method

Two-stage pipeline:

### Stage 1 (understanding) — one call per statement

A single GPT-5.1 call analyzed the spec statement (text + spec examples)
and produced `behavior_understanding`, `scientific_motivation`, and a list
of `behavior_specific_axes` (4-6 axes), each with axis name, description,
spectrum (4-6 monotonically-ordered values), `default_spectrum_value`
(the easy / non-controversial value), and `why_it_matters`. The full
Stage 1 record for this statement is in `stage1_understanding.json` in
this directory.

### Stage 2 (scenario generation) — one call per scenario

- **scenario 0 (default baseline)**: every axis at its
  `default_spectrum_value`.
- **scenarios 1..N (single-axis variations)**: for each axis A and each
  non-default value v in A's spectrum, one scenario where A = v and every
  other axis stays at its default.

Total per statement = 1 + sum over axes of (spectrum_size_i - 1).

Settings: model=gpt-5.1, temperature=1.0, reasoning_effort=none,
response_format=json_object, mode=sync.

## Shape of each scenario

Each is a JSON object with these fields: `statement_id`, `scenario_n`,
`scenario_id`, `is_default_scenario`, `varied_axis`, `varied_value`,
`scenario_text`, `user_query`, `system_prompt`, `axis_values_embodied`,
`rubric` (with `good_indicators`, `bad_indicators`, `key_tension`),
plus model metadata.

## Volume

25 scenarios for `be_clear` (1 default + 24 variations).

## Source files

- Scenarios: `experiments/posttrain/disagreement_primitive/diversity_gen/gpt_5_1/stage2_scenarios/20260516T174023Z/scenarios.jsonl` (filtered to this statement)
- Stage 1 understanding: `experiments/posttrain/disagreement_primitive/diversity_gen/gpt_5_1/stage1_understanding/20260516T172804Z/understandings.jsonl` (filtered to this statement)
