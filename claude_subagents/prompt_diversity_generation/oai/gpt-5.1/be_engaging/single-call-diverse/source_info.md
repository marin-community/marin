# Source info — single-call-diverse scenarios

## Generation method

A **single GPT-5.1 call per statement** returns N+1 scenarios in one JSON
array, where N = number of axes for that statement.

- **Scenario 1 (DEFAULT BASELINE)**: every axis at its
  `default_spectrum_value` (the easy / non-controversial value identified
  in Stage 1).
- **Scenarios 2..N+1 (SINGLE-AXIS VARIATIONS)**: one scenario per axis.
  Each variation moves one axis to a non-default value (the LM picks
  which non-default value); every other axis stays at its default.

The L1 universal prefix + L2 per-statement prefix are identical to the
rubric-default-style strategy (same axes, same defaults, same understanding
record). The difference is the L3 per-call instruction: instead of asking
for one scenario per call, the LM is asked for N+1 scenarios in one
response.

### Critical diversity constraint baked into the prompt

The prompt explicitly requires each of the N+1 scenarios be set in a
completely different real-world context — different domain, persona,
topic, cultural reference. It cites the failure mode of an earlier
strategy that produced 10 scenarios all anchored to one topic, and
instructs the LM to deliberately pick distinct contexts.

The output contains a `context_summary` field per scenario for post-hoc
verification of diversity.

## Settings

- **Model**: gpt-5.1
- **Temperature**: 1.0
- **reasoning_effort**: "none"
- **response_format**: `{"type": "json_object"}`
- **max_completion_tokens**: 16000
- **Mode**: sync

## Shape of each scenario

`statement_id`, `scenario_n`, `scenario_id`, `is_default_scenario`,
`varied_axis`, `varied_value`, `strategy = "single_call_diverse"`,
`scenario_text`, `user_query`, `system_prompt`, `axis_values_embodied`,
`rubric` (good_indicators / bad_indicators / key_tension),
`context_summary` (one-sentence description of the distinct context).

## Volume

N+1 scenarios per statement, where N = number of axes (5-6 for most
statements, so ~6-7 scenarios total).

## Source files (this comparison)

- Scenarios: `experiments/posttrain/disagreement_primitive/diversity_gen/gpt_5_1/stage2_scenarios/scd/<run_id>/scenarios.jsonl` (filtered to this statement)
