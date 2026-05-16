# Source info — legacy-independent scenarios

## Generation method

Single GPT-5.1 call per statement. Settings: model=gpt-5.1, temperature=0,
reasoning_effort=none, response_format=json_object.

Generator script:
`experiments/posttrain/disagreement_primitive/e8_paired_indirection.py::stage2_generate_scenarios`
(line ~348). System prompt: `SCENARIO_GEN_SYSTEM` (line ~109). Per-statement
prompt is just the statement text plus "Generate 20 borderline scenarios per
the schema."

## Shape

Each scenario is `{"user_query": "..."}`. No explicit axis structure. The
system prompt asks the LM to "Generate exactly 20 distinct scenarios" that
are borderline and "span different angles (different user types, contexts,
framings) — not paraphrases of one base case."

## Volume

20 scenarios per statement.

## Source file

`experiments/posttrain/disagreement_primitive/e8_scenarios.jsonl`
