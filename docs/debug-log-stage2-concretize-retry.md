# Debugging log for stage2-concretize-retry

Investigate why prompt-generation Stage 2 sometimes returned fewer concretized scenarios than requested, and add enough structure to identify and retry the exact missing covering-array configs.

## Initial status

The `us-central1` prompt-generation smoke run succeeded but logged a Stage 2 shortfall for `ask_clarifying_questions`: one concretization batch returned `3/4` scenarios and the pipeline silently continued with `66` prompts instead of `67`.

At the start of this debug pass:
- Stage 2 prompt output used generic `<scenario>...</scenario>` and `<rubric>...</rubric>` tags.
- Stage 2 parsing in `generate_prompts.py` matched scenarios purely by order.
- The code could tell that a batch was short, but not which covering-array config was missing.
- Missing concretizations were silently dropped before Stage 3 extraction.

## Hypothesis 1

The missing-config identity is being lost because Stage 2 responses are unlabeled. If the model omits or malformed one block, the parser only sees a shorter list and cannot map the gap back to a specific `cfg_*`.

## Changes to make

- Update `lib/marin/src/marin/alignment/prompts/concretize.py` so each requested config carries an explicit `cfg_*` identity and requires indexed output tags like `<scenario_cfg_017>`.
- Update `lib/marin/src/marin/alignment/generate_prompts.py` to:
  - parse concretization responses by config id instead of by position
  - record per-attempt diagnostics including requested ids, returned ids, missing ids, and raw response text
  - retry only the missing configs as singleton requests
  - fail Stage 2 if any configs are still missing after `concretize_max_attempts`
- Plumb `concretize_max_attempts` through `lib/marin/src/marin/alignment/align.py`.
- Add regression coverage in `tests/test_alignment.py` for indexed parsing and a local retry flow that recovers a missing config.

## Future Work

- [ ] Persist partial prompt-generation artifacts even when Stage 2 fails after retries, so failed raw responses survive job failure.
- [ ] Consider capturing prompt text alongside `raw_response` for every concretization attempt if future debugging needs full request/response reconstruction.
- [ ] Consider a dedicated retry budget for rubric-only failures if empty rubrics become common.

## Results

Implemented.

New behavior:
- Stage 2 now requests explicit config-scoped tags such as `<scenario_cfg_000>` and `<rubric_cfg_000>`.
- `generate_prompts.py` now stores `concretization_attempts` in each statement’s `ideation.json`.
- Each attempt record includes:
  - `attempt`
  - `requested_config_ids`
  - `requested_configs`
  - `returned_config_ids`
  - `missing_config_ids`
  - `missing_rubric_config_ids`
  - `raw_response`
- Missing configs are retried as singleton concretization requests up to `concretize_max_attempts`.
- If any config is still missing after the retry budget, Stage 2 raises with the exact unresolved `cfg_*` ids instead of silently dropping them.

Validation:
- `./infra/pre-commit.py --fix lib/marin/src/marin/alignment/generate_prompts.py lib/marin/src/marin/alignment/prompts/concretize.py lib/marin/src/marin/alignment/align.py tests/test_alignment.py`
- `uv run pytest tests/test_alignment.py -q`
- Result: `71 passed`
