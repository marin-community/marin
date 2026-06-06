# Debugging log for v6e prefill-heavy diagnostics

Investigate why the Levanter-only v6e diagnostic for
`prefill_b8_i2048_o128_n1` reported slower `no_lm_head` and
`lm_head_no_sampling` rows than the normal measured row.

## Initial status

Iris job `/dlwh/qwen3-v6e8-prefilldiag-prefill-b8-i2048-o128-n1-20260606-1504`
succeeded. The normal `levanter:auto` row decoded one `1022`-token iteration,
while the diagnostic rows decoded `510,256,256` token groups. This made the
diagnostic rows unsuitable for direct LM-head or sampling attribution.

## Hypothesis 1

The diagnostic generation path uses a different prefill-drain schedule than
the production `generate()` path. In long-prompt cases split across several
prefill admissions, that makes the diagnostic rows decode after each pending
prefill admission instead of after all currently admissible prefill work is
queued.

## Changes to make

Update `_generate_diagnostic()` in `levanter.inference.engine` so it drains all
pending prefill admissions before each diagnostic decode submission, matching
the production `generate()` scheduling shape. Tighten the diagnostic
multi-prefill engine test to require one decode-token iteration for the small
logical batch.

## Results

Focused validation passed:

- `uv run --package marin-levanter --group test pytest lib/levanter/tests/inference/test_engine.py -q`
  passed with `12 passed`.
- `./infra/pre-commit.py --files lib/levanter/src/levanter/inference/engine.py lib/levanter/tests/inference/test_engine.py docs/debug-log-v6e-prefill-diagnostic.md --fix`
  passed.

## Hypothesis 2

The corrected diagnostic run showed production `levanter:auto` with high
`decode submit s` while `lm_head_no_sampling` had near-zero submit time. Code
inspection found that production `generate()` started the submit timer before
draining pending prefills, while `_generate_diagnostic()` started it immediately
before submitting the diagnostic decode loop. The production metric therefore
included prefill admission work and was not directly comparable to the
diagnostic submit field.

## Changes to make

Start the production `decode_submit_seconds_per_iteration` timer immediately
before `_run_generation_loop(...)`, after pending prefill admissions have been
drained. This preserves serving behavior and fixes metric attribution for
future benchmark rows.

## Future work

- [ ] Rerun one corrected Levanter-only v6e diagnostic if clean LM-head and
      sampling attribution is still needed.
