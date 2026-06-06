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

## Future work

- [ ] Rerun one corrected Levanter-only v6e diagnostic if clean LM-head and
      sampling attribution is still needed.
